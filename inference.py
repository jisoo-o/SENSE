import os
import glob
import json
import tempfile
import utils
import argparse
import torch
import torch.nn as nn
import commons
from data_utils import (
  EEGAudioLoader,
  EEGAudioCollate
)
from torch.utils.data import DataLoader
from EEGModule import EEGModule
from models import SpeechDecoder
from scipy.io.wavfile import write

OUTPUT_DIR = './logs'


def _get_per_subject_lines(filelist_path):
    """Parse a filelist and group lines by subject ID (first path component, e.g. 'sub-23')."""
    lines = utils.load_filepaths_and_eeg(filelist_path)
    subjects = {}
    for line in lines:
        sub_id = line.split('/')[0]
        subjects.setdefault(sub_id, []).append(line)
    return subjects


def _make_loader_from_lines(lines, hps_data, collate_fn):
    """Create an EEGAudioLoader DataLoader from an in-memory list of filelist lines."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write('\n'.join(lines))
        temp_path = f.name
    dataset = EEGAudioLoader(temp_path, hps_data)
    os.unlink(temp_path)
    return DataLoader(dataset, num_workers=1, shuffle=False, batch_size=1,
                      pin_memory=True, drop_last=False, collate_fn=collate_fn)


def _find_checkpoint(run_name, checkpoint_idx, prefix):
    """Find checkpoint file, supporting both flat and subdir layouts.

    Flat:   logs/{run_name}/{prefix}_{checkpoint_idx}.pth
    Subdir: logs/{run_name}/{checkpoint_idx}/{prefix}_*.pth  (milestone format)
    """
    flat = os.path.join('./logs', run_name, f"{prefix}_{checkpoint_idx}.pth")
    if os.path.exists(flat):
        return flat
    subdir = os.path.join('./logs', run_name, str(checkpoint_idx))
    if os.path.isdir(subdir):
        matches = sorted(glob.glob(os.path.join(subdir, f"{prefix}_*.pth")))
        if matches:
            return matches[-1]
    raise FileNotFoundError(
        f"Checkpoint not found: tried {flat} and {subdir}/{prefix}_*.pth"
    )


def synthesize(eeg_enc, audio_gen, eval_loader, suffix, hps, output_key, run_name, cfg_scale=0.0, eeg_sem_proj=None, mask_indices=None):

    eeg_enc.eval()
    audio_gen.eval()

    texts_map = {}

    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, _texts, _clip_embs, _is_congruent, _phoneme, _phoneme_lengths) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)

            # Channel masking: zero out specified channels before EEG encoding
            if mask_indices is not None:
                x[:, mask_indices, :] = 0.0

            x, x_mask_output, mid_output, eeg_decoder_output = eeg_enc(x)

            mid_output_lengths = x_lengths.clone() * mid_output.size(2) / x.size(2)
            mid_output_lengths = mid_output_lengths.long()

            # Optional: pass CLIP embeddings for CFG-guided generation
            c_text = None
            if cfg_scale > 0.0:
                c_text = _clip_embs.cuda(0)

            # ESC: compute EEG-derived semantic conditioning vector
            g_eeg = None
            if eeg_sem_proj is not None:
                aud_mask = commons.sequence_mask(mid_output_lengths, mid_output.size(2)).unsqueeze(1).float()
                aud_pooled = (mid_output * aud_mask).sum(2) / aud_mask.sum(2).clamp(min=1)
                g_eeg = eeg_sem_proj(aud_pooled).unsqueeze(-1)  # [B, inter_channels, 1]

            y_hat, _, mask, *_ = audio_gen.infer(mid_output, mid_output_lengths, max_len=1000, noise_scale=1,
                                                  c_text=c_text, cfg_scale=cfg_scale, g_eeg=g_eeg)

            out_dir = os.path.join(OUTPUT_DIR, run_name, 'synthesized', output_key, suffix)
            os.makedirs(out_dir, exist_ok=True)

            write(os.path.join(out_dir, f'{batch_idx}_gt.wav'), hps.data.sampling_rate, y[0,:,:y_hat.size(2)][0].cpu().float().numpy())
            write(os.path.join(out_dir, f'{batch_idx}_syn.wav'), hps.data.sampling_rate, y_hat[0,:,:y_hat.size(2)][0].cpu().float().numpy())

            texts_map[str(batch_idx)] = {
                'text': _texts[0],
                'congruent': bool(_is_congruent[0].item())
            }

    out_dir = os.path.join(OUTPUT_DIR, run_name, 'synthesized', output_key, suffix)
    with open(os.path.join(out_dir, 'texts.json'), 'w') as f:
        json.dump(texts_map, f, indent=2)


def run(args):
    seed = getattr(args, 'seed', 777)
    torch.manual_seed(seed)

    output_tag = getattr(args, 'output_tag', None)
    output_key = f"{args.checkpoint_idx}/{output_tag}" if output_tag else str(args.checkpoint_idx)

    config_dir = os.path.join('./logs', args.run_name, 'config.json')
    hps = utils.get_hparams_from_file(config_dir)
    print(hps)

    collate_fn = EEGAudioCollate()

    eval_loader_both = DataLoader(EEGAudioLoader(hps.data.validation_files_both, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    eval_loader_audio = DataLoader(EEGAudioLoader(hps.data.validation_files_audio, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    eval_loader_subject = DataLoader(EEGAudioLoader(hps.data.validation_files_subject, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

    eval_loaders = [eval_loader_both, eval_loader_audio, eval_loader_subject]

    eeg_module = EEGModule(
        n_layers_cnn=hps.model.eeg_module.n_layers_cnn,
        use_s4=hps.model.eeg_module.use_s4,
        n_layers_s4=hps.model.eeg_module.n_layers_s4,
        embedding_size=hps.model.inter_channels,
        is_mask=False,
        in_channels=hps.model.eeg_module.in_channels,
        use_channel_attention=getattr(hps.model.eeg_module, 'use_channel_attention', False),
        use_n400_prior=getattr(hps.model.eeg_module, 'use_n400_prior', False),
        acoustic_prior_indices=getattr(hps.model.eeg_module, 'acoustic_prior_indices', None),
        prior_weight=getattr(hps.model.eeg_module, 'prior_weight', 2.0),
        use_gnn=getattr(hps.model.eeg_module, 'use_gnn', False),
        graph_type=getattr(hps.model.eeg_module, 'graph_type', 'fixed'),
        d_node=getattr(hps.model.eeg_module, 'd_node', 64),
        n_gat_heads=getattr(hps.model.eeg_module, 'n_gat_heads', 4),
        n_gat_layers_gnn=getattr(hps.model.eeg_module, 'n_gat_layers_gnn', 2),
        use_gat_attention=getattr(hps.model.eeg_module, 'use_gat_attention', False),
        use_gnn_skip=getattr(hps.model.eeg_module, 'use_gnn_skip', False),
    ).cuda()

    net_g = SpeechDecoder(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()

    utils.load_checkpoint(_find_checkpoint(args.run_name, args.checkpoint_idx, "G"), net_g, None)
    utils.load_checkpoint(_find_checkpoint(args.run_name, args.checkpoint_idx, "E"), eeg_module, None)

    # Load ESC eeg_sem_proj if checkpoint exists (v9+)
    eeg_sem_proj = None
    try:
        s_path = _find_checkpoint(args.run_name, args.checkpoint_idx, "S")
        esc_ckpt = torch.load(s_path, map_location='cpu', weights_only=False)
        if 'eeg_sem_proj' in esc_ckpt:
            eeg_sem_proj = nn.Linear(hps.model.inter_channels, hps.model.inter_channels).cuda()
            eeg_sem_proj.load_state_dict(esc_ckpt['eeg_sem_proj'])
            eeg_sem_proj.eval()
            print(f"[ESC] Loaded eeg_sem_proj from {s_path}")
    except FileNotFoundError:
        pass

    net_g.eval()
    eeg_module.eval()

    cfg_scale = getattr(args, 'cfg_scale', 0.0)

    # Parse channel mask indices (comma-separated string → list of ints)
    mask_indices = None
    raw_mask = getattr(args, 'mask_channel_indices', None)
    if raw_mask:
        mask_indices = [int(i) for i in raw_mask.split(',')]
        print(f"[Mask] Zeroing out {len(mask_indices)} channels: {mask_indices}")

    for val_type, eval_loader in list(zip(['both', 'audio', 'subject'], eval_loaders)):
        synthesize(eeg_module, net_g, eval_loader, val_type, hps, output_key, args.run_name, cfg_scale=cfg_scale, eeg_sem_proj=eeg_sem_proj, mask_indices=mask_indices)

    # Per-subject synthesis: creates subject_{sub_id}/ dirs alongside the combined subject/ dir
    if getattr(args, 'per_subject', False):
        subject_lines = _get_per_subject_lines(hps.data.validation_files_subject)
        for sub_id in sorted(subject_lines.keys()):
            print(f"[Per-subject] Synthesizing for {sub_id} ({len(subject_lines[sub_id])} samples)")
            sub_loader = _make_loader_from_lines(subject_lines[sub_id], hps.data, collate_fn)
            synthesize(eeg_module, net_g, sub_loader, f"subject_{sub_id}", hps, output_key,
                       args.run_name, cfg_scale=cfg_scale, eeg_sem_proj=eeg_sem_proj, mask_indices=mask_indices)

    return output_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="fesde")
    parser.add_argument('--checkpoint_idx', default=0)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--cfg_scale', type=float, default=0.0,
                        help='CFG scale for guided generation (0=EEG only, >0=use CLIP text)')
    parser.add_argument('--output_tag', type=str, default=None,
                        help='Subdirectory tag for synthesized output (default: checkpoint_idx only)')
    parser.add_argument('--mask_channel_indices', type=str, default=None,
                        help='Comma-separated EEG channel indices to zero out, e.g. "42,53,114"')
    parser.add_argument('--per_subject', action='store_true',
                        help='Also synthesize per-subject splits from validation_files_subject '
                             '(creates subject_{sub_id}/ dirs alongside the combined subject/ dir)')
    args = parser.parse_args()

    run(args)
