import os
import json
import argparse
import itertools
import math
from datetime import timedelta
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.io.wavfile import write


import numpy as np
import librosa
import tempfile
from pymcd.mcd import Calculate_MCD

import commons
import utils
from data_utils import EEGAudioLoader, EEGAudioCollate
from models import SpeechDecoder, MultiPeriodDiscriminator, PhonemePredictor
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
  eeg_loss,
  InfoNCELoss,
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from EEGModule import EEGModule

import wandb

torch.backends.cudnn.benchmark = True
global_step = 0
best_eval_loss = float('inf')
patience_counter = 0

import warnings
warnings.filterwarnings('ignore')


def get_loss_weight(base_weight, global_step, warmup_steps):
  """Linear warmup from 0 to base_weight over warmup_steps."""
  if warmup_steps <= 0:
    return base_weight
  return base_weight * min(1.0, global_step / warmup_steps)



def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = int(os.environ.get('WORLD_SIZE', 1))
  rank = int(os.environ.get('LOCAL_RANK', 0))

  hps = utils.get_hparams()

  if 'MASTER_ADDR' not in os.environ:
    os.environ['MASTER_ADDR'] = 'localhost'
  if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = hps.port

  run(rank=rank, n_gpus=n_gpus, hps=hps)


def run(rank, n_gpus, hps):
  global global_step, best_eval_loss, patience_counter

  if hps.wandb and rank == 0:
    wandb.login()
    wandb.init(project='sense', name=hps.run_name)

  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank, timeout=timedelta(minutes=60))
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = EEGAudioLoader(hps.data.training_files, hps.data)
  collate_fn = EEGAudioCollate()
  train_sampler = DistributedSampler(train_dataset, num_replicas=n_gpus, rank=rank, shuffle=True)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_size=hps.train.batch_size, sampler=train_sampler)
  if rank == 0:
    eval_loader_both = DataLoader(EEGAudioLoader(hps.data.validation_files_both, hps.data), num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    eval_loader_audio = DataLoader(EEGAudioLoader(hps.data.validation_files_audio, hps.data), num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    eval_loader_subject = DataLoader(EEGAudioLoader(hps.data.validation_files_subject, hps.data), num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
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
    skip_gate_init=getattr(hps.model.eeg_module, 'skip_gate_init', 0.0),
    normalize_gnn_skip=getattr(hps.model.eeg_module, 'normalize_gnn_skip', False),
  ).cuda(rank)

  net_g = SpeechDecoder(
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

  # Phoneme predictor (CTC loss auxiliary task on EEG encoder output)
  net_p = None
  if getattr(hps.model, 'use_phoneme_predictor', False):
    net_p = PhonemePredictor(
        num_layers=getattr(hps.model, 'num_conformer_layers', 1),
        num_class=getattr(hps.model, 'vocab_size', 51),
        encoder_dim=hps.model.inter_channels,
    ).cuda(rank)

  # Auxiliary: z_aud → CLIP alignment projection
  use_semantic_loss = getattr(hps.train, 'use_semantic_loss', False)
  aud_proj = None
  if use_semantic_loss:
    aud_proj = nn.Linear(hps.model.inter_channels, 512).cuda(rank)

  # ESC: EEG semantic conditioning projection (inter_channels → inter_channels)
  # Aligns pooled EEG representation with VITS conditioning space for inference-time use
  use_eeg_sem_cond = getattr(hps.train, 'use_eeg_sem_cond', False)
  eeg_sem_proj = None
  if use_eeg_sem_cond:
    eeg_sem_proj = nn.Linear(hps.model.inter_channels, hps.model.inter_channels).cuda(rank)

  aux_params = list(aud_proj.parameters()) if aud_proj is not None else []
  phone_params = list(net_p.parameters()) if net_p is not None else []
  sem_params = list(eeg_sem_proj.parameters()) if eeg_sem_proj is not None else []

  optim_g = torch.optim.AdamW(
      net_g.parameters(),
      hps.train.learning_rate,
      betas=hps.train.betas,
      eps=hps.train.eps)
  optim_eeg = torch.optim.AdamW(
      list(eeg_module.parameters()) + aux_params + phone_params + sem_params,
      hps.train.learning_rate,
      betas=hps.train.betas,
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate,
      betas=hps.train.betas,
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
  eeg_module = DDP(eeg_module, device_ids=[rank], find_unused_parameters=True)
  if net_p is not None:
    net_p = DDP(net_p, device_ids=[rank], find_unused_parameters=True)

  if hps.train.pretrained_audio:
    print('loading pretrained audio generator from ' + hps.train.pretrained_audio)
    pretrained_audio_ckpt = torch.load(hps.train.pretrained_audio, map_location='cpu', weights_only=False)
    pretrained_stated_dict = pretrained_audio_ckpt['model']
    new_state_dict = {}
    for k, v in net_g.module.state_dict().items():
      if k.split('.')[0] != 'enc_proj':
        new_state_dict[k] = pretrained_stated_dict[k]
      else:
        new_state_dict[k] = v
    net_g.module.load_state_dict(new_state_dict)

    print('finished loading pretrained audio generator')


    if hps.train.freeze_modules:
      if "dec" in hps.train.freeze_modules:
        for param in net_g.module.dec.parameters():
          param.requires_grad = False
      if "enc_q" in hps.train.freeze_modules:
        for param in net_g.module.enc_q.parameters():
          param.requires_grad = False
      if "flow" in hps.train.freeze_modules:
        for param in net_g.module.flow.parameters():
          param.requires_grad = False
      if "eeg" in hps.train.freeze_modules:
        for param in eeg_module.module.parameters():
          param.requires_grad = False
          print('freezing eeg module')

  if hps.train.pretrained_eeg:
    print('loading pretrained eeg module', hps.train.pretrained_eeg)
    _ = utils.load_checkpoint(hps.train.pretrained_eeg, eeg_module, None)
    if "eeg" in hps.train.freeze_modules:
      for param in eeg_module.module.parameters():
        param.requires_grad = False
        print('freezing eeg module')

  # Load pretrained ESC projection (e.g., from Stage 1 Broderick pretrain)
  pretrained_esc = getattr(hps.train, 'pretrained_esc', '')
  if pretrained_esc and os.path.isfile(pretrained_esc) and eeg_sem_proj is not None:
    print('loading pretrained ESC projection', pretrained_esc)
    esc_ckpt = torch.load(pretrained_esc, map_location='cpu', weights_only=False)
    if 'eeg_sem_proj' in esc_ckpt:
      eeg_sem_proj.load_state_dict(esc_ckpt['eeg_sem_proj'])
      print('  loaded eeg_sem_proj from pretrained ESC')

  epoch_str = 1
  global_step = 0

  if True: # continue from past run
    try:
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "E_*.pth"), eeg_module, optim_eeg)
      global_step = (epoch_str - 1) * len(train_loader)
    except:
      pass
    # Load aud_proj checkpoint
    try:
      if aud_proj is not None:
        aux_ckpt_path = utils.latest_checkpoint_path(hps.model_dir, "A_*.pth")
        aux_ckpt = torch.load(aux_ckpt_path, map_location='cpu', weights_only=False)
        if 'aud_proj' in aux_ckpt:
          aud_proj.load_state_dict(aux_ckpt['aud_proj'])
    except:
      pass
    # Load phoneme predictor checkpoint
    try:
      if net_p is not None:
        utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "P_*.pth"), net_p, None)
    except:
      pass
    # Load eeg_sem_proj checkpoint (ESC)
    try:
      if eeg_sem_proj is not None:
        esc_path = utils.latest_checkpoint_path(hps.model_dir, "S_*.pth")
        esc_ckpt = torch.load(esc_path, map_location='cpu', weights_only=False)
        if 'eeg_sem_proj' in esc_ckpt:
          eeg_sem_proj.load_state_dict(esc_ckpt['eeg_sem_proj'])
    except:
      pass
    try:
      meta_path = os.path.join(hps.model_dir, "training_meta.pth")
      if os.path.isfile(meta_path):
        meta = torch.load(meta_path, map_location='cpu', weights_only=False)
        best_eval_loss = meta['best_eval_loss']
        patience_counter = meta['patience_counter']
    except:
      pass


  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_eeg = torch.optim.lr_scheduler.ExponentialLR(optim_eeg, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler('cuda', enabled=False)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    train_sampler.set_epoch(epoch)
    if rank==0:
      should_stop = train_and_evaluate(rank, epoch, hps, [net_g, net_d, eeg_module], [optim_g, optim_d, optim_eeg], [scheduler_g, scheduler_d, scheduler_eeg], scaler, [train_loader, eval_loaders], logger, [writer, writer_eval], aud_proj=aud_proj, net_p=net_p, eeg_sem_proj=eeg_sem_proj)
    else:
      should_stop = train_and_evaluate(rank, epoch, hps, [net_g, net_d, eeg_module], [optim_g, optim_d, optim_eeg], [scheduler_g, scheduler_d, scheduler_eeg], scaler, [train_loader, None], None, None, aud_proj=aud_proj, net_p=net_p, eeg_sem_proj=eeg_sem_proj)
    stop_tensor = torch.tensor([1 if should_stop else 0], device='cuda')
    dist.broadcast(stop_tensor, src=0)
    if stop_tensor.item() == 1:
      if rank == 0:
        logger.info("Training stopped by early stopping.")
      break
    scheduler_g.step()
    scheduler_d.step()
    scheduler_eeg.step()

  dist.destroy_process_group()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, aud_proj=None, net_p=None, eeg_sem_proj=None):
  net_g, net_d, eeg_module = nets
  optim_g, optim_d, optim_eeg = optims
  scheduler_g, scheduler_d, scheduler_eeg = schedulers
  train_loader = loaders[0]
  if writers is not None:
    writer, writer_eval = writers

  global global_step, best_eval_loss, patience_counter
  early_stopping_patience = getattr(hps.train, 'early_stopping_patience', None)
  use_semantic_loss = getattr(hps.train, 'use_semantic_loss', False)
  use_eeg_sem_cond = getattr(hps.train, 'use_eeg_sem_cond', False)
  semantic_loss_type = getattr(hps.train, 'semantic_loss_type', 'infonce')
  infonce_loss_fn = None
  if use_semantic_loss and semantic_loss_type == 'infonce':
    infonce_loss_fn = InfoNCELoss(
      embed_dim=512,
      bank_size=getattr(hps.train, 'infonce_bank_size', 256),
      temperature=getattr(hps.train, 'infonce_temperature', 0.07)
    ).cuda(rank)
  warmup_steps = getattr(hps.train, 'warmup_steps', 0)
  unfreeze_eeg_step = getattr(hps.train, 'unfreeze_eeg_step', 0)
  _should_stop = False

  net_g.train()
  net_d.train()
  eeg_module.train()
  if net_p is not None:
    net_p.train()

  congruency_aware = getattr(hps.train, 'congruency_aware_sem', True)

  _eeg_unfrozen = "eeg" not in getattr(hps.train, 'freeze_modules', [])
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, _texts, _clip_embs, _is_congruent, phoneme, phoneme_lengths) in enumerate(train_loader):
    # Gradual unfreeze: unfreeze EEG encoder after unfreeze_eeg_step
    if not _eeg_unfrozen and unfreeze_eeg_step > 0 and global_step >= unfreeze_eeg_step:
      for param in eeg_module.module.parameters():
        param.requires_grad = True
      hps.train.freeze_modules = [m for m in hps.train.freeze_modules if m != "eeg"]
      _eeg_unfrozen = True
      if rank == 0:
        logger.info(f"Unfreezing EEG encoder at step {global_step}")

    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    phoneme = phoneme.cuda(rank, non_blocking=True)
    phoneme_lengths = phoneme_lengths.cuda(rank, non_blocking=True)

    with autocast('cuda', enabled=False):
      x, x_mask_output, mid_output, eeg_decoder_out = eeg_module(x)
      mid_output_lengths = x_lengths.clone() * mid_output.size(2) / x.size(2)
      mid_output_lengths = mid_output_lengths.long()

      # CTC loss via phoneme predictor (auxiliary task on EEG encoder output)
      loss_ctc = torch.tensor(0.0, device=x.device)
      if net_p is not None:
        phoneme_preds = net_p(mid_output, mid_output_lengths, phoneme)
        loss_ctc = torch.nn.CTCLoss(zero_infinity=True).cuda()(
            phoneme_preds.transpose(0, 1).log_softmax(2),
            phoneme[:, 1:],
            phoneme_lengths - 1,
            phoneme_lengths - 1,
        )

      # Pass CLIP text embeddings as conditioning (c_text)
      clip_embs = _clip_embs.cuda(rank, non_blocking=True)

      # Shuffled CLIP control: randomly permute CLIP targets within batch
      # so EEG-semantic alignment trains against wrong sentences.
      # Generator conditioning (c_text) uses original clip_embs;
      # only alignment/ESC losses use shuffled version.
      shuffle_clip = getattr(hps.train, 'shuffle_clip_targets', False)
      if shuffle_clip:
        perm = torch.randperm(clip_embs.size(0), device=clip_embs.device)
        clip_embs_shuffled = clip_embs[perm]
      else:
        clip_embs_shuffled = clip_embs

      _eeg_input = mid_output if not getattr(hps.train, 'detach_eeg', True) else mid_output.detach()

      # --- v11: EEG-conditioning of generator (optional) ---
      # With probability eeg_cond_prob, replace CLIP conditioning with EEG semantic vector
      # so the decoder learns to handle g_eeg and it is in-distribution at inference.
      g_override = None
      eeg_cond_prob = getattr(hps.train, 'eeg_cond_prob', 0.0)
      if use_eeg_sem_cond and eeg_sem_proj is not None and eeg_cond_prob > 0.0:
        if torch.rand(1).item() < eeg_cond_prob:
          with torch.no_grad():
            _aud_lengths_gc = (x_lengths.float() * mid_output.size(2) / x.size(2)).long()
            _aud_mask_gc = commons.sequence_mask(_aud_lengths_gc, mid_output.size(2)).unsqueeze(1).float()
            _aud_pooled_gc = (mid_output.detach() * _aud_mask_gc).sum(dim=2) / _aud_mask_gc.sum(dim=2).clamp(min=1)
          g_override = eeg_sem_proj(_aud_pooled_gc).detach().unsqueeze(-1)  # [B, C, 1]

      y_hat, _, _, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(_eeg_input, mid_output_lengths, spec, spec_lengths,
                                                   c_text=clip_embs, g_override=g_override)

      mel = spec_to_mel_torch(
          spec,
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.mel_fmin,
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1),
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.hop_length,
          hps.data.win_length,
          hps.data.mel_fmin,
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast('cuda', enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast('cuda', enabled=False):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast('cuda', enabled=False):
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        loss_eeg_module = eeg_loss(x, eeg_decoder_out, x_lengths)

        # --- Congruency mask (NPC=True / NPI=False) ---
        is_cong = _is_congruent.cuda(rank, non_blocking=True) if congruency_aware \
                  else torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        # --- L_align: z_aud → projection → CLIP space (InfoNCE, congruent only) ---
        loss_align = torch.tensor(0.0, device=x.device)
        if use_semantic_loss and aud_proj is not None:
          aud_lengths = (x_lengths.float() * mid_output.size(2) / x.size(2)).long()
          aud_mask = commons.sequence_mask(aud_lengths, mid_output.size(2)).unsqueeze(1)
          aud_pooled = (mid_output * aud_mask).sum(dim=2) / aud_mask.sum(dim=2).clamp(min=1)
          aud_projected = aud_proj(aud_pooled)
          c_sem_current = get_loss_weight(hps.train.c_sem, global_step, warmup_steps)
          if is_cong.any():
            proj_cong = aud_projected[is_cong]
            clip_cong = clip_embs_shuffled[is_cong] if shuffle_clip else clip_embs[is_cong]
            if semantic_loss_type == 'infonce':
              loss_align = infonce_loss_fn(proj_cong, clip_cong) * c_sem_current
            else:
              # Cosine distance fallback
              proj_norm = F.normalize(proj_cong, dim=-1)
              clip_norm = F.normalize(clip_cong, dim=-1)
              loss_align = (1 - (proj_norm * clip_norm).sum(dim=-1)).mean() * c_sem_current
          loss_eeg_module = loss_eeg_module + loss_align

        # --- ESC: EEG Semantic Conditioning alignment ---
        # Aligns eeg_sem_proj(pool(mid_output)) with clip_proj(clip_embs) in 192-dim
        # conditioning space. Cosine distance (stable), congruent pairs only.
        # At inference, eeg_sem_proj output replaces null_embedding to condition generation.
        loss_esc = torch.tensor(0.0, device=x.device)
        if use_eeg_sem_cond and eeg_sem_proj is not None:
          if aud_proj is None:
            # aud_pooled not yet computed (use_semantic_loss=False); compute here
            aud_lengths = (x_lengths.float() * mid_output.size(2) / x.size(2)).long()
            aud_mask = commons.sequence_mask(aud_lengths, mid_output.size(2)).unsqueeze(1).float()
            aud_pooled = (mid_output * aud_mask).sum(dim=2) / aud_mask.sum(dim=2).clamp(min=1)
          g_eeg_vec = eeg_sem_proj(aud_pooled)                       # [B, inter_channels]
          _clip_for_esc = clip_embs_shuffled if shuffle_clip else clip_embs
          g_clip_tgt = net_g.module.clip_proj(_clip_for_esc).detach()    # [B, inter_channels]
          c_esc = getattr(hps.train, 'c_esc', 0.5)
          c_esc_norm = getattr(hps.train, 'c_esc_norm', 0.0)
          if is_cong.any():
            g_eeg_cong = g_eeg_vec[is_cong]
            g_cli_cong = g_clip_tgt[is_cong]
            g_eeg_n = F.normalize(g_eeg_cong, dim=-1)
            g_cli_n = F.normalize(g_cli_cong, dim=-1)
            loss_cos = (1 - (g_eeg_n * g_cli_n).sum(-1)).mean()
            # Norm matching: align L2 scale of g_eeg with g_clip so decoder conditioning
            # distribution matches what the flow/decoder was trained to expect
            loss_norm_match = F.mse_loss(g_eeg_cong.norm(dim=-1), g_cli_cong.norm(dim=-1))
            loss_esc = (loss_cos + c_esc_norm * loss_norm_match) * c_esc
          loss_eeg_module = loss_eeg_module + loss_esc

        c_ctc = getattr(hps.train, 'c_ctc', 0.3)
        loss_eeg_phone = loss_eeg_module + c_ctc * loss_ctc

    optim_g.zero_grad()
    _detach_eeg = getattr(hps.train, 'detach_eeg', True)
    scaler.scale(loss_gen_all).backward(retain_graph=not _detach_eeg)
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)

    optim_eeg.zero_grad()
    if "eeg" not in hps.train.freeze_modules:
      scaler.scale(loss_eeg_phone).backward()
    scaler.unscale_(optim_eeg)
    grad_norm_eeg_enc = commons.clip_grad_value_(eeg_module.parameters(), None)
    scaler.step(optim_eeg)

    scaler.update()

    # Sync early stopping decision BEFORE eval (all ranks participate, prevents NCCL timeout)
    if global_step % hps.train.eval_interval == 0:
      stop_tensor = torch.zeros(1, device=f'cuda:{rank}')
      if _should_stop:
        stop_tensor.fill_(1)
      dist.broadcast(stop_tensor, src=0)
      if stop_tensor.item() == 1:
        return True

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])

        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g, "loss/eeg/total":loss_eeg_module
                       }
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl})
        if use_semantic_loss:
          scalar_dict["loss/eeg/align"] = loss_align
        if net_p is not None:
          scalar_dict["loss/eeg/ctc"] = loss_ctc
        if use_eeg_sem_cond:
          scalar_dict["loss/eeg/esc"] = loss_esc

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = {
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }
        utils.summarize(
          writer=writer,
          global_step=global_step,
          images=image_dict,
          scalars=scalar_dict)

        if hps.wandb:
          wandb.log(scalar_dict)
          image_dict_wandb = {k:wandb.Image(v, caption=k) for k,v in image_dict.items()}
          wandb.log(image_dict_wandb)

      if global_step % hps.train.eval_interval == 0:
        eval_losses = []
        eval_mcds = []
        eval_mel_corrs = []
        for val_type, eval_loader in list(zip(['both', 'audio', 'subject'], loaders[1])):
          eval_loss, eval_mcd, eval_mel_corr = evaluate(hps, [net_g, eeg_module], eval_loader, writer_eval, val_type)
          eval_losses.append(eval_loss)
          eval_mcds.append(eval_mcd)
          eval_mel_corrs.append(eval_mel_corr)
          logger.info(f"Eval [{val_type}] mel_loss={eval_loss:.4f} MCD={eval_mcd:.4f} Mel-Corr={eval_mel_corr:.4f}")
        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        avg_mcd = sum(eval_mcds) / len(eval_mcds)
        avg_mel_corr = sum(eval_mel_corrs) / len(eval_mel_corrs)
        logger.info(f"Eval [avg] mel_loss={avg_eval_loss:.4f} MCD={avg_mcd:.4f} Mel-Corr={avg_mel_corr:.4f}")

        if hps.wandb:
          for val_type, mcd_val, corr_val in zip(['both', 'audio', 'subject'], eval_mcds, eval_mel_corrs):
            wandb.log({f"eval/{val_type}/mcd": mcd_val, f"eval/{val_type}/mel_corr": corr_val, "global_step": global_step})
          wandb.log({"eval/avg/mcd": avg_mcd, "eval/avg/mel_corr": avg_mel_corr, "eval/avg/mel_loss": avg_eval_loss, "global_step": global_step})

        full_eval_interval = getattr(hps.train, 'full_eval_interval', 0)
        if full_eval_interval and global_step % full_eval_interval == 0:
          full_eval_losses = []
          full_eval_mel_corrs = []
          for val_type, eval_loader in list(zip(['both', 'audio', 'subject'], loaders[1])):
            fe_loss, _, fe_corr = evaluate(hps, [net_g, eeg_module], eval_loader, writer_eval, val_type, full_eval=True)
            full_eval_losses.append(fe_loss)
            full_eval_mel_corrs.append(fe_corr)
            logger.info(f"FullEval [{val_type}] mel_loss={fe_loss:.4f} Mel-Corr={fe_corr:.4f}")
          avg_fe_loss = sum(full_eval_losses) / len(full_eval_losses)
          avg_fe_corr = sum(full_eval_mel_corrs) / len(full_eval_mel_corrs)
          logger.info(f"FullEval [avg] mel_loss={avg_fe_loss:.4f} Mel-Corr={avg_fe_corr:.4f}")
          if hps.wandb:
            for val_type, fe_loss, fe_corr in zip(['both', 'audio', 'subject'], full_eval_losses, full_eval_mel_corrs):
              wandb.log({f"full_eval/{val_type}/mel_loss": fe_loss, f"full_eval/{val_type}/mel_corr": fe_corr, "global_step": global_step})
            wandb.log({"full_eval/avg/mel_loss": avg_fe_loss, "full_eval/avg/mel_corr": avg_fe_corr, "global_step": global_step})

          if avg_fe_loss < best_eval_loss:
            best_eval_loss = avg_fe_loss
            patience_counter = 0
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_best.pth"))
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_best.pth"))
            utils.save_checkpoint(eeg_module, optim_eeg, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "E_best.pth"))
            if aud_proj is not None:
              torch.save({'aud_proj': aud_proj.state_dict()},
                         os.path.join(hps.model_dir, "A_best.pth"))
            if net_p is not None:
              utils.save_checkpoint(net_p, None, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "P_best.pth"))
            if eeg_sem_proj is not None:
              torch.save({'eeg_sem_proj': eeg_sem_proj.state_dict()},
                         os.path.join(hps.model_dir, "S_best.pth"))
            logger.info(f"New best (full eval) mel_loss: {best_eval_loss:.4f} at step {global_step}")
          else:
            patience_counter += 1
            logger.info(f"No improvement (full eval). Patience: {patience_counter}/{early_stopping_patience}")

          torch.save({'best_eval_loss': best_eval_loss, 'patience_counter': patience_counter},
                     os.path.join(hps.model_dir, "training_meta.pth"))

        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        utils.save_checkpoint(eeg_module, optim_eeg, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "E_{}.pth".format(global_step)))
        if aud_proj is not None:
          torch.save({'aud_proj': aud_proj.state_dict()},
                     os.path.join(hps.model_dir, "A_{}.pth".format(global_step)))
        if net_p is not None:
          utils.save_checkpoint(net_p, None, hps.train.learning_rate, epoch,
                                os.path.join(hps.model_dir, "P_{}.pth".format(global_step)))
        if eeg_sem_proj is not None:
          torch.save({'eeg_sem_proj': eeg_sem_proj.state_dict()},
                     os.path.join(hps.model_dir, f"S_{global_step}.pth"))
        utils.cleanup_checkpoints(hps.model_dir, prefixes=("G_", "D_", "E_", "A_", "P_", "S_"), keep_recent=3)

        milestone_steps = getattr(hps.train, 'milestone_steps', [])
        if global_step in milestone_steps:
          milestone_dir = os.path.join(hps.model_dir, f"milestone_{global_step}")
          os.makedirs(milestone_dir, exist_ok=True)
          import shutil
          for prefix in ["G", "D", "E"]:
            shutil.copy2(os.path.join(hps.model_dir, f"{prefix}_{global_step}.pth"), milestone_dir)
          if aud_proj is not None:
            shutil.copy2(os.path.join(hps.model_dir, f"A_{global_step}.pth"), milestone_dir)
          if net_p is not None:
            shutil.copy2(os.path.join(hps.model_dir, f"P_{global_step}.pth"), milestone_dir)
          if eeg_sem_proj is not None:
            shutil.copy2(os.path.join(hps.model_dir, f"S_{global_step}.pth"), milestone_dir)
          logger.info(f"Milestone checkpoint saved at step {global_step} -> {milestone_dir}")

        if early_stopping_patience and patience_counter >= early_stopping_patience:
          logger.info(f"Early stopping at step {global_step}. Best eval loss: {best_eval_loss:.4f}")
          _should_stop = True

        max_steps = getattr(hps.train, 'max_steps', 0)
        if max_steps and global_step >= max_steps:
          logger.info(f"Reached max_steps ({max_steps}). Stopping training.")
          _should_stop = True

    global_step += 1

  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))
  return False


def evaluate(hps, net_g, eval_loader, writer_eval, val_type, full_eval=False):
    generator, eeg_module = net_g
    generator.eval()
    eeg_module.eval()

    sr = hps.data.sampling_rate
    all_mel_losses = []
    all_mel_corrs = []
    logged_av = False

    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, _texts, _clip_embs, _is_congruent, _phoneme, _phoneme_lengths) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)

        if not full_eval:
          x = x[:1]; x_lengths = x_lengths[:1]
          spec = spec[:1]; spec_lengths = spec_lengths[:1]
          y = y[:1]; y_lengths = y_lengths[:1]

        x_out, x_mask_output, mid_output, eeg_decoder_out = eeg_module.module(x)
        mid_output_lengths = x_lengths.clone() * mid_output.size(2) / x.size(2)
        mid_output_lengths = mid_output_lengths.long()

        # Inference: EEG only (no c_text, cfg_scale=0)
        y_hat, _, mask, *_ = generator.module.infer(mid_output, mid_output_lengths, max_len=1000)
        y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

        mel = spec_to_mel_torch(
          spec,
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.mel_fmin,
          hps.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1).float(),
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.hop_length,
          hps.data.win_length,
          hps.data.mel_fmin,
          hps.data.mel_fmax
        )

        min_len = min(mel.size(2), y_hat_mel.size(2))
        all_mel_losses.append(F.l1_loss(mel[:, :, :min_len], y_hat_mel[:, :, :min_len]).item())

        for i in range(x.size(0)):
          gt_wav = y[i, 0, :y_lengths[i]].cpu().float().numpy()
          gen_wav = y_hat[i, 0, :y_hat_lengths[i]].cpu().float().numpy()
          wav_min_len = min(len(gt_wav), len(gen_wav))
          if wav_min_len == 0:
            continue
          mel_gt = librosa.feature.melspectrogram(y=gt_wav[:wav_min_len], sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=80)
          mel_syn = librosa.feature.melspectrogram(y=gen_wav[:wav_min_len], sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=80)
          min_frames = min(mel_gt.shape[1], mel_syn.shape[1])
          all_mel_corrs.append(np.corrcoef(mel_gt[:, :min_frames].flatten(), mel_syn[:, :min_frames].flatten())[0, 1] * 100)

        if not logged_av:
          image_dict = {
            f"{val_type}/gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
          }
          audio_dict = {
            f"{val_type}/audio": y_hat[0,:,:y_hat_lengths[0]].cpu().float().numpy()
          }
          if global_step == 0:
            image_dict.update({f"{val_type}/gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
            audio_dict.update({f"{val_type}/gt/audio": y[0,:,:y_lengths[0]].cpu().float().numpy()})
          utils.summarize(
            writer=writer_eval,
            global_step=global_step,
            images=image_dict,
            audios=audio_dict,
            audio_sampling_rate=hps.data.sampling_rate
          )
          if hps.wandb:
            image_dict_wandb = {k:wandb.Image(v, caption=k) for k,v in image_dict.items()}
            audio_dict_wandb = {k:wandb.Audio(v[0], caption=k, sample_rate=hps.data.sampling_rate) for k,v in audio_dict.items()}
            wandb.log(image_dict_wandb)
            wandb.log(audio_dict_wandb)
          logged_av = True

        if not full_eval:
          break

    eval_mel_loss = float(np.mean(all_mel_losses)) if all_mel_losses else float('nan')
    mel_corr = float(np.mean(all_mel_corrs)) if all_mel_corrs else float('nan')

    # MCD: 1-sample fast eval only
    mcd = float('nan')
    if not full_eval:
      gt_wav = y[0, 0, :y_lengths[0]].cpu().float().numpy()
      gen_wav = y_hat[0, 0, :y_hat_lengths[0]].cpu().float().numpy()
      with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f_gt, \
           tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as f_syn:
        write(f_gt.name, sr, gt_wav)
        write(f_syn.name, sr, gen_wav)
        mcd_calculator = Calculate_MCD(MCD_mode="dtw")
        mcd = mcd_calculator.calculate_mcd(f_gt.name, f_syn.name)

    generator.train()
    eeg_module.train()

    return eval_mel_loss, mcd, mel_corr


if __name__ == "__main__":
  main()
