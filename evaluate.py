import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
import argparse
import glob
import json
import numpy as np
import librosa
import torch
from datetime import datetime
from tqdm import tqdm
from pymcd.mcd import Calculate_MCD
from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn

PESQ_SR = 16000

# ---------------------------------------------------------------------------
# CLAP (Contrastive Language-Audio Pretraining) — ASR-free semantic scoring
# Computes cosine_sim( CLAP_audio(decoded_wav), CLAP_text(original_text) )
# without any intermediate Whisper transcription step.
# ---------------------------------------------------------------------------
CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
CLAP_SR = 48000  # native sample rate expected by the CLAP audio encoder


def load_clap_model(device):
    from transformers import ClapModel, ClapProcessor
    print(f"Loading CLAP model ({CLAP_MODEL_ID}) on {device}...")
    model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device)
    processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    model.eval()
    return model, processor


def compute_clap_score(syn_path, text, clap_model, clap_processor, device):
    """Direct audio-to-text semantic similarity via CLAP (no ASR required).

    Resamples syn_path to CLAP_SR, encodes with the CLAP audio encoder, then
    computes cosine similarity with the CLAP text embedding of ``text``.
    Returns a float in [-1, 1] (higher = more semantically aligned).
    """
    import torch.nn.functional as F
    audio, sr = librosa.load(syn_path, sr=None)
    if sr != CLAP_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=CLAP_SR)

    audio_inputs = clap_processor(audios=audio, return_tensors="pt", sampling_rate=CLAP_SR)
    text_inputs  = clap_processor(text=[text], return_tensors="pt", padding=True)

    with torch.no_grad():
        audio_emb = clap_model.get_audio_features(
            **{k: v.to(device) for k, v in audio_inputs.items()})
        text_emb  = clap_model.get_text_features(
            **{k: v.to(device) for k, v in text_inputs.items()})

    audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
    text_emb  = text_emb  / text_emb.norm(dim=-1, keepdim=True)
    return (audio_emb @ text_emb.T).item()


# ---------------------------------------------------------------------------

METRIC_LABELS = [
    ('mcd',              'MCD'),
    ('mel_corr',         'Mel-Corr'),
    ('pesq',             'PESQ'),
    ('stoi',             'STOI'),
    ('bertscore',        'BERTScore'),
    ('bertscore_cong',   'BERTScore[cong]'),
    ('bertscore_incong', 'BERTScore[incong]'),
    ('wer',              'WER'),
    ('wer_cong',         'WER[cong]'),
    ('wer_incong',       'WER[incong]'),
    ('clip_sim',         'CLIP-Sim'),
    ('clip_sim_cong',    'CLIP-Sim[cong]'),
    ('clip_sim_incong',  'CLIP-Sim[incong]'),
    # CLAP-Score: ASR-free direct audio↔text semantic similarity
    ('clap_score',       'CLAP-Score'),
    ('clap_score_cong',  'CLAP-Score[cong]'),
    ('clap_score_incong','CLAP-Score[incong]'),
]

SEMANTIC_KEYS = {
    'bertscore', 'bertscore_cong', 'bertscore_incong',
    'wer', 'wer_cong', 'wer_incong',
    'clip_sim', 'clip_sim_cong', 'clip_sim_incong',
}

# CLAP keys are gated by --clap flag (independently of --semantic / Whisper)
CLAP_KEYS = {'clap_score', 'clap_score_cong', 'clap_score_incong'}


def _fmt(v):
    return f"{v:.4f}" if not np.isnan(v) else "  n/a"


def _nan_to_none(v):
    return None if (isinstance(v, float) and np.isnan(v)) else v


def mel_correlation(gt_path, syn_path, sr=22050, n_mels=80):
    gt, _ = librosa.load(gt_path, sr=sr)
    syn, _ = librosa.load(syn_path, sr=sr)
    min_len = min(len(gt), len(syn))
    gt, syn = gt[:min_len], syn[:min_len]

    mel_gt = librosa.feature.melspectrogram(y=gt, sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=n_mels)
    mel_syn = librosa.feature.melspectrogram(y=syn, sr=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=n_mels)

    min_frames = min(mel_gt.shape[1], mel_syn.shape[1])
    corr = np.corrcoef(mel_gt[:, :min_frames].flatten(), mel_syn[:, :min_frames].flatten())[0, 1] * 100
    return corr


def pesq_score(gt_path, syn_path, sr=22050):
    try:
        gt, _ = librosa.load(gt_path, sr=sr)
        syn, _ = librosa.load(syn_path, sr=sr)
        min_len = min(len(gt), len(syn))
        gt, syn = gt[:min_len], syn[:min_len]
        gt_16k = librosa.resample(gt, orig_sr=sr, target_sr=PESQ_SR)
        syn_16k = librosa.resample(syn, orig_sr=sr, target_sr=PESQ_SR)
        return pesq_fn(PESQ_SR, gt_16k, syn_16k, 'wb')
    except Exception:
        return None


def stoi_score(gt_path, syn_path, sr=22050):
    try:
        gt, _ = librosa.load(gt_path, sr=sr)
        syn, _ = librosa.load(syn_path, sr=sr)
        min_len = min(len(gt), len(syn))
        return stoi_fn(gt[:min_len], syn[:min_len], sr, extended=False)
    except Exception:
        return None


def transcribe_audio(wav_path, whisper_model):
    result = whisper_model.transcribe(wav_path, language="en")
    return result["text"].strip()


def compute_clip_similarity(text1, text2, clip_model, clip_tokenizer, device):
    inputs1 = clip_tokenizer(text1, return_tensors="pt", padding=True, truncation=True).to(device)
    inputs2 = clip_tokenizer(text2, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        emb1 = clip_model.get_text_features(**inputs1)
        emb2 = clip_model.get_text_features(**inputs2)
    emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
    emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
    return (emb1 @ emb2.T).item()


def _compute_bertscore_wer(pred_texts, ref_texts):
    if not pred_texts:
        return float('nan'), float('nan')
    from bert_score import score as bertscore_fn
    from jiwer import wer as wer_fn
    P, R, F1 = bertscore_fn(pred_texts, ref_texts, lang="en", verbose=False, rescale_with_baseline=True)
    return F1.mean().item(), wer_fn(ref_texts, pred_texts)


def evaluate_split(split_dir, mcd_calculator, whisper_model=None,
                   clip_model=None, clip_tokenizer=None,
                   clap_model=None, clap_processor=None,
                   device='cpu'):
    gt_files = sorted(glob.glob(os.path.join(split_dir, "*_gt.wav")))
    if not gt_files:
        print(f"  No files found in {split_dir}")
        return None

    texts_path = os.path.join(split_dir, "texts.json")
    texts_map = {}
    if os.path.isfile(texts_path):
        with open(texts_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            if isinstance(v, dict):
                texts_map[k] = v
            else:
                texts_map[k] = {'text': v, 'congruent': True}

    mcd_scores = []
    corr_scores = []
    pesq_scores = []
    stoi_scores = []
    pred_texts_cong, ref_texts_cong = [], []
    pred_texts_incong, ref_texts_incong = [], []
    clip_sims_cong, clip_sims_incong = [], []
    clap_scores_cong, clap_scores_incong = [], []

    split_name = os.path.basename(split_dir)
    for gt_path in tqdm(gt_files, desc=f"  {split_name}"):
        idx = os.path.basename(gt_path).replace("_gt.wav", "")
        syn_path = os.path.join(split_dir, f"{idx}_syn.wav")
        if not os.path.exists(syn_path):
            continue

        mcd = mcd_calculator.calculate_mcd(gt_path, syn_path)
        corr = mel_correlation(gt_path, syn_path)
        p_val = pesq_score(gt_path, syn_path)
        s_val = stoi_score(gt_path, syn_path)

        mcd_scores.append(mcd)
        corr_scores.append(corr)
        if p_val is not None:
            pesq_scores.append(p_val)
        if s_val is not None:
            stoi_scores.append(s_val)

        entry = texts_map.get(idx)

        # ── ASR-dependent metrics (Whisper → BERTScore, WER, CLIP-Sim) ──
        if entry and whisper_model is not None:
            gt_text = entry['text']
            is_cong = entry.get('congruent', True)
            try:
                pred_text = transcribe_audio(syn_path, whisper_model)
                if is_cong:
                    pred_texts_cong.append(pred_text)
                    ref_texts_cong.append(gt_text)
                else:
                    pred_texts_incong.append(pred_text)
                    ref_texts_incong.append(gt_text)

                if clip_model is not None:
                    sim = compute_clip_similarity(
                        pred_text, gt_text, clip_model, clip_tokenizer, device)
                    if is_cong:
                        clip_sims_cong.append(sim)
                    else:
                        clip_sims_incong.append(sim)
            except Exception as e:
                print(f"  Warning: semantic eval error for {idx}: {e}")

        # ── CLAP-Score: ASR-free audio↔original-text similarity ─────────
        if entry and clap_model is not None:
            gt_text = entry['text']
            is_cong = entry.get('congruent', True)
            try:
                score = compute_clap_score(
                    syn_path, gt_text, clap_model, clap_processor, device)
                if is_cong:
                    clap_scores_cong.append(score)
                else:
                    clap_scores_incong.append(score)
            except Exception as e:
                print(f"  Warning: CLAP score error for {idx}: {e}")

    results = {
        'mcd':      np.mean(mcd_scores)  if mcd_scores  else float('nan'),
        'mel_corr': np.mean(corr_scores) if corr_scores else float('nan'),
        'pesq':     np.mean(pesq_scores) if pesq_scores else float('nan'),
        'stoi':     np.mean(stoi_scores) if stoi_scores else float('nan'),
    }

    bs_cong,   wer_cong   = _compute_bertscore_wer(pred_texts_cong,   ref_texts_cong)
    bs_incong, wer_incong = _compute_bertscore_wer(pred_texts_incong, ref_texts_incong)
    bs_all,    wer_all    = _compute_bertscore_wer(
        pred_texts_cong + pred_texts_incong,
        ref_texts_cong  + ref_texts_incong)

    results['bertscore']        = bs_all
    results['bertscore_cong']   = bs_cong
    results['bertscore_incong'] = bs_incong
    results['wer']              = wer_all
    results['wer_cong']         = wer_cong
    results['wer_incong']       = wer_incong

    all_clip = clip_sims_cong + clip_sims_incong
    results['clip_sim']         = np.mean(all_clip)         if all_clip         else float('nan')
    results['clip_sim_cong']    = np.mean(clip_sims_cong)   if clip_sims_cong   else float('nan')
    results['clip_sim_incong']  = np.mean(clip_sims_incong) if clip_sims_incong else float('nan')

    all_clap = clap_scores_cong + clap_scores_incong
    results['clap_score']        = np.mean(all_clap)           if all_clap           else float('nan')
    results['clap_score_cong']   = np.mean(clap_scores_cong)   if clap_scores_cong   else float('nan')
    results['clap_score_incong'] = np.mean(clap_scores_incong) if clap_scores_incong else float('nan')

    return results


def print_results(split, results, semantic=False, clap=False):
    print(f"\n[{split}]")
    print(f"  MCD:        {results['mcd']:.4f}")
    print(f"  Mel-Corr:   {results['mel_corr']:.4f}")
    print(f"  PESQ:       {results['pesq']:.4f}")
    print(f"  STOI:       {results['stoi']:.4f}")
    if semantic:
        print(f"  BERTScore:  {_fmt(results['bertscore'])}  [cong: {_fmt(results.get('bertscore_cong', float('nan')))}  incong: {_fmt(results.get('bertscore_incong', float('nan')))}]")
        print(f"  WER:        {_fmt(results['wer'])}  [cong: {_fmt(results.get('wer_cong', float('nan')))}  incong: {_fmt(results.get('wer_incong', float('nan')))}]")
        print(f"  CLIP-Sim:   {_fmt(results['clip_sim'])}  [cong: {_fmt(results.get('clip_sim_cong', float('nan')))}  incong: {_fmt(results.get('clip_sim_incong', float('nan')))}]")
    if clap:
        print(f"  CLAP-Score: {_fmt(results['clap_score'])}  [cong: {_fmt(results.get('clap_score_cong', float('nan')))}  incong: {_fmt(results.get('clap_score_incong', float('nan')))}]")


def print_mean_std(split_run_results, splits, semantic=False, clap=False):
    """Print mean±std across runs for each split, then overall."""
    for split in splits:
        run_list = split_run_results.get(split, [])
        if not run_list:
            continue
        print(f"\n[{split}]")
        for k, label in METRIC_LABELS:
            if k in SEMANTIC_KEYS and not semantic:
                continue
            if k in CLAP_KEYS and not clap:
                continue
            vals = [r[k] for r in run_list if not np.isnan(r.get(k, float('nan')))]
            if vals:
                print(f"  {label:22s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    overall = {k: [] for k, _ in METRIC_LABELS}
    for run_list in split_run_results.values():
        for r in run_list:
            for k, _ in METRIC_LABELS:
                v = r.get(k, float('nan'))
                if not np.isnan(v):
                    overall[k].append(v)
    if any(overall.values()):
        print(f"\n[Overall]")
        for k, label in METRIC_LABELS:
            if k in SEMANTIC_KEYS and not semantic:
                continue
            if k in CLAP_KEYS and not clap:
                continue
            if overall[k]:
                print(f"  {label:22s}  {np.mean(overall[k]):.4f} ± {np.std(overall[k]):.4f}")


def save_results_json(path, meta, splits_results, split_run_results=None):
    """
    Single run:  splits_results = {split: result_dict}, split_run_results = None
    Multi-run:   splits_results = {},                   split_run_results = {split: [result_dict, ...]}
    """
    data = {"meta": meta}

    if split_run_results is None:
        data["splits"] = {
            split: {k: _nan_to_none(v) for k, v in res.items()}
            for split, res in splits_results.items()
        }
        overall = {k: [] for k, _ in METRIC_LABELS}
        for res in splits_results.values():
            for k, _ in METRIC_LABELS:
                v = res.get(k, float('nan'))
                if not np.isnan(v):
                    overall[k].append(v)
        data["overall"] = {
            k: _nan_to_none(float(np.mean(vals))) if vals else None
            for k, vals in overall.items()
        }
    else:
        data["runs"] = {
            split: [{k: _nan_to_none(v) for k, v in r.items()} for r in run_list]
            for split, run_list in split_run_results.items()
        }
        summary = {}
        for split, run_list in split_run_results.items():
            if not run_list:
                continue
            summary[split] = {}
            for k, _ in METRIC_LABELS:
                vals = [r[k] for r in run_list if not np.isnan(r.get(k, float('nan')))]
                if vals:
                    summary[split][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        data["summary"] = summary

        overall_vals = {k: [] for k, _ in METRIC_LABELS}
        for run_list in split_run_results.values():
            for r in run_list:
                for k, _ in METRIC_LABELS:
                    v = r.get(k, float('nan'))
                    if not np.isnan(v):
                        overall_vals[k].append(v)
        data["overall"] = {
            k: {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None
            for k, vals in overall_vals.items()
        }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--checkpoint_idx", type=str, default="best")
    parser.add_argument("--semantic", action="store_true",
                        help="Enable ASR-based metrics (Whisper → BERTScore, WER, CLIP-Sim)")
    parser.add_argument("--clap", action="store_true",
                        help="Enable CLAP-Score: ASR-free audio↔text semantic similarity")
    parser.add_argument("--whisper_size", type=str, default="large-v3",
                        help="Whisper model size (tiny/base/small/medium/large/large-v3)")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of repeated inference runs for mean±std evaluation")
    parser.add_argument("--base_seed", type=int, default=1,
                        help="Base seed; run i uses seed base_seed+i")
    parser.add_argument("--output_tag", type=str, default=None,
                        help="Subdirectory tag written by inference.py (e.g. 'sem_mask')")
    parser.add_argument("--per_subject", action="store_true",
                        help="Evaluate per-subject splits (subject_{sub_id}/ dirs) and report "
                             "per-subject metrics plus their average")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip inference and evaluate already-synthesized wavs "
                             "(requires --n_runs to match the number of existing runs)")
    args = parser.parse_args()

    mcd_calculator = Calculate_MCD(MCD_mode="dtw")

    whisper_model = None
    clip_model = None
    clip_tokenizer = None
    clap_model = None
    clap_processor = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.semantic:
        import whisper
        print(f"Loading Whisper model ({args.whisper_size})...")
        whisper_model = whisper.load_model(args.whisper_size, device=device)

        from transformers import CLIPModel, CLIPTokenizer
        print("Loading CLIP model (openai/clip-vit-base-patch32)...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

    if args.clap:
        clap_model, clap_processor = load_clap_model(device)

    splits = ["both", "audio", "subject"]
    print(f"\n{'='*60}")
    print(f"Evaluation: {args.run_name} (checkpoint: {args.checkpoint_idx})")
    if args.semantic:
        print(f"Semantic metrics: ON (Whisper={args.whisper_size})")
    if args.clap:
        print(f"CLAP-Score: ON ({CLAP_MODEL_ID})")
    if args.n_runs > 1:
        print(f"Runs: {args.n_runs} (seeds {args.base_seed}~{args.base_seed + args.n_runs - 1})")
    print(f"{'='*60}")

    tag_suffix = f"_{args.output_tag}" if getattr(args, 'output_tag', None) else ""
    suffix = f"_n{args.n_runs}runs" if args.n_runs > 1 else ""
    suffix += "_semantic" if args.semantic else ""
    suffix += "_clap" if args.clap else ""
    save_path = os.path.join(
        "./logs", args.run_name, "eval_results",
        f"{args.checkpoint_idx}{tag_suffix}{suffix}.json"
    )
    meta = {
        "run_name": args.run_name,
        "checkpoint_idx": args.checkpoint_idx,
        "output_tag": getattr(args, 'output_tag', None),
        "n_runs": args.n_runs,
        "semantic": args.semantic,
        "clap": args.clap,
        "whisper_size": args.whisper_size if args.semantic else None,
        "clap_model_id": CLAP_MODEL_ID if args.clap else None,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
    }

    # ── Single run (pre-generated wavs) ─────────────────────────────
    if args.n_runs == 1:
        synth_base = str(args.checkpoint_idx)
        if getattr(args, 'output_tag', None):
            synth_base = os.path.join(synth_base, args.output_tag)
        base_dir = os.path.join("./logs", args.run_name, "synthesized", synth_base)
        if not os.path.exists(base_dir):
            print(f"Directory not found: {base_dir}")
            print("Run inference.py first to generate synthesized wavs.")
            return

        splits_results = {}
        all_results = {k: [] for k, _ in METRIC_LABELS}

        for split in splits:
            split_dir = os.path.join(base_dir, split)
            if not os.path.isdir(split_dir):
                continue
            results = evaluate_split(
                split_dir, mcd_calculator,
                whisper_model, clip_model, clip_tokenizer,
                clap_model, clap_processor, device)
            if results is None:
                continue
            splits_results[split] = results
            print_results(split, results, args.semantic, args.clap)
            for k, _ in METRIC_LABELS:
                v = results.get(k, float('nan'))
                if not np.isnan(v):
                    all_results[k].append(v)

        if all_results['mcd']:
            print(f"\n[Overall]")
            for k, label in METRIC_LABELS:
                if k in SEMANTIC_KEYS and not args.semantic:
                    continue
                if k in CLAP_KEYS and not args.clap:
                    continue
                if all_results[k]:
                    print(f"  {label:22s}  {np.mean(all_results[k]):.4f}")

        # Per-subject evaluation: subject_{sub_id}/ dirs created by inference --per_subject
        if getattr(args, 'per_subject', False):
            sub_dirs = sorted(glob.glob(os.path.join(base_dir, "subject_sub-*")))
            per_sub_results = {}
            for sub_dir in sub_dirs:
                sub_key = os.path.basename(sub_dir)  # e.g. "subject_sub-23"
                results = evaluate_split(
                    sub_dir, mcd_calculator,
                    whisper_model, clip_model, clip_tokenizer,
                    clap_model, clap_processor, device)
                if results is None:
                    continue
                splits_results[sub_key] = results
                per_sub_results[sub_key] = results
                print_results(sub_key, results, args.semantic, args.clap)

            if len(per_sub_results) > 1:
                avg_results = {}
                for k, _ in METRIC_LABELS:
                    vals = [r.get(k, float('nan')) for r in per_sub_results.values()
                            if not np.isnan(r.get(k, float('nan')))]
                    avg_results[k] = float(np.mean(vals)) if vals else float('nan')
                splits_results['subject_avg'] = avg_results
                print_results('subject[avg]', avg_results, args.semantic, args.clap)

        if splits_results:
            save_results_json(save_path, meta, splits_results)

    # ── Multi-run: inference × N → mean ± std ───────────────────────
    else:
        import types
        from inference import run as inference_run

        split_run_results = {split: [] for split in splits}

        for i in range(args.n_runs):
            seed = args.base_seed + i
            output_tag = f"run{i}_seed{seed}"
            print(f"\n[Run {i+1}/{args.n_runs}] seed={seed}")

            if getattr(args, 'eval_only', False):
                # Skip inference; reconstruct output_key from the standard multi-run naming
                output_key = f"{args.checkpoint_idx}/{output_tag}"
                print(f"  [eval-only] Skipping inference, using: synthesized/{output_key}")
            else:
                inf_args = types.SimpleNamespace(
                    run_name=args.run_name,
                    checkpoint_idx=args.checkpoint_idx,
                    seed=seed,
                    output_tag=output_tag,
                    per_subject=getattr(args, 'per_subject', False),
                )
                output_key = inference_run(inf_args)

            run_base_dir = os.path.join("./logs", args.run_name, "synthesized", output_key)
            for split in splits:
                split_dir = os.path.join(run_base_dir, split)
                if not os.path.isdir(split_dir):
                    continue
                results = evaluate_split(
                    split_dir, mcd_calculator,
                    whisper_model, clip_model, clip_tokenizer,
                    clap_model, clap_processor, device)
                if results:
                    split_run_results[split].append(results)
                    print_results(split, results, args.semantic, args.clap)

            # Per-subject evaluation in multi-run
            if getattr(args, 'per_subject', False):
                sub_dirs = sorted(glob.glob(os.path.join(run_base_dir, "subject_sub-*")))
                run_per_sub = {}
                for sub_dir in sub_dirs:
                    sub_key = os.path.basename(sub_dir)  # e.g. "subject_sub-23"
                    results = evaluate_split(
                        sub_dir, mcd_calculator,
                        whisper_model, clip_model, clip_tokenizer,
                        clap_model, clap_processor, device)
                    if results:
                        split_run_results.setdefault(sub_key, []).append(results)
                        run_per_sub[sub_key] = results
                        print_results(sub_key, results, args.semantic, args.clap)

                if len(run_per_sub) > 1:
                    avg_results = {}
                    for k, _ in METRIC_LABELS:
                        vals = [r.get(k, float('nan')) for r in run_per_sub.values()
                                if not np.isnan(r.get(k, float('nan')))]
                        avg_results[k] = float(np.mean(vals)) if vals else float('nan')
                    split_run_results.setdefault('subject_avg', []).append(avg_results)

        print(f"\n{'='*60}")
        print(f"Mean ± Std over {args.n_runs} runs")
        print(f"{'='*60}")
        all_splits = list(split_run_results.keys())
        print_mean_std(split_run_results, all_splits, args.semantic, args.clap)
        save_results_json(save_path, meta, {}, split_run_results)

    print()


if __name__ == "__main__":
    main()

# Usage examples:
#
# CLAP-Score only (no Whisper, no GPU for ASR):
#   python evaluate.py --run_name v7 --checkpoint_idx milestone_100000 --n_runs 5 --clap
#
# CLAP + legacy semantic metrics:
#   python evaluate.py --run_name v7 --checkpoint_idx milestone_100000 --n_runs 5 --semantic --clap
#
# Legacy semantic only:
#   python evaluate.py --run_name v7 --checkpoint_idx milestone_100000 --n_runs 5 --semantic --whisper_size large-v3
