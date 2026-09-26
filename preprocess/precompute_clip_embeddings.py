"""
Precompute CLIP text embeddings for every unique sentence in the N400 filelists.
Writes a dict[str, Tensor] cache consumed by EEGAudioLoader.

The cache for the released filelists is already provided at
clip_cache/n400_clip_embeddings.pt; rerun this only for a different sentence set.

Usage:
    python preprocess/precompute_clip_embeddings.py -c configs/sense.json
"""

import os
import argparse
import json
import torch
import torch.nn.functional as F


# ── Sentence extraction ────────────────────────────────────────────────────────

def extract_unique_sentences(config):
    """Extract all unique word/sentence stimuli from train + all test filelists."""
    filelist_keys = [
        'training_files',
        'validation_files_both',
        'validation_files_audio',
        'validation_files_subject',
    ]
    sentences = set()
    for key in filelist_keys:
        path = config['data'].get(key)
        if path and os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '||' in line:
                        text = line.split('||', 1)[1].split('||')[0]  # second field
                        sentences.add(text)
    return sorted(sentences)


# ── Embedding backends ─────────────────────────────────────────────────────────

def embed_clip(sentences, model_name, batch_size, device):
    """CLIP ViT-B/32 text encoder (original, dim=512)."""
    from transformers import CLIPModel, CLIPTokenizer
    print(f'Loading CLIP model: {model_name}')
    model = CLIPModel.from_pretrained(model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model.eval()

    cache = {}
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           return_tensors='pt').to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = F.normalize(feats, dim=-1)
        for j, text in enumerate(batch):
            cache[text] = feats[j].cpu()
        print(f'  [{i + len(batch)}/{len(sentences)}] processed')

    return cache


def embed_clap(sentences, model_name, batch_size, device):
    """CLAP audio-text text encoder (dim=512).

    Uses laion/clap-htsat-unfused by default.
    Requires transformers >= 4.36.
    """
    try:
        from transformers import ClapModel, ClapProcessor
    except ImportError:
        raise ImportError(
            "CLAP requires transformers >= 4.36. "
            "Run: pip install --upgrade transformers")

    print(f'Loading CLAP model: {model_name}')
    model = ClapModel.from_pretrained(model_name).to(device)
    processor = ClapProcessor.from_pretrained(model_name)
    model.eval()

    cache = {}
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = processor(text=batch, return_tensors='pt',
                           padding=True, truncation=True)
        # move only tensor values to device
        inputs = {k: v.to(device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = F.normalize(feats, dim=-1)
        for j, text in enumerate(batch):
            cache[text] = feats[j].cpu()
        print(f'  [{i + len(batch)}/{len(sentences)}] processed')

    return cache


def embed_sbert(sentences, model_name, batch_size, device):
    """Sentence-BERT text encoder (dim=768 for all-mpnet-base-v2).

    Requires sentence-transformers.
    Run: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "SBERT requires sentence-transformers. "
            "Run: pip install sentence-transformers")

    print(f'Loading SBERT model: {model_name}')
    model = SentenceTransformer(model_name, device=device)

    cache = {}
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        # normalize_embeddings=True → unit-norm, same convention as CLIP
        feats = model.encode(batch, convert_to_tensor=True,
                             normalize_embeddings=True,
                             show_progress_bar=False)
        for j, text in enumerate(batch):
            cache[text] = feats[j].cpu()
        print(f'  [{i + len(batch)}/{len(sentences)}] processed')

    return cache


# ── Default model names ────────────────────────────────────────────────────────

_DEFAULT_MODELS = {
    'clip':  'openai/clip-vit-base-patch32',
    'clap':  'laion/clap-htsat-unfused',
    'sbert': 'sentence-transformers/all-mpnet-base-v2',
}

_EXPECTED_DIMS = {
    'clip':  512,
    'clap':  512,
    'sbert': 768,
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Precompute semantic text embeddings for N400 stimuli')
    parser.add_argument('-c', '--config', type=str,
                        default='configs/sense.json')
    parser.add_argument('--model_type', type=str, default='clip',
                        choices=['clip', 'clap', 'sbert'],
                        help='Embedding backend (default: clip)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='HuggingFace model ID; defaults per model_type')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output .pt path; defaults to clip_cache/n400_{type}_embeddings.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # Resolve defaults
    model_name = args.model_name or _DEFAULT_MODELS[args.model_type]
    output     = args.output or f'clip_cache/n400_{args.model_type}_embeddings.pt'

    with open(args.config, 'r') as f:
        config = json.load(f)

    sentences = extract_unique_sentences(config)
    print(f'Found {len(sentences)} unique sentences')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    embed_fn = {'clip': embed_clip, 'clap': embed_clap, 'sbert': embed_sbert}
    cache = embed_fn[args.model_type](sentences, model_name, args.batch_size, device)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    torch.save(cache, output)

    # Verify
    sample_key = sentences[0]
    sample_emb = cache[sample_key]
    actual_dim = sample_emb.shape[0]
    expected   = _EXPECTED_DIMS[args.model_type]
    dim_ok     = '✓' if actual_dim == expected else f'(expected {expected})'

    print(f'\nSaved {len(cache)} embeddings → {output}')
    print(f'Embedding dim : {actual_dim} {dim_ok}')
    print(f'Sample        : "{sample_key}" → norm={sample_emb.norm().item():.4f}')


if __name__ == '__main__':
    main()
