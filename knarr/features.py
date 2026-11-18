"""WavLM-large layer 0 + 3 features exactly as F4N2 was trained (copied from formant_tracker's analysis/wavlm_features.py, release f4n2_v1).
Mono, 16 kHz, peak 0.9; the WavLM feature extractor z-normalises each chunk; signals over 30 s run in 30 s chunks with 2 s overlap, stitched by
global frame index with the seam at the centre of each overlap. One row per 20 ms frame, 2,048 values (1,024 per layer)."""
from math import gcd

import numpy as np

SR = 16000
HOP = 320
MIN_SAMPLES = 400
CHUNK_S, OVERLAP_S = 30, 2
WAVLM_REPO = "microsoft/wavlm-large"
WAVLM_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"   # the snapshot F4N2 was trained with
WAVLM_SHA256 = "fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d"   # pytorch_model.bin


class InsufficientAudio(ValueError):
    """the signal is shorter than WavLM's minimum (25 ms)"""


def to_16k(y, sr):
    from scipy.signal import resample_poly
    y = np.asarray(y, np.float64)
    y = y.mean(1) if y.ndim > 1 else y
    if int(sr) != SR:
        g = gcd(SR, int(sr)); y = resample_poly(y, SR // g, int(sr) // g)
    return (0.9 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def stitch(chunks, n_samples, chunk=CHUNK_S * SR, overlap=OVERLAP_S * SR):
    parts = []
    for k, (s, g) in enumerate(chunks):
        g0 = s // HOP
        lo = 0 if k == 0 else (s + overlap // 2) // HOP
        hi = g0 + len(g) if k == len(chunks) - 1 else (chunks[k + 1][0] + overlap // 2) // HOP
        parts.append(g[max(0, lo - g0):max(0, hi - g0)])
    return np.concatenate(parts, 0) if parts else np.zeros((0, chunks[0][1].shape[1] if chunks else 0), np.float32)


def chunk_starts(n_samples, chunk=CHUNK_S * SR, overlap=OVERLAP_S * SR):
    if n_samples <= chunk: return [0]
    out, s = [], 0
    while True:
        out.append(s)
        if s + chunk >= n_samples: return out
        s += chunk - overlap


def load_wavlm(device="cpu"):
    """WavLM-large at the pinned revision (downloaded once into the Hugging Face cache, about 1.2 GB); fp16 on CUDA as in training"""
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained(WAVLM_REPO, revision=WAVLM_REVISION)
    wl = WavLMModel.from_pretrained(WAVLM_REPO, revision=WAVLM_REVISION).eval().to(device)
    if str(device).startswith("cuda"): wl = wl.half()
    return fe, wl


def wavlm_l0_l3_features(y16, fe, wl, device="cpu"):
    """y16: float32 16 kHz mono conditioned by to_16k. Returns (T, 2048) float32, T = WavLM's own frame count."""
    import torch
    if len(y16) < MIN_SAMPLES: raise InsufficientAudio(f"{len(y16)} samples < {MIN_SAMPLES}")
    dtype = next(wl.parameters()).dtype
    def run(seg):
        with torch.no_grad():
            x = fe(seg, sampling_rate=SR, return_tensors="pt")["input_values"].to(device).to(dtype)
            hs = wl(x, output_hidden_states=True).hidden_states
            return torch.cat([hs[0][0], hs[3][0]], -1).float().cpu().numpy()
    chunks = []
    for s in chunk_starts(len(y16)):
        seg = y16[s:s + CHUNK_S * SR]
        if len(seg) < MIN_SAMPLES: break
        chunks.append((s, run(seg)))
    return stitch(chunks, len(y16))
