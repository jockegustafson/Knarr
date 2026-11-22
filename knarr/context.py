"""Knarr's context model: how likely creak is on a word given only what leads into it, never the word's own voice.

    from knarr.context import context_scores
    context_scores("talk.wav", [("hej", 0.51, 0.80), ("då", 1.20, 1.55)], channel=0)   # -> [0.07, 0.41]

Features per word (as in its training data, KTH 2026-11-22): the 0.5 s before the word (median F0 in semitones and median energy in dB re
the speaker's median over the file, their slopes per 100 ms, the voiced share; none at the start of an utterance), the pauses before and
after it, its position in the utterance (s and words to the end, last word), its length in letters, whether it is a filler, and the
utterance's speech rate. Utterances are the words split at pauses of 1 s or more. F0 is REAPER's (25-500 Hz, 10 ms), on each utterance
+- 0.3 s. A one-hidden-layer net (weights/context_v1.json) trained to predict Knarr v2 >= 0.5 on 2.1 M words of 34 Swedish corpora;
held-out speakers: AUC 0.76 against Knarr, 0.69 against the Swedish test set's labels (Knarr itself 0.88).

What it is for: words where Knarr and the context disagree are the informative ones to check by ear (knarr-inspect --pick disagree): creak
where the context does not lead one to expect it (noise, laughter, or real but unusual creak), and expected creak that Knarr does not hear.
Needs pyreaper (pip install "knarr[gate]")."""
import json
from importlib import resources

import numpy as np

SR, PAD, LEAD, UTT_GAP = 16000, 0.3, 50, 1.0
FILL = {"eh", "öh", "ehm", "öhm", "mm", "hm", "mhm", "äh", "ah", "m", "eeh", "öhh"}
_W = None


def _weights():
    global _W
    if _W is None:
        W = json.loads((resources.files("knarr") / "weights" / "context_v1.json").read_text(encoding="utf-8"))
        _W = {k: np.asarray(v, np.float64) if isinstance(v, list) and k != "features" else v for k, v in W.items()}
    return _W


def _utterances(words):
    """words sorted by start, split at pauses >= 1 s -> lists of indices"""
    order = sorted(range(len(words)), key=lambda i: words[i][1]); out = []
    for i in order:
        if out and words[i][1] - words[out[-1][-1]][2] < UTT_GAP: out[-1].append(i)
        else: out.append([i])
    return out


def _analyse(y):
    """REAPER F0 (Hz, 0 = unvoiced) and frame energy (dB, 25 ms RMS) on a 10 ms grid"""
    from .gate import _quiet
    import pyreaper
    x = np.clip(y / (np.abs(y).max() + 1e-9) * 0.9 * 32767, -32768, 32767).astype(np.int16)
    with _quiet(): _, _, f0_t, f0, _ = pyreaper.reaper(x, SR, minf0=25.0, maxf0=500.0, frame_period=0.01)
    n = len(y) // 160; grid = np.arange(n) * 0.01
    k = np.clip(np.searchsorted(f0_t, grid), 0, max(len(f0_t) - 1, 0))
    F = np.where(f0[k] > 0, f0[k], 0.0) if len(f0_t) else np.zeros(n)
    fr = np.lib.stride_tricks.sliding_window_view(np.pad(y, (200, 200)), 400)[::160][:n]
    E = 20 * np.log10(np.sqrt((fr.astype(np.float64) ** 2).mean(1)) + 1e-7)
    return F, E


def _slope(x):
    ok = np.isfinite(x)
    if ok.sum() < 3: return np.nan
    k = np.where(ok)[0]; return -np.polyfit(k * 0.01, x[ok], 1)[0] / 10.0


def features(y, words):
    """y: 16 kHz mono; words: [(word, start, end)] of one speaker -> (feature matrix in the model's column order, index order = words)"""
    W = _weights(); utts = _utterances(words); segs = []
    for u in utts:
        t0 = max(0.0, words[u[0]][1] - PAD); t1 = min(len(y) / SR, words[u[-1]][2] + PAD)
        F, E = _analyse(y[int(t0 * SR):int(t1 * SR)].astype(np.float64)); segs.append((t0, F, E))
    vf = np.concatenate([F[F > 0] for _, F, _ in segs]) if segs else np.zeros(0); ve = np.concatenate([E[F > 0] for _, F, E in segs]) if segs else np.zeros(0)
    rf = float(np.median(vf)) if len(vf) else 100.0; re_ = float(np.median(ve)) if len(ve) else -30.0
    rows = {}
    for u, (t0, F, E) in zip(utts, segs):
        v = F > 0; fst = np.where(v, 12 * np.log2(np.maximum(F, 1e-3) / rf), np.nan); est = np.where(v, E - re_, np.nan)
        W_ = [words[i] for i in u]; t_last = W_[-1][2]; rate = len(W_) / max(t_last - W_[0][1], 0.1)
        for wi, i in enumerate(u):
            w, s, e = words[i]; a = min(int((s - t0) * 100), len(F)); lo = max(0, a - LEAD)
            if a - lo < 10 or v[lo:a].sum() < 3: pre = [0.0, 0.0, 0.0, 0.0, float(v[lo:a].mean()) if a > lo else 0.0, 1.0]
            else:
                pre = [float(np.nanmedian(fst[lo:a])), float(np.nanmedian(est[lo:a])), _slope(fst[lo:a]), _slope(est[lo:a]), float(v[lo:a].mean()), 0.0]
                pre = [0.0 if (isinstance(x, float) and np.isnan(x)) else x for x in pre]
            bare = "".join(c for c in (w or "").lower() if c.isalpha())
            rows[i] = {"f0_pre": pre[0], "en_pre": pre[1], "f0_pre_slope": pre[2], "en_pre_slope": pre[3], "voiced_pre": pre[4],
                       "t_end": np.log1p(max(0.0, t_last - e)), "w_end": np.log1p(len(u) - 1 - wi), "last": float(wi == len(u) - 1),
                       "pause_before": min(2.0, s - W_[wi - 1][2]) if wi else 2.0, "pause_after": min(2.0, W_[wi + 1][1] - e) if wi < len(u) - 1 else 2.0,
                       "nchar": len(w or ""), "filler": float(bare in FILL), "rate": rate, "no_lead": pre[5]}
    return np.array([[rows[i][c] for c in W["features"]] for i in range(len(words))], np.float64)


def predict(X):
    W = _weights(); Z = (X - W["mu"]) / W["sd"]; h = np.maximum(0, Z @ W["W1"] + W["b1"]); z = (h @ W["W2"] + W["b2"]).reshape(-1)
    return 1 / (1 + np.exp(-z))


def context_scores(audio, words, channel=None):
    """creak expected from context, per word (0-1), in the order of `words` ([(word, start, end)] of one speaker)"""
    from .detector import CreakDetector
    y = CreakDetector.load_audio(audio, channel)
    if isinstance(y, tuple): y, sr = y
    else: sr = SR
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, sr); y = resample_poly(y, SR // g, sr // g)
    words = [(w, float(s), float(e)) for w, s, e in words]
    return predict(features(np.asarray(y, np.float64), words)).round(4).tolist() if words else []
