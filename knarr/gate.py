"""The voicing gate of F4N2 ("gate V"): REAPER with a low F0 floor (min 25 Hz, max 500 Hz) marks glottal epochs; on a 10 ms grid, a time
is voiced when the epoch interval around it is 2-40 ms. A model frame counts for a word only if a voiced time lies within 15 ms of its centre.
REAPER is run on stretches around the words (word +- 0.5 s, overlapping stretches merged), because on a whole long file it takes minutes.
Needs pyreaper (pip install pyreaper; it imports pkg_resources, so setuptools < 81)."""
import os
from contextlib import contextmanager

import numpy as np

MINF0, MAXF0, PAD = 25.0, 500.0, 0.5


def _cflush():
    try:
        import ctypes
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


@contextmanager
def _quiet():
    """REAPER's C code prints "Residual symmetry ..." for every call (C-buffered, so it is flushed before the descriptors are restored):
    send file descriptors 1 and 2 to /dev/null meanwhile"""
    try: saved = [os.dup(1), os.dup(2)]; null = os.open(os.devnull, os.O_WRONLY)
    except OSError: yield; return
    try:
        _cflush(); os.dup2(null, 1); os.dup2(null, 2); yield
    finally:
        _cflush(); os.dup2(saved[0], 1); os.dup2(saved[1], 2); [os.close(f) for f in saved + [null]]


def reaper_epochs(y, sr, minf0=MINF0, maxf0=MAXF0):
    try:
        import pyreaper
    except ImportError as ex:
        raise ImportError("the voicing gate needs pyreaper: pip install pyreaper 'setuptools<81' (or score with gate=False)") from ex
    y = np.asarray(y, np.float64).reshape(-1); scale = 0.9 / max(1.0, float(np.abs(y).max()) if y.size else 0.0)
    x16 = np.clip(y * scale * 32767.0, -32768, 32767).astype(np.int16)
    with _quiet(): pm_t, pm, _, _, _ = pyreaper.reaper(x16, int(sr), minf0=float(minf0), maxf0=float(maxf0))
    pm_t, pm = np.asarray(pm_t, "f8"), np.asarray(pm)
    return np.sort(pm_t[pm == 1])


def voiced_times(y16, a, b, sr=16000):
    """voiced 10 ms grid times (s, file time) in the stretch a..b"""
    seg = y16[int(a * sr):int(b * sr)].astype(np.float64)
    if len(seg) < 1600: return np.zeros(0)
    try: mk = reaper_epochs(seg, sr)
    except ImportError: raise
    except Exception: return np.zeros(0)
    if len(mk) < 2: return np.zeros(0)
    tc = (np.arange(int((b - a) / 0.01)) + 0.5) * 0.01
    i = np.searchsorted(mk, tc, side="right") - 1; ok = (i >= 0) & (i < len(mk) - 1)
    T = np.full(len(tc), np.nan); T[ok] = mk[i[ok] + 1] - mk[i[ok]]
    return a + tc[(T >= 0.002) & (T <= 0.040)]


def stretches(intervals, dur, pad=PAD):
    """word intervals +- pad, merged where they overlap"""
    iv = sorted((max(0.0, a - pad), min(dur, b + pad)) for a, b in intervals); out = []
    for a, b in iv:
        if out and a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out


def voiced_for_words(y16, intervals, sr=16000):
    dur = len(y16) / sr; parts = [voiced_times(y16, a, b, sr) for a, b in stretches(intervals, dur)]
    return np.concatenate(parts) if parts else np.zeros(0)
