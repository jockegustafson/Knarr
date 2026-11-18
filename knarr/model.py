"""The F4N2 network and its exact word scorer, copied from formant_tracker (src/formant_tracker/analysis/f4_word.py, release f4n2_v1,
2026-10-30) so this package needs nothing from that repository. Do not change the numbers here: they are the trained geometry.

Word value: for each of the three seeds, the network runs on a 150-frame (3 s) window centred on the word (features padded with 0.0 beyond
the recording, then normalised with the seed's mu / sd); the perceptual head's sigmoid is pooled as the mean of the top-3 frames inside the
word (gate V: only frames with a REAPER-low voiced time within +-15 ms, >= 2 needed; ungated: the word's frames, >= 1); the word value is the
mean over seeds. Too few supported frames -> INSUFFICIENT_SUPPORT, never a number.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

WIN = 150
MECH = ("ipi", "irr", "m23", "long", "cq", "lf0", "sup")
INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"


class Net(nn.Module):
    def __init__(self, d_in=2048, h=256):
        super().__init__(); self.inp = nn.Sequential(nn.Linear(d_in, h), nn.GELU())
        self.blocks = nn.ModuleList([nn.Conv1d(h, h, 3, padding=d, dilation=d) for d in (1, 2, 4)]); self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])
        self.heads = nn.ModuleDict({k: nn.Linear(h, 1) for k in MECH + ("stod", "perc")})

    def forward(self, x):
        z = self.inp(x)
        for c, n in zip(self.blocks, self.norms): z = n(z + Fn.gelu(c(z.transpose(1, 2)).transpose(1, 2)))
        return {k: h(z)[..., 0] for k, h in self.heads.items()} | {"_z": z}


def fr(t):
    """model frame index of time t (s): frames are 20 ms, frame k centred at 0.0125 + 0.02 k"""
    return int(max(0, np.floor((t - 0.0125) / 0.02 + 0.5)))


def frame_centres(n):
    return 0.0125 + 0.02 * np.arange(n)


def word_window(a, b, n):
    """(start frame, word mask over the 150-frame window)"""
    fa, fb = fr(a), max(fr(a) + 1, fr(b)); s = max(0, min((fa + fb) // 2 - WIN // 2, n - WIN))
    m = np.zeros(WIN, bool); m[max(0, fa - s):max(0, min(WIN, fb - s))] = True
    if not m.any(): m[min(WIN - 1, max(0, fa - s))] = True
    return s, m


def window_feats(X, s):
    w = np.asarray(X[s:s + WIN], np.float32)
    return w if len(w) >= WIN else np.concatenate([w, np.zeros((WIN - len(w), X.shape[1]), np.float32)])


def topk_pool(p, m, min_kept):
    """p, m: (N, WIN); mean of the top-3 kept probabilities; NaN where fewer than min_kept frames are kept"""
    q = np.where(m, p, -1.0); k = np.sort(q, 1)[:, ::-1][:, :3]; valid = k >= 0; n = m.sum(1)
    s = (np.clip(k, 0, None) * valid).sum(1) / np.maximum(valid.sum(1), 1); s[n < min_kept] = np.nan
    return s, n


def keep_voiced(frame_t, voiced_t, tol=0.015):
    """gate V: a model frame is kept if a voiced REAPER-low time lies within +-15 ms of its centre"""
    if voiced_t is None or len(voiced_t) == 0: return np.zeros(len(frame_t), bool)
    v = np.sort(np.asarray(voiced_t, float)); i = np.searchsorted(v, frame_t - tol)
    return (i < len(v)) & (v[np.minimum(i, len(v) - 1)] <= frame_t + tol)


def score_words(X, intervals, nets, voiced_t=None, gate=True, device="cpu", batch=256):
    """X: (T, 2048) features. intervals: list of (a, b) seconds relative to X's frame 0. nets: list of (Net, mu, sd).
    voiced_t: REAPER-low voiced times (s) for gate V. Returns a list of dicts {score | status, n_support, per_seed}."""
    n = len(X); kept = keep_voiced(frame_centres(n), voiced_t) if gate else None; out = []
    for b0 in range(0, len(intervals), batch):
        iv = intervals[b0:b0 + batch]; W, M = [], []
        for a, b in iv:
            s, m = word_window(a, b, n); W.append(window_feats(X, s))
            if gate:
                k = np.zeros(WIN, bool); seg = kept[s:s + WIN]; k[:len(seg)] = seg; m = m & k
            M.append(m)
        W = np.stack(W); M = np.stack(M); per = []
        with torch.no_grad():
            for net, mu, sd in nets:
                x = (torch.tensor(W, device=device) - torch.as_tensor(mu, device=device)) / torch.as_tensor(sd, device=device)
                p = torch.sigmoid(net(x)["perc"]).float().cpu().numpy(); per.append(topk_pool(p, M, 2 if gate else 1))
        for j in range(len(iv)):
            ns = int(per[0][1][j]); vals = [float(q[0][j]) for q in per]
            if any(np.isnan(v) for v in vals): out.append({"status": INSUFFICIENT_SUPPORT, "n_support": ns})
            else: out.append({"score": float(np.mean(vals)), "n_support": ns, "per_seed": [round(v, 6) for v in vals]})
    return out


def frame_evidence(X, nets, device="cpu", hop=50):
    """uncalibrated local evidence: seed-mean sigmoid per model frame from overlapping 150-frame windows, central `hop` frames kept.
    For display only: it is not a frame probability or a creak boundary."""
    n = len(X); P = np.zeros(n); cnt = np.zeros(n)
    starts = list(range(0, max(1, n - WIN + 1), hop)) or [0]
    if starts[-1] + WIN < n: starts.append(max(0, n - WIN))
    for st in starts:
        w = window_feats(X, st)[None]; L = min(WIN, n - st)
        with torch.no_grad():
            pr = np.mean([torch.sigmoid(net((torch.tensor(w, device=device) - torch.as_tensor(mu, device=device)) / torch.as_tensor(sd, device=device))["perc"])[0].float().cpu().numpy()
                          for net, mu, sd in nets], 0)
        lo = 0 if st == 0 else (WIN - hop) // 2; hi = L if st + WIN >= n else (WIN + hop) // 2
        P[st + lo:st + hi] += pr[lo:hi]; cnt[st + lo:st + hi] += 1
    return np.where(cnt > 0, P / np.maximum(cnt, 1), np.nan)
