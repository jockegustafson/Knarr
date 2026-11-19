"""CreakDetector: load F4N2 (three seeds) and WavLM once, then score words in audio files or arrays."""
import hashlib
import json
from importlib import resources

import numpy as np
import torch

from .features import SR, InsufficientAudio, load_wavlm, to_16k, wavlm_l0_l3_features
from .gate import voiced_for_words
from .model import INSUFFICIENT_SUPPORT, Net, frame_centres, frame_evidence, score_words


MODELS = {"f4n2": ("MANIFEST.json", "f4n2_v1.npz"), "knarr": ("MANIFEST_knarr.json", "knarr_v2.npz"), "knarr_e": ("MANIFEST_knarr_e.json", "knarr_e.npz")}


def _weights(model="f4n2"):
    """the three seeds of a model from the package (npz: per seed its state dict and the feature mu / sd), hash-checked"""
    ref = resources.files("knarr") / "weights"; mf, wf = MODELS[model]
    man = json.loads((ref / mf).read_text())
    raw = (ref / wf).read_bytes()
    if hashlib.sha256(raw).hexdigest() != man["npz_sha256"]: raise RuntimeError(f"{wf} does not match its {mf}")
    import io
    z = np.load(io.BytesIO(raw)); out = []
    for seed in man["seeds"]:
        st = {k[len(seed) + 1:]: torch.from_numpy(z[k].copy()) for k in z.files if k.startswith(seed + "/") and k[len(seed) + 1:] not in ("mu", "sd")}
        out.append((st, z[f"{seed}/mu"], z[f"{seed}/sd"]))
    return out, man


class CreakDetector:
    """Creak word detector.
    model: "knarr" (default) = Knarr v2, F4N2 fine-tuned on Swedish dialogue with channel augmentation (2026-11-19; Swedish test set
           AUC 0.881 vs 0.796 for gated F4N2, pool-weighted 0.881 vs 0.740; scores stable under mp3, phone band, noise and reverb);
           "knarr_e" = the first Knarr (2026-11-18, AUC 0.874); both meant to be used without the voicing gate;
           "f4n2" = Joakim Gustafson's release f4n2_v1.
    device: "cuda", "cuda:1", "mps" or "cpu" (CPU works, about 10x slower for WavLM).
    gate: True = the voicing gate (needs pyreaper; words with < 2 voiced frames get status INSUFFICIENT_SUPPORT); False = score every word.
          Default: True for f4n2 (as released), False for knarr (as trained and tested)."""

    def __init__(self, device=None, gate=None, model="knarr"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu"); self.model = model
        self.gate = (model == "f4n2") if gate is None else gate
        W, self.manifest = _weights(model); self.nets = []
        for st, mu, sd in W:
            n = Net(); n.load_state_dict(st); n.eval().to(self.device)
            self.nets.append((n, torch.as_tensor(mu, device=self.device), torch.as_tensor(sd, device=self.device)))
        self.fe, self.wl = load_wavlm(self.device)

    # ---------------------------------------------------------------- input
    @staticmethod
    def load_audio(path, channel=None):
        """mono float array + sample rate; channel: None = mean of the channels, else that channel (0-based)"""
        import soundfile as sf
        y, sr = sf.read(str(path), dtype="float64", always_2d=True)
        y = y[:, channel] if channel is not None else y.mean(1)
        return y, sr

    def features(self, y, sr):
        """(y16, X): the conditioned 16 kHz signal and its (T, 2048) features; frame k is centred at 0.0125 + 0.02 k s"""
        y16 = to_16k(y, sr); return y16, wavlm_l0_l3_features(y16, self.fe, self.wl, self.device)

    # ---------------------------------------------------------------- scoring
    def score(self, y, sr, words, gate=None):
        """words: list of (start, end) or (label, start, end) in seconds from the start of y.
        Returns one dict per word: {label, start, end, score (0-1, None if not scored), status ("ok" | INSUFFICIENT_SUPPORT), n_support, per_seed}."""
        gate = self.gate if gate is None else gate
        W = [(None, float(w[0]), float(w[1])) if len(w) == 2 else (w[0], float(w[1]), float(w[2])) for w in words]
        try: y16, X = self.features(y, sr)
        except InsufficientAudio: return [{"label": l, "start": a, "end": b, "score": None, "status": INSUFFICIENT_SUPPORT, "n_support": 0} for l, a, b in W]
        iv = [(a, b) for _, a, b in W]
        voiced = voiced_for_words(y16, iv, SR) if gate else None
        res = score_words(X, iv, self.nets, voiced_t=voiced, gate=gate, device=self.device); out = []
        for (l, a, b), r in zip(W, res):
            out.append({"label": l, "start": a, "end": b, "score": r.get("score"), "status": r.get("status", "ok"), "n_support": r["n_support"],
                        "per_seed": r.get("per_seed")})
        return out

    def score_file(self, path, words, channel=None, gate=None):
        """score words of an audio file (any rate, any number of channels); the whole file is encoded once"""
        y, sr = self.load_audio(path, channel); return self.score(y, sr, words, gate)

    def evidence(self, y, sr):
        """(times, values): frame-level creak evidence for plotting, 20 ms frames. Uncalibrated display lane, not a frame probability."""
        _, X = self.features(y, sr); return frame_centres(len(X)), frame_evidence(X, self.nets, device=self.device)
