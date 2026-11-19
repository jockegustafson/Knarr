"""Exports Knarr v2 (ablation v, TTS report section 60.14) for the knarr / f4n2_creak packages: per seed the state dict without the
training-only heads (v_*, d_creak) and the feature mu / sd -> <weights dir>/knarr_v2.npz + MANIFEST_knarr.json; the previous model's
manifest is kept as MANIFEST_knarr_e.json.
  python export_knarr_v.py <weights dir>   [~/knarr_data/venv]"""
import hashlib, io, json, shutil, sys
from pathlib import Path
import numpy as np, torch
R = Path.home() / "knarr_data/runs"; o = Path(sys.argv[1]); arr, src = {}, {}
for s in (1, 2, 3):
    p = R / f"v_s{s}.pt"; ck = torch.load(p, weights_only=False, map_location="cpu"); k = f"Knarr_v2_s{s}"
    for n, v in ck["state"].items():
        if n.startswith("heads.v_") or n.startswith("heads.d_creak"): continue
        arr[f"{k}/{n}"] = v.float().numpy()
    arr[f"{k}/mu"] = np.asarray(ck["mu"], np.float32); arr[f"{k}/sd"] = np.asarray(ck["sd"], np.float32)
    src[k] = {"best_step": ck["step"], "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
buf = io.BytesIO(); np.savez(buf, **arr); raw = buf.getvalue(); (o / "knarr_v2.npz").write_bytes(raw)
if not (o / "MANIFEST_knarr_e.json").exists(): shutil.copy(o / "MANIFEST_knarr.json", o / "MANIFEST_knarr_e.json")
T = json.load(open(R / "test_eval.json"))["auc"]["Knarr v"]
(o / "MANIFEST_knarr.json").write_text(json.dumps({"model": "knarr_v2", "date": "2026-11-19", "seeds": list(src), "npz_sha256": hashlib.sha256(raw).hexdigest(),
    "source_checkpoints": src,
    "what": "Knarr v2 (TTS report section 60.14): Knarr e retrained with channel augmentation (mp3, phone band, low-pass, white/pink noise, reverb; "
            "half of all training windows read from the degraded copy, clean targets) and without formant_tracker's reserved Spontal SPLIT_V4 TEST channels and persons",
    "word_rule": "as F4N2 (150-frame window centred on the word, top-3 frames of the perc head, mean over seeds) but WITHOUT the voicing gate",
    "test": f"Swedish creak test set (274 words, user labels): AUC {T['all']} (Knarr e 0.874, F4N2 gated 0.796); women {T['women']}, men {T['men']}; pool-weighted {T['pool_weighted']}",
    "robustness": "development words, 8 channel conditions: mean |score change| 0.074 (Knarr e 0.123), mean signed change -0.00 (Knarr e -0.09)"}, indent=1))
print("wrote", o / "knarr_v2.npz", round(len(raw) / 1e6, 1), "MB")
