"""Writes knarr/weights/f4n2_v1.npz + MANIFEST.json from the verified formant_tracker release (releases/f4n2_v1, read-only):
per seed its state dict (float32 arrays, key "<seed>/<param>") and the feature normalisation ("<seed>/mu", "<seed>/sd").
  python export_weights.py <package dir>   [env formant, on the 5090]"""
import hashlib, io, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path.home() / "formant_tracker/src"))
from formant_tracker.analysis import f4_release as REL
d, man, _ = REL.verify(check_wavlm=False)
arr, seeds = {}, []
for k in sorted(man["models"]):
    ck = torch.load(d / man["models"][k]["path"], weights_only=False, map_location="cpu"); seeds.append(k)
    for p, v in ck["state"].items(): arr[f"{k}/{p}"] = v.float().numpy()
    arr[f"{k}/mu"] = np.asarray(ck["mu"], np.float32); arr[f"{k}/sd"] = np.asarray(ck["sd"], np.float32)
buf = io.BytesIO(); np.savez(buf, **arr); raw = buf.getvalue()
o = Path(sys.argv[1]) / "knarr/weights"; o.mkdir(parents=True, exist_ok=True); (o / "f4n2_v1.npz").write_bytes(raw)
(o / "MANIFEST.json").write_text(json.dumps({"release": "f4n2_v1", "date": man["date"], "seeds": seeds, "npz_sha256": hashlib.sha256(raw).hexdigest(),
    "source_checkpoints": {k: man["models"][k]["sha256"] for k in seeds}, "wavlm": {"repo": man["wavlm"]["repo"], "revision": man["wavlm"]["revision"], "layers": man["wavlm"]["layers"]},
    "word_rule": man["word_rule"], "gate_rule": man["gate_rule"], "threshold_provenance": man["threshold_provenance"]}, indent=1))
print("wrote", o, len(raw) / 1e6, "MB", seeds)
