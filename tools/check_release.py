"""Checks the package against the F4N2 scores written by the release scorer (creak_annot.py -> lung/creak/spontal.jsonl) on one Spontal
channel: same words, same file, package installed in a fresh venv.  python check_release.py [device]"""
import json, sys
from collections import defaultdict
import numpy as np
from knarr import CreakDetector
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
rows = defaultdict(list)
for l in open("/data/jkgu/voxtream/lung/creak/spontal.jsonl"):
    r = json.loads(l); rows[(r["audio"], r.get("ch"))].append(r)
(audio, ch), rs = sorted(rows.items(), key=lambda kv: -len(kv[1]))[5]
words = [(w[0], w[1], w[2]) for r in rs for w in r["words"]]; ref = [w[3] for r in rs for w in r["words"]]
det = CreakDetector(device=dev); out = det.score_file(audio, words, channel=ch)
a = np.array([o["score"] if o["score"] is not None else np.nan for o in out]); b = np.array([np.nan if v is None else v for v in ref], float)
both = np.isfinite(a) & np.isfinite(b)
import sys as _s; _s.stdout.flush(); print(audio, "ch", ch, len(words), "words; scored: package", int(np.isfinite(a).sum()), "release", int(np.isfinite(b).sum()), "both", int(both.sum()))
d = np.abs(a[both] - b[both]); print("abs diff median %.5f  p99 %.5f  max %.5f" % (np.median(d), np.percentile(d, 99), d.max()))
print("corr %.5f  same side of 0.5: %.4f" % (np.corrcoef(a[both], b[both])[0, 1], np.mean((a[both] >= .5) == (b[both] >= .5))))
print("scored by only one:", int((np.isfinite(a) ^ np.isfinite(b)).sum()))
