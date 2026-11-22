"""knarr AUDIO WORDS [--model knarr|f4n2] [--channel N] [--no-gate] [--context] [--device cuda|cpu] [-o out.tsv]

WORDS: a Praat TextGrid (the first interval tier, or --tier NAME; empty intervals skipped) or a TSV/CSV: with a header naming the columns
start, end and optionally word (any order, so this tool's own output works), or without a header: start, end[, word]. Writes TSV: word, start, end, score, status, n_support.
--context adds a column `context`: how likely creak is on the word from what leads into it alone (knarr.context; needs pyreaper)."""
import argparse
import csv
import sys


def read_words(path, tier=None):
    if str(path).lower().endswith(".textgrid"):
        import re
        txt = open(path, encoding="utf-8", errors="replace").read()
        tiers = re.split(r'item \[\d+\]:', txt)[1:]
        for t in tiers:
            name = re.search(r'name = "(.*?)"', t).group(1)
            if 'IntervalTier' not in t or (tier and name != tier): continue
            iv = re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*?)"', t, re.S)
            return [(w.strip(), float(a), float(b)) for a, b, w in iv if w.strip()]
        sys.exit(f"no interval tier {tier or ''} in {path}")
    rows = [r for r in csv.reader(open(path, encoding="utf-8"), delimiter="\t" if str(path).endswith(".tsv") else ",") if r]
    head = [c.strip().lower() for c in rows[0]] if rows else []
    if "start" in head and "end" in head:                 # a header names the columns (any order; "word" or "label" optional)
        i, j = head.index("start"), head.index("end"); k = head.index("word") if "word" in head else head.index("label") if "label" in head else None
        return [(r[k] if k is not None else None, float(r[i]), float(r[j])) for r in rows[1:]]
    out = []
    for r in rows:                                        # no header: start, end[, word]
        try: out.append((r[2] if len(r) >= 3 else None, float(r[0]), float(r[1])))
        except ValueError: continue
    return out


def main():
    ap = argparse.ArgumentParser(description="Knarr / F4N2 creak word detector"); ap.add_argument("audio"); ap.add_argument("words")
    ap.add_argument("--channel", type=int); ap.add_argument("--tier"); ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--device"); ap.add_argument("-o", "--out"); ap.add_argument("--model", default="knarr", choices=["f4n2", "knarr", "knarr_e"])
    ap.add_argument("--context", action="store_true", help="add the context model's creak expectation per word (needs pyreaper)")
    a = ap.parse_args()
    from . import CreakDetector
    det = CreakDetector(device=a.device, gate=False if a.no_gate else None, model=a.model); res = det.score_file(a.audio, read_words(a.words, a.tier), channel=a.channel)
    ctx = None
    if a.context:
        from .context import context_scores
        ctx = context_scores(a.audio, [(r["label"], r["start"], r["end"]) for r in res], channel=a.channel)
    f = open(a.out, "w", newline="") if a.out else sys.stdout; w = csv.writer(f, delimiter="\t")
    w.writerow(["word", "start", "end", "score", "status", "n_support"] + (["context"] if ctx is not None else []))
    for k, r in enumerate(res):
        w.writerow([r["label"] or "", f'{r["start"]:.3f}', f'{r["end"]:.3f}', "" if r["score"] is None else f'{r["score"]:.4f}', r["status"], r["n_support"]]
                   + ([f"{ctx[k]:.4f}"] if ctx is not None else []))


if __name__ == "__main__":
    main()
