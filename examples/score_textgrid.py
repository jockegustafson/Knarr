"""Score every word of a TextGrid's word tier with F4N2 and write the scores as a new tier "creak" (and a TSV).

    python score_textgrid.py dialog.wav dialog.TextGrid --tier words [--channel 0] [--threshold 0.5]
    -> dialog.creak.TextGrid (the input tiers + "creak": the score per word, "" where F4N2 could not score it)
       dialog.creak.tsv
"""
import argparse
from pathlib import Path

from knarr import CreakDetector
from knarr.cli import read_words


def write_textgrid(src, out, res, xmax):
    """the original file with one more interval tier appended (long text format)"""
    import re
    txt = open(src, encoding="utf-8", errors="replace").read()
    n = int(re.search(r"size = (\d+)", txt).group(1))
    txt = re.sub(r"size = \d+", f"size = {n + 1}", txt, count=1)
    iv, t = [], 0.0
    for r in res:                                         # fill gaps so the tier is continuous
        if r["start"] > t + 1e-6: iv.append((t, r["start"], ""))
        iv.append((r["start"], r["end"], "" if r["score"] is None else f'{r["score"]:.3f}')); t = r["end"]
    if t < xmax: iv.append((t, xmax, ""))
    tier = [f"    item [{n + 1}]:", '        class = "IntervalTier"', '        name = "creak"', "        xmin = 0", f"        xmax = {xmax}",
            f"        intervals: size = {len(iv)}"]
    for k, (a, b, s) in enumerate(iv, 1):
        tier += [f"        intervals [{k}]:", f"            xmin = {a}", f"            xmax = {b}", f'            text = "{s}"']
    open(out, "w", encoding="utf-8").write(txt.rstrip() + "\n" + "\n".join(tier) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("audio"); ap.add_argument("textgrid"); ap.add_argument("--tier")
    ap.add_argument("--channel", type=int); ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    words = read_words(a.textgrid, a.tier)
    det = CreakDetector()
    res = det.score_file(a.audio, words, channel=a.channel)
    scored = [r for r in res if r["score"] is not None]
    print(f"{len(res)} words, {len(scored)} scored, {sum(r['score'] >= a.threshold for r in scored)} at or above {a.threshold}")
    stem = Path(a.textgrid).with_suffix("")
    with open(f"{stem}.creak.tsv", "w") as f:
        f.write("word\tstart\tend\tscore\tstatus\n")
        for r in res: f.write(f'{r["label"]}\t{r["start"]:.3f}\t{r["end"]:.3f}\t{"" if r["score"] is None else round(r["score"], 4)}\t{r["status"]}\n')
    import soundfile as sf
    write_textgrid(a.textgrid, f"{stem}.creak.TextGrid", res, sf.info(a.audio).duration)
    print("wrote", f"{stem}.creak.tsv", f"{stem}.creak.TextGrid")


if __name__ == "__main__":
    main()
