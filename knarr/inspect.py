"""knarr-inspect: a creak listening page for a set of words (one self-contained HTML file; open it in any browser, nothing is uploaded).

    knarr-inspect AUDIO WORDS [-o creak_inspect.html] [--channel N] [--tier NAME] [--scores knarr.tsv] [--pick all|uncertain|creaky|disagree]
                  [--max 200] [--seed 7] [--show-score] [--title TEXT] [--parts N]

WORDS as for `knarr` (TSV/CSV with start, end[, word], or a Praat TextGrid). --scores takes `knarr`'s own output for the same words: with
--pick uncertain only words scored 0.3-0.7 are kept (the ones worth a human ear), with --pick creaky those >= 0.5, and with --pick disagree
the words where Knarr and the context model (knarr.context: creak expected from the lead-up alone) disagree, as in the lab's round 4: 40 %
"unexpected" (Knarr >= 0.7, context in the file's lowest 20 %), 40 % "missed" (context in the top 10 %, Knarr <= 0.15), 20 % Knarr-unsure
(0.35-0.65); --show-score reveals the score after the word is labelled. Words shorter than 0.08 s are skipped; --max words are drawn at random (--seed).

Per word the page shows, on one time axis (0 = word start, the word +- 0.25 s): the waveform, zoomable to single glottal cycles, with
REAPER's glottal pulse marks (min F0 25 Hz, as the voicing gate); a wideband (4 ms, pulses as vertical striations) or narrowband (30 ms,
harmonics) spectrogram; the pitch (REAPER F0 line and the pulse rate 1/interval as dots, log 30-400 Hz); and the period of every cycle,
with a cycle that differs by more than 20 % from the previous one marked. Listening: the word, the word at half speed (pitch an octave
down, so single pulses become audible), the 2.5 s before it, a loop, or a dragged selection. Labels: creaky / not creaky / unsure / no speech
("nospeech": noise or a wrongly timed word, to be left out), a creak
type (phrase creak, glottal onset or stop, breathy-creaky) and any number of marked creaky stretches inside the word. They are kept in
the browser (localStorage) and saved with "Download labels (JSON)": [{code, word, src: {audio, start, end}, label, type, spans: [{s, e}]}],
span times in s from the word start.
With --parts N the words go to N script files beside the page (<out>_items_<k>.js, loaded by the page), for hosts that cap the size of
one file; a page of 300 words with its clips is about 15 MB.
If Huanchen Cai's VoiceMapping (github.com/HuanchenCai/VoiceMapping) is importable as `voicemap`, its audio-only IAIF cycle starts are added
as a second pulse layer (F0 30-500 Hz, no clarity filter). The context clip is mp3 when ffmpeg is on the PATH, otherwise 16 kHz WAV.
Needs pyreaper (pip install "knarr[gate]")."""
import argparse, base64, csv, io, json, random, shutil, subprocess, sys
from importlib import resources
from pathlib import Path

import numpy as np

from .cli import read_words
from .gate import _quiet

SR, PADV, PADA, PRE, POST, MIN_DUR = 16000, 0.25, 0.6, 2.5, 0.4, 0.08


def load(path, channel=None):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y[:, channel] if channel is not None and channel < y.shape[1] else y.mean(1)
    if sr != SR: g = gcd(SR, sr); y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def reaper(y):
    try:
        import pyreaper
    except ImportError as ex:
        raise SystemExit("knarr-inspect needs pyreaper: pip install pyreaper 'setuptools<81'") from ex
    x = np.clip(y / (np.abs(y).max() + 1e-9) * 0.9 * 32767, -32768, 32767).astype(np.int16)
    with _quiet(): pm_t, pm, f0_t, f0, _ = pyreaper.reaper(x, SR, minf0=25.0, maxf0=500.0, frame_period=0.01)
    return np.asarray(pm_t)[np.asarray(pm) == 1], np.asarray(f0_t), np.asarray(f0)


def iaif(y):
    """VoiceMapping IAIF cycle starts (s), or [] when voicemap is not installed"""
    try:
        from voicemap.analyzer import VoiceMapAnalyzer
        from voicemap.config import VoiceMapConfig
    except ImportError:
        return []
    an = VoiceMapAnalyzer(VoiceMapConfig(sample_rate=SR, min_frequency=30, max_period_samples=int(SR / 30), min_samples=int(SR / 500),
                                         pitch_floor_hz=30.0, n_min_midi=0, n_min_spl=0, clarity_threshold=0.0))
    with _quiet(): trig = an.voice_only_cycle_detection(an.preprocess_voice(y.astype(np.float64)))
    return list(np.where(trig > 0.5)[0] / SR)


def clip_uri(y):
    y = (y / (np.abs(y).max() + 1e-6) * 0.7).astype(np.float32)
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-b:a", "32k", "-f", "mp3", "-"],
                           input=y.tobytes(), capture_output=True)
        if r.returncode == 0 and r.stdout: return "data:audio/mpeg;base64," + base64.b64encode(r.stdout).decode()
    import soundfile as sf
    b = io.BytesIO(); sf.write(b, y, SR, format="WAV", subtype="PCM_16"); return "data:audio/wav;base64," + base64.b64encode(b.getvalue()).decode()


def item(y, s, e):
    """the page's data for the word [s, e] of the 16 kHz signal y"""
    a0 = max(0.0, s - PADA); ya = y[int(a0 * SR):int((e + PADA) * SR)]
    v0, v1 = max(0.0, s - PADV), min(len(y) / SR, e + PADV); lo, hi = v0 - a0, v1 - a0
    keep = lambda ts: [round(float(t - lo), 5) for t in ts if lo <= t < hi]
    pm, f0_t, f0 = reaper(ya)
    grid = np.arange(0, hi - lo, 0.01) + lo
    f0v = [round(float(f0[np.argmin(np.abs(f0_t - t))]), 1) if len(f0_t) else 0.0 for t in grid]
    f0v = [x if x > 0 else 0.0 for x in f0v]
    yv = ya[int(lo * SR):int(hi * SR)]; pcm = (yv / (np.abs(yv).max() + 1e-9) * 32000).astype("<i2")
    t0 = max(0.0, e - PRE); yc = y[int(t0 * SR):int((e + POST) * SR)]
    return {"pcm": base64.b64encode(pcm.tobytes()).decode(), "sr": SR, "vws": round(s - v0, 3), "vwe": round(e - v0, 3),
            "reaper": keep(pm), "iaif": keep(iaif(ya)), "f0": f0v, "audio": clip_uri(yc), "ws": round(s - t0, 3), "we": round(e - t0, 3)}


def context(words, k):
    w, s, e = words[k]
    before = [x[0] or "·" for x in words[:k] if x[2] > e - PRE and x[1] < s]
    after = []
    for x in words[k + 1:k + 4]:
        if x[1] > e + POST or x[0] == w: break
        after.append(x[0] or "·")
    return " ".join(before + [w or "·"] + after)


def read_scores(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    return {round(float(r["start"]), 2): float(r["score"]) for r in rows if r.get("score") not in (None, "", "nan")}


def disagree(y, words, cand, sc, n_max, seed):
    """the words where Knarr and the context model disagree (see the module doc), at most n_max, in time order"""
    from .context import features, predict
    import numpy as np
    p = predict(features(np.asarray(y, np.float64), words)); pc = np.argsort(np.argsort(p)) / max(len(p) - 1, 1)     # context percentile in the file
    v = {k: sc[round(words[k][1], 2)] for k in cand}; rnd = random.Random(seed)
    groups = [([k for k in cand if v[k] >= 0.7 and pc[k] <= 0.2], 0.4), ([k for k in cand if pc[k] >= 0.9 and v[k] <= 0.15], 0.4),
              ([k for k in cand if 0.35 <= v[k] <= 0.65], 0.2)]
    out = []
    for g, share in groups:
        g = [k for k in g if k not in out]; rnd.shuffle(g); out += g[:max(1, round(n_max * share))]
    return sorted(out[:n_max])


def build(audio, words, out, channel=None, scores=None, pick="all", n_max=200, seed=7, show_score=False, title="Creak Inspection", parts=0):
    y = load(audio, channel); words = sorted(((w, float(s), float(e)) for w, s, e in words), key=lambda x: x[1])
    cand = [k for k, (w, s, e) in enumerate(words) if e - s >= MIN_DUR and e <= len(y) / SR]
    sc = read_scores(scores) if scores else {}
    if pick != "all":
        if not sc: raise SystemExit("--pick needs --scores (knarr's output for these words)")
        cand = [k for k in cand if round(words[k][1], 2) in sc]
    if pick == "disagree":
        cand = disagree(y, words, cand, sc, n_max, seed)
    elif pick != "all":
        ok = (lambda v: 0.3 <= v <= 0.7) if pick == "uncertain" else (lambda v: v >= 0.5)
        cand = [k for k in cand if ok(sc[round(words[k][1], 2)])]
    if len(cand) > n_max: cand = sorted(random.Random(seed).sample(cand, n_max))
    items = []
    for n, k in enumerate(cand):
        w, s, e = words[k]; v = sc.get(round(s, 2))
        it = {"code": f"W{n + 1:03d}", "word": w or "·", "ctx": context(words, k), "src": {"audio": Path(audio).name, "start": s, "end": e},
              "reveal": f"Knarr score {v:.2f}" if show_score and v is not None else "", **item(y, s, e)}
        items.append(it); print(f"\r{n + 1}/{len(cand)} words", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    t = resources.files("knarr").joinpath("inspect.html").read_text(encoding="utf-8")
    page = "".join(c if c.isalnum() else "_" for c in f"{Path(audio).stem}_{Path(out).stem}")[:80]
    note = ("The score Knarr gave the word is shown after you label it." if show_score else "")
    data = json.dumps(items, ensure_ascii=False)
    if parts:                          # the items in script files beside the page (hosts with a size limit per file)
        names, k = [], -(-len(items) // parts)
        for j in range(parts):
            name = f"{Path(out).stem}_items_{j + 1}.js"; names.append(name)
            (Path(out).parent / name).write_text("(window.ITEMS_PARTS = window.ITEMS_PARTS || []).push("
                                                 + json.dumps(items[j * k:(j + 1) * k], ensure_ascii=False) + ");\n", encoding="utf-8")
        data = "[].concat(...(window.ITEMS_PARTS || []))"
        assert t.count("\n<script>\n") == 1; t = t.replace("\n<script>\n", "\n" + "".join(f'<script src="{n}"></script>\n' for n in names) + "<script>\n")
    for a, b in (("/*TITLE*/", title), ("/*NOTE*/", note), ("/*F0SRC*/", "REAPER F0"),
                 ("/*ITEMS*/[]", data), ('/*PAGE*/"page"', json.dumps(page))):
        assert a in t, a; t = t.replace(a, b)
    Path(out).write_text(t, encoding="utf-8")
    print(f"wrote {out}: {len(items)} words, {Path(out).stat().st_size / 1e6:.1f} MB" + (f" + {parts} item files" if parts else ""), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="knarr-inspect: a creak listening page for a set of words")
    ap.add_argument("audio"); ap.add_argument("words"); ap.add_argument("-o", "--out", default="creak_inspect.html")
    ap.add_argument("--channel", type=int); ap.add_argument("--tier"); ap.add_argument("--scores")
    ap.add_argument("--pick", default="all", choices=["all", "uncertain", "creaky", "disagree"]); ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--show-score", action="store_true"); ap.add_argument("--title", default="Creak Inspection")
    ap.add_argument("--parts", type=int, default=0, help="write the words to N script files beside the page (for hosts with a per-file size limit)")
    a = ap.parse_args()
    build(a.audio, read_words(a.words, a.tier), a.out, a.channel, a.scores, a.pick, a.max, a.seed, a.show_score, a.title, a.parts)


if __name__ == "__main__":
    main()
