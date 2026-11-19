"""knarr-demo: a local web page for trying the creak detectors (record or load audio -> spectrogram with the creak decisions on top).

    knarr-demo [--port 8765] [--model knarr|f4n2] [--device cuda|mps|cpu] [--host 127.0.0.1]

then open http://localhost:8765 in a browser on the same machine (the microphone works on localhost; from another machine the browser
needs https, so load files there instead). The page decodes any audio the browser can play, sends it as 16 kHz mono, and the server answers:
  - a log spectrogram (0-8 kHz, 10 ms hop)
  - the detector run over the whole recording: windows of WIN_S = 0.2 s every HOP_S = 0.05 s scored with the word rule (top-3 frames of the
    window), so a creaky stretch shows as a run of high scores
  - optionally the words of a TextGrid or TSV you give, scored as words (the validated use)
Nothing leaves the machine. Recordings longer than MAX_S = 300 s are cut."""
import argparse, base64, io, json, os, tempfile, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import numpy as np
from .cli import read_words
from .model import score_words
from .gate import voiced_for_words

WIN_S, HOP_S, MAX_S = 0.2, 0.05, 300.0
DET = None; LOCK = threading.Lock()


def spectrogram(y, sr=16000, n_fft=512, hop=160):
    if len(y) < n_fft: y = np.pad(y, (0, n_fft - len(y)))
    w = np.hanning(n_fft).astype(np.float32); n = 1 + (len(y) - n_fft) // hop
    fr = np.lib.stride_tricks.as_strided(y, (n, n_fft), (y.strides[0] * hop, y.strides[0])) * w
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-6); top = S.max()
    S = np.clip((S - (top - 80)) / 80, 0, 1)                                    # 80 dB range
    return {"frames": int(n), "bins": int(S.shape[1]), "hop_s": hop / sr, "fmax": sr / 2,
            "data": base64.b64encode((S * 255).astype(np.uint8).tobytes()).decode()}


def analyze(y, words=None):
    det = DET; y = np.asarray(y, np.float32)[: int(MAX_S * 16000)]; dur = len(y) / 16000
    with LOCK:
        y16, X = det.features(y, 16000)
        iv = [(t, t + WIN_S) for t in np.arange(0, max(dur - WIN_S, 0) + 1e-9, HOP_S)]
        voiced = voiced_for_words(y16, iv, 16000) if det.gate else None
        win = score_words(X, iv, det.nets, voiced_t=voiced, gate=det.gate, device=det.device)
        wres = []
        if words:
            v2 = voiced_for_words(y16, [(a, b) for _, a, b in words], 16000) if det.gate else None
            for (lab, a, b), r in zip(words, score_words(X, [(a, b) for _, a, b in words], det.nets, voiced_t=v2, gate=det.gate, device=det.device)):
                wres.append({"word": lab, "start": a, "end": b, "score": r.get("score")})
    return {"dur": dur, "model": det.model, "gate": det.gate, "win_s": WIN_S, "hop_s": HOP_S, "spec": spectrogram(y16),
            "windows": [{"t": round(a + WIN_S / 2, 3), "score": r.get("score")} for (a, b), r in zip(iv, win)], "words": wres}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype):
        b = body if isinstance(body, bytes) else body.encode(); self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (resources.files("knarr") / "demo.html").read_bytes(), "text/html; charset=utf-8")
        else: self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/analyze": return self._send(404, "not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            y = np.frombuffer(base64.b64decode(req["pcm"]), np.float32)          # 16 kHz mono float32 from the page
            words = None
            if req.get("words"):
                suffix = ".TextGrid" if "IntervalTier" in req["words"] else ".tsv"
                with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f: f.write(req["words"])
                try: words = read_words(f.name, req.get("tier") or None)
                finally: os.unlink(f.name)
            self._send(200, json.dumps(analyze(y, words)), "application/json")
        except SystemExit as ex:
            self._send(400, json.dumps({"error": str(ex)}), "application/json")
        except Exception as ex:
            self._send(500, json.dumps({"error": f"{type(ex).__name__}: {ex}"}), "application/json")


def main():
    global DET
    ap = argparse.ArgumentParser(description="Knarr creak demo page (local)")
    ap.add_argument("--port", type=int, default=8765); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="knarr", choices=["knarr", "knarr_e", "f4n2"]); ap.add_argument("--device"); ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    from . import CreakDetector
    print("loading the detector and WavLM (first run downloads WavLM-large, ~1.2 GB) ...", flush=True)
    DET = CreakDetector(device=a.device, model=a.model)
    url = f"http://localhost:{a.port}"; print(f"Knarr demo ({a.model}, {DET.device}) at {url}  - Ctrl-C to stop", flush=True)
    if not a.no_browser: threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
