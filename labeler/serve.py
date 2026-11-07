#!/usr/bin/env python3
"""Local labeling server: python serve.py [audio_dir] [port]."""

import csv
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
AUDIO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "corpus" / "model"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
DENOISED = AUDIO.with_name(AUDIO.name + "_denoised")
OUT = HERE / f"labels_{AUDIO.name}.csv"
FIELDS = ["file", "sentence", "version", "text", "liaison", "close_mid_vowel"]


def items():
    out = []
    for wav in sorted(AUDIO.glob("*.wav")):
        sid, _, ver = wav.stem.partition("_")
        lab = wav.with_suffix(".lab")
        text = lab.read_text(encoding="utf-8").strip() if lab.exists() else ""
        out.append({"file": wav.name, "sentence": sid, "version": ver, "text": text})
    return out


def load():
    if not OUT.exists():
        return {}
    with OUT.open(encoding="utf-8", newline="") as f:
        return {
            r["file"]: {"liaison": r["liaison"], "close_mid_vowel": r["close_mid_vowel"]} for r in csv.DictReader(f)
        }


def save(labels):
    rows = []
    for it in items():
        lab = labels.get(it["file"], {})
        rows.append({**it, "liaison": lab.get("liaison", ""), "close_mid_vowel": lab.get("close_mid_vowel", "")})
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = unquote(self.path.split("?")[0])
        if p == "/":
            self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif p == "/api/data":
            self.send(
                200,
                json.dumps(
                    {"items": items(), "labels": load(), "out": OUT.name, "has_denoised": DENOISED.is_dir()}
                ).encode(),
                "application/json",
            )
        elif p.startswith("/audio/"):
            kind, _, name = p[7:].partition("/")
            f = (DENOISED if kind == "denoised" and DENOISED.is_dir() else AUDIO) / Path(name).name
            if f.suffix == ".wav" and f.exists():
                self.send(200, f.read_bytes(), "audio/wav")
            else:
                self.send(404, b"", "text/plain")
        else:
            self.send(404, b"", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            return self.send(404, b"", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        save(json.loads(self.rfile.read(n)))
        self.send(200, b"{}", "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if not OUT.exists():
        save({})
    print(f"audio: {AUDIO}\ndenoised: {DENOISED}\nsaving to: {OUT}\nhttp://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
