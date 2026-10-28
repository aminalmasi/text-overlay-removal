#!/usr/bin/env python3
"""HTTP service: one process = one copy of the models on the GPU, one photo at a time.

    TL_MODELS=/models TL_PREPARED=/models/prepared python -m eraser.server                  # 127.0.0.1:8080
    TL_TOKEN=... python -m eraser.server --host 0.0.0.0 --port 8080                         # reachable from outside

POST /erase  (JSON; header "Authorization: Bearer <TL_TOKEN>" when a token is set)
    {"id": "listing-123-photo-2",                 stable id; it seeds the repaints
     "image_b64": "<base64 JPEG/PNG bytes>",      upright: no EXIF rotation tag (see README)
     "worker_reply": {"text": "...", "boxes": [...]},   P92's reply, unchanged
     "format": "jpeg" | "png",                    optional, default jpeg
     "quality": 95}                               optional, JPEG quality 50..100
->  200 {"status": "erased", "image_b64": "<the erased photo>", "info": {...}}
    200 {"status": "held" | "nothing_to_erase" | "too_large" | "rotated_input", "info": {...}}
        (no image: ship your original bytes unchanged)
    400 bad request · 401 missing/wrong token · 413 body too large · 503 busy (queue full) or GPU out of memory

GET /health -> {"ok": true, "gpu": ..., "precision": ..., "loader": ..., "load_s": ..., "busy": ..., "waiting": n}

Photos are erased one at a time (a lock around the GPU work); at most --max-waiting requests wait for it, the rest
get 503 at once. A second process on another port fits on an A6000 but adds at most ~10% throughput.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import gc
import hmac
import io
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image                 # noqa: E402
from eraser.engine import Eraser      # noqa: E402

MAX_BODY = 40 * 2**20


def make_handler(eraser: Eraser, token: str | None, max_waiting: int):
    import torch
    gpu = torch.cuda.get_device_name(0)
    lock, slots = threading.Lock(), threading.BoundedSemaphore(max_waiting + 1)
    waiting = [0]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self):
            if not token:
                return True
            got = self.headers.get("Authorization", "")
            return hmac.compare_digest(got.encode(), f"Bearer {token}".encode())

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": "not found"})
            if not self._authorised():
                return self._send(401, {"error": "missing or wrong token"})
            self._send(200, {"ok": True, "gpu": gpu, "precision": eraser.precision, "loader": eraser.loader,
                             "load_s": eraser.load_seconds, "busy": lock.locked(), "waiting": waiting[0]})

        def do_POST(self):
            if self.path != "/erase":
                return self._send(404, {"error": "not found"})
            if not self._authorised():
                return self._send(401, {"error": "missing or wrong token"})
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = 0
            if n <= 0 or n > MAX_BODY:
                return self._send(413, {"error": f"body must be 1..{MAX_BODY} bytes"})
            try:
                req = json.loads(self.rfile.read(n))
                pid = str(req["id"])
                fmt = req.get("format", "jpeg")
                quality = int(req.get("quality", 95))
                if fmt not in ("jpeg", "png") or not 50 <= quality <= 100:
                    raise ValueError("format must be jpeg or png, quality 50..100")
                photo = Image.open(io.BytesIO(base64.b64decode(req["image_b64"], validate=True)))
                reply = req.get("worker_reply") or {}
                if not isinstance(reply, dict):
                    raise ValueError("worker_reply must be the worker's JSON object")
            except (KeyError, ValueError, TypeError, binascii.Error, OSError) as e:
                return self._send(400, {"error": f"bad request: {e!r}"})
            if not slots.acquire(blocking=False):
                return self._send(503, {"error": "busy: queue full, retry later"})
            try:
                waiting[0] += 1
                with lock:
                    waiting[0] -= 1
                    res = eraser.erase(photo, reply, photo_id=pid)
                payload = {"status": res["status"], "info": res["info"]}
                if res["status"] == "erased":
                    buf = io.BytesIO()
                    extra = {"icc_profile": res["icc_profile"]} if res.get("icc_profile") else {}
                    if fmt == "png":
                        res["image"].save(buf, "PNG", **extra)
                    else:
                        res["image"].save(buf, "JPEG", quality=quality, subsampling=0, **extra)
                    payload["image_b64"] = base64.b64encode(buf.getvalue()).decode()
                self._send(200, payload)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                self._send(503, {"error": "GPU out of memory on this photo; ship the original"})
            except Exception as e:
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()
                self._send(500, {"error": repr(e)})
            finally:
                slots.release()

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--precision", default="nf4", choices=["nf4", "bf16"])
    ap.add_argument("--prepared", default=os.environ.get("TL_PREPARED"))
    ap.add_argument("--max-waiting", type=int, default=8, help="requests allowed to queue for the GPU")
    ap.add_argument("--insecure", action="store_true", help="allow a non-local --host without TL_TOKEN")
    a = ap.parse_args()
    token = os.environ.get("TL_TOKEN") or None
    if a.host not in ("127.0.0.1", "localhost", "::1") and not token and not a.insecure:
        sys.exit("refusing to listen on a public address without TL_TOKEN (or pass --insecure on a private network)")
    eraser = Eraser(precision=a.precision, prepared=a.prepared)
    eraser.warmup()                   # the first repaint pays one-off kernel set-up; do it before taking traffic
    print(json.dumps({"listening": f"{a.host}:{a.port}", "token": bool(token), "load_s": eraser.load_seconds,
                      "loader": eraser.loader}), flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(eraser, token, a.max_waiting)).serve_forever()


if __name__ == "__main__":
    main()
