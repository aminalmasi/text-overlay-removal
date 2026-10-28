#!/usr/bin/env python3
"""Smoke-test a running eraser service: the normal case and the edge cases it must survive.

    TL_TOKEN=... python tests/service_smoke.py http://127.0.0.1:8080

Uses bench_samples/ (first sample) plus synthetic pictures. Exit code 0 when every check passes.
"""
import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent.parent
URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8080"
TOKEN = os.environ.get("TL_TOKEN")
fails = 0


def post(payload, token=TOKEN, raw=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with urllib.request.urlopen(urllib.request.Request(URL + "/erase", data=body, headers=headers), timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def b64(im, fmt="JPEG", **kw):
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return base64.b64encode(buf.getvalue()).decode()


def check(cond, what):
    global fails
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    fails += not cond


req = json.loads((HERE / "bench_samples/requests.jsonl").read_text(encoding="utf-8").splitlines()[0])
photo = Image.open(HERE / req["photo"])

code, res = post({"id": req["id"], "image_b64": b64(photo, quality=98), "worker_reply": req["worker_reply"]})
check(code == 200 and res.get("status") == "erased" and "image_b64" in res, f"real sample {req['id']}: erased ({res.get('status')})")

if TOKEN:
    code, _ = post({"id": "x", "image_b64": "", "worker_reply": {}}, token="wrong")
    check(code == 401, "wrong token -> 401")
code, _ = post(None, raw=b"{not json")
check(code == 400, "malformed JSON -> 400")
code, _ = post({"id": "x", "image_b64": "!!notbase64!!", "worker_reply": {}})
check(code == 400, "bad base64 -> 400")

pic = Image.new("RGB", (1000, 800), (90, 110, 130))
code, res = post({"id": "nothing", "image_b64": b64(pic), "worker_reply": {"boxes": []}})
check(code == 200 and res.get("status") == "nothing_to_erase" and "image_b64" not in res, "no boxes -> nothing_to_erase, no image")

bad = {"boxes": [{"box_2d": [1, 2, 3]}, {"box_2d": None}, {"box_2d": [-5, 10, 1200, 900], "physically_in_scene": "no"}, "x"]}
code, res = post({"id": "malformed", "image_b64": b64(pic), "worker_reply": bad})
check(code == 200 and res.get("status") == "nothing_to_erase", "malformed boxes -> handled, nothing erased")

# a mark wholly inside a protected surface (a plate covering ~6% of the frame): the stencil is empty
plate = [450, 350, 550, 650]
mark = [470, 400, 530, 600]
reply = {"boxes": [{"box_2d": plate, "text_box_2d": plate, "physically_in_scene": True, "carrier_present": False, "kind": "text"},
                   {"box_2d": mark, "text_box_2d": mark, "physically_in_scene": False, "carrier_present": False, "kind": "text"}]}
code, res = post({"id": "on-plate", "image_b64": b64(photo, quality=98), "worker_reply": reply})
check(code == 200 and res.get("status") in ("nothing_to_erase", "erased"), f"mark entirely on a protected plate -> no crash ({res.get('status')})")

big = Image.new("RGB", (3000, 2500), (120, 120, 120))
code, res = post({"id": "big", "image_b64": b64(big, quality=80), "worker_reply": req["worker_reply"]})
check(code == 200 and res.get("status") == "too_large", "7.5 MP photo -> too_large, original kept")

exif = Image.Exif()
exif[0x0112] = 6
code, res = post({"id": "rotated", "image_b64": b64(photo, exif=exif.tobytes()), "worker_reply": req["worker_reply"]})
check(code == 200 and res.get("status") == "rotated_input", "EXIF-rotated photo -> rotated_input, original kept")

with urllib.request.urlopen(urllib.request.Request(URL + "/health", headers={"Authorization": "Bearer " + TOKEN} if TOKEN else {}), timeout=30) as r:
    h = json.loads(r.read())
check(h.get("ok") is True, f"health: {h}")

print("\nALL CHECKS PASSED" if not fails else f"\n{fails} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
