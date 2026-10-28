#!/usr/bin/env python3
"""Call the running service with one photo and P92's reply.

    TL_TOKEN=... python -m eraser.client_example http://localhost:8080 photo.jpg p92_reply.json out.jpg

Writes the erased photo to out.jpg, or a copy of the original when nothing was erased.
"""
import base64
import json
import os
import shutil
import sys
import urllib.request

url, photo, reply, out = sys.argv[1:5]
body = json.dumps({"id": os.path.basename(photo), "image_b64": base64.b64encode(open(photo, "rb").read()).decode(),
                   "worker_reply": json.load(open(reply, encoding="utf-8")), "format": "jpeg", "quality": 95}).encode()
headers = {"Content-Type": "application/json"}
if os.environ.get("TL_TOKEN"):
    headers["Authorization"] = "Bearer " + os.environ["TL_TOKEN"]
req = urllib.request.Request(url.rstrip("/") + "/erase", data=body, headers=headers)
with urllib.request.urlopen(req, timeout=600) as r:
    res = json.loads(r.read())
if "image_b64" in res:
    open(out, "wb").write(base64.b64decode(res["image_b64"]))
else:
    shutil.copyfile(photo, out)              # held / nothing_to_erase / too_large / rotated_input: ship the original
print(res["status"], json.dumps(res["info"]))
