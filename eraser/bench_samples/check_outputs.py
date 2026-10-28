#!/usr/bin/env python3
"""Compare a fresh install's outputs on the 24 sample photos with the reference outputs.

    python -m eraser.cli --requests bench_samples/requests.jsonl --out /tmp/tl_check --debug
    python bench_samples/check_outputs.py /tmp/tl_check

The references were made by this package (clean install, RTX A5000) and are bit-identical to the reviewed run on
all 24 photos, masks and final pictures. On the same GPU type expect identical results. On another GPU
type (A6000, L40) the masks should still match almost exactly (no FLUX involved), while FLUX's repaints can
differ slightly because its kernels are not bit-reproducible across GPU architectures: compare those by eye in
<out>/<id>.png against reference/<id>.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def main():
    out = Path(sys.argv[1])
    ref = {r["id"]: r for r in map(json.loads, open(HERE / "reference/results.jsonl"))}
    got = {r["id"]: r for r in map(json.loads, open(out / "results.jsonl"))}
    bad = 0
    print(f"{'photo':6} {'status':16} {'mask IoU':>9} {'final diff':>11}")
    for pid, r in sorted(ref.items()):
        g = got.get(pid)
        if g is None:
            print(f"{pid:6} MISSING"); bad += 1
            continue
        iou = diff = None
        a, b = out / "debug" / f"{pid}_stencil.png", HERE / "reference" / f"{pid}_stencil.png"
        if a.exists() and b.exists():
            x, y = np.asarray(Image.open(a)) > 127, np.asarray(Image.open(b)) > 127
            iou = float((x & y).sum() / max(1, (x | y).sum()))
        if (out / f"{pid}.png").exists():
            diff = float(np.abs(np.asarray(Image.open(out / f"{pid}.png").convert("RGB"), np.int16)
                                - np.asarray(Image.open(HERE / "reference" / f"{pid}.png").convert("RGB"), np.int16)).mean())
        same = g["status"] == r["status"]
        flag = "" if same and (iou is None or iou >= 0.99) else "  <-- check"
        bad += bool(flag)
        print(f"{pid:6} {g['status'] + ('' if same else ' (ref ' + r['status'] + ')'):16} "
              f"{'' if iou is None else f'{iou:.4f}':>9} {'' if diff is None else f'{diff:.2f}/255':>11}{flag}")
    s = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    if s:
        print(f"\n{s.get('gpu')}: median {s['per_photo_s']['median']} s, mean {s['per_photo_s']['mean']} s per photo, "
              f"{s.get('photos_per_hour')} photos/h, peak GPU memory {s['vram_gib']['process_peak_nvidia_smi']} GiB, "
              f"model load {s.get('model_load_s')} s ({s.get('loader')})")
    print("\nOK: masks and statuses match the reference" if not bad else f"\n{bad} photo(s) to check")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
