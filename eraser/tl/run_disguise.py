#!/usr/bin/env python3
"""The first repaint with the hole's shape disguised.

FLUX-Fill takes the mask as an input, so it sees the hole's silhouette. When the hole is exactly
the shape of the overlay's carrier -- a rounded label, a translucent panel, a text band -- it
recognises the shape and paints that kind of thing back, text included: a dark label with fake
words on s089, the pink "iQ" panel over the tree on s111, a faint poster grid on s061.

So FLUX is given a different hole: the stencil grown by an irregular margin, a smooth random field
deciding how far it reaches at each point, so no straight edge, corner or band survives. Only the
stencil is then pasted back (through the same outward-only feather), so no more of the photograph
changes than today: halo 0 is untouched. Everything else is the signed-off first pass: same seed,
prompt, LoRA, guidance, crop logic, calibration and guard.

    python3 run_disguise.py --ids s014 s061 --margin 0.01 0.03 --tag DG1_h0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent   # deploy copy: this folder
sys.path.insert(0, str(ROOT))
HERE = ROOT / "experiments/creativity_v1"
sys.path.insert(0, str(HERE))

from pipeline import erase                          # noqa: E402
from pipeline.run import seed_for                   # noqa: E402
from run_maskfix import guard_mask, keep_protected  # noqa: E402


def disguise(stencil, low, high, seed):
    """The stencil plus a margin between low and high (fractions of the long side), varying
    smoothly and randomly along the boundary."""
    h, w = stencil.shape
    side = max(h, w)
    rng = np.random.default_rng(seed)
    coarse = rng.random((max(2, h // 48), max(2, w // 48))).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    dist = cv2.distanceTransform((~stencil).astype(np.uint8), cv2.DIST_L2, 5)
    reach = (low + (high - low) * field) * side
    return stencil | (dist < reach)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", type=Path, default=HERE / "records/items_W88.json")
    ap.add_argument("--base", default="W88_h0", help="masks/<base>: the stencils to disguise")
    ap.add_argument("--ids", nargs="+", help="default: every photo in --items")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--margin", nargs=2, type=float, required=True, metavar=("LOW", "HIGH"))
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    items = {str(i["id"]): i for i in json.loads(a.items.read_text(encoding="utf-8"))}
    ids = (a.ids or sorted(items))[a.shard::a.shards]
    out_dir, mask_dir = HERE / "results" / a.tag, HERE / "masks" / a.tag
    shown_dir = HERE / "masks" / f"{a.tag}_shown"
    for d in (out_dir, mask_dir, shown_dir):
        d.mkdir(parents=True, exist_ok=True)
    from pipeline import load_all
    t0 = time.perf_counter()
    models = load_all()
    print(json.dumps({"models_loaded_seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    records = HERE / (f"records/{a.tag}.shard{a.shard}.jsonl" if a.shards > 1 else f"records/{a.tag}.jsonl")
    with records.open("a", encoding="utf-8", buffering=1) as handle:
        for image_id in ids:
            if (out_dir / f"{image_id}.png").exists():
                continue
            if not (HERE / "masks" / a.base / f"{image_id}.png").exists():
                continue
            item = items[image_id]
            source = Image.open(item["source"]).convert("RGB")
            stencil_img = Image.open(HERE / "masks" / a.base / f"{image_id}.png").convert("L")
            stencil = np.asarray(stencil_img) > 127
            guard = guard_mask(item, source.size) > 0.5
            shown = disguise(stencil, a.margin[0], a.margin[1], seed_for(image_id)) & ~guard
            started = time.perf_counter()
            painted = erase.run(models["flux"], models["embeds"], source,
                                Image.fromarray((shown * 255).astype(np.uint8), "L"),
                                seed=seed_for(image_id))
            # keep only the stencil from FLUX's answer, through the pipeline's own outward feather
            result = Image.composite(painted, source, erase.feathered(stencil_img))
            result, restored = keep_protected(source, result, guard)
            result.save(out_dir / f"{image_id}.png")
            stencil_img.save(mask_dir / f"{image_id}.png")
            Image.fromarray((shown * 255).astype(np.uint8), "L").save(shown_dir / f"{image_id}.png")
            row = {"id": image_id, "margin": a.margin, "stencil_pct": round(100 * float(stencil.mean()), 2),
                   "shown_pct": round(100 * float(shown.mean()), 2),
                   "seconds": round(time.perf_counter() - started, 2)}
            handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
