#!/usr/bin/env python3
"""The second layer, targeted: repair only what FLUX generated, and never outside the first mask.

The rule after the repaint-v2 review (2026-10-11): a fix may not change the photograph beyond
the mask that was meant to be repainted. v2 broke it -- its disguised repaint was pasted through the
pipeline's outward feather, which reaches ~24 px past the mask, so FLUX's fresh paint of the disguise
margin leaked into the real photo (up to 0.77% of the frame outside the mask on s174).

So: start from the first repaint as it is, and act only where FLUX generated text.
  1 find it      CRAFT (region score > 0.25) inside the first-pass hole, blobs of 20 px or more
  2 outline it   SAM2, prompted with a box around each cluster of blobs (grown by a quarter), segments
                 the generated object FLUX painted around the letters -- the band on s089, a label --
                 so its outline is repaired with it, not left behind
  3 confine it   repair = (SAM2's outline | the blobs grown 4 px) & the first-pass mask & not protected
  4 repaint it   FLUX (disguised view: it sees the repair area grown by an irregular margin, so the
                 band's silhouette is not handed back to it), two seeds; pasted through a feather that
                 lies entirely INSIDE the repair area -- alpha is 0 everywhere outside it
  5 keep it      only if CRAFT text inside the repair area drops by at least half, and CRAFT text in
                 the whole first-pass hole does not grow
No LaMa anywhere. Photos where CRAFT finds nothing are left exactly as they are.

Output, as a run <tag>: results/<tag>_h0 (the first repaint used), results/L_<tag>_h0 (after the
repair), masks/<tag>_h0 (the first-pass mask), masks/L_<tag>_h0 (the repair area), the starting-mask
parts for the review page, records/items_<tag>.json. Then the plate carve and the gate as usual.

    python3 run_layer2_targeted.py --chain X92LP3,X92LAC,X92L --tag X92LT --keep-first X92LP3
"""
from __future__ import annotations

import argparse
import json
import shutil
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

from pipeline import erase, masking                 # noqa: E402
from pipeline.run import seed_for                   # noqa: E402
from run_maskfix import guard_mask, keep_protected  # noqa: E402
from run_disguise import disguise                   # noqa: E402
from run_layer2_sam import survivors                # noqa: E402
import detect_leftover as D                         # noqa: E402

LEVEL, MIN_AREA = 0.25, 20


def inside_feather(mask, px=2):
    """A soft edge that lies entirely inside the mask: alpha is exactly 0 outside it."""
    core = cv2.erode(mask.astype(np.uint8), np.ones((2 * px + 1,) * 2, np.uint8))
    alpha = cv2.GaussianBlur(core.astype(np.float32), (0, 0), max(0.5, px * 0.75))
    return np.clip(alpha, 0, 1) * mask


def paste_inside(painted, base, mask, px=2):
    a = inside_feather(mask, px)[..., None]
    out = np.asarray(painted, np.float32) * a + np.asarray(base, np.float32) * (1 - a)
    return Image.fromarray(out.round().clip(0, 255).astype(np.uint8), "RGB")


def text_mass(craft, image, region):
    score, _ = D.region_score(craft, np.asarray(image))
    return int(((score > LEVEL) & region).sum()), score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", required=True, help="first-pass runs, best first")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--keep-first", default=None,
                    help="also write photos taken from this run even without a repair (e.g. a patch-fix run)")
    ap.add_argument("--margin", nargs=2, type=float, default=(0.01, 0.035))
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    chain = a.chain.split(",")
    items = {t: {i["id"]: i for i in json.loads((HERE / f"records/items_{t}.json").read_text(encoding="utf-8"))}
             for t in chain}
    low = a.tag.lower()
    dirs = {k: HERE / v for k, v in {"first": f"results/{a.tag}_h0", "after": f"results/L_{a.tag}_h0",
                                     "mask": f"masks/{a.tag}_h0", "repair": f"masks/L_{a.tag}_h0",
                                     "core": f"masks/core_{low}", "craft": f"masks/core_{low}c_craft"}.items()}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    from pipeline import load_all
    from craft_model import load_craft
    models, craft = load_all(), load_craft(D.CRAFT_W, "cuda")
    ids = sorted(set().union(*items.values()))[a.shard::a.shards]
    written = []
    with (HERE / f"records/{a.tag}.shard{a.shard}.jsonl").open("a", encoding="utf-8", buffering=1) as handle:
        for image_id in ids:
            t = next(t for t in chain if image_id in items[t])
            first_p, mask_p = HERE / f"results/{t}_h0/{image_id}.png", HERE / f"masks/{t}_h0/{image_id}.png"
            if not (first_p.exists() and mask_p.exists()):
                continue
            item = items[t][image_id]
            source = Image.open(item["source"]).convert("RGB")
            first = Image.open(first_p).convert("RGB")
            stencil = np.asarray(Image.open(mask_p).convert("L")) > 127
            guard = guard_mask(item, source.size) > 0.5
            before_all, score = text_mass(craft, first, stencil)
            blobs = survivors((score > LEVEL) & stencil, MIN_AREA)
            result, repair, row = first, np.zeros_like(stencil), {"id": image_id, "from": t, "blobs": len(blobs)}
            if blobs:
                hot = np.zeros_like(stencil)
                for b, _, _ in blobs:
                    hot |= b
                side = max(stencil.shape)
                clusters = cv2.dilate(hot.astype(np.uint8), np.ones((max(3, side // 50),) * 2, np.uint8))
                n, lab, st, _ = cv2.connectedComponentsWithStats(clusters, 8)
                boxes = []
                for c in range(1, n):
                    x, y, w, h = st[c, :4]
                    gx, gy = int(0.25 * w / 2) + 2, int(0.25 * h / 2) + 2
                    boxes.append((int(max(0, x - gx)), int(max(0, y - gy)),         # plain ints: SAM2's processor
                                  int(min(stencil.shape[1], x + w + gx)), int(min(stencil.shape[0], y + h + gy))))
                outline = np.zeros_like(stencil)
                for got in masking.sam_shapes(models["sam"], models["processor"], first, boxes):
                    if got is not None:
                        outline |= got > 127
                grown = cv2.dilate(hot.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                repair = (outline | grown) & stencil & ~guard
                before, _ = text_mass(craft, first, repair)
                best = None
                tries = []
                for k in range(1, a.tries + 1):
                    seed = seed_for(image_id) + 104729 + 7919 * k
                    shown = disguise(repair, a.margin[0], a.margin[1], seed) & ~guard
                    started = time.perf_counter()
                    painted = erase.run(models["flux"], models["embeds"], first,
                                        Image.fromarray((shown * 255).astype(np.uint8), "L"), seed=seed)
                    cand = paste_inside(painted, first, repair)
                    cand, _ = keep_protected(source, cand, guard)
                    after, _ = text_mass(craft, cand, repair)
                    after_all, _ = text_mass(craft, cand, stencil)
                    tries.append({"seed": seed, "text_in_repair": after, "text_in_hole": after_all,
                                  "seconds": round(time.perf_counter() - started, 1)})
                    if after_all <= before_all and (best is None or after < best[0]):
                        best = (after, cand)
                accept = best is not None and best[0] <= 0.5 * max(before, 1)
                if accept:
                    result = best[1]
                row.update({"repair_pct": round(100 * float(repair.mean()), 2), "sam_pct": round(100 * float(outline.mean()), 2),
                            "text_before": before, "tries": tries, "accepted": bool(accept)})
                if not accept:
                    repair = np.zeros_like(stencil)
            changed = bool(repair.any())
            row["written"] = bool(blobs) or t == a.keep_first   # with blobs: replaces the old LaMa layer 2
            handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if not row["written"]:
                continue
            first.save(dirs["first"] / f"{image_id}.png")
            result.save(dirs["after"] / f"{image_id}.png")
            shutil.copyfile(mask_p, dirs["mask"] / f"{image_id}.png")
            if changed:
                Image.fromarray((repair * 255).astype(np.uint8), "L").save(dirs["repair"] / f"{image_id}.png")
            base = t if not items[t][image_id].get("patchfix_from") else items[t][image_id]["patchfix_from"]
            for src_dir, dst in ((f"core_{t.lower()}", dirs["core"]), (f"core_{t.lower()}c_craft", dirs["craft"])):
                p = HERE / "masks" / src_dir / f"{image_id}.png"
                if p.exists():
                    shutil.copyfile(p, dst / f"{image_id}.png")
            written.append({**item, "layer2_targeted_from": t})
    (HERE / f"records/items_{a.tag}.shard{a.shard}.json").write_text(json.dumps(written, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"tag": a.tag, "shard": a.shard, "written": len(written)}))


if __name__ == "__main__":
    main()
