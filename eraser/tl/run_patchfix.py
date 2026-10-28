#!/usr/bin/env python3
"""Find first-repaint holes that show as a patch, and repaint just those -- keeping the new paint
only when it measurably blends better.

On smooth surroundings FLUX sometimes fills a watermark-sized hole with a flat tone a shade off, or
with speckle (s032, s045, s060 in the compare_best review, 2026-10-11). patchcheck.py measures each
hole's grain and the colour step across its edge and flags those past its thresholds. A flagged
hole is repainted with the disguised hole (FLUX sees the hole grown by an irregular margin, which
cleared s032's speckle before) and a new seed, twice; only the hole itself is pasted back. The best
try replaces the first repaint only if its score is lower than the original's -- so a false alarm
costs a few seconds and never makes a photo worse.

v2 (after looking at v1's output): a repaint can lower the score by inventing something -- s151's
blank phone header came back with fake text and was accepted. A candidate is now refused when CRAFT
finds more text inside the hole than in the first repaint. And a tone step that a repaint does not
remove (s060, s045: a flat patch a shade off) gets a colour correction: on smooth surroundings the
hole is shifted by the median Lab step measured across its edge, tried on the first repaint and on
every repainted candidate, kept only when it scores better.

Each photo is taken from the first run in --chain that has it (the best-current composite), and the
accepted ones are written as a new run <tag>: results/<tag>_h0 (the first repaint), masks/<tag>_h0
(the same stencil), the starting-mask parts for the review page, and records/items_<tag>.json, so
layer 2, the plate carve and the gate run on it as on any other first pass.

    python3 run_patchfix.py --chain X92LAC,X92L --tag X92LP
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

from pipeline import erase                          # noqa: E402
from pipeline.run import seed_for                   # noqa: E402
from run_maskfix import guard_mask, keep_protected  # noqa: E402
from run_disguise import disguise                   # noqa: E402
import patchcheck as P                              # noqa: E402
import detect_leftover as D                         # noqa: E402
from run_layer2_targeted import paste_inside        # noqa: E402

bgr = lambda im: cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def tone_fixed(source, painted, stencil, region):
    """painted with the region shifted by the median Lab step across its edge (smooth ground only)."""
    O = cv2.cvtColor(bgr(source), cv2.COLOR_BGR2LAB).astype(np.float32)
    R = cv2.cvtColor(bgr(painted), cv2.COLOR_BGR2LAB).astype(np.float32)
    din = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    dout = cv2.distanceTransform((~stencil).astype(np.uint8), cv2.DIST_L2, 5)
    near = cv2.dilate(region.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    band_in, band_out = region & (din >= 3) & (din <= 6), near & ~stencil & (dout >= 3) & (dout <= 6)
    if band_in.sum() < 40 or band_out.sum() < 40:
        return painted
    step = np.median(R[band_in], axis=0) - np.median(O[band_out], axis=0)
    R[region] -= step
    out = cv2.cvtColor(R.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def text_in(craft, image, region):
    """CRAFT text pixels (region score > 0.25) inside the region."""
    score, _ = D.region_score(craft, np.asarray(image))
    return int(((score > 0.25) & region).sum())


def worst(source, painted, stencil, region):
    """The badness of the worst flagged-measure hole overlapping `region` in `painted`."""
    rows = [r for r in P.holes(bgr(source), bgr(painted), stencil) if (r["mask"] & region).any()]
    return max((P.badness(r) for r in rows), default=0.0), rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chain", required=True, help="runs, best first: the composite's order")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--margin", nargs=2, type=float, default=(0.01, 0.035))
    ap.add_argument("--tries", type=int, default=2)
    a = ap.parse_args()
    chain = a.chain.split(",")
    items = {t: {i["id"]: i for i in json.loads((HERE / f"records/items_{t}.json").read_text(encoding="utf-8"))}
             for t in chain}
    low = a.tag.lower()
    out_res, out_mask, out_shown = HERE / f"results/{a.tag}_h0", HERE / f"masks/{a.tag}_h0", HERE / f"masks/{a.tag}_shown"
    out_core, out_craft = HERE / f"masks/core_{low}", HERE / f"masks/core_{low}c_craft"
    for d in (out_res, out_mask, out_shown, out_core, out_craft):
        d.mkdir(parents=True, exist_ok=True)
    ids = sorted(set().union(*items.values()))
    from pipeline import load_all
    models = craft = None
    accepted, log = [], []
    records = HERE / f"records/{a.tag}.jsonl"
    with records.open("a", encoding="utf-8", buffering=1) as handle:
        for image_id in ids:
            t = next(t for t in chain if image_id in items[t])
            first_p, mask_p = HERE / f"results/{t}_h0/{image_id}.png", HERE / f"masks/{t}_h0/{image_id}.png"
            if not (first_p.exists() and mask_p.exists()):
                continue
            item = items[t][image_id]
            source = Image.open(item["source"]).convert("RGB")
            first = Image.open(first_p).convert("RGB")
            stencil = np.asarray(Image.open(mask_p).convert("L")) > 127
            rows = P.holes(bgr(source), bgr(first), stencil)
            flags = [r for r in rows if P.flagged(r)]
            if not flags:
                continue
            region = np.zeros_like(stencil)
            for r in flags:
                region |= r["mask"]
            before = max(P.badness(r) for r in flags)
            if models is None:
                models = load_all()
            guard = guard_mask(item, source.size) > 0.5
            region_img = Image.fromarray((region * 255).astype(np.uint8), "L")
            if craft is None:
                from craft_model import load_craft
                craft = load_craft(D.CRAFT_W, "cuda")
            text0 = text_in(craft, first, region)
            best = None
            tries = []
            cand = tone_fixed(source, first, stencil, region)       # colour correction alone
            score, _ = worst(source, cand, stencil, region)
            tries.append({"seed": None, "tone": True, "score": round(score, 2), "text_px": text_in(craft, cand, region) - text0})
            if tries[-1]["text_px"] <= 20:
                best = (score, cand, region)
            for k in range(1, a.tries + 1):
                seed = seed_for(image_id) + 7919 * k
                shown = disguise(region, a.margin[0], a.margin[1], seed) & ~guard
                started = time.perf_counter()
                painted = erase.run(models["flux"], models["embeds"], first,
                                    Image.fromarray((shown * 255).astype(np.uint8), "L"), seed=seed)
                candidate = paste_inside(painted, first, region)        # never outside the hole
                candidate, _ = keep_protected(source, candidate, guard)
                for tone in (False, True):
                    c = tone_fixed(source, candidate, stencil, region) if tone else candidate
                    score, _ = worst(source, c, stencil, region)
                    invented = text_in(craft, c, region) - text0
                    tries.append({"seed": seed, "tone": tone, "score": round(score, 2), "text_px": invented,
                                  "seconds": round(time.perf_counter() - started, 1)})
                    if invented > 20:
                        continue                    # the repaint wrote something: never accept it
                    if best is None or score < best[0]:
                        best = (score, c, shown)
            keep = best is not None and best[0] < 0.9 * before      # a real improvement, not noise
            row = {"id": image_id, "from": t, "holes": [{k: v for k, v in r.items() if k != "mask"} for r in flags],
                   "before": round(before, 2), "tries": tries, "accepted": bool(keep)}
            handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            log.append(row)
            if not keep:
                continue
            best[1].save(out_res / f"{image_id}.png")
            shutil.copyfile(mask_p, out_mask / f"{image_id}.png")
            Image.fromarray((best[2] * 255).astype(np.uint8), "L").save(out_shown / f"{image_id}.png")
            for src_dir, dst in ((f"core_{t.lower()}", out_core), (f"core_{t.lower()}c_craft", out_craft)):
                p = HERE / "masks" / src_dir / f"{image_id}.png"
                if p.exists():
                    shutil.copyfile(p, dst / f"{image_id}.png")
            accepted.append({**item, "patchfix_from": t})
    (HERE / f"records/items_{a.tag}.json").write_text(json.dumps(accepted, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"tag": a.tag, "flagged_photos": len(log), "accepted": [r["id"] for r in log if r["accepted"]],
                      "rejected": [r["id"] for r in log if not r["accepted"]]}))


if __name__ == "__main__":
    main()
