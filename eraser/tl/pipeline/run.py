#!/usr/bin/env python3
"""End to end: a photograph in, the same photograph without its overlay out.

    python -m pipeline.run listing.jpg -o cleaned.jpg
    python -m pipeline.run inbox/ -o outbox/ --save-masks

Localization needs GEMINI_API_KEY. Everything after it runs locally on one GPU in about
10 GiB, which is small enough that the models can sit alongside other work.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image, ImageOps

from . import config as C
from . import erase, localize, masking

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def seed_for(image_id):
    return zlib.crc32(str(image_id).encode()) % (2**31)


def load_lama(device="cuda"):
    """LaMa is used only for change detection while building the mask, not to erase."""
    sys.path.insert(0, str(C.LAMA_REPO))
    from saicinpainting.training.trainers import load_checkpoint
    config = OmegaConf.create(yaml.safe_load((C.LAMA_DIR / "config.yaml").read_text()))
    config.training_model.predict_only = True
    config.visualizer.kind = "noop"
    model = load_checkpoint(config, C.LAMA_DIR / "models/best.ckpt", strict=False,
                            map_location="cpu")
    model.freeze()
    return model.to(device).eval()


def lama_fill(model, image, mask, device="cuda"):
    width, height = image.size
    pad_w, pad_h = (-width) % 8, (-height) % 8
    batch = {
        "image": torch.from_numpy(
            np.asarray(ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=0), np.float32)
            / 255.0).permute(2, 0, 1)[None].to(device),
        "mask": torch.from_numpy(
            np.asarray(ImageOps.expand(mask, (0, 0, pad_w, pad_h), fill=0), np.float32)
            / 255.0)[None, None].to(device),
    }
    batch["mask"] = (batch["mask"] > 0).to(batch["image"].dtype)
    with torch.inference_mode():
        result = model(batch)["inpainted"][0].permute(1, 2, 0)
    return Image.fromarray(result.clamp(0, 1).mul(255).byte().cpu().numpy()[:height, :width],
                           "RGB")


def load_all(device="cuda"):
    """Every model the local half of the pipeline needs, loaded once."""
    from transformers import Sam2Model, Sam2Processor
    pipe, embeds = erase.load(device)
    sam = Sam2Model.from_pretrained(C.SAM_PATH, local_files_only=True).to(device).eval()
    processor = Sam2Processor.from_pretrained(C.SAM_PATH, local_files_only=True)
    return {"flux": pipe, "embeds": embeds, "sam": sam, "processor": processor,
            "lama": load_lama(device)}


def clean(image_path, out_path=None, models=None, regions=None, save_mask=None,
          keep_in_scene=False):
    """Remove the overlay from one photograph.

    `regions` skips the Gemini call when the localization already exists. Returns the
    cleaned image, or the original unchanged when nothing was found -- an image with no
    overlay must come back untouched, not merely unharmed.
    """
    models = models or load_all()
    image = Image.open(image_path).convert("RGB")
    if regions is None:
        regions = localize.localize(image_path)
    boxes = localize.boxes_from(regions, keep_in_scene=keep_in_scene)
    if not boxes:
        if out_path:
            image.save(out_path)
        return image, {"regions": 0, "mask_pct": 0.0, "changed": False}

    def repaint(picture, mask):
        return lama_fill(models["lama"], picture, mask)

    mask, halo = masking.build(image, boxes, models["sam"], models["processor"], repaint)
    if mask is None or mask.getbbox() is None:
        if out_path:
            image.save(out_path)
        return image, {"regions": len(boxes), "mask_pct": 0.0, "changed": False}

    result = erase.run(models["flux"], models["embeds"], image, halo,
                       seed=seed_for(Path(image_path).stem))
    if save_mask:
        halo.save(save_mask)
    if out_path:
        result.save(out_path)
    return result, {"regions": len(boxes),
                    "mask_pct": round(100 * float((np.asarray(halo) > 127).mean()), 2),
                    "changed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="an image, or a directory of them")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--regions", type=Path,
                        help="localization already done: {image_stem: [regions]} as JSON")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--keep-in-scene", action="store_true",
                        help="also remove writing that belongs to the photographed world")
    args = parser.parse_args()

    paths = ([args.source] if args.source.is_file()
             else sorted(p for p in args.source.rglob("*") if p.suffix.lower() in SUFFIXES))
    if not paths:
        raise SystemExit(f"no images under {args.source}")
    cached = json.loads(args.regions.read_text()) if args.regions else {}

    if args.source.is_file():
        args.out.parent.mkdir(parents=True, exist_ok=True)
    else:
        args.out.mkdir(parents=True, exist_ok=True)

    models = load_all()
    for index, path in enumerate(paths, 1):
        destination = args.out if args.source.is_file() else args.out / f"{path.stem}.png"
        mask_path = None
        if args.save_masks:
            mask_path = destination.with_name(destination.stem + "_mask.png")
        started = time.perf_counter()
        _, info = clean(path, destination, models=models,
                        regions=cached.get(path.stem), save_mask=mask_path,
                        keep_in_scene=args.keep_in_scene)
        print(json.dumps({"n": index, "of": len(paths), "id": path.stem,
                          "seconds": round(time.perf_counter() - started, 2), **info}),
              flush=True)


if __name__ == "__main__":
    main()
