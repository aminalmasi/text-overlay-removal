"""The GPU half of text_layover: erase what the Gemini worker (P92) boxed, in one process.

    from eraser.engine import Eraser
    eraser = Eraser()                                     # loads every model once (~10-25 s)
    out = eraser.erase(photo, worker_reply, photo_id="listing-123-photo-2")
    out["image"]        # PIL image to ship: the erased photo, or the original when held / nothing to do
    out["status"]       # "erased" | "held" | "nothing_to_erase" | "too_large" | "rotated_input"
    out["info"]         # what fired, mask size, seconds per stage

The steps, in the order the reviewed runs used (text_layover review pages, October 2026):

  0 line split    a region with no carrier becomes one region per line box (P92's line_boxes)
  1 refine        one CRAFT pass over the photo; box edges snap to escaping words, emoji regions widen,
                  plain-text boxes Gemini shifted are moved/stretched onto their text
  2 start         starting mask: SAM2 outline | LaMa change region | CRAFT strokes, widened 2 px
  3 stencil       row-gap closing, halo 0, protected surfaces (physically_in_scene true) restored
  4 repaint1      FLUX.1-Fill + objectRemoval LoRA, composited, protected pixels restored
  5 patch         noise-patch check; when flagged: 2 disguised repaints + tone fix, CRAFT invention guard,
                  pasted only inside the hole, kept only if >= 10% better
  6 layer2        CRAFT on the repaint; leftover text grown with SAM2, FLUX + LaMa inside the stencil
  7 carve         writing that crosses a protected plate: only the strokes, LaMa
  8 gate          largest patch > 16% of the frame and > 25% of the centre -> ship the original

Every function used below is the reviewed code, copied unchanged into ../tl (see VENDORED.md). The seed
of every repaint comes from `photo_id`, so the same photo and id always give the same output.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TL = Path(__file__).resolve().parent.parent / "tl"
sys.path.insert(0, str(TL))

import torch                                                         # noqa: E402
from pipeline import config as C, erase, localize, masking           # noqa: E402
from pipeline.run import lama_fill, load_lama, seed_for              # noqa: E402
from run_maskfix import BRIDGE, boxes_of, close_rows, guard_mask, keep_protected   # noqa: E402
from run_layer2_sam import eligible, sam_extend, survivors, rect     # noqa: E402
from run_plate_carve import carve, regions                          # noqa: E402
from run_disguise import disguise                                    # noqa: E402
from run_layer2_targeted import paste_inside                         # noqa: E402
from run_patchfix import tone_fixed                                  # noqa: E402
from make_items_lines import split_lines                             # noqa: E402
import refine_boxes as RB                                            # noqa: E402
import patchcheck as P                                               # noqa: E402
import detect_leftover as D                                          # noqa: E402

LEVEL = 0.25                    # CRAFT region-score threshold used by every reviewed step
GATE_LIMIT, GATE_CENTRE = 16, 25
# The reviewed photos were at most 2.1 megapixels (long side <= 2048). CRAFT and LaMa run on the whole photo at
# its own size, so much larger photos are untested and can run out of GPU memory: they are shipped unchanged.
MAX_PIXELS = int(os.environ.get("TL_MAX_PIXELS", 2048 * 2048))


def _box(b):
    """[ymin, xmin, ymax, xmax] on Gemini's 0-1000 grid as ints, clamped; None when unusable."""
    if not isinstance(b, (list, tuple)) or len(b) != 4 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in b):
        return None
    y0, x0, y1, x1 = (int(round(min(1000, max(0, v)))) for v in b)
    return [y0, x0, y1, x1] if y0 < y1 and x0 < x1 else None


def clean_regions(regions):
    """Drop what the reviewed code could not handle, so a malformed reply never crashes or erases the wrong
    place: regions without a usable box_2d are dropped, a malformed text_box_2d is removed (box_2d is used), and a
    region whose physically_in_scene is not a boolean is protected, never erased. A well-formed P92 reply passes
    through unchanged."""
    out = []
    for r in regions if isinstance(regions, list) else []:
        if not isinstance(r, dict):
            continue
        box = _box(r.get("box_2d"))
        if box is None:
            continue
        r = {**r, "box_2d": box}
        if "text_box_2d" in r:
            tb = _box(r["text_box_2d"])
            if tb is None:
                r.pop("text_box_2d")
            else:
                r["text_box_2d"] = tb
        if not isinstance(r.get("physically_in_scene"), bool):
            r["physically_in_scene"] = True
        if "line_boxes" in r and not isinstance(r["line_boxes"], list):
            r.pop("line_boxes")
        out.append(r)
    return out


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Timer:
    def __init__(self, times, name):
        self.times, self.name = times, name

    def __enter__(self):
        _sync()
        self.t = time.perf_counter()

    def __exit__(self, *a):
        _sync()
        self.times[self.name] = self.times.get(self.name, 0.0) + time.perf_counter() - self.t


def _bgr(im):
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def _load_flux_prepared(prepared: Path, device="cuda"):
    """FLUX from prepare_models.py output: the NF4 transformer saved already quantized and the prompt
    already encoded, so neither the 23 GB bf16 transformer nor the 9.5 GB T5 encoder is read or put on the
    GPU at start-up. Same weights, same prompt embeddings, same outputs as erase.load()."""
    from diffusers import FluxFillPipeline, FluxTransformer2DModel
    transformer = FluxTransformer2DModel.from_pretrained(prepared / "transformer_nf4", torch_dtype=torch.bfloat16)
    pipe = FluxFillPipeline.from_pretrained(
        C.FLUX_PATH, transformer=transformer, text_encoder=None, text_encoder_2=None, tokenizer=None,
        tokenizer_2=None, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.load_lora_weights(str(C.FLUX_LORA.parent), weight_name=C.FLUX_LORA.name, adapter_name="removal")
    pipe.set_adapters(["removal"], [C.LORA_SCALE])
    pipe.vae.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)
    saved = torch.load(prepared / "prompt_embeds.pt", map_location=device)
    if saved.get("prompt") != C.PROMPT:
        raise RuntimeError("prompt_embeds.pt was made for a different repaint prompt; rerun prepare_models.py")
    return pipe, (saved["prompt_embeds"], saved["pooled_prompt_embeds"])


class Eraser:
    """All models resident on one GPU. Not thread-safe: one photo at a time per instance."""

    def __init__(self, precision: str = "nf4", prepared: str | os.PathLike | None = None):
        # The reviewed helpers run on "cuda", i.e. the first visible GPU: choose the GPU with CUDA_VISIBLE_DEVICES.
        device = "cuda"
        if precision not in ("nf4", "bf16"):
            raise ValueError("precision is nf4 (reviewed) or bf16 (needs a 48 GB GPU, not reviewed)")
        t0 = time.perf_counter()
        prepared = Path(prepared or os.environ.get("TL_PREPARED", "")) if (prepared or os.environ.get("TL_PREPARED")) else None
        C.USE_NF4 = precision == "nf4"
        steps, t = {}, time.perf_counter()
        if precision == "nf4" and prepared and (prepared / "transformer_nf4").exists() and (prepared / "prompt_embeds.pt").exists():
            pipe, embeds = _load_flux_prepared(prepared, device)
            self.loader = "prepared"
        else:
            pipe, embeds = erase.load(device)          # the reviewed loader: quantizes at start-up, encodes the prompt
            self.loader = "original"
        steps["flux"], t = time.perf_counter() - t, time.perf_counter()
        from transformers import Sam2Model, Sam2Processor
        from craft_model import load_craft
        sam = Sam2Model.from_pretrained(C.SAM_PATH, local_files_only=True).to(device).eval()
        processor = Sam2Processor.from_pretrained(C.SAM_PATH, local_files_only=True)
        steps["sam2"], t = time.perf_counter() - t, time.perf_counter()
        lama = load_lama(device)
        steps["lama"], t = time.perf_counter() - t, time.perf_counter()
        self.models = {"flux": pipe, "embeds": embeds, "sam": sam, "processor": processor, "lama": lama}
        self.craft = load_craft(D.CRAFT_W, device)
        steps["craft"] = time.perf_counter() - t
        self.load_steps = {k: round(v, 1) for k, v in steps.items()}
        gc.collect()
        torch.cuda.empty_cache()
        self.precision = precision
        self.load_seconds = round(time.perf_counter() - t0, 1)

    def warmup(self):
        """One FLUX repaint, one LaMa fill and one CRAFT pass on a synthetic picture, so the first real photo
        does not pay the one-off kernel set-up (several seconds)."""
        pic = Image.new("RGB", (1024, 768), (180, 170, 160))
        m = np.zeros((768, 1024), np.uint8)
        m[300:420, 200:800] = 255
        mask = Image.fromarray(m, "L")
        D.region_score(self.craft, np.asarray(pic))
        lama_fill(self.models["lama"], pic, mask)
        erase.run(self.models["flux"], self.models["embeds"], pic, mask, seed=0)
        _sync()

    # ------------------------------------------------------------------------------------------------
    def erase(self, photo, worker_reply: dict, photo_id: str, debug: bool = False, split: bool = True) -> dict:
        """photo: a PIL image or a path. worker_reply: P92's JSON reply (needs "boxes"). photo_id: any stable
        string; it seeds the repaints. split=False takes boxes that were already line-split (test harness)."""
        opened = photo if isinstance(photo, Image.Image) else Image.open(photo)
        icc = opened.info.get("icc_profile")
        if opened.width * opened.height > MAX_PIXELS:          # checked before the pixels are decoded
            return {"image": None, "status": "too_large", "icc_profile": icc,
                    "info": {"pixels": opened.width * opened.height, "max_pixels": MAX_PIXELS}}
        orientation = (opened.getexif() or {}).get(0x0112, 1)
        image = opened.convert("RGB")
        if orientation not in (None, 1):
            # Gemini's boxes and these pixels must describe the same orientation, and an EXIF-rotated photo may
            # have been seen upright by Gemini. Never guess: send both the same upright bytes (see README).
            return {"image": image, "status": "rotated_input", "icc_profile": icc, "info": {"exif_orientation": orientation}}
        boxes = clean_regions((worker_reply or {}).get("boxes") if isinstance(worker_reply, dict) else None)
        if split:
            boxes, _ = split_lines(boxes)
        if not any(b.get("physically_in_scene") is False for b in boxes):
            return {"image": image, "status": "nothing_to_erase", "info": {"regions": len(boxes)}}
        final, info, stages = self._run(image, boxes, str(photo_id), debug)
        if info.get("skipped") or not (info["stencil_pct"] > 0 or info["carved"]):
            return {"image": image, "status": "nothing_to_erase", "icc_profile": icc, "info": info,
                    **({"stages": stages} if debug else {})}
        status = "held" if info["held"] else "erased"
        out = {"image": image if info["held"] else final, "status": status, "icc_profile": icc, "info": info}
        if debug:
            out["stages"] = {**stages, "final_before_gate": final}
        return out

    # ------------------------------------------------------------------------------------------------
    def _run(self, image, boxes, image_id, debug):
        models, craft = self.models, self.craft
        times, stages = {}, {}
        size, arr = image.size, np.asarray(image)
        item = {"id": image_id, "boxes": boxes}
        # 1 refine: one CRAFT pass on the photo, reused by the strokes and the carve
        with _Timer(times, "refine"):
            score, affinity = D.region_score(craft, arr)
            h, w = score.shape
            # The reviewed refine step read this map back from an 8-bit PNG (craft_full.py: clip * 255, truncated);
            # thresholding the same 8-bit values reproduces its boxes exactly. Strokes and carve use the float map.
            saved = (np.clip(score, 0, 1) * 255).astype(np.uint8) / 255.0
            words = cv2.dilate((saved > LEVEL).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 160), 3)))
            _, labels, stats, _ = cv2.connectedComponentsWithStats(words, 8)
            I = cv2.integral(cv2.dilate((saved > LEVEL).astype(np.uint8), np.ones((3, 3), np.uint8)))
            refined = []
            for r in item["boxes"]:
                if r.get("physically_in_scene") is False and isinstance(r.get("box_2d"), list) and len(r["box_2d"]) == 4:
                    r, _ = RB.refine(r, words, labels, stats, (w, h), True, I)
                refined.append(r)
            item = {**item, "boxes": refined}
        # 2 start: SAM2 | changed region | CRAFT strokes
        with _Timer(times, "start"):
            b1000 = localize.boxes_from(item["boxes"])
            core, _ = masking.build(image, b1000, models["sam"], models["processor"],
                                    lambda pic, m: lama_fill(models["lama"], pic, m))
            core = (np.asarray(core) > 127) if core is not None else np.zeros((h, w), bool)
            wb = [masking.widen(masking.to_pixels(b, size), size) for b in b1000]
            rectm = masking.rect_of([b for b in wb if b[2] > b[0] and b[3] > b[1]], (h, w)) > 0
            hot = ((score > LEVEL) & rectm).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(hot, 8)
            keep = np.isin(lab, [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 20])
            strokes = (cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0) & rectm
            strokes = np.asarray(masking.dilate(Image.fromarray((strokes * 255).astype(np.uint8), "L"), C.MASK_WIDEN)) > 127
            core |= strokes
        if not core.any():
            return image, {"skipped": "empty mask", "seconds": times}, stages
        # 3 stencil
        with _Timer(times, "stencil"):
            core = close_rows(core, boxes_of(item, size, False), max(40, int(BRIDGE * max(core.shape))))
            guard = guard_mask(item, size) > 0.5
            stencil = core & ~guard
        # The whole mask can lie on protected surfaces (a watermark entirely on a plate or a screen). The reviewed
        # run_maskfix then kept the photo unchanged and skipped the repaint; only the plate carve can still act.
        has_hole = bool(stencil.any())
        # 4 first repaint
        with _Timer(times, "repaint1"):
            if has_hole:
                first = erase.run(models["flux"], models["embeds"], image,
                                  Image.fromarray((stencil * 255).astype(np.uint8), "L"), seed=seed_for(image_id))
                first, _ = keep_protected(image, first, guard)
            else:
                first = image
        if debug:
            stages["stencil"] = Image.fromarray((stencil * 255).astype(np.uint8), "L")
            stages["repaint1"] = first
        # 5 patch check (+ fix only when flagged)
        info = {"patch_flagged": False, "patch_fixed": False}
        with _Timer(times, "patch"):
            flags = [r for r in P.holes(_bgr(image), _bgr(first), stencil) if P.flagged(r)] if has_hole else []
            if flags:
                info["patch_flagged"] = True
                region = np.zeros_like(stencil)
                for r in flags:
                    region |= r["mask"]
                before = max(P.badness(r) for r in flags)
                text0 = int(((D.region_score(craft, np.asarray(first))[0] > LEVEL) & region).sum())
                best = None
                for k in (1, 2):
                    seed = seed_for(image_id) + 7919 * k
                    shown = disguise(region, 0.01, 0.035, seed) & ~guard
                    painted = erase.run(models["flux"], models["embeds"], first,
                                        Image.fromarray((shown * 255).astype(np.uint8), "L"), seed=seed)
                    c = paste_inside(painted, first, region)
                    c2 = tone_fixed(image, c, stencil, region)
                    for cand in (c, c2):
                        rows = [r for r in P.holes(_bgr(image), _bgr(cand), stencil) if (r["mask"] & region).any()]
                        s = max((P.badness(r) for r in rows), default=0.0)
                        inv = int(((D.region_score(craft, np.asarray(cand))[0] > LEVEL) & region).sum()) - text0
                        if inv <= 20 and (best is None or s < best[0]):
                            best = (s, cand)
                if best is not None and best[0] < 0.9 * before:
                    first = best[1]
                    info["patch_fixed"] = True
        if debug:
            stages["after_patch"] = first
        # 6 layer 2 (CRAFT -> SAM2 -> FLUX + LaMa), first-layer guard rule
        with _Timer(times, "layer2"):
            current = first
            info["layer2_fired"] = False
            if has_hole:
                region2, _ = D.region_score(craft, np.asarray(current))
                box_region = np.zeros_like(stencil)
                for r in item["boxes"]:
                    if r.get("physically_in_scene"):
                        continue
                    l, t, rr, b = masking.widen(masking.to_pixels(r["box_2d"], size), size)
                    box_region[max(0, t):b, max(0, l):rr] = True
                found = survivors(eligible(region2, stencil, box_region, "mask", LEVEL, 0.55), 20)
                repair = np.zeros_like(stencil)
                for blob, _, _ in found:
                    repair |= blob
                if found:
                    repair |= sam_extend(models, current, [b for _, b, _ in found], stencil.shape) & stencil
                    repair = cv2.dilate((repair * 255).astype(np.uint8), rect(4)) > 127
                repair &= stencil & ~guard
                info["layer2_fired"] = bool(repair.any())
                if repair.any():
                    m = Image.fromarray((repair * 255).astype(np.uint8), "L")
                    current = erase.run(models["flux"], models["embeds"], current, m, seed=seed_for(image_id) + 104729)
                    current = lama_fill(models["lama"], current, m)
                    px = np.asarray(current).copy()
                    px[guard] = arr[guard]
                    current = Image.fromarray(px, "RGB")
        # 7 plate carve
        with _Timer(times, "carve"):
            protect, erase_area = regions(item, size, True), regions(item, size, False)
            info["carved"] = False
            if (protect & erase_area).any():
                picked, _ = carve(arr, score, affinity, erase_area, protect)
                if picked is not None:
                    current = lama_fill(models["lama"], current, Image.fromarray((picked * 255).astype(np.uint8), "L"))
                    info["carved"] = True
        # 8 gate
        with _Timer(times, "gate"):
            from scipy import ndimage
            lab, cnt = ndimage.label(stencil)
            biggest = central = 0.0
            if cnt:
                sizes = ndimage.sum(stencil, lab, range(1, cnt + 1))
                biggest = 100 * float(sizes.max()) / stencil.size
                patch = lab == int(np.argmax(sizes)) + 1
                hh, ww = stencil.shape
                heart = np.zeros_like(stencil)
                heart[hh // 4: hh - hh // 4, ww // 4: ww - ww // 4] = True
                central = 100 * float((patch & heart).sum() / heart.sum())
            info["held"] = bool(biggest > GATE_LIMIT and central > GATE_CENTRE)
        info.update({"largest_patch_pct": round(biggest, 2), "largest_patch_centre_pct": round(central, 1),
                     "stencil_pct": round(100 * float(stencil.mean()), 2), "regions": len(item["boxes"]),
                     "seconds": {k: round(v, 3) for k, v in times.items()},
                     "seconds_total": round(sum(times.values()), 2)})
        return current, info, stages
