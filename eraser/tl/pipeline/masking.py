"""From the localizer's boxes to the pixels that will be repainted.

Two independent signals, unioned, then confined to the boxes:

  SAM2 outline      what object is at this box. Prompted with geometry only -- SAM2 has no
                    text encoder, so it cannot be asked for "text"; it answers "what is
                    here", which on a marketplace photograph is often the product. Its
                    answer is therefore filtered by plausible area, not trusted outright.
  change region     run a throwaway repaint of the plain rectangle and see what moved.
                    Catches soft glow, drop shadow and anti-aliased halo that no character
                    score sees.

Nothing here decides *whether* a region is an overlay -- that is the localizer's job, and
the reason it cannot be delegated is that every signal available at this level fires on
ordinary photographed text just as readily. Measured on 86 photographs with no overlay:
CRAFT fires on 85, Grounding DINO on 72.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

from . import config as C


def to_pixels(box, size):
    """Gemini's 0-1000 [ymin, xmin, ymax, xmax] to pixel [left, top, right, bottom]."""
    ymin, xmin, ymax, xmax = box
    width, height = size
    return (round(xmin / 1000 * width), round(ymin / 1000 * height),
            round(xmax / 1000 * width), round(ymax / 1000 * height))


def widen(box, size, fraction=C.BOX_WIDEN):
    left, top, right, bottom = box
    pad_x = max(4, int(fraction * (right - left)))
    pad_y = max(4, int(fraction * (bottom - top)))
    return (max(0, left - pad_x), max(0, top - pad_y),
            min(size[0], right + pad_x), min(size[1], bottom + pad_y))


def dilate(mask, px):
    """Square dilation, pixel-identical to ImageFilter.MaxFilter(2*px+1) and ~1900x faster.

    PIL evaluates the window per pixel; at px=18 that is a 37x37 max over the whole frame
    and measured 4601 ms against cv2's 2.4 ms. It was 30% of the pipeline before this.
    """
    if px <= 0:
        return mask
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1,) * 2)
    return Image.fromarray(cv2.dilate(np.asarray(mask), element), mask.mode)


def rect_of(boxes, shape):
    canvas = np.zeros(shape, np.uint8)
    for left, top, right, bottom in boxes:
        canvas[top:bottom, left:right] = 255
    return canvas


def sam_shapes(model, processor, image, boxes):
    """SAM2 from box prompts. Returns one boolean mask per box, or None where implausible."""
    if not boxes:
        return []
    inputs = processor(images=image, input_boxes=[[list(b) for b in boxes]],
                       return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    array = masks.numpy()
    if array.ndim == 3:
        array = array[None]
    scores = outputs.iou_scores.cpu().numpy().reshape(array.shape[0], -1)
    results = []
    for index, box in enumerate(boxes):
        box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        best, best_score = None, -1.0
        for option in range(array.shape[1]):
            candidate = array[index][option].astype(bool)
            area = candidate.sum()
            if area < C.SAM_MIN_AREA * box_area or area > C.SAM_MAX_AREA * box_area:
                continue
            if scores[index][option] > best_score:
                best, best_score = candidate, scores[index][option]
        results.append(None if best is None else cv2.dilate(
            (best * 255).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (C.SAM_DILATE * 2 + 1,) * 2)))
    return results


def changed_region(source, repainted, inside):
    """What a repaint of the plain rectangle actually altered."""
    difference = np.abs(np.asarray(source, np.float32)
                        - np.asarray(repainted, np.float32)).max(axis=2)
    binary = ((difference > C.CHANGE_THRESHOLD) & inside).astype(np.uint8) * 255
    disc = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1,) * 2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, disc(C.CHANGE_OPEN))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, disc(C.CHANGE_CLOSE))
    return cv2.dilate(binary, disc(C.CHANGE_DILATE))


def build(image, boxes_1000, sam, processor, repaint):
    """The mask, and the halo-grown mask the eraser is actually given.

    `repaint` is a callable (image, mask) -> image used once for change detection. It is
    the most expensive part of this stage by far -- around 6 s against 0.2 s for everything
    else -- because it runs a full repaint whose output is then discarded.
    """
    size = image.size
    shape = (size[1], size[0])
    boxes = [widen(to_pixels(b, size), size) for b in boxes_1000]
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return None, None

    rect = rect_of(boxes, shape)
    inside = rect > 0
    rect_mask = Image.fromarray(rect, "L")

    canvas = np.zeros(shape, np.uint8)
    for box, shape_mask in zip(boxes, sam_shapes(sam, processor, image, boxes)):
        if shape_mask is None:
            left, top, right, bottom = box
            shape_mask = np.zeros(shape, np.uint8)
            shape_mask[top:bottom, left:right] = 255
        canvas = np.maximum(canvas, shape_mask)

    changed = changed_region(image, repaint(image, rect_mask), inside) > 127
    combo = (canvas > 127) | changed
    if not combo.any():
        combo = inside

    mask = dilate(Image.fromarray((combo * 255).astype(np.uint8), "L"), C.MASK_WIDEN)
    return mask, dilate(mask, C.HALO)
