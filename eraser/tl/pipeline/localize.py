"""Ask Gemini where the overlay is -- and whether there is one at all.

This is the only stage that separates an *applied* overlay from text that was
*photographed*: a price sticker added in an editor, versus the price printed on the
product's own packaging. Nothing else in the pipeline can make that distinction, and it
was not for want of trying. Measured on 86 photographs containing no overlay:

    CRAFT, full frame ................ fires on 85 / 86
    Grounding DINO, 14 phrases ....... fires on 72 / 86
    a hand-built carrier classifier .. no separation (0.28 vs 0.33)
    SAM2 ............................. has no text encoder; cannot be asked

Everything downstream is confined to the boxes returned here, so a region this stage never
reports is a region no later stage can recover.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import config as C

RETRIES, BACKOFF = 3, 5.0


def load_prompt():
    return C.PROMPT_FILE.read_text(encoding="utf-8")


def load_schema():
    return json.loads(C.SCHEMA_FILE.read_text(encoding="utf-8"))


def localize(image_path, client=None, model=C.GEMINI_MODEL):
    """Return the raw list of regions for one image.

    Each region carries `box_2d` (0-1000, [ymin, xmin, ymax, xmax]), `kind`
    (text / logo_watermark / emoji_symbol / sticker_badge / mixed), `carrier_present`,
    `carrier_kind`, `carrier_opacity` and `physically_in_scene`.

    Nothing identifying the image is ever sent: no filename, no path, no label. The model
    sees the pixels and the prompt, and that is all.
    """
    from google import genai
    from google.genai import types

    client = client or genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    data = Path(image_path).read_bytes()
    suffix = Path(image_path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"

    last = None
    for attempt in range(RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=data, mime_type=mime), load_prompt()],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=load_schema(),
                    temperature=0.0),
            )
            payload = json.loads(response.text)
            return payload.get("regions", payload if isinstance(payload, list) else [])
        except Exception as error:            # rate limits and transient 5xx
            last = error
            if attempt + 1 < RETRIES:
                time.sleep(BACKOFF * (attempt + 1))
    raise RuntimeError(f"localization failed after {RETRIES} attempts: {last}")


def boxes_from(regions, keep_in_scene=False):
    """The boxes the mask stage should act on.

    `physically_in_scene` marks writing that belongs to the photographed world -- a shop
    sign, a licence plate, print on a shirt. Removing those changes the goods rather than
    the advert, so by default they are left alone.
    """
    out = []
    for region in regions:
        if not keep_in_scene and region.get("physically_in_scene"):
            continue
        box = region.get("box_2d")
        if box and len(box) == 4:
            out.append(box)
    return out
