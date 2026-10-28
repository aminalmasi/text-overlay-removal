"""Every tuned constant, with the measurement that fixed it.

Nothing here is a default. Each value below was chosen against evidence, and the comment
records what that evidence was, because several of these look arbitrary and are not --
and two of them were bugs for a long time before they were measured.
"""
from __future__ import annotations

import os
from pathlib import Path

MODELS = Path(os.environ.get("TL_MODELS", "/models"))
FLUX_PATH = MODELS / "FLUX.1-Fill-dev"
FLUX_LORA = MODELS / "FluxFill-ObjectRemoval-LoRA/objectRemovalv1_r32.safetensors"
SAM_PATH = MODELS / "sam2.1-hiera-large"
CRAFT_WEIGHTS = MODELS / "CRAFT/craft_mlt_25k.pth"
LAMA_DIR = MODELS / "LaMa-big/big-lama"
LAMA_REPO = Path(__file__).resolve().parent.parent / "third_party/lama"

# --- localization ---------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-pro"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts/P61.md"
SCHEMA_FILE = Path(__file__).resolve().parent.parent / "prompts/response_schema_p61.json"

# --- mask -----------------------------------------------------------------
BOX_WIDEN = 0.06          # per side, min 4px. Gemini's boxes clip anti-aliased edges.
SAM_MIN_AREA, SAM_MAX_AREA = 0.05, 1.6     # of the prompt box; outside this SAM2 has
                                           # segmented the product rather than the overlay
SAM_DILATE = 4
CHANGE_THRESHOLD = 22     # /255, what a repaint actually altered
CHANGE_OPEN, CHANGE_CLOSE, CHANGE_DILATE = 3, 9, 4
MASK_WIDEN = 2            # `combo_w2`. Median 8.5% of frame; chosen by review over both
                          # the untouched combo (8.07%) and a CRAFT-derived mask.

# --- halo -----------------------------------------------------------------
# The previous pipeline ran LaMa on dilate(mask, 18), and that band -- not the model -- is
# why it cleared carrier rims a mask-only eraser left behind. Measured on 16 images, text
# remaining fell 6/16 -> 2/16 and the boundary step 3.0 -> 1.3 as this went 0 -> 18.
# It is not maximised: coverage above ~8% of frame is the strongest predictor of failure
# in the review data (AUC 0.75), so this is deliberately short of the ladder's best number.
HALO = 15

# --- working resolution ---------------------------------------------------
# Both bounds were bugs before they were measured. A 272x160 crop made FLUX emit granular
# noise (it is trained near 1024 square); a 1013x1527 crop squeezed to 512x768 and upscaled
# back made a smudge. Both were mistaken for model failures.
MIN_SIDE, MAX_SIDE, MULTIPLE, MAX_ASPECT = 768, 1280, 16, 2.2
CROP_PAD = 0.18

# --- erase ----------------------------------------------------------------
STEPS = 30
GUIDANCE = 30.0           # FLUX-dev is guidance-distilled: this is a conditioning input it
                          # was trained on, not classifier-free guidance. Low values do NOT
                          # tame it -- guidance 3 gave the worst texture of the whole grid.
STRENGTH = 1.0            # how much noise the repaint starts from. 1.0 is the pipeline's
                          # default and means pure noise: the model is free to put anything
                          # in the hole, which is where an invented cat comes from. Below 1.0
                          # the latent starts from the photograph itself with less noise
                          # added, so the answer stays near what was there -- and what was
                          # there includes the overlay, so this trades invention against
                          # leftover. Never varied before v9.
LORA_SCALE = 1.0          # the actual restraint dial. At guidance 30, scale 1.0 repaints
                          # 70/255 against scale 0's 82, and scale 0 hallucinated whole
                          # paragraphs of fake text.
PROMPT = ("plain empty surface, seamless continuation of the surrounding material, "
          "smooth and uniform")
# FluxFillPipeline exposes no negative_prompt and no true_cfg_scale in diffusers 0.39, so
# "do not write text" cannot be said. With an empty prompt and guidance 30 there is nothing
# to steer toward, the model's prior takes over, and its prior for a banner-shaped hole in a
# marketplace photograph is text. The positive prompt is what stops that.

USE_NF4 = True            # 23 GiB transformer -> ~6 GiB. Median difference 5.28/255,
                          # judged indistinguishable. Needs compute capability >= 7.5.
CACHE_PROMPT = True       # the prompt never varies, so encode once and free both text
                          # encoders: 8.9 GiB saved, output pixel-identical (6/6).

# --- composite ------------------------------------------------------------
FEATHER = 8               # swept 3/8/14/20: the boundary step is 2.16 / 1.50 / 1.60 / 1.69.
                          # A wider feather does not reduce the tone offset, it spreads it,
                          # and past 8 it trades a crisp edge for a visible soft halo.
CALIBRATE = True          # fit the repair's tone against the unmasked pixels of the same
                          # crop, where the correct answer is already known.
CALIBRATE_GAIN = (0.7, 1.4)
CALIBRATE_MAX_OFFSET = 40.0
