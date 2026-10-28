"""Repaint the masked region and put it back without leaving a visible patch.

The erase itself is one FLUX.1-Fill pass. Most of what is written here is about the two
things that surround it, both of which mattered more than the choice of model:

  the crop     FLUX is trained near 1024 square. Hand it a 272x160 strip and it emits
               granular noise; hand it a downscaled 1013x1527 and upscale the answer back
               and it emits a smudge. Both were read as model failures for a long time.
  the paste    the repaired area lands a few levels off in tone and slightly softer than
               its surroundings, in the shape of the mask -- which reads as a leftover
               carrier and is not one. Measured: by +20px of halo the carrier is entirely
               inside the mask (0.4% of the interior unrepainted), and what survives is the
               repair announcing itself. Enlarging the mask makes that rectangle bigger,
               not fainter.
"""
from __future__ import annotations

import gc

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter

from . import config as C


def load(device="cuda"):
    """FLUX.1-Fill with the removal LoRA, optionally 4-bit, prompt pre-encoded."""
    from diffusers import FluxFillPipeline

    transformer = None
    if C.USE_NF4:
        from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
        transformer = FluxTransformer2DModel.from_pretrained(
            C.FLUX_PATH, subfolder="transformer", local_files_only=True,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))

    pipe = FluxFillPipeline.from_pretrained(
        C.FLUX_PATH, torch_dtype=torch.bfloat16, local_files_only=True,
        **({"transformer": transformer} if transformer is not None else {}))
    pipe.load_lora_weights(str(C.FLUX_LORA.parent), weight_name=C.FLUX_LORA.name,
                           adapter_name="removal")
    pipe.set_adapters(["removal"], [C.LORA_SCALE])

    free = torch.cuda.mem_get_info()[0] / 2**30
    if transformer is not None:
        # a bitsandbytes model is already placed and must not be moved again; the rest of
        # the pipeline still has to be. Offloading is 5x slower and only worth it if the
        # weights genuinely will not fit.
        if free < 16:
            pipe.enable_model_cpu_offload()
        else:
            for part in ("text_encoder", "text_encoder_2", "vae"):
                module = getattr(pipe, part, None)
                if module is not None:
                    module.to(device)
    elif free < 40:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    embeds = None
    if C.CACHE_PROMPT:
        with torch.inference_mode():
            prompt_embeds, pooled, _ = pipe.encode_prompt(
                prompt=C.PROMPT, prompt_2=C.PROMPT, device=device,
                num_images_per_prompt=1, max_sequence_length=512)
        embeds = (prompt_embeds.clone(), pooled.clone())
        pipe.text_encoder = pipe.text_encoder_2 = None
        pipe.tokenizer = pipe.tokenizer_2 = None
        gc.collect()
        torch.cuda.empty_cache()
    return pipe, embeds


def crop_box(mask, size):
    """Frame the mask, then widen until the aspect ratio is one the model can hold."""
    box = mask.getbbox()
    if box is None:
        return (0, 0, size[0], size[1])
    pad = max(24, int(C.CROP_PAD * max(box[2] - box[0], box[3] - box[1])))
    left, top = max(0, box[0] - pad), max(0, box[1] - pad)
    right, bottom = min(size[0], box[2] + pad), min(size[1], box[3] + pad)
    width, height = right - left, bottom - top
    if width > C.MAX_ASPECT * height:
        grow = (width / C.MAX_ASPECT - height) / 2
        top, bottom = max(0, int(top - grow)), min(size[1], int(bottom + grow))
    elif height > C.MAX_ASPECT * width:
        grow = (height / C.MAX_ASPECT - width) / 2
        left, right = max(0, int(left - grow)), min(size[0], int(right + grow))
    return (left, top, right, bottom)


def working_size(width, height):
    longest = max(width, height)
    scale = (C.MAX_SIDE / longest if longest > C.MAX_SIDE
             else C.MIN_SIDE / longest if longest < C.MIN_SIDE else 1.0)
    return (max(C.MULTIPLE, int(round(width * scale / C.MULTIPLE)) * C.MULTIPLE),
            max(C.MULTIPLE, int(round(height * scale / C.MULTIPLE)) * C.MULTIPLE))


def calibrate(generated, crop, crop_mask):
    """Fit the repair's tone using pixels whose correct value is already known.

    The model repaints the whole crop, but only the masked part is kept -- so its rendering
    of the unmasked pixels is a free reference, because the original holds the answer for
    exactly those. Any systematic shift shows there, and the same correction applies inside
    the mask, which has no reference of its own.
    """
    reference = np.asarray(crop, np.float32)
    outside = np.asarray(crop_mask, np.uint8) < 128
    if outside.sum() < 500:
        return generated
    fixed = generated.copy()
    for channel in range(3):
        source = generated[..., channel][outside]
        target = reference[..., channel][outside]
        if source.std() < 1e-3:
            continue
        gain, offset = np.polyfit(source, target, 1)
        low, high = C.CALIBRATE_GAIN
        if not (low < gain < high) or abs(offset) > C.CALIBRATE_MAX_OFFSET:
            continue        # a runaway fit would do more harm than the offset it corrects
        fixed[..., channel] = gain * generated[..., channel] + offset
    return np.clip(fixed, 0, 255)


def run(pipe, embeds, image, mask, seed=0):
    """One repaint of `mask`, calibrated and composited back into `image`."""
    box = crop_box(mask, image.size)
    width, height = box[2] - box[0], box[3] - box[1]
    target = working_size(width, height)
    crop = image.crop(box).resize(target, Image.Resampling.LANCZOS)
    crop_mask = mask.crop(box).resize(target, Image.Resampling.NEAREST)

    text = ({"prompt_embeds": embeds[0], "pooled_prompt_embeds": embeds[1]}
            if embeds is not None else {"prompt": C.PROMPT})
    generated = pipe(**text, image=crop, mask_image=crop_mask,
                     height=target[1], width=target[0], guidance_scale=C.GUIDANCE,
                     num_inference_steps=C.STEPS, strength=C.STRENGTH,
                     generator=torch.Generator(device="cpu").manual_seed(seed)).images[0]

    array = np.asarray(generated, np.float32)
    if C.CALIBRATE:
        array = calibrate(array, crop, crop_mask)
    generated = Image.fromarray(array.astype(np.uint8), "RGB")

    patch = image.copy()
    patch.paste(generated.resize((width, height), Image.Resampling.LANCZOS), box[:2])
    return Image.composite(patch, image, feathered(mask))


def feathered(mask):
    """Soften the mask outward only. A plain Gaussian also eats inward, and that is a bug.

    The feather exists to hide the seam at the mask's outer edge. Blurring the mask itself
    does that, and also hollows out anything thinner than the kernel: a letter stroke eight
    pixels wide, blurred at sigma 8, never reaches full strength anywhere along its spine,
    so the composite hands back a mixture of the repair and the photograph. What survives is
    a faint copy of the original, in the original colour, in the exact shape of the text --
    which reads as an overlay that was never erased, and is in fact one that was erased and
    then partly painted back on.

    Measured over the 125: at halo 10 the blur leaves 29% of the mask below full strength
    and a median 6.1% of the original surviving inside it; at halo 0, where the mask hugs
    the strokes, 37.5% and 7.9%. Taking the brighter of the mask and its blur keeps the
    interior at full strength and puts the entire soft skirt outside, where it was always
    meant to be.
    """
    return ImageChops.lighter(mask, mask.filter(ImageFilter.GaussianBlur(C.FEATHER)))
