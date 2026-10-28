#!/usr/bin/env python3
"""One-time preparation on the GPU node, after download_models.py. Makes start-up faster and lighter:

  prepared/prompt_embeds.pt    the repaint prompt encoded once (T5 + CLIP). The prompt never changes, so the
                               9.5 GB T5 encoder never has to be read or put on the GPU again.
  prepared/transformer_nf4/    the FLUX transformer saved already quantized to NF4 (~6.7 GB instead of 23 GB
                               to read, and no quantization step at start-up).

Both are made with the reviewed loader (pipeline/erase.py load), so the weights and embeddings are the same
ones the reviewed runs used. Run it once:

    TL_MODELS=/models python prepare_models.py --out /models/prepared
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tl"))

import torch                                      # noqa: E402
from pipeline import config as C, erase          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    C.USE_NF4 = True
    pipe, embeds = erase.load("cuda")
    torch.save({"prompt": C.PROMPT, "prompt_embeds": embeds[0].cpu(), "pooled_prompt_embeds": embeds[1].cpu()},
               a.out / "prompt_embeds.pt")
    # save the quantized base transformer WITHOUT the LoRA: the engine loads the LoRA on top, as erase.load does
    pipe.unload_lora_weights()
    pipe.transformer.save_pretrained(a.out / "transformer_nf4")
    print(f"prepared in {time.perf_counter() - t0:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
