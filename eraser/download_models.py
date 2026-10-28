#!/usr/bin/env python3
"""Download the exact model files the reviewed runs used: pinned revisions, and every weight file checked
by SHA-256 (configs and tokenizers are pinned by the revision).

    export HF_TOKEN=hf_...        # FLUX.1-Fill-dev is gated: accept its license on huggingface.co first
    python download_models.py --models /models

Downloads about 33 GB. Afterwards run prepare_models.py once.
"""
import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

FLUX = ("black-forest-labs/FLUX.1-Fill-dev", "358293da0354175698b67ec8299acf928313a78a")
FLUX_SHA256 = {   # every weight file the engine loads, as Hugging Face lists them at that revision
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": "4cef6b166b2163ba9a01ccb68bf53054556852b69b32c0565dfe0741d28982d7",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": "cd99fdd45ac404a9c98c939bdc986647cd5160ca48b8e679e81d3d59c2b9a1e0",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": "ed878852603579083b4a4ea80544504faca9f42ebd41aaa103c3023ee2ebfbd5",
    "vae/diffusion_pytorch_model.safetensors": "8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94",
    "text_encoder/model.safetensors": "893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082",
    "text_encoder_2/model-00001-of-00002.safetensors": "ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e",
    "text_encoder_2/model-00002-of-00002.safetensors": "a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00",
}
LORA = ("lrzjason/ObjectRemovalFluxFill", "867f2e6adb15232efd8d07bbb289e0f9f7551791", "objectRemovalv1_r32.safetensors",
        "66e85b8d4fea938b8377ac01bdceab94c7f75dbec23f669b45418e192f3a7c16")
SAM = ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610",
       {"model.safetensors": "dc407dce21301fd94abb395c5099b4f2c455fdc8a8f261ac3d0ea6d4cd197230"})
LAMA = ("smartywu/big-lama", "05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9", "big-lama.zip",
        "f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6")
CRAFT = ("boomb0om/CRAFT-text-detector", "78216c17652819119d0788143b82290fa15cd6e6", "craft_mlt_25k.pth",
         "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def check(path: Path, want: str):
    got = sha256(path)
    if got != want:
        sys.exit(f"CHECKSUM MISMATCH {path}: {got} != {want}")
    print(f"ok  {path.name}  sha256 {got[:12]}...")


def one_file(repo, rev, name, want, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / name).exists():
        shutil.copy(hf_hub_download(repo, name, revision=rev), dest / name)
    check(dest / name, want)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True, help="becomes TL_MODELS")
    a = ap.parse_args()
    m = a.models
    m.mkdir(parents=True, exist_ok=True)

    print("FLUX.1-Fill-dev (gated, ~32 GB) ...")
    snapshot_download(FLUX[0], revision=FLUX[1], local_dir=m / "FLUX.1-Fill-dev",
                      allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
                                      "tokenizer/*", "tokenizer_2/*", "transformer/*", "vae/*", "LICENSE.md"])
    for name, want in FLUX_SHA256.items():
        check(m / "FLUX.1-Fill-dev" / name, want)
    one_file(LORA[0], LORA[1], LORA[2], LORA[3], m / "FluxFill-ObjectRemoval-LoRA")
    snapshot_download(SAM[0], revision=SAM[1], local_dir=m / "sam2.1-hiera-large",
                      allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json"])
    for name, want in SAM[2].items():
        check(m / "sam2.1-hiera-large" / name, want)
    one_file(LAMA[0], LAMA[1], LAMA[2], LAMA[3], m / "LaMa-big")
    if not (m / "LaMa-big/big-lama/models/best.ckpt").exists():
        with zipfile.ZipFile(m / "LaMa-big/big-lama.zip") as z:
            z.extractall(m / "LaMa-big")
    check(m / "LaMa-big/big-lama/models/best.ckpt", "fccb7adffd53ec0974ee5503c3731c2c2f1e7e07856fd9228cdcc0b46fd5d423")
    one_file(CRAFT[0], CRAFT[1], CRAFT[2], CRAFT[3], m / "CRAFT")
    print(f"\nall models in {m}. Next: TL_MODELS={m} python prepare_models.py --out {m}/prepared")


if __name__ == "__main__":
    main()
