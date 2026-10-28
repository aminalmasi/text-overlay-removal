# GPU eraser (deployment package)

Erases the text that Gemini's worker boxed on a listing photo (see [`../gemini/`](../gemini/README.md) for the reply format),
with the pipeline that was reviewed photo by photo in October 2026. One process holds every model on one GPU and serves photos one at a time,
over HTTP or in batches.

> **Licence — read first.** FLUX.1 Fill [dev] is released under the *FLUX.1 [dev] Non-Commercial License*:
> non-commercial and non-production use only. Running this on a live marketplace needs a commercial licence
> from Black Forest Labs. The other models are SAM 2.1 (Apache-2.0), LaMa (Apache-2.0), CRAFT (MIT) and the
> object-removal LoRA by lrzjason (MIT, trained on top of FLUX.1-Fill-dev).

## Contents

| Path | What |
|---|---|
| `eraser/engine.py` | `Eraser`: loads the models once; `erase(photo, worker_reply, photo_id)` runs the 9 steps |
| `eraser/server.py` | HTTP service: `POST /erase`, `GET /health` |
| `eraser/cli.py` | batch mode + benchmark (seconds per stage, photos per hour, GPU memory) |
| `eraser/client_example.py` | how to call the service |
| `download_models.py` | downloads the exact model files (pinned revisions, SHA-256 checked), ~33 GB |
| `prepare_models.py` | one-time: saves the prompt encoding and the NF4 transformer, for a lighter start |
| `bench_samples/` | `bench_a6000.sh` (speed, two processes per card, bf16) and `check_outputs.py` (compares an install with reference outputs). Both expect `bench_samples/requests.jsonl` with your own photos: the 24 sample photos, their worker replies and the reference outputs are private and not included |
| `tests/service_smoke.py` | 10 checks against a running service (normal case and edge cases) |
| `tl/` | the reviewed research code, copied unchanged except for paths |
| `requirements.txt`, `setup_venv.sh`, `Dockerfile` | environment, pinned to the reviewed versions |

## The steps (in the reviewed order)

| # | Step | Model | When |
|--:|---|---|---|
| 0 | line split: a region with no carrier becomes one region per line box | — | always |
| 1 | refine: snap box edges to the text, widen emoji regions, re-align shifted boxes | CRAFT | always |
| 2 | starting mask: SAM2 outline ∪ LaMa change region ∪ CRAFT strokes | SAM2.1, LaMa, CRAFT | always |
| 3 | stencil: close gaps between lines, halo 0, restore protected surfaces | — | always |
| 4 | first repaint: FLUX.1-Fill + objectRemoval LoRA, NF4, 30 steps | FLUX | always |
| 5 | patch fix: noise patch → 2 disguised repaints, pasted only inside the hole | FLUX, CRAFT | ~6% of photos |
| 6 | layer 2: leftover text → SAM2 extension → FLUX + LaMa inside the stencil | CRAFT, SAM2, FLUX, LaMa | ~16% |
| 7 | plate carve: writing over a protected plate, strokes only | LaMa | rare |
| 8 | size gate: largest patch > 16% of the frame and > 25% of the centre → ship the original | — | ~2% held |

## Install

Needs: an NVIDIA GPU with ≥ 24 GB (A6000 recommended), a recent driver, ~50 GB disk, ≥ 32 GB RAM, Python 3.10
or Docker, and a Hugging Face token whose account has accepted the FLUX.1-Fill-dev licence.

```bash
# 1. environment (or: docker build -t tl-eraser .)
bash setup_venv.sh /opt/tl-venv && source /opt/tl-venv/bin/activate

# 2. models (~33 GB, checksums verified)
export HF_TOKEN=hf_...
python download_models.py --models /models

# 3. one-time preparation (~5 min)
TL_MODELS=/models python prepare_models.py --out /models/prepared

# 4. run a batch (one JSON line per photo: {"id", "photo": path, "worker_reply"}); --debug keeps every stage
export TL_MODELS=/models TL_PREPARED=/models/prepared
python -m eraser.cli --requests my_requests.jsonl --out /tmp/tl_out --debug

# 5. serve
python -m eraser.server --port 8080          # local only: 127.0.0.1:8080
TL_TOKEN=$(openssl rand -hex 16) python -m eraser.server --host 0.0.0.0 --port 8080   # reachable; send the token
python tests/service_smoke.py http://127.0.0.1:8080                                  # 10 checks
```

With Docker, steps 2–3 run once on the host (or in the container with `-v /models:/models`), then
`docker run --gpus all -e TL_TOKEN=... -p 8080:8080 -v /models:/models tl-eraser`.

## API

`POST /erase` with JSON:

```json
{"id": "listing-123-photo-2",
 "image_b64": "<base64 of the photo bytes>",
 "worker_reply": {"text": "...", "boxes": [ ... P92's reply, unchanged ... ]},
 "format": "jpeg", "quality": 95}
```

With `Authorization: Bearer <TL_TOKEN>` when the server has a token. Reply: `{"status": ..., "info": {...}}`,
plus `image_b64` only when the status is `erased`; otherwise ship your original bytes unchanged.

| status | meaning | image returned |
|---|---|---|
| `erased` | the overlay was removed | the erased photo |
| `held` | the hole would be too big (size gate) | none: ship the original |
| `nothing_to_erase` | no region with `physically_in_scene: false`, or an empty mask | none |
| `too_large` | over `TL_MAX_PIXELS` (default 2048×2048; tested up to 2.1 MP) | none |
| `rotated_input` | the photo carries an EXIF rotation tag (see below) | none |

Errors: 400 bad request, 401 token, 413 body over 40 MB, 503 queue full (`--max-waiting`, default 8) or GPU out
of memory on that photo. In every error case, ship the original.

`info` says what fired (`patch_flagged`, `patch_fixed`, `layer2_fired`, `carved`), the mask size and the seconds
per stage. Requests are served one at a time; put a queue in front.

Two things the caller must get right:
- **Same pixels for Gemini and the eraser.** Send both the same bytes. If photos carry an EXIF rotation tag,
  rotate them upright and strip the tag first (e.g. `PIL.ImageOps.exif_transpose`); the eraser refuses a tagged
  photo (`rotated_input`) rather than guess which orientation Gemini saw.
- **A stable `id`.** It seeds every repaint, so the same photo and id always give the same output.

## Speed and memory

Measured on an RTX A5000 (this package, 111 reviewed photos, GPU time per photo without file writes); the A6000
column is the A5000 × 0.80, the ratio of the same repaint on the same photos in earlier runs. The earlier, unpackaged
pipeline measured 21.1 s median on an A6000 itself. `bench_samples/bench_a6000.sh` measures a new node.

| Per erased photo | RTX A5000 | RTX A6000 (est.) |
|---|--:|--:|
| median | 25.0 s | 20.0 s |
| mean | 31.6 s | 25.3 s |
| p90 | 64 s | 51 s |
| photos per hour, one process | 114 | 142 |
| model load at start-up (network disk) | 78 s (126 without prepare) | faster on a local SSD |
| GPU memory, whole process, peak | 13.4 GiB (15.8 without prepare) | same |

FLUX is 98% of the GPU time and keeps the GPU ~95% busy, so:
- **1 process per GPU** is the default. An A6000 (or a 48 GB L40) holds up to three 13.4 GiB processes, an A5000 one,
  but extra processes share the same busy GPU: a second one adds at most ~10% photos per hour, and each photo then
  takes about twice as long. `bench_samples/bench_a6000.sh` measures this on the node.
- More photos per hour = more GPUs.

## Optimisations

Applied (output unchanged):
- every model loaded once in one process; the CRAFT text map is computed once per photo and reused by
  refine, the mask and the plate carve (the experiment runs ran each step as its own process);
- `prepare_models.py`: the repaint prompt is encoded once and saved, so the 9.5 GB T5 encoder is never loaded
  again, and the NF4 transformer is saved already quantized (6.7 GB to read instead of 23 GB). Both produce
  the same weights and embeddings as the reviewed loader (checked bit for bit, see below). Start-up: 15 s on a
  warm disk (78 s cold on network storage, against 126 s for the reviewed loader), and the process peaks at 13.4
  GiB instead of 15.8 because the 9.5 GB text encoder never touches the GPU;
- a warm-up repaint at service start, so the first real photo doesn't pay the one-off kernel set-up.

Possible, but each changes the output and needs a review before use:

| Option | Expected gain | Risk |
|---|---|---|
| `--precision bf16` (no quantization; A6000 only, 1 process, ~30 GB) | ~10–30% faster repaint | small pixel differences (NF4 vs bf16 measured earlier: median 5.28/255, judged indistinguishable) |
| patch fix: stop after the first accepted repaint | ~26 s on the 6% of photos it fires on | the better of two repaints is no longer chosen |
| 20 FLUX steps instead of 30 | ~−33% GPU time | more texture and invention |
| crop cap 1024 px instead of 1280 | large overlays up to ~35% faster | less detail in big holes |

Not worth it: `torch.compile`, TF32, batching two photos per FLUX call (gains under ~5–10% at this GPU load,
and outputs drift).

## Validation

Run before hand-off (RTX A5000):

- **All 113 photos of the reviewed P92 run**, compared stage by stage with the reviewed outputs: masks median
  IoU 1.0000; first repaint bit-identical on 111 of 113, final picture on 109 of 111, patch-fix results 2 of 2,
  size-gate decision 113 of 113. The two exceptions (s043, s073) came from the reviewed run reading its text map
  back from an 8-bit PNG; the engine now reads it the same way, and both became bit-identical.
- **Loaders:** the prepared loader and the reviewed loader give bit-identical results (4 of 4 photos).
- **Clean install:** a new environment built from `requirements.txt` alone (nothing inherited) ran the 24 bench
  samples from raw P92 replies (so the line split ran too): all 24 bit-identical to the reviewed run, masks and
  final pictures. Then the HTTP service with a token passed `tests/service_smoke.py`: a real photo, wrong token,
  malformed JSON and base64, no boxes, malformed boxes, a mark entirely on a protected plate (this used to crash),
  a 7.5 MP photo (`too_large`) and an EXIF-rotated photo (`rotated_input`).
- **Independent code review** before hand-off; its findings are fixed: missing LaMa dependencies (scikit-learn,
  pandas, imgaug), the empty-mask crash, input sanitising, size and rotation guards, colour profile kept on
  re-encode, token auth and a bounded queue on the server.

Not covered by the review photos: patch fix v3 followed by layer 2 on the same photo (each was reviewed, not
the pair), and photos above 2.1 MP (refused as `too_large`).

## Troubleshooting

- `CHECKSUM MISMATCH` during download: the file is not the reviewed one. Delete it and download again.
- Out of memory: check that nothing else is on the GPU; NF4 needs ~13.4 GiB per process (15.8 without
  prepare); `bf16` needs ~30 GB.
- Slow first start: `prepare_models.py` was not run, or `TL_PREPARED` is not set (`/health` shows `"loader"`).
