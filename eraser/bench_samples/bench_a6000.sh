#!/bin/bash
# The numbers to measure on the rented node, BEFORE starting the eraser services (the GPU must be free).
# Each run uses the 24 sample photos (~10-15 min each):
#   bash gpu_eraser/bench_samples/bench_a6000.sh $TL_OUT/bench
# 1) NF4, one process (the reviewed setting)       -> <out>/nf4_1/summary.json
# 2) NF4, two processes sharing the GPU            -> <out>/nf4_2a, nf4_2b (their photos per hour add up)
# 3) bf16, one process (needs ~30 GB; NOT reviewed; skipped on cards under 40 GB)
set -u
BASE=$([ -d /workspace ] && echo /workspace || echo /opt)
source "${TL_ROOT:-$BASE/tl}/env.sh"
PY=${PY:-$TL_ROOT/venv-eraser/bin/python}
OUT=${1:-$TL_OUT/bench}
mkdir -p "$OUT"
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "the GPU is in use (eraser services running?): stop them first (bash node/stop_services.sh)"; exit 1
fi
cd "$(dirname "$0")/.."
R=bench_samples/requests.jsonl
"$PY" -m eraser.cli --requests $R --out $OUT/nf4_1 > $OUT.nf4_1.log 2>&1
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$MEM" -ge 40000 ]; then          # two 13.4 GiB processes do not fit on a 24 GB card
  "$PY" -m eraser.cli --requests $R --out $OUT/nf4_2a --offset 0 --step 2 > $OUT.nf4_2a.log 2>&1 &
  "$PY" -m eraser.cli --requests $R --out $OUT/nf4_2b --offset 1 --step 2 > $OUT.nf4_2b.log 2>&1 &
  wait
else echo "skipping the 2-process run: ${MEM} MiB GPU"; fi
if [ "$MEM" -ge 40000 ]; then "$PY" -m eraser.cli --requests $R --out $OUT/bf16_1 --precision bf16 > $OUT.bf16_1.log 2>&1
else echo "skipping bf16: ${MEM} MiB GPU"; fi
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
def s(n):
    p = o / n / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None
one, a, b, bf = s("nf4_1"), s("nf4_2a"), s("nf4_2b"), s("bf16_1")
if one: print(f"NF4, 1 process : {one['photos_per_hour']} photos/h, median {one['per_photo_s']['median']} s, peak {one['vram_gib']['process_peak_nvidia_smi']} GiB")
if a and b:
    wall = max(a["wall_s"], b["wall_s"]); n = a["photos"] + b["photos"]
    print(f"NF4, 2 processes: {3600 * n / wall:.0f} photos/h together (wall clock), median {a['per_photo_s']['median']} / {b['per_photo_s']['median']} s per photo")
if bf: print(f"bf16, 1 process : {bf['photos_per_hour']} photos/h, median {bf['per_photo_s']['median']} s, peak {bf['vram_gib']['process_peak_nvidia_smi']} GiB")
PY
