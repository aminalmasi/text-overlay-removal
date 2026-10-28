#!/usr/bin/env python3
"""Batch mode: erase many photos with the models loaded once, and report time and memory.

Requests file (JSON lines), one photo per line, P92's reply as the worker returned it:
    {"id": "listing-123-photo-2", "photo": "/data/in/123_2.jpg", "worker_reply": {"text": "...", "boxes": [...]}}

    python -m eraser.cli --requests requests.jsonl --out /data/out
    python -m eraser.cli --requests requests.jsonl --out /data/out --precision bf16      # 48 GB GPU only

Writes <out>/<id>.png (the photo to ship: erased, or the original when held), <out>/results.jsonl (status and
info per photo) and <out>/summary.json (seconds per stage, photos per hour, peak GPU memory). --debug also
writes the stencil, the first repaint and the patch-fix result per photo under <out>/debug/.
--items reads the research harness's items_<tag>.json (boxes already line-split) instead of --requests.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eraser.engine import Eraser   # noqa: E402


class VramSampler(threading.Thread):
    """This process's GPU memory as nvidia-smi sees it (CUDA context + allocator), every 0.3 s."""
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, False

    def run(self):
        pid = str(os.getpid())
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                for line in out.splitlines():
                    p, m = [x.strip() for x in line.split(",")]
                    if p == pid:
                        self.peak = max(self.peak, int(m))
            except Exception:
                pass
            time.sleep(0.3)


def p90(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(0.9 * (len(v) - 1))))] if v else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--requests", type=Path)
    src.add_argument("--items", type=Path, help="research harness: items_<tag>.json, boxes already line-split")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--precision", default="nf4", choices=["nf4", "bf16"])
    ap.add_argument("--prepared", default=os.environ.get("TL_PREPARED"), help="prepare_models.py output (faster start)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    a = ap.parse_args()

    if a.requests:
        jobs = [json.loads(l) for l in a.requests.read_text(encoding="utf-8").splitlines() if l.strip()]
        jobs = [(j["id"], j["photo"], j.get("worker_reply") or {}, True) for j in jobs]
    else:
        jobs = [(i["id"], i["source"], {"boxes": i["boxes"]}, False) for i in json.loads(a.items.read_text(encoding="utf-8"))]
    jobs = jobs[a.offset::a.step]
    if a.limit:
        jobs = jobs[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    if a.debug:
        (a.out / "debug").mkdir(exist_ok=True)

    sampler = VramSampler()
    sampler.start()
    import torch
    eraser = Eraser(precision=a.precision, prepared=a.prepared)
    load_peak = sampler.peak
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"loaded_s": eraser.load_seconds, "load_steps_s": eraser.load_steps, "loader": eraser.loader, "precision": a.precision,
                      "gpu": torch.cuda.get_device_name(0), "photos": len(jobs)}), flush=True)

    rows, wall0 = [], time.perf_counter()
    with open(a.out / "results.jsonl", "w", encoding="utf-8") as log:
        for k, (pid, photo, reply, split) in enumerate(jobs):
            t = time.perf_counter()
            try:
                res = eraser.erase(photo, reply, photo_id=pid, debug=a.debug, split=split)
            except Exception as e:                       # one bad photo must not stop the batch
                row = {"id": pid, "status": "error", "error": repr(e)}
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                continue
            if res["image"] is not None:              # too_large: nothing decoded, ship the original bytes
                res["image"].save(a.out / f"{pid}.png", **({"icc_profile": res["icc_profile"]} if res.get("icc_profile") else {}))
            for name, im in (res.get("stages") or {}).items():
                im.save(a.out / "debug" / f"{pid}_{name}.png")
            row = {"id": pid, "status": res["status"], "wall_s": round(time.perf_counter() - t, 2), "warmup": k == 0, **res["info"]}
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    wall = time.perf_counter() - wall0
    sampler.stop = True

    timed = [r for r in rows if not r["warmup"] and r["status"] != "nothing_to_erase" and "seconds" in r]
    stages = ["refine", "start", "stencil", "repaint1", "patch", "layer2", "carve", "gate"]
    summary = {
        "gpu": torch.cuda.get_device_name(0), "precision": a.precision, "loader": eraser.loader,
        "model_load_s": eraser.load_seconds, "load_steps_s": eraser.load_steps, "photos": len(rows), "timed_photos": len(timed),
        "status": {s: sum(r["status"] == s for r in rows) for s in ("erased", "held", "nothing_to_erase", "too_large", "rotated_input")},
        "errors": sum(1 for l in open(a.out / "results.jsonl") if '"error"' in l),
        "wall_s": round(wall, 1),
        "per_photo_s": {"median": round(st.median([r["wall_s"] for r in timed]), 2) if timed else None,
                        "mean": round(st.mean([r["wall_s"] for r in timed]), 2) if timed else None,
                        "p90": round(p90([r["wall_s"] for r in timed]), 2)},
        "photos_per_hour": round(3600 * len(timed) / sum(r["wall_s"] for r in timed), 1) if timed else None,
        "stage_median_s": {s: round(st.median([r["seconds"].get(s, 0) for r in timed]), 3) for s in stages} if timed else {},
        "stage_mean_s": {s: round(st.mean([r["seconds"].get(s, 0) for r in timed]), 3) for s in stages} if timed else {},
        "fired": {"patch_flagged": sum(r.get("patch_flagged", False) for r in timed), "patch_fixed": sum(r.get("patch_fixed", False) for r in timed),
                  "layer2": sum(r.get("layer2_fired", False) for r in timed), "carved": sum(r.get("carved", False) for r in timed)},
        "vram_gib": {"process_peak_at_load_nvidia_smi": round(load_peak / 1024, 2),
                     "process_peak_nvidia_smi": round(sampler.peak / 1024, 2),
                     "torch_peak_allocated_working": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     "torch_peak_reserved_working": round(torch.cuda.max_memory_reserved() / 2**30, 2)},
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
