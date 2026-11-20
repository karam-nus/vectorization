#!/usr/bin/env python3
"""Compile code/bench/bench.c five ways, run the book's experiments, write results.json.

    python tools/measurements/run_bench.py            # ~3 minutes on one core
    python tools/measurements/run_bench.py --quick    # smaller sizes, ~40 s

Every measured number in the book comes from the JSON this script writes. The machine
description is recorded alongside the numbers; never quote a number without it.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "code", "bench", "bench.c")
OUT = os.path.join(HERE, "results.json")

# name -> gcc flags. "scalar" switches off both the loop and the SLP vectorizer.
VARIANTS = {
    "scalar": ["-O2", "-fno-tree-vectorize", "-fno-tree-slp-vectorize", "-march=x86-64"],
    "sse2": ["-O3", "-march=x86-64"],
    "avx2": ["-O3", "-march=x86-64-v3"],
    "avx512": ["-O3", "-march=x86-64-v4", "-mprefer-vector-width=512"],
    "avx512_fast": ["-O3", "-march=x86-64-v4", "-mprefer-vector-width=512", "-ffast-math"],
}

# kernel -> (flop per element, bytes moved per element) for derived metrics.
COST = {
    "saxpy": (2, 12), "leaky": (1, 8), "strided": (1, 8), "dot": (2, 8), "prefix": (1, 8),
    "step_aos": (6, 48), "step_soa": (6, 36),
    "dot_avx512_1acc": (2, 8), "dot_avx512_2acc": (2, 8), "dot_avx512_4acc": (2, 8),
    "dot_avx512_8acc": (2, 8), "leaky_avx512": (1, 8), "strided_gather": (1, 8),
}


def machine():
    model = ""
    flags = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and not model:
                    model = line.split(":", 1)[1].strip()
                if line.startswith("flags") and not flags:
                    flags = line.split(":", 1)[1]
    except OSError:
        pass
    cpu_model = subprocess.run(["bash", "-c", "lscpu | awk -F: '/^Model:/{print $2}'"],
                               capture_output=True, text=True).stdout.strip()
    gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    isa = [f for f in ("avx2", "avx512f", "avx512_fp16", "avx512_bf16", "avx512_vnni", "amx_tile")
           if f" {f} " in f" {flags} "]
    return {"cpu": model, "cpu_model_number": cpu_model, "compiler": gcc,
            "isa": isa, "python": platform.python_version(), "threads": 1,
            "method": "pinned to one core; best of 3 processes x best of 5 timed trials"}


def build(tmp):
    bins = {}
    for name, flags in VARIANTS.items():
        exe = os.path.join(tmp, f"bench_{name}")
        cmd = ["gcc", *flags, f'-DVARIANT="{name}"', "-o", exe, SRC, "-lm"]
        subprocess.run(cmd, check=True)
        bins[name] = exe
    return bins


PIN = ["taskset", "-c", "1"] if shutil.which("taskset") else []


def run(exe, kernel, n, stride=1, repeats=3):
    """Best of `repeats` processes, each already best-of-5 trials: cloud VMs are noisy."""
    r = None
    for _ in range(repeats):
        out = subprocess.run([*PIN, exe, kernel, str(n), str(stride)], capture_output=True, text=True)
        if out.returncode != 0:
            return None
        cur = json.loads(out.stdout)
        if r is None or cur["sec"] < r["sec"]:
            r = cur
    if kernel in COST:
        flop, byte = COST[kernel]
        r["ns_per_elem"] = r["sec"] / n * 1e9
        r["gflops"] = flop * n / r["sec"] / 1e9
        r["gbps"] = byte * n / r["sec"] / 1e9
    return r


def run_amx():
    """Compile and run code/bench/amx_peak.c; return the best of 3 runs, or None without AMX."""
    if "amx_tile" not in machine()["isa"]:
        return None
    src = os.path.join(ROOT, "code", "bench", "amx_peak.c")
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "amx_peak")
        if subprocess.run(["gcc", "-O2", "-mamx-tile", "-mamx-bf16", "-o", exe, src]).returncode:
            return None
        best = None
        for _ in range(3):
            out = subprocess.run([*PIN, exe], capture_output=True, text=True)
            if out.returncode:
                return None
            r = json.loads(out.stdout)
            if best is None or r["sec"] < best["sec"]:
                best = r
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["amx"], help="run one experiment and merge it into results.json")
    args = ap.parse_args()
    q = args.quick
    if args.only == "amx":
        res = json.load(open(OUT))
        res["experiments"]["amx_peak"] = run_amx()
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"updated {OUT}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        bins = build(tmp)
        res = {"machine": machine(), "flags": VARIANTS, "experiments": {}}
        E = res["experiments"]

        # E1 — register width ladder, data in L1 (n = 4096 fp32 = 16 KB per array)
        n = 4096
        rows = []
        for v, exe in bins.items():
            for k in ("saxpy", "leaky", "dot", "prefix", "step_soa", "step_aos"):
                rows.append(run(exe, k, n))
        for k in ("dot_avx512_8acc", "leaky_avx512"):
            rows.append(run(bins["avx512"], k, n))
        E["width_ladder_L1"] = [r for r in rows if r]
        print("E1 done", file=sys.stderr)

        # E2 — the same saxpy at four memory levels
        sizes = [4096, 65536, 1 << 20, 1 << 26] if not q else [4096, 65536, 1 << 20, 1 << 23]
        E["saxpy_memory_levels"] = [run(bins[v], "saxpy", s) for v in ("scalar", "sse2", "avx2", "avx512")
                                    for s in sizes]
        print("E2 done", file=sys.stderr)

        # E3 — stride sweep, 1 M outputs
        n = 1 << 20
        rows = []
        for s in (1, 2, 4, 8, 16, 32):
            rows.append(run(bins["scalar"], "strided", n, s))
            rows.append(run(bins["avx512"], "strided", n, s))
            rows.append(run(bins["avx512"], "strided_gather", n, s))
        E["stride_sweep"] = [r for r in rows if r]
        print("E3 done", file=sys.stderr)

        # E4 — accumulator count for a reduction, data in L1
        n = 4096
        rows = [run(bins["scalar"], "dot", n), run(bins["avx512"], "dot", n), run(bins["avx512_fast"], "dot", n)]
        rows += [run(bins["avx512"], f"dot_avx512_{k}acc", n) for k in (1, 2, 4, 8)]
        E["dot_accumulators_L1"] = [r for r in rows if r]
        print("E4 done", file=sys.stderr)

        # E5 — matmul loop order
        N = 512 if not q else 256
        rows = []
        for v in ("scalar", "avx512", "avx512_fast"):
            for k in ("mm_ijk", "mm_ikj"):
                r = run(bins[v], k, N)
                if r:
                    r["gflops"] = 2 * N ** 3 / r["sec"] / 1e9
                rows.append(r)
        E["matmul_loop_order"] = [r for r in rows if r]
        print("E5 done", file=sys.stderr)

        # E6 — pairwise distances, the Chapter 20 ladder (C rungs)
        n = 1024 if not q else 256
        rows = []
        for v in ("scalar", "avx512", "avx512_fast"):
            for k in ("pdist_naive", "pdist_outer"):
                r = run(bins[v], k, n)
                if r:
                    r["ns_per_pair_dim"] = r["sec"] / (n * n * 64) * 1e9
                rows.append(r)
        E["pdist_c"] = [r for r in rows if r]
        print("E6 done", file=sys.stderr)

        # E7 — compute roofs: register-only FMA chains at 1, 4, 8, 16 fp32 lanes
        iters = 2_000_000
        rows = []
        for lanes in (1, 4, 8, 16):
            r = run(bins["avx512"], f"peak_w{lanes}", iters)
            if r:
                r["lanes"] = lanes
                r["gflops"] = iters * 12 * lanes * 2 / r["sec"] / 1e9
                rows.append(r)
        E["peak_fma"] = rows
        print("E7 done", file=sys.stderr)

        # E8 — Intel AMX bf16 tile multiply throughput (needs amx_tile; skipped elsewhere)
        E["amx_peak"] = run_amx()
        print("E8 done", file=sys.stderr)

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
