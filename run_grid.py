"""run_grid.py -- runs the experiment grid (resumable; one JSON + NPZ per run).
usage: python run_grid.py SHARD NSHARDS [GROUP ...]"""
import os
import sys, os, time, json, itertools
import numpy as np
from data import load_panel, build_task
from train import run

ROOT = os.environ.get("MILO_RES", "results/runs")
MAIN = ["CE", "LS", "GCE", "FwdT", "CORAL", "SORD", "MSE", "HetG", "LD", "LD-flat", "LT", "LT-noN", "LT-noC"]


def jobs():
    J = []
    for task in ["A", "C", "B"]:                                         # main grid, 5 seeds
        for seed, bb, m in itertools.product(range(5), ["MLP", "GRU"], MAIN):
            J.append(("main", task, {}, m, bb, seed, ""))
    for seed, bb, m in itertools.product(range(3), ["MLP", "GRU"], MAIN):  # Task C with S1 noise
        J.append(("cs1", "C", {"scen": "S1"}, m, bb, seed, "_S1"))
    for seed, bb, s in itertools.product(range(3), ["MLP", "GRU"], [0.0625, 0.125, 0.1875, 0.3125, 0.375, 0.5]):
        J.append(("miss", "C", {}, f"LT@{s}", bb, seed, ""))             # misspecified assumed noise
    for seed, s, m in itertools.product(range(3), [0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375],
                                        ["CE", "LS", "FwdT", "HetG", "LD", "LT"]):
        J.append(("sweep", "C", {"s_inject": s}, m, "MLP", seed, f"_s{s:g}"))
    for task, seed, bb, m in itertools.product(["A", "C"], range(3), ["TCN", "TRF"], ["CE", "FwdT", "HetG", "LD", "LT"]):
        J.append(("backbone", task, {}, m, bb, seed, ""))
    return J


if __name__ == "__main__":
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    groups = set(sys.argv[3:]) or None
    stns, dates, V = load_panel()
    cache = {}
    J = [j for j in jobs() if groups is None or j[0] in groups]
    mine = J[shard::nsh]
    t0 = time.time()
    for i, (grp, task, kw, m, bb, seed, tag) in enumerate(mine):
        out = os.path.join(ROOT, grp)
        stem = f"{task}{tag}_{m}_{bb}_s{seed}"
        if os.path.exists(os.path.join(out, stem + ".json")):
            continue
        key = (task, seed if task in "AC" else 0, tuple(sorted(kw.items())))
        if key not in cache:
            if len(cache) > 6:
                cache.clear()
            cache[key] = build_task(task, stns, dates, V, seed=seed, **kw)
        r = run(cache[key], m, bb, seed, outdir=out, tag=tag)
        print(f"[{shard}] {i + 1}/{len(mine)} {grp} {stem} ep={r['epochs']} t={r['train_time_s']}s "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"[{shard}] DONE in {time.time() - t0:.0f}s", flush=True)
