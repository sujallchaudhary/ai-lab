"""
Modal.com Remote Runner
========================
Distributes the ISBOA benchmark experiment across Modal containers
using .starmap() — one task per (function, algorithm), all 30 runs inside.

Usage:
  1. pip install modal
  2. modal setup            (one-time auth)
  3. modal run run_modal.py

Results are downloaded to the local ./results/ folder when done.
"""

import modal
import os

# --------------- Modal setup ---------------

app = modal.App("isboa-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "opfunu>=1.0",
        "setuptools<81",
    )
    .add_local_file("algorithms.py", "/root/algorithms.py", copy=True)
    .add_local_file("run_experiment.py", "/root/run_experiment.py", copy=True)
)

# --------------- Configuration ---------------

POP_SIZE = 30
MAX_FES = 60_000
NUM_RUNS = 30
DIM_MAP = {2014: 30, 2017: 30, 2020: 10, 2022: 10}
FUNC_RANGE = {2014: range(1, 31), 2017: range(1, 30), 2020: range(1, 11), 2022: range(1, 13)}
ALGO_NAMES = [
    "ISBOA", "SBOA", "GWO", "WOA", "SCA", "SSA", "HHO", "MPA", "AOA",
]


# --------------- Remote worker (one task = all 30 runs for one func+algo) -----

@app.function(image=image, timeout=1800, max_containers=100)
def batch_run(year: int, fnum: int, algo_name: str):
    """Run all NUM_RUNS seeds for a single (year, function, algorithm) combo."""
    import sys
    sys.path.insert(0, "/root")

    import numpy as np
    from algorithms import ALGORITHMS
    from run_experiment import _make_cec_func

    dim = DIM_MAP[year]
    result = _make_cec_func(year, fnum, dim)

    rows = []
    for seed in range(1, NUM_RUNS + 1):
        if result is None:
            rows.append({"Year": year, "Function": f"F{fnum}",
                         "Algorithm": algo_name, "Run": seed, "Error": float("inf")})
            continue
        obj, lb, ub, f_bias = result
        np.random.seed(seed)
        try:
            _, best_fit, _ = ALGORITHMS[algo_name](obj, lb, ub, dim, POP_SIZE, MAX_FES)
        except Exception:
            best_fit = float("inf")
        rows.append({"Year": year, "Function": f"F{fnum}",
                     "Algorithm": algo_name, "Run": seed, "Error": best_fit})
    return rows


# --------------- Local entrypoint ---------------

@app.local_entrypoint()
def main():
    import pandas as pd

    # Build task list: one per (year, function, algorithm) = 729 tasks
    tasks = []
    for year in sorted(DIM_MAP.keys()):
        for fnum in FUNC_RANGE[year]:
            for algo_name in ALGO_NAMES:
                tasks.append((year, fnum, algo_name))

    total = len(tasks)
    print("=" * 60)
    print(f"  ISBOA Benchmark — {total} tasks via Modal .starmap()")
    print(f"  Each task runs {NUM_RUNS} seeds internally")
    print("=" * 60)

    # Fan out across containers
    all_rows = []
    for i, batch in enumerate(batch_run.starmap(tasks)):
        all_rows.extend(batch)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i + 1}/{total}] tasks done  ({len(all_rows)} total runs)")

    df = pd.DataFrame(all_rows)

    # ---------- Post-processing (local) ----------
    from run_experiment import (
        compute_stats, compute_rankings,
        wilcoxon_test, win_tie_loss, generate_report,
    )

    local_results = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(local_results, exist_ok=True)

    csv_path = os.path.join(local_results, "raw_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    summary = compute_stats(df)
    rankings = compute_rankings(summary)
    rankings.to_csv(os.path.join(local_results, "summary.csv"), index=False)

    wilcox_df = wilcoxon_test(df, target="ISBOA")
    wilcox_df.to_csv(os.path.join(local_results, "wilcoxon.csv"), index=False)
    wtl_df = win_tie_loss(wilcox_df)
    wtl_df.to_csv(os.path.join(local_results, "win_tie_loss.csv"), index=False)

    report_path = os.path.join(local_results, "report.md")
    try:
        generate_report(summary, rankings, wilcox_df, wtl_df, report_path)
    except Exception as exc:
        print(f"  Report generation failed: {exc}")

    print("\n" + "=" * 60)
    print("  Experiment complete! Results in ./results/")
    print("=" * 60)
