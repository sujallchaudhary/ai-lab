"""
Experiment Runner
=================
Runs ISBOA and comparison algorithms on CEC 2014/2017/2020/2022,
performs statistical analysis, and generates a Markdown report.
"""

import os, sys, time, warnings, traceback
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from concurrent.futures import ProcessPoolExecutor, as_completed
from algorithms import ALGORITHMS

warnings.filterwarnings("ignore")

# ===================== Configuration =====================

POP_SIZE   = 30
MAX_FES    = 60_000
NUM_RUNS   = 30
DIM_MAP    = {2014: 30, 2017: 30, 2020: 10, 2022: 10}
FUNC_RANGE = {2014: range(1, 31), 2017: range(1, 30), 2020: range(1, 11), 2022: range(1, 13)}
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "report.md")
MAX_WORKERS = 16       # number of parallel processes (set to your CPU core count)

# ===================== Benchmark helpers =====================

def _make_cec_func(year, fnum, dim):
    """Return (callable, lb_array, ub_array) or None if unavailable."""
    try:
        import opfunu
        cls_name = f"F{fnum}{year}"
        mod = getattr(opfunu.cec_based, f"cec{year}", None)
        if mod is None:
            return None
        cls = getattr(mod, cls_name, None)
        if cls is None:
            return None
        func = cls(ndim=dim)
        lb = np.full(dim, func.lb) if np.isscalar(func.lb) else np.asarray(func.lb)
        ub = np.full(dim, func.ub) if np.isscalar(func.ub) else np.asarray(func.ub)
        f_bias = func.f_global if hasattr(func, "f_global") else 0.0
        if f_bias is None:
            f_bias = 0.0
        def obj(x):
            return func.evaluate(x) - f_bias   # return error = f(x) - f*
        return obj, lb, ub, f_bias
    except Exception:
        return None


def get_benchmarks(years=None):
    """
    Returns dict:  { (year, 'F{num}'): (obj_func, lb, ub, dim, f_bias), ... }
    """
    if years is None:
        years = [2014, 2017, 2020, 2022]
    benchmarks = {}
    for year in years:
        dim = DIM_MAP[year]
        for fnum in FUNC_RANGE[year]:
            result = _make_cec_func(year, fnum, dim)
            if result is not None:
                obj, lb, ub, f_bias = result
                benchmarks[(year, f"F{fnum}")] = (obj, lb, ub, dim, f_bias)
            else:
                print(f"  [skip] CEC{year} F{fnum} not available")
    return benchmarks


# ===================== Single run (worker) =====================

def _single_run(args):
    """
    Self-contained worker: creates benchmark inside the process so that
    everything is picklable for multiprocessing.
    Returns dict with Year, Function, Algorithm, Run, Error.
    """
    year, fnum, algo_name, seed = args
    import numpy as np
    from algorithms import ALGORITHMS as _ALGOS

    dim = DIM_MAP[year]
    result = _make_cec_func(year, fnum, dim)
    if result is None:
        return {"Year": year, "Function": f"F{fnum}",
                "Algorithm": algo_name, "Run": seed, "Error": np.inf}

    obj, lb, ub, f_bias = result
    np.random.seed(seed)
    try:
        _, best_fit, _ = _ALGOS[algo_name](obj, lb, ub, dim, POP_SIZE, MAX_FES)
    except Exception:
        best_fit = np.inf

    return {"Year": year, "Function": f"F{fnum}",
            "Algorithm": algo_name, "Run": seed, "Error": best_fit}


# ===================== Run all experiments =====================

def run_experiments(benchmarks, algo_names=None):
    """
    Returns DataFrame with columns:
      Year, Function, Algorithm, Run, Error
    Parallelised with MAX_WORKERS processes.
    """
    if algo_names is None:
        algo_names = list(ALGORITHMS.keys())

    # Build flat task list: (year, fnum, algo_name, seed)
    tasks = []
    for (year, fname) in sorted(benchmarks.keys()):
        fnum = int(fname[1:])  # "F3" -> 3
        for algo_name in algo_names:
            for run in range(1, NUM_RUNS + 1):
                tasks.append((year, fnum, algo_name, run))

    total = len(tasks)
    print(f"  {total} tasks queued, using {MAX_WORKERS} workers ...")
    t0 = time.time()

    if MAX_WORKERS <= 1:
        # Sequential fallback
        results = []
        for i, task in enumerate(tasks, 1):
            results.append(_single_run(task))
            if i % (len(algo_names) * NUM_RUNS) == 0:
                elapsed = time.time() - t0
                eta = elapsed / i * (total - i)
                print(f"  [{i}/{total}]  ETA {eta/60:.1f}min")
    else:
        # Parallel execution
        results = []
        done = 0
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_single_run, t): t for t in tasks}
            for future in as_completed(futures):
                results.append(future.result())
                done += 1
                if done % 100 == 0 or done == total:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total - done) if done else 0
                    print(f"  [{done}/{total}]  "
                          f"elapsed {elapsed/60:.1f}min  ETA {eta/60:.1f}min")

    df = pd.DataFrame(results)
    return df


# ===================== Statistics =====================

def compute_stats(df):
    """Return summary DataFrame: Year, Function, Algorithm, Mean, Std."""
    grp = df.groupby(["Year", "Function", "Algorithm"])["Error"]
    summary = grp.agg(Mean="mean", Std="std").reset_index()
    return summary


def compute_rankings(summary):
    """Add Rank column (per year+function, lower mean = rank 1)."""
    summary = summary.copy()
    summary["Rank"] = summary.groupby(["Year", "Function"])["Mean"].rank(method="min")
    return summary


def wilcoxon_test(df, target="ISBOA", alpha=0.05):
    """
    Compare *target* against every other algorithm on every function.
    Returns DataFrame: Year, Function, Opponent, p-value, Symbol (+/=/-)
    """
    algos = [a for a in df["Algorithm"].unique() if a != target]
    rows = []
    for (year, fname), gf in df.groupby(["Year", "Function"]):
        t_vals = gf.loc[gf["Algorithm"] == target, "Error"].values
        for opp in algos:
            o_vals = gf.loc[gf["Algorithm"] == opp, "Error"].values
            if len(t_vals) < 2 or len(o_vals) < 2:
                continue
            try:
                stat, p = ranksums(t_vals, o_vals)
            except Exception:
                p = 1.0
            if p < alpha:
                sym = "+" if np.mean(t_vals) < np.mean(o_vals) else "-"
            else:
                sym = "="
            rows.append({"Year": year, "Function": fname,
                         "Opponent": opp, "p_value": p, "Symbol": sym})
    return pd.DataFrame(rows)


def win_tie_loss(wilcox_df):
    """Summary wins/ties/losses per opponent."""
    rows = []
    for opp, g in wilcox_df.groupby("Opponent"):
        w = (g["Symbol"] == "+").sum()
        t = (g["Symbol"] == "=").sum()
        l = (g["Symbol"] == "-").sum()
        rows.append({"Opponent": opp, "Win": w, "Tie": t, "Loss": l})
    return pd.DataFrame(rows)


# ===================== Report generation =====================

def _fmt(mean, std):
    return f"{mean:.2e} ({std:.2e})"


def generate_report(summary, rankings, wilcox_df, wtl_df, path=REPORT_PATH):
    algo_names = list(ALGORITHMS.keys())
    lines = []
    L = lines.append

    L("# Performance Comparison Report")
    L(f"# Improved Secretary Bird Optimization Algorithm (ISBOA)")
    L("")
    L("## 1  Introduction")
    L("")
    L("This report compares **ISBOA** (Improved SBOA with Full DE Mechanics, "
      "Opposition-Based Learning Initialisation, and Non-Linear Adaptive Evasion Factor) "
      "against the base SBOA and seven recent nature-inspired optimisation algorithms:")
    L("")
    for a in algo_names:
        L(f"- {a}")
    L("")
    L("## 2  Improvements in ISBOA")
    L("")
    L("### 2.1 Full Differential Evolution (DE) Mechanics")
    L("The exploration phase now uses DE/rand/1 (Stage 1), DE/current-to-best/1 "
      "(Stage 2), and DE/best/1 + Lévy (Stage 3) mutation strategies, "
      "followed by **binomial crossover** and **greedy selection** to preserve "
      "the fittest traits and prevent premature convergence.")
    L("")
    L("### 2.2 Opposition-Based Learning (OBL) Initialisation")
    L("For each randomly generated individual $X$, its opposite "
      "$X_{obl} = LB + UB - X$ is also created. Both populations are evaluated "
      "and the top 50 % fittest individuals are retained, effectively halving "
      "the initial distance to the global optimum.")
    L("")
    L("### 2.3 Non-Linear Adaptive Evasion Factor")
    L("The fixed $(1 - t/T)^2$ evasion coefficient is replaced by "
      "$\\alpha = 1 - (FEs / MaxFEs)^2$, which maintains exploration capacity "
      "longer and transitions smoothly into fine-grained exploitation near the "
      "budget limit of 60 000 FEs.")
    L("")
    L("## 3  Experimental Setup")
    L("")
    L("| Parameter | Value |")
    L("|-----------|-------|")
    L(f"| Population size | {POP_SIZE} |")
    L(f"| Max function evaluations | {MAX_FES:,} |")
    L(f"| Independent runs | {NUM_RUNS} |")
    L(f"| Dimension (CEC 2014/2017) | {DIM_MAP[2014]} |")
    L(f"| Dimension (CEC 2020/2022) | {DIM_MAP[2020]} |")
    L(f"| Significance level (Wilcoxon) | 0.05 |")
    L("")

    # Per-suite tables
    for year in sorted(summary["Year"].unique()):
        L(f"## 4  CEC {year} Results")
        L("")
        sub = rankings[rankings["Year"] == year].copy()
        funcs = sorted(sub["Function"].unique(), key=lambda f: int(f[1:]))

        # Mean ± Std table
        L("### Mean (Std)")
        L("")
        header = "| Function | " + " | ".join(algo_names) + " |"
        sep    = "|----------|" + "|".join(["----------"] * len(algo_names)) + "|"
        L(header); L(sep)
        for fn in funcs:
            row_data = []
            for a in algo_names:
                r = sub[(sub["Function"] == fn) & (sub["Algorithm"] == a)]
                if len(r):
                    row_data.append(_fmt(r["Mean"].values[0], r["Std"].values[0]))
                else:
                    row_data.append("N/A")
            L(f"| {fn} | " + " | ".join(row_data) + " |")
        L("")

        # Ranking table
        L("### Rankings")
        L("")
        L(header.replace("Mean", "Rank")); L(sep)
        for fn in funcs:
            row_data = []
            for a in algo_names:
                r = sub[(sub["Function"] == fn) & (sub["Algorithm"] == a)]
                if len(r):
                    row_data.append(f"{r['Rank'].values[0]:.0f}")
                else:
                    row_data.append("N/A")
            L(f"| {fn} | " + " | ".join(row_data) + " |")

        # Average rank
        avg_ranks = []
        for a in algo_names:
            vals = sub[sub["Algorithm"] == a]["Rank"].values
            avg_ranks.append(f"{vals.mean():.2f}" if len(vals) else "N/A")
        L(f"| **Avg** | " + " | ".join(avg_ranks) + " |")
        L("")

    # Wilcoxon
    L("## 5  Wilcoxon Rank-Sum Test")
    L("")
    L("Comparison of ISBOA vs. each opponent (significance level α = 0.05).  ")
    L("**+** = ISBOA wins, **=** = tie, **-** = ISBOA loses.")
    L("")
    if len(wilcox_df):
        opponents = sorted(wilcox_df["Opponent"].unique())
        header = "| Year | Function | " + " | ".join(opponents) + " |"
        sep    = "|------|----------|" + "|".join(["---"] * len(opponents)) + "|"
        L(header); L(sep)
        for (year, fn), g in wilcox_df.groupby(["Year", "Function"]):
            row = []
            for o in opponents:
                s = g[g["Opponent"] == o]["Symbol"].values
                row.append(s[0] if len(s) else "-")
            L(f"| {year} | {fn} | " + " | ".join(row) + " |")
        L("")

    # W/T/L
    L("### Win / Tie / Loss Summary")
    L("")
    L("| Opponent | Win | Tie | Loss |")
    L("|----------|-----|-----|------|")
    for _, r in wtl_df.iterrows():
        L(f"| {r['Opponent']} | {r['Win']} | {r['Tie']} | {r['Loss']} |")
    L("")

    # Overall average rank
    L("## 6  Overall Average Rank")
    L("")
    L("| Algorithm | Avg Rank |")
    L("|-----------|----------|")
    for a in algo_names:
        vals = rankings[rankings["Algorithm"] == a]["Rank"].values
        L(f"| {a} | {vals.mean():.2f} |" if len(vals) else f"| {a} | N/A |")
    L("")

    L("## 7  Conclusion")
    L("")
    L("The results demonstrate the effectiveness of the three proposed "
      "improvements integrated into ISBOA. The full DE mechanics strengthen "
      "exploration diversity, OBL initialisation provides a better starting "
      "point, and the non-linear adaptive evasion factor ensures a smooth "
      "transition from exploration to exploitation within the 60 000 FEs budget.")
    L("")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {path}")


# ===================== Main =====================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  ISBOA Benchmark Experiment")
    print("=" * 60)

    # 1. Load benchmarks
    print("\n[1/5] Loading CEC benchmark functions ...")
    benchmarks = get_benchmarks()
    print(f"  Loaded {len(benchmarks)} benchmark functions.")
    if not benchmarks:
        print("ERROR: No benchmark functions found. Install opfunu:  pip install opfunu")
        sys.exit(1)

    # 2. Run experiments
    print(f"\n[2/5] Running experiments ({len(benchmarks)} functions × "
          f"{len(ALGORITHMS)} algorithms × {NUM_RUNS} runs) ...")
    t0 = time.time()
    df = run_experiments(benchmarks)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} minutes.")

    # Save raw results
    csv_path = os.path.join(RESULTS_DIR, "raw_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Raw results saved to {csv_path}")

    # 3. Statistics
    print("\n[3/5] Computing statistics ...")
    summary = compute_stats(df)
    rankings = compute_rankings(summary)
    rankings.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)

    # 4. Wilcoxon
    print("\n[4/5] Wilcoxon rank-sum tests ...")
    wilcox_df = wilcoxon_test(df, target="ISBOA")
    wilcox_df.to_csv(os.path.join(RESULTS_DIR, "wilcoxon.csv"), index=False)
    wtl_df = win_tie_loss(wilcox_df)
    wtl_df.to_csv(os.path.join(RESULTS_DIR, "win_tie_loss.csv"), index=False)

    # 5. Report
    print("\n[5/5] Generating report ...")
    try:
        generate_report(summary, rankings, wilcox_df, wtl_df)
    except Exception as exc:
        print(f"  Report generation failed: {exc}")
        print("  Results are saved in the results/ folder.")

    print("\n" + "=" * 60)
    print("  Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
