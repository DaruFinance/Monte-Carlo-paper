"""
Gold-standard MC analysis for the paper revision: per-asset MC rank
aggregation and the placebo run that anchors the Section sec:fp-pitfall
discussion.

Inputs (relative to project root, override with MC_PAPER_DATA env var):
  - results/raw_data/<asset>_mc_path_ranks.csv  (Rust mc_path_ranks output)

Outputs:
  - results/tables/gold_mc_<asset>_agg.json (9 instruments)
  - results/tables/gold_mc_placebo.json
"""
import os  # noqa: F401

# ======================================================================
# === Section: Gold MC per-asset aggregation ===
# ======================================================================

"""
Parametrized gold-standard (stationary bar-return bootstrap) MC analysis.
Works across any of the 9 paper instruments.

Usage: python3 gold_mc_analysis.py <asset>
  asset in {btc,doge,bnb,sol,eurusd,usdjpy,eurgbp,xauusd,wti}

Reads:
  results/raw_data/<asset>_mc_metrics.parquet  (INIT=1 fractional units)
  results/raw_data/<asset>_window_pairs.csv                  (actual IS/OOS baseline metrics, INIT=1000)
  results/raw_data/<asset>_corrected_ranks.csv               (actual MDD/Calmar/Ulcer + trade-shuffle ranks)
Writes:
  results/tables/gold_mc_<asset>_summary.md
  results/tables/gold_mc_<asset>_agg.json      (headline numbers for cross-asset table)
"""
from __future__ import annotations
import os
import sys, json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ASSET = sys.argv[1].lower()
ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"
PARQUET = os.environ.get("MC_METRICS_DIR", "") + f"/{ASSET}_mc_metrics.parquet"
ROI_DIVISOR = 100.0
MDD_DIVISOR = 1000.0


def con():
    c = duckdb.connect()
    c.execute("SET memory_limit='6GB'; SET threads=4")
    return c


def compute_ranks() -> pd.DataFrame:
    wp = pd.read_csv(DATA / f"{ASSET}_window_pairs.csv",
                     usecols=["strategy", "window_i", "baseline_is_pf",
                              "baseline_is_sharpe", "baseline_is_roi", "baseline_is_trades"])
    wp["actual_is_roi_frac"] = wp["baseline_is_roi"] / ROI_DIVISOR
    cr = pd.read_csv(DATA / f"{ASSET}_corrected_ranks.csv",
                     usecols=["strategy", "window", "actual_mdd", "actual_calmar",
                              "actual_ulcer", "actual_roi"])
    cr["window_i"] = cr["window"].str.replace("W", "", regex=False).astype(int)
    cr["actual_is_mdd_frac"] = cr["actual_mdd"] / MDD_DIVISOR
    cr["actual_is_ulcer_frac"] = cr["actual_ulcer"] / MDD_DIVISOR
    cr["actual_is_calmar_dimless"] = cr["actual_calmar"]
    actuals = wp.merge(cr[["strategy", "window_i", "actual_is_mdd_frac",
                           "actual_is_ulcer_frac", "actual_is_calmar_dimless"]],
                       on=["strategy", "window_i"], how="inner")
    c = con()
    c.register("actuals", actuals)
    ranks = c.execute(f"""
    WITH boot AS (
      SELECT strategy, window_i, is_pf, is_roi, is_sharpe, is_mdd, is_calmar, is_ulcer
      FROM read_parquet('{PARQUET}')
    ), joined AS (
      SELECT b.strategy, b.window_i,
             a.baseline_is_pf, a.baseline_is_sharpe, a.actual_is_roi_frac,
             a.actual_is_mdd_frac, a.actual_is_ulcer_frac, a.actual_is_calmar_dimless,
             b.is_pf, b.is_roi, b.is_sharpe, b.is_mdd, b.is_calmar, b.is_ulcer
      FROM boot b JOIN actuals a USING (strategy, window_i)
    )
    SELECT strategy, window_i, COUNT(*) AS n_perms,
      AVG(CASE WHEN baseline_is_pf > is_pf THEN 1.0 ELSE 0.0 END)*100 AS rank_pf,
      AVG(CASE WHEN actual_is_roi_frac > is_roi THEN 1.0 ELSE 0.0 END)*100 AS rank_roi,
      AVG(CASE WHEN baseline_is_sharpe > is_sharpe THEN 1.0 ELSE 0.0 END)*100 AS rank_sharpe,
      AVG(CASE WHEN actual_is_mdd_frac < is_mdd THEN 1.0 ELSE 0.0 END)*100 AS rank_mdd,
      AVG(CASE WHEN actual_is_calmar_dimless > is_calmar THEN 1.0 ELSE 0.0 END)*100 AS rank_calmar,
      AVG(CASE WHEN actual_is_ulcer_frac < is_ulcer THEN 1.0 ELSE 0.0 END)*100 AS rank_ulcer
    FROM joined GROUP BY strategy, window_i
    """).df()
    return ranks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[{ASSET}] computing IS ranks ...", flush=True)
    ranks = compute_ranks()
    n_cells = len(ranks); n_strat = ranks["strategy"].nunique()
    print(f"[{ASSET}] {n_cells:,} cells, {n_strat:,} strategies", flush=True)

    # cross-MC correlation + filter lift
    cr = pd.read_csv(DATA / f"{ASSET}_corrected_ranks.csv",
                     usecols=["strategy", "window", "mdd_rank", "calmar_rank",
                              "ulcer_rank", "roi_rank_broken"])
    cr["window_i"] = cr["window"].str.replace("W", "", regex=False).astype(int)
    merged = ranks.merge(cr, on=["strategy", "window_i"], how="inner")
    corrs = {}
    for gold, shuf, lab in [("rank_mdd", "mdd_rank", "MDD"), ("rank_calmar", "calmar_rank", "Calmar"),
                            ("rank_ulcer", "ulcer_rank", "Ulcer"), ("rank_roi", "roi_rank_broken", "ROI*")]:
        corrs[lab] = {"pearson": float(merged[gold].corr(merged[shuf])),
                      "spearman": float(merged[gold].corr(merged[shuf], method="spearman"))}

    wp = pd.read_csv(DATA / f"{ASSET}_window_pairs.csv",
                     usecols=["strategy", "window_i", "baseline_oos_pf"])
    m = merged.merge(wp, on=["strategy", "window_i"], how="inner")
    m = m[m["baseline_oos_pf"].notna()]
    oos = (m["baseline_oos_pf"] > 1.0).astype(float)
    base = oos.mean() * 100
    lifts = {}
    for lab, col in [("gold_mdd", "rank_mdd"), ("gold_calmar", "rank_calmar"),
                     ("gold_ulcer", "rank_ulcer"), ("gold_roi", "rank_roi"),
                     ("ts_mdd", "mdd_rank"), ("ts_calmar", "calmar_rank"),
                     ("ts_ulcer", "ulcer_rank"), ("ts_roi", "roi_rank_broken")]:
        mask = m[col] >= 50
        rate = oos[mask].mean() * 100 if mask.sum() else float("nan")
        lifts[lab] = {"pool": int(mask.sum()), "rate": round(rate, 2), "lift": round(rate - base, 2)}

    rd = {}
    for metric, lab in [("rank_pf", "PF"), ("rank_roi", "ROI"), ("rank_sharpe", "Sharpe"),
                        ("rank_mdd", "MDD"), ("rank_calmar", "Calmar"), ("rank_ulcer", "Ulcer")]:
        v = ranks[metric].dropna()
        rd[lab] = {"mean": round(float(v.mean()), 1), "below50": round(float((v < 50).mean()*100), 1),
                   "above90": round(float((v > 90).mean()*100), 1), "std": round(float(v.std()), 1)}

    agg = {"asset": ASSET, "n_cells": n_cells, "n_strategies": n_strat,
           "baseline_oos_rate": round(base, 2), "rank_dist": rd, "lifts": lifts, "corrs": corrs}
    (OUT / f"gold_mc_{ASSET}_agg.json").write_text(json.dumps(agg, indent=2))
    print(f"[{ASSET}] wrote agg json. gold-MDD p50 lift = {lifts['gold_mdd']['lift']:+.2f}pp; "
          f"mean PF rank = {rd['PF']['mean']}; MDD corr r = {corrs['MDD']['pearson']:+.3f}", flush=True)


if __name__ == "__main__":
    main()

# ======================================================================
# === Section: Gold MC placebo ===
# ======================================================================

"""
Matched-pool placebo + IS-matched control for the gold-standard MC filter.
Uses ONLY existing data: the *_mc_metrics.parquet (one pass for per-cell gold
ranks) + window_pairs.csv / corrected_ranks.csv. No new permutations.

For each asset, for the Gold-MDD p50 filter:
  - gold_lift           : OOS PF>1 lift of cells with gold-MDD rank >= 50
  - matched-pool placebo: 2000 random subsets of the same size -> lift CI
                          (tests: real selection vs pool-size/base-rate noise)
  - IS-MDD control      : same-size subset chosen by best raw IS MDD
  - IS-PF control       : same-size subset chosen by best raw IS PF
                          (tests: does the gold rank add anything over picking
                           good in-sample quality directly?)
Writes results/tables/gold_mc_placebo.json
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"
ASSETS = ["btc", "doge", "bnb", "sol", "eurusd", "usdjpy", "eurgbp", "xauusd", "wti"]
N_PLACEBO = 2000
RNG = np.random.default_rng(42)
ROI_DIVISOR, MDD_DIVISOR = 100.0, 1000.0


def per_cell(asset):
    """Per (strategy,window): gold-MDD rank + actual IS MDD/PF + OOS PF>1."""
    pq = os.environ.get("MC_METRICS_DIR", "") + f"/{asset}_mc_metrics.parquet"
    wp = pd.read_csv(DATA / f"{asset}_window_pairs.csv",
                     usecols=["strategy", "window_i", "baseline_is_pf", "baseline_oos_pf"])
    cr = pd.read_csv(DATA / f"{asset}_corrected_ranks.csv", usecols=["strategy", "window", "actual_mdd"])
    cr["window_i"] = cr["window"].str.replace("W", "", regex=False).astype(int)
    cr["actual_is_mdd_frac"] = cr["actual_mdd"] / MDD_DIVISOR
    actuals = wp.merge(cr[["strategy", "window_i", "actual_is_mdd_frac"]], on=["strategy", "window_i"], how="inner")
    c = duckdb.connect(); c.execute("SET memory_limit='6GB'; SET threads=4")
    c.register("actuals", actuals)
    ranks = c.execute(f"""
      WITH boot AS (SELECT strategy, window_i, is_mdd FROM read_parquet('{pq}')),
      j AS (SELECT b.strategy, b.window_i, a.actual_is_mdd_frac, b.is_mdd
            FROM boot b JOIN actuals a USING (strategy, window_i))
      SELECT strategy, window_i,
             AVG(CASE WHEN actual_is_mdd_frac < is_mdd THEN 1.0 ELSE 0.0 END)*100 AS rank_mdd
      FROM j GROUP BY strategy, window_i
    """).df()
    m = ranks.merge(actuals, on=["strategy", "window_i"], how="inner")
    m = m[m["baseline_oos_pf"].notna()].copy()
    m["oos"] = (m["baseline_oos_pf"] > 1.0).astype(float)
    return m


def analyze(asset):
    m = per_cell(asset)
    base = m["oos"].mean() * 100
    n = len(m)
    oos = m["oos"].to_numpy()
    # gold-MDD p50 filter
    sel = (m["rank_mdd"] >= 50).to_numpy()
    nf = int(sel.sum())
    gold_lift = oos[sel].mean() * 100 - base
    # matched-pool placebo
    pl = np.array([oos[RNG.choice(n, nf, replace=False)].mean() * 100 - base for _ in range(N_PLACEBO)])
    ci_lo, ci_hi = np.percentile(pl, [2.5, 97.5])
    # IS-matched controls (same size nf): best raw IS MDD (lowest), best IS PF (highest)
    idx_mdd = np.argsort(m["actual_is_mdd_frac"].to_numpy())[:nf]          # lowest MDD = best
    is_mdd_lift = oos[idx_mdd].mean() * 100 - base
    idx_pf = np.argsort(-m["baseline_is_pf"].to_numpy())[:nf]              # highest PF = best
    is_pf_lift = oos[idx_pf].mean() * 100 - base
    return {"asset": asset, "n_pool": n, "n_filt": nf, "baseline": round(base, 2),
            "gold_mdd_lift": round(gold_lift, 2),
            "placebo_mean": round(float(pl.mean()), 3),
            "placebo_ci": [round(float(ci_lo), 2), round(float(ci_hi), 2)],
            "gold_outside_ci": bool(gold_lift < ci_lo or gold_lift > ci_hi),
            "is_mdd_lift": round(is_mdd_lift, 2), "is_pf_lift": round(is_pf_lift, 2)}


def main():
    res = []
    for a in ASSETS:
        r = analyze(a); res.append(r)
        print(f"{a:7} pool={r['n_pool']:>6} filt={r['n_filt']:>6} base={r['baseline']:>5} | "
              f"gold={r['gold_mdd_lift']:+.2f} placeboCI=[{r['placebo_ci'][0]:+.2f},{r['placebo_ci'][1]:+.2f}] "
              f"out={r['gold_outside_ci']} | IS-MDD={r['is_mdd_lift']:+.2f} IS-PF={r['is_pf_lift']:+.2f}", flush=True)
    (OUT / "gold_mc_placebo.json").write_text(json.dumps(res, indent=2))
    print("wrote gold_mc_placebo.json")


if __name__ == "__main__":
    main()
