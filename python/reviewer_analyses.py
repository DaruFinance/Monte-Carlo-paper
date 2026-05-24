"""
Reviewer-response analyses for the MC paper revision.

Produces Table 16 (cost-sensitivity), placebo, and Sharpe-rank tables across
the 9 instruments.

Inputs (relative to project root, override with MC_PAPER_DATA env var):
  - results/raw_data/<asset>_corrected_ranks.csv
  - results/raw_data/<asset>_window_pairs.csv

Outputs:
  - results/tables/table16_cost_sensitivity_corrected.{csv,tex}
  - results/tables/table16_cost_sensitivity_NOTE.txt
  - results/tables/matched_pool_placebo_corrected.csv
  - results/tables/continuous_sharpe_corrected.csv
"""
import os  # noqa: F401

# ======================================================================
# === Section: Table 16 (cost sensitivity) ===
# ======================================================================

"""
Build corrected Table 16 (cost sensitivity) for the 4 crypto assets.

Schema mirrors the original paper Table 16 but adds the corrected
path-dependent MC filters (MC-MDD p50, MC-Calmar p50, MC-Ulcer p50)
plus the artefactual MC-ROI* p50 for reference.

For each (asset, cost_level), report:
  Pool size   : strategies with <cost>_is_pf > 1
  IS OOS%     : among the pool, fraction with <cost>_oos_pf > 1
  MC+IS OOS%  : same but additionally filtered by MC rank >= threshold
  Lift (pp)   : MC+IS OOS% - IS OOS%

The MC rank uses BASELINE trade pnls (we don't have elevated-cost trade
pnls; the original paper made the same simplification implicitly by
re-running strategies at elevated costs and re-applying the SAME MC
filter built on baseline data).

Cost levels:
  Baseline     : 1x fees, 1x slip   (col prefix: baseline)
  Fee+100%     : 2x fees, 1x slip   (col prefix: fee)
  Slip+200%    : 1x fees, 3x slip   (col prefix: sli)
  ENT+IND      : entry & indicator drift (col prefix: entind)
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"

ASSETS = ["btc", "doge", "bnb", "sol", "eurusd", "usdjpy", "eurgbp", "xauusd", "wti"]

# (label, is_pf_col, oos_pf_col)
COST_LEVELS = [
    ("Baseline (1x fee, 1x slip)", "baseline_is_pf", "baseline_oos_pf"),
    ("Fee+100% (2x fee)",          "fee_is_pf",      "fee_oos_pf"),
    ("Slip+200% (3x slip)",        "sli_is_pf",      "sli_oos_pf"),
    ("Entry+Indicator drift",      "entind_is_pf",   "entind_oos_pf"),
]

# (label, rank_column, threshold)
MC_FILTERS = [
    ("MC-MDD p50",         "mdd_rank",         50.0),
    ("MC-Calmar p50",      "calmar_rank",      50.0),
    ("MC-Ulcer p50",       "ulcer_rank",       50.0),
    ("MC-ROI* p50 (art.)", "roi_rank_broken",  50.0),
]


def load_one(asset: str) -> pd.DataFrame:
    wp = pd.read_csv(DATA / f"{asset}_window_pairs.csv")
    mc = pd.read_csv(DATA / f"{asset}_corrected_ranks.csv")
    mc["window_i"] = mc["window"].str.replace("W", "", regex=False).astype(int)
    merged = wp.merge(
        mc[["strategy", "window_i", "mdd_rank", "calmar_rank",
            "ulcer_rank", "roi_rank_broken"]],
        on=["strategy", "window_i"], how="inner",
    )
    return merged


def build_rows(asset: str, m: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for cost_label, is_col, oos_col in COST_LEVELS:
        if is_col not in m.columns or oos_col not in m.columns:
            continue
        sub = m[m[is_col].notna() & m[oos_col].notna()].copy()
        in_pool = sub[is_col] > 1.0
        pool = sub[in_pool]
        pool_n = len(pool)
        if pool_n == 0:
            continue
        oos_prof = (pool[oos_col] > 1.0).astype(float)
        is_oos_rate = oos_prof.mean() * 100
        for mc_label, mc_col, thr in MC_FILTERS:
            passing = pool[mc_col] >= thr
            n_pass = int(passing.sum())
            mc_oos_rate = oos_prof[passing.values].mean() * 100 if n_pass > 0 else np.nan
            lift = mc_oos_rate - is_oos_rate
            rows.append({
                "Asset": asset.upper(),
                "Cost Level": cost_label,
                "MC Filter": mc_label,
                "IS Pool (n)": pool_n,
                "MC+IS Pool (n)": n_pass,
                "IS OOS PF>1 %": round(is_oos_rate, 2),
                "MC+IS OOS PF>1 %": round(mc_oos_rate, 2) if not np.isnan(mc_oos_rate) else np.nan,
                "Lift (pp)": round(lift, 2) if not np.isnan(lift) else np.nan,
            })
    return rows


def write_tex(df: pd.DataFrame, out_path: Path):
    """Compact LaTeX table — one row per (asset × cost × MC filter)."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Corrected Table~16: MC lift at elevated transaction "
                 r"cost levels, using path-dependent MC filters (MC-MDD/Calmar/Ulcer "
                 r"at the 50th percentile). The artefactual MC-ROI$^*$ filter row "
                 r"is included for reference. Pool = strategy-windows with the "
                 r"perturbation-specific IS PF\,$>$\,1; lift = MC+IS OOS PF\,$>$\,1 "
                 r"\% minus IS-only OOS PF\,$>$\,1 \%. Under the corrected filters, "
                 r"lift is near zero at every cost level on every asset, confirming "
                 r"that the cost-sensitivity conclusion of the original paper does "
                 r"not change qualitatively after the FP-summation correction.}")
    lines.append(r"\label{tab:cost_sensitivity_corrected}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Asset & Cost Level / Filter & IS Pool & MC+IS Pool & "
                 r"IS OOS\% & MC+IS OOS\% & Lift (pp) \\")
    lines.append(r"\midrule")
    cur_asset = None
    cur_cost = None
    for _, r in df.iterrows():
        if r["Asset"] != cur_asset:
            if cur_asset is not None:
                lines.append(r"\midrule")
            cur_asset = r["Asset"]
            cur_cost = None
        asset_cell = r["Asset"] if r["Asset"] != cur_asset else ""
        if r["Cost Level"] != cur_cost:
            cur_cost = r["Cost Level"]
            cost_cell = r["Cost Level"].replace("%", r"\%") + r" / " + r["MC Filter"].replace("%", r"\%")
        else:
            cost_cell = r"\hspace{1em}" + r["MC Filter"].replace("%", r"\%")
        lift = r["Lift (pp)"]
        lift_str = f"${lift:+.2f}$" if pd.notna(lift) else "--"
        is_pct = r["IS OOS PF>1 %"]
        mc_pct = r["MC+IS OOS PF>1 %"]
        is_str = f"{is_pct:.2f}" if pd.notna(is_pct) else "--"
        mc_str = f"{mc_pct:.2f}" if pd.notna(mc_pct) else "--"
        lines.append(
            f"{r['Asset']} & {cost_cell} & "
            f"{r['IS Pool (n)']:,} & {r['MC+IS Pool (n)']:,} & "
            f"{is_str} & {mc_str} & "
            f"{lift_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    all_rows: list[dict] = []
    for asset in ASSETS:
        print(f"=== {asset.upper()} ===")
        m = load_one(asset)
        rows = build_rows(asset, m)
        print(f"  {len(rows)} rows")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "table16_cost_sensitivity_corrected.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    tex_path = OUT / "table16_cost_sensitivity_corrected.tex"
    write_tex(df, tex_path)
    print(f"Wrote {tex_path}")

    # Summary print
    print("\n=== Summary: lifts by (cost level × MC filter), averaged over assets ===")
    pivot = df.pivot_table(
        index=["Cost Level", "MC Filter"], values="Lift (pp)", aggfunc="mean"
    ).round(2)
    print(pivot.to_string())


if __name__ == "__main__":
    main()

# ======================================================================
# === Section: Placebo + Sharpe tables ===
# ======================================================================

"""
Extend the placebo and continuous-Sharpe analyses from §4 to the full
9-instrument universe.

Placebo (Table 10 / `tab:placebo`):
  For each asset, sample a random pass/fail label per strategy-window with
  the same pass rate as the artefactual MC-ROI* p50 filter (100 draws),
  measure OOS-profitability lift, report 95% CI. Compare to the actual
  MC-ROI* lift.

Continuous Sharpe (Table 6 / `tab:filter_ranking_sharpe`):
  Median OOS Sharpe ratio by filter condition, pooled across all 9 assets
  and per-asset.

Outputs:
  - results/tables/matched_pool_placebo_corrected.csv (9 assets)
  - results/tables/continuous_sharpe_corrected.csv    (9 assets pooled)
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"

ASSETS = [
    ("BTC", "btc"), ("DOGE", "doge"), ("BNB", "bnb"), ("SOL", "sol"),
    ("EUR/USD", "eurusd"), ("USD/JPY", "usdjpy"), ("EUR/GBP", "eurgbp"),
    ("XAU/USD", "xauusd"), ("WTI", "wti"),
]

# Filters defined on corrected metrics. (col, threshold)
FILTERS = {
    "No filter":            ("__none__", None),
    "IS PF$>$1":            ("__pf__",   None),
    "MC-MDD p50":           ("mdd_rank",        50),
    "MC-Calmar p50":        ("calmar_rank",     50),
    "MC-Ulcer p50":         ("ulcer_rank",      50),
    "MC-MDD p50 + IS PF$>$1":    ("mdd_rank_pf",    50),
    "MC-Calmar p50 + IS PF$>$1": ("calmar_rank_pf", 50),
    "MC-ROI* p50 (artefactual)": ("roi_rank_broken", 50),
}


def load_merged(short: str) -> pd.DataFrame:
    wp = pd.read_csv(DATA / f"{short}_window_pairs.csv")
    mc = pd.read_csv(DATA / f"{short}_corrected_ranks.csv")
    mc["window_i"] = mc["window"].str.replace("W", "", regex=False).astype(int)
    m = wp.merge(
        mc[["strategy", "window_i", "mdd_rank", "calmar_rank",
            "ulcer_rank", "roi_rank_broken"]],
        on=["strategy", "window_i"], how="inner",
    )
    return m


def placebo_table(n_placebo: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    rows = []
    for label, short in ASSETS:
        m = load_merged(short)
        m = m[m["baseline_oos_pf"].notna() & m["roi_rank_broken"].notna()].copy()
        n = len(m)
        if n == 0:
            continue
        oos_prof = (m["baseline_oos_pf"] > 1.0).astype(float).to_numpy()
        baseline = oos_prof.mean() * 100
        # MC-ROI* p50 filter (artefactual)
        roi_pass = (m["roi_rank_broken"] >= 50).to_numpy()
        n_pass = int(roi_pass.sum())
        roi_rate = oos_prof[roi_pass].mean() * 100 if n_pass else np.nan
        roi_lift = roi_rate - baseline
        # Placebo: random pass/fail with matched pool size
        placebo_rates = np.empty(n_placebo)
        for i in range(n_placebo):
            idx = rng.choice(n, size=n_pass, replace=False)
            placebo_rates[i] = oos_prof[idx].mean() * 100
        plb_mean = placebo_rates.mean()
        plb_lo = float(np.percentile(placebo_rates, 2.5))
        plb_hi = float(np.percentile(placebo_rates, 97.5))
        plb_lift_lo = plb_lo - baseline
        plb_lift_hi = plb_hi - baseline
        ratio = abs(roi_lift) / max(abs(plb_lift_hi), abs(plb_lift_lo), 0.01)
        rows.append({
            "Asset": label,
            "N": n,
            "MC-ROI* lift (pp)": round(roi_lift, 2),
            "Placebo mean (pp)": round(plb_mean - baseline, 3),
            "Placebo 95% CI lo (pp)": round(plb_lift_lo, 2),
            "Placebo 95% CI hi (pp)": round(plb_lift_hi, 2),
            "Ratio": round(ratio, 1),
            "MC outside CI": bool(roi_lift < plb_lift_lo or roi_lift > plb_lift_hi),
        })
        print(f"{label:>8}: ROI* lift={roi_lift:+.2f}  placebo CI [{plb_lift_lo:+.2f}, {plb_lift_hi:+.2f}]  ratio {ratio:.1f}x")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "matched_pool_placebo_corrected.csv", index=False)
    print(f"Wrote {OUT / 'matched_pool_placebo_corrected.csv'}")
    return df


def continuous_sharpe_table():
    """Median OOS Sharpe by filter condition, pooled across 9 assets + per-asset."""
    all_m = []
    for label, short in ASSETS:
        m = load_merged(short)
        m["__asset__"] = label
        all_m.append(m[["__asset__", "baseline_is_pf", "baseline_oos_sharpe",
                       "mdd_rank", "calmar_rank", "ulcer_rank", "roi_rank_broken"]])
    pooled = pd.concat(all_m, ignore_index=True)
    pooled = pooled[pooled["baseline_oos_sharpe"].notna()].copy()
    base_median = pooled["baseline_oos_sharpe"].median()

    def median_for_filter(col: str, thr):
        if col == "__none__":
            return pooled["baseline_oos_sharpe"].median()
        if col == "__pf__":
            mask = pooled["baseline_is_pf"] > 1.0
        elif col == "mdd_rank_pf":
            mask = (pooled["baseline_is_pf"] > 1.0) & (pooled["mdd_rank"] >= thr)
        elif col == "calmar_rank_pf":
            mask = (pooled["baseline_is_pf"] > 1.0) & (pooled["calmar_rank"] >= thr)
        else:
            mask = pooled[col] >= thr
        return pooled.loc[mask, "baseline_oos_sharpe"].median()

    rows = []
    for filter_label, (col, thr) in FILTERS.items():
        med = median_for_filter(col, thr)
        delta = med - base_median if pd.notna(med) else np.nan
        rows.append({
            "Filter": filter_label,
            "Median OOS Sharpe": round(med, 3) if pd.notna(med) else np.nan,
            "Delta": round(delta, 3) if pd.notna(delta) else np.nan,
        })
        print(f"{filter_label:>32}: median {med:+.3f}  Δ {delta:+.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "continuous_sharpe_corrected.csv", index=False)
    print(f"Wrote {OUT / 'continuous_sharpe_corrected.csv'}")
    return df


def main():
    print("=== Placebo (9 instruments) ===")
    placebo_table()
    print()
    print("=== Continuous OOS Sharpe by filter (9-instrument pool) ===")
    continuous_sharpe_table()


if __name__ == "__main__":
    main()
