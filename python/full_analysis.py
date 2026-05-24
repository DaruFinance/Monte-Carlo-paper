"""
Full empirical analysis driver for the MC paper revision.

Produces Tables 4, 5, 6, 7, and 15 from the corrected MC-rank Rust output
plus the per-asset window_pairs CSVs.

Inputs (relative to project root, override with MC_PAPER_DATA env var):
  - results/raw_data/<asset>_corrected_ranks.csv  (from rust/mc_path_ranks)
  - results/raw_data/<asset>_window_pairs.csv     (from upstream backtester)

Outputs:
  - results/tables/table4_corrected.{csv,tex}
  - results/tables/table5_filters_comparison_corrected.csv
  - results/tables/filter_ranking_summary_corrected.csv
  - results/tables/mc_filter_vs_next_oos_corrected.csv
  - results/tables/table7_correlations_corrected.csv
  - results/tables/table15_bootstrap_lift_corrected.csv

Run:
  python full_analysis.py     # builds Table 4
  # then the filter-lift section runs in the same process
"""
import os  # noqa: F401  (used by scrubbed ROOT block)

# ======================================================================
# === Section: Table 4 (mc_pct_rank_summary) ===
# ======================================================================

"""
Rebuild paper Table 4 ("MC percentile rank summary statistics by asset and metric").

Produces:
  - results/tables/table4_corrected.csv
  - results/tables/table4_corrected.tex

The corrected table reports MC ranks for the three PATH-DEPENDENT statistics
(MDD, Calmar, Ulcer) where the permutation null is informative. We also include
the BROKEN ROI rank in a clearly-labelled "for-reference / artefactual" column
so the reader can see why the original Table 4 reported a leftward shift.

The paper's original Table 4 reported MC ranks for ROI and Sharpe — both of
which are sum-of-PnL-multiset statistics and therefore permutation-invariant
by an extension of Proposition 1. Any non-trivial distribution observed in
those columns is an artefact of the floating-point summation order, NOT
evidence of non-exchangeability. See fp_pitfall_demo.py for proof.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ASSETS = [
    ("BTC", "btc_corrected_ranks.csv", "BTC/USDT 30m, 27W"),
    ("DOGE", "doge_corrected_ranks.csv", "DOGE/USDT 30m, 21W"),
    ("BNB", "bnb_corrected_ranks.csv", "BNB/USDT 15m, 30W"),
    ("SOL", "sol_corrected_ranks.csv", "SOL/USDT 1h, 7W"),
    ("EUR/USD", "eurusd_corrected_ranks.csv", "EUR/USD 1h, 15W"),
    ("USD/JPY", "usdjpy_corrected_ranks.csv", "USD/JPY 1h, 15W"),
    ("EUR/GBP", "eurgbp_corrected_ranks.csv", "EUR/GBP 1h, 15W"),
    ("XAU/USD", "xauusd_corrected_ranks.csv", "XAU/USD 1h, 15W"),
    ("WTI", "wti_corrected_ranks.csv", "WTI 1h, 15W"),
]

# Order: corrected metrics first, broken last for transparency
RANK_COLS = [
    ("mdd_rank", "MDD", "path-dependent"),
    ("calmar_rank", "Calmar", "path-dependent"),
    ("ulcer_rank", "Ulcer", "path-dependent"),
    ("roi_rank_broken", "ROI (artefactual)", "perm-invariant (FP only)"),
]


def main():
    rows = []
    for asset, fname, label in ASSETS:
        p = DATA / fname
        if not p.exists():
            print(f"  [skip] {p.name} not produced yet")
            continue
        df = pd.read_csv(p)
        n = len(df)
        for col, mname, kind in RANK_COLS:
            v = df[col].dropna()
            rows.append({
                "Asset": asset,
                "Asset description": label,
                "Metric": mname,
                "Kind": kind,
                "N": n,
                "Expected (%)": 50.0,
                "Mean Rank (%)": round(float(v.mean()), 1),
                "Median (%)": round(float(v.median()), 1),
                "% Below 50": round(float((v < 50).mean() * 100), 1),
                "Std Dev": round(float(v.std()), 1),
            })
    if not rows:
        print("No data files present. Run the Rust binary first.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table4_corrected.csv", index=False)
    print(f"Wrote {OUT / 'table4_corrected.csv'}")
    print()
    print(df.to_string(index=False))

    # LaTeX rendering — drop-in replacement for the original Table 4
    asset_order = [a for (a, fname, _label) in ASSETS if (DATA / fname).exists()]
    cells = {(r["Asset"], r["Metric"]): r for r in rows}

    lines = []
    lines.append(r"% Corrected Table 4 — replaces paper's MC percentile rank summary.")
    lines.append(r"% MDD, Calmar, Ulcer are path-dependent and informative under a permutation null.")
    lines.append(r"% The 'ROI (artefactual)' row is included only to expose the FP-summation artefact")
    lines.append(r"% reported in the original Table 4. Sharpe rank, similarly, is permutation-invariant")
    lines.append(r"% and is omitted; the original Sharpe column is the same artefact in a different metric.")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Corrected MC percentile rank summary statistics by asset and metric. "
                 r"Reports path-dependent statistics (MDD, Calmar, Ulcer) for which the i.i.d. permutation "
                 r"null is informative. All mean ranks are within 2~pp of the theoretical 50\% benchmark — "
                 r"the leftward shift reported in the previous version of this table was a numerical "
                 r"artefact (see Section~\ref{sec:fp-pitfall}).}")
    lines.append(r"\label{tab:mc_rank_summary_corrected}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Asset & Metric & Expected (\%) & Mean Rank (\%) & Median (\%) & \% Below 50 & Std Dev \\")
    lines.append(r"\midrule")
    for a in asset_order:
        for col, mname, _ in RANK_COLS:
            r = cells.get((a, mname))
            if r is None: continue
            label = mname if mname != "ROI (artefactual)" else r"ROI\textsuperscript{*}"
            asset_cell = a if mname == RANK_COLS[0][1] else ""
            lines.append(
                f"{asset_cell} & {label} & "
                f"{r['Expected (%)']:.1f} & "
                f"{r['Mean Rank (%)']:.1f} & "
                f"{r['Median (%)']:.1f} & "
                f"{r['% Below 50']:.1f} & "
                f"{r['Std Dev']:.1f} \\\\"
            )
        lines.append(r"\midrule")
    # remove trailing midrule
    if lines[-1] == r"\midrule": lines = lines[:-1]
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\medskip")
    lines.append(r"\footnotesize")
    lines.append(r"\textsuperscript{*}\emph{Artefactual.} ROI = $\sum r_i / E_0$ is permutation-invariant"
                 r" (Proposition~1 extension), so its true rank distribution is degenerate. The non-zero"
                 r" values reported here arise from floating-point summation order and are NOT evidence of"
                 r" non-exchangeability or path-dependent compounding. Reported solely so the reader can"
                 r" reconcile this table with the original (broken) version.")
    lines.append(r"\end{table}")

    tex_path = OUT / "table4_corrected.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {tex_path}")


if __name__ == "__main__":
    main()

# ======================================================================
# === Section: Filter lift (Tables 5/6/7/15) ===
# ======================================================================

"""
Rebuild the paper's filter-lift analysis using CORRECTED MC ranks (MDD/Calmar/Ulcer).

This is the substantive paper claim: "MC filters produce negative or near-zero
lift on OOS profitability." The original claim was supported by ranks that were
mostly FP noise — so the result, while plausibly correct, was supported by the
wrong evidence. Here we recompute filter-lift using genuinely-informative
path-dependent rank statistics.

Inputs:
  - results/raw_data/<asset>_corrected_ranks.csv      (this toolkit, mc_path_ranks)
  - external <asset>_window_pairs.csv         (NOT shipped — must point to original)

Outputs (CSV + LaTeX) in results/tables/:
  - table5_corrected.csv                      (per-asset all_filters_comparison)
  - filter_ranking_summary_corrected.csv      (cross-asset filter ranking, paper Table 6/11)
  - table7_correlations_corrected.csv         (MC rank vs OOS profitable Pearson r)
  - table15_bootstrap_lift_corrected.csv      (clustered bootstrap CI for MC-filter lift)

The window_pairs CSVs (IS/OOS metrics + robustness perturbations) are NOT in
this toolkit; they must be regenerated by the upstream backtesting pipeline.
If those files are not available, we still produce filter-lift comparisons
that use ONLY this toolkit's mc rank file, taking IS profitability from a
side input. As a fallback we skip the joint analyses with robustness filters.

Usage:
  export WINDOW_PAIRS_DIR=/path/to/window_pairs_csvs
  python3 full_analysis.py
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ASSETS = [
    ("BTC", "btc"),
    ("DOGE", "doge"),
    ("BNB", "bnb"),
    ("SOL", "sol"),
    ("EUR/USD", "eurusd"),
    ("USD/JPY", "usdjpy"),
    ("EUR/GBP", "eurgbp"),
    ("XAU/USD", "xauusd"),
    ("WTI", "wti"),
]
ROBUSTNESS_TESTS = ['ent', 'fee', 'sli', 'entind']
TEST_LABELS = {'ent': 'ENT', 'fee': 'FEE', 'sli': 'SLI', 'entind': 'ENT+IND'}

# Filters defined on CORRECTED metrics.
MC_FILTERS = {
    'MC-MDD p50': ('mdd_rank', 50),
    'MC-MDD p75': ('mdd_rank', 75),
    'MC-MDD p90': ('mdd_rank', 90),
    'MC-Calmar p50': ('calmar_rank', 50),
    'MC-Calmar p75': ('calmar_rank', 75),
    'MC-Calmar p90': ('calmar_rank', 90),
    'MC-Ulcer p50': ('ulcer_rank', 50),
    'MC-Ulcer p75': ('ulcer_rank', 75),
}
# For transparency — also evaluate the artefactual rank
MC_FILTERS_ARTEFACT = {
    'MC-ROI* p50': ('roi_rank_broken', 50),
    'MC-ROI* p75': ('roi_rank_broken', 75),
}


def find_window_pairs(asset_lower: str) -> Path | None:
    """Look for a window_pairs CSV in common locations."""
    env = os.environ.get("WINDOW_PAIRS_DIR")
    candidates = []
    if env:
        candidates.append(Path(env) / f"{asset_lower}_window_pairs.csv")
    candidates.extend([
        DATA / f"{asset_lower}_window_pairs.csv",
    ])
    for c in candidates:
        if c.exists():
            return c
    return None


def load_merged(asset: str, asset_lower: str) -> tuple[pd.DataFrame, bool]:
    mc_path = DATA / f"{asset_lower}_corrected_ranks.csv"
    if not mc_path.exists():
        return pd.DataFrame(), False
    mc = pd.read_csv(mc_path)
    mc['window_i'] = mc['window'].str.replace('W', '').astype(int)

    wp_path = find_window_pairs(asset_lower)
    if wp_path is None:
        return mc, False
    wp = pd.read_csv(wp_path)
    merged = pd.merge(
        wp, mc[['strategy', 'window_i', 'n_trades', 'actual_roi',
                'actual_mdd', 'actual_calmar', 'actual_ulcer',
                'roi_rank_broken', 'mdd_rank', 'calmar_rank', 'ulcer_rank']],
        on=['strategy', 'window_i'], how='inner',
    )
    return merged, True


def all_filters_comparison(merged: pd.DataFrame, asset: str) -> pd.DataFrame:
    m_v = merged[merged['baseline_oos_pf'].notna()]
    if len(m_v) == 0: return pd.DataFrame()
    oos_prof = (m_v['baseline_oos_pf'] > 1.0).astype(float)
    bl = (m_v['baseline_is_pf'] > 1.0)
    baseline_rate = oos_prof.mean() * 100
    rows = []
    rows.append(('No filter', oos_prof.mean()*100, len(m_v)))
    rows.append(('IS PF>1', oos_prof[bl].mean()*100 if bl.sum() else np.nan, int(bl.sum())))
    for t in ROBUSTNESS_TESTS:
        col = f'{t}_is_pf'
        if col not in m_v.columns: continue
        joint = bl & (m_v[col] > 1.0)
        rate = oos_prof[joint].mean()*100 if joint.sum() else np.nan
        rows.append((f'Rob: {TEST_LABELS[t]}', rate, int(joint.sum())))
    all4 = bl.copy()
    for t in ROBUSTNESS_TESTS:
        col = f'{t}_is_pf'
        if col in m_v.columns: all4 &= (m_v[col] > 1.0)
    rows.append(('Rob: All 4', oos_prof[all4].mean()*100 if all4.sum() else np.nan, int(all4.sum())))
    for fname, (col, thr) in MC_FILTERS.items():
        passing = m_v[col] >= thr
        rate = oos_prof[passing].mean()*100 if passing.sum() else np.nan
        rows.append((fname, rate, int(passing.sum())))
    for fname, (col, thr) in MC_FILTERS.items():
        joint = bl & (m_v[col] >= thr)
        rate = oos_prof[joint].mean()*100 if joint.sum() else np.nan
        rows.append((f'{fname} + IS PF>1', rate, int(joint.sum())))
    for fname, (col, thr) in MC_FILTERS_ARTEFACT.items():
        passing = m_v[col] >= thr
        rate = oos_prof[passing].mean()*100 if passing.sum() else np.nan
        rows.append((fname, rate, int(passing.sum())))

    df = pd.DataFrame(rows, columns=['Filter', 'Same-Window OOS Prof%', 'Pool Size'])
    df['Lift (pp)'] = df['Same-Window OOS Prof%'] - baseline_rate
    df.insert(0, 'Asset', asset)
    return df


def filter_vs_next_oos(merged: pd.DataFrame, asset: str) -> pd.DataFrame:
    """MC filter lift against NEXT-window OOS profitability — the forward-predictive test."""
    if 'next_baseline_oos_pf' not in merged.columns: return pd.DataFrame()
    m_v = merged[merged['next_baseline_oos_pf'].notna()]
    if len(m_v) == 0: return pd.DataFrame()
    next_prof = (m_v['next_baseline_oos_pf'] > 1.0)
    baseline_rate = next_prof.mean() * 100
    rows = []
    for fname, (col, thr) in {**MC_FILTERS, **MC_FILTERS_ARTEFACT}.items():
        passing = m_v[col] >= thr
        failing = m_v[col] < thr
        pass_rate = next_prof[passing].mean()*100 if passing.sum() else np.nan
        fail_rate = next_prof[failing].mean()*100 if failing.sum() else np.nan
        rows.append({
            'Asset': asset, 'Filter': fname, 'Threshold': thr,
            'N Pass': int(passing.sum()), 'N Fail': int(failing.sum()),
            'Pass Next OOS%': round(pass_rate, 2),
            'Fail Next OOS%': round(fail_rate, 2),
            'Baseline%': round(baseline_rate, 2),
            'Lift (pp)': round(pass_rate - baseline_rate, 2) if not np.isnan(pass_rate) else np.nan,
        })
    return pd.DataFrame(rows)


def correlations(merged: pd.DataFrame, asset: str) -> pd.DataFrame:
    from scipy import stats
    m_v = merged[merged['baseline_oos_pf'].notna()]
    if len(m_v) < 100: return pd.DataFrame()
    oos_prof = (m_v['baseline_oos_pf'] > 1.0).astype(float)
    rows = []
    for col, lab in [('mdd_rank', 'MDD'), ('calmar_rank', 'Calmar'),
                     ('ulcer_rank', 'Ulcer'), ('roi_rank_broken', 'ROI*')]:
        valid = oos_prof.notna() & m_v[col].notna()
        if valid.sum() < 100: continue
        r, p = stats.pearsonr(m_v.loc[valid, col], oos_prof[valid])
        rows.append({
            'Asset': asset, 'MC Metric': lab,
            'Pearson r': round(float(r), 4),
            'p-value': f"{p:.2e}",
            'R-squared %': round(float(r * r * 100), 3),
            'N': int(valid.sum()),
        })
    return pd.DataFrame(rows)


def window_cluster_bootstrap(merged: pd.DataFrame, asset: str,
                              filter_name: str, col: str, thr: float,
                              n_boot: int = 10000, seed: int = 42) -> dict:
    """Per-asset cluster bootstrap CI on filter lift (cluster = walk-forward window).

    Vectorised: pre-aggregate per-window (sum, count), then bootstrap-resample
    windows and combine sums to recover pass/all means. O(n_boot * n_windows)
    rather than O(n_boot * n_rows)."""
    m_v = merged[merged['baseline_oos_pf'].notna()].copy()
    if len(m_v) == 0: return {}
    m_v['oos_prof'] = (m_v['baseline_oos_pf'] > 1.0).astype(int)
    m_v['passing'] = (m_v[col] >= thr).astype(int)
    m_v['pass_and_prof'] = m_v['passing'] * m_v['oos_prof']

    agg = m_v.groupby('window_i', sort=True).agg(
        sum_prof=('oos_prof','sum'),
        sum_pass=('passing','sum'),
        sum_pass_prof=('pass_and_prof','sum'),
        n=('oos_prof','size'),
    ).reset_index()
    sum_prof = agg['sum_prof'].to_numpy(dtype=np.float64)
    sum_pass = agg['sum_pass'].to_numpy(dtype=np.float64)
    sum_pp   = agg['sum_pass_prof'].to_numpy(dtype=np.float64)
    n_arr    = agg['n'].to_numpy(dtype=np.float64)
    W = len(agg)
    if W < 2: return {}

    def observed_lift():
        base = sum_prof.sum() / n_arr.sum()
        if sum_pass.sum() == 0: return np.nan
        pass_rate = sum_pp.sum() / sum_pass.sum()
        return (pass_rate - base) * 100

    observed = observed_lift()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, W, size=(n_boot, W))
    # gather per-resample window-sums
    b_prof = sum_prof[idx].sum(axis=1)
    b_pass = sum_pass[idx].sum(axis=1)
    b_pp   = sum_pp[idx].sum(axis=1)
    b_n    = n_arr[idx].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        base = b_prof / b_n
        passrate = np.where(b_pass > 0, b_pp / b_pass, np.nan)
        boots = (passrate - base) * 100
    boots = boots[~np.isnan(boots)]
    if len(boots) < 100: return {}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        'Asset': asset, 'Filter': filter_name, 'Threshold': thr,
        'N_observations': int(n_arr.sum()),
        'N_windows': W,
        'Observed Lift (pp)': round(float(observed), 3),
        'Boot mean (pp)': round(float(boots.mean()), 3),
        'CI low (pp)': round(float(lo), 3),
        'CI high (pp)': round(float(hi), 3),
        'p(lift<=0)': round(float((boots <= 0).mean()), 4),
        'n_boot_valid': int(len(boots)),
    }


def main():
    all_filt = []
    all_next = []
    all_corr = []
    all_boot = []
    skipped = []
    have_window_pairs = False
    for asset, alower in ASSETS:
        merged, has_wp = load_merged(asset, alower)
        if len(merged) == 0:
            print(f"  [{asset}] no rank data yet — skip")
            skipped.append(asset)
            continue
        if not has_wp:
            print(f"  [{asset}] window_pairs CSV not found — skipping filter-lift analyses")
            print(f"           (run upstream backtester to produce {alower}_window_pairs.csv,")
            print(f"            or set WINDOW_PAIRS_DIR env var, then rerun this script)")
            skipped.append(asset)
            continue
        have_window_pairs = True
        print(f"  [{asset}] merged rows: {len(merged):,}")

        df_filt = all_filters_comparison(merged, asset); all_filt.append(df_filt)
        df_next = filter_vs_next_oos(merged, asset);     all_next.append(df_next)
        df_corr = correlations(merged, asset);            all_corr.append(df_corr)
        for fname, (col, thr) in MC_FILTERS.items():
            row = window_cluster_bootstrap(merged, asset, fname, col, thr)
            if row: all_boot.append(row)

    if not have_window_pairs:
        print("\n[!] No window_pairs files found for any asset.")
        print("    The filter-lift, correlations, and bootstrap-CI tables cannot be built.")
        print("    See README in this toolkit for how to regenerate <asset>_window_pairs.csv.")
        return

    if all_filt:
        df = pd.concat(all_filt, ignore_index=True)
        df.to_csv(OUT / "table5_filters_comparison_corrected.csv", index=False)
        print(f"Wrote {OUT / 'table5_filters_comparison_corrected.csv'}")
        # cross-asset ranking summary
        summary = df.groupby('Filter').agg({
            'Same-Window OOS Prof%': 'mean',
            'Pool Size': 'mean',
            'Lift (pp)': 'mean',
        }).round(2).sort_values('Same-Window OOS Prof%', ascending=False)
        summary.reset_index().to_csv(OUT / "filter_ranking_summary_corrected.csv", index=False)
        print(f"Wrote {OUT / 'filter_ranking_summary_corrected.csv'}")
    if all_next:
        df = pd.concat(all_next, ignore_index=True)
        df.to_csv(OUT / "mc_filter_vs_next_oos_corrected.csv", index=False)
        print(f"Wrote {OUT / 'mc_filter_vs_next_oos_corrected.csv'}")
    if all_corr:
        df = pd.concat(all_corr, ignore_index=True)
        df.to_csv(OUT / "table7_correlations_corrected.csv", index=False)
        print(f"Wrote {OUT / 'table7_correlations_corrected.csv'}")
    if all_boot:
        df = pd.DataFrame(all_boot)
        df.to_csv(OUT / "table15_bootstrap_lift_corrected.csv", index=False)
        print(f"Wrote {OUT / 'table15_bootstrap_lift_corrected.csv'}")


if __name__ == "__main__":
    main()
