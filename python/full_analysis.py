"""
Comprehensive empirical MC-vs-robustness analysis for the 4 crypto assets.

Produces the per-asset summary tables feeding:
  - Table 4  (MC percentile rank summary statistics, mc_pct_rank_summary.csv)
  - Table 5  (MC filter performance, all_filters_comparison.csv)
  - Table 6  (median OOS Sharpe by filter condition)
  - Table 7  (Pearson correlations MC rank vs OOS, mc_correlations.csv)
  - Table 9  (placebo-filter comparison inputs)
  - filter_ranking_summary.csv (cross-asset ranking used by Tables 11/12)

Figure generation has been removed from this script — all paper figures are
built by regenerate_all_figures.py. This script produces tables only.

Inputs (relative to project root):
  - results/raw_data/<asset>_mc_perwindow.csv
  - results/raw_data/<asset>_window_pairs.csv

Outputs:
  - results/tables/mc_pct_rank_summary.csv
  - results/tables/mc_filter_vs_next_oos.csv
  - results/tables/all_filters_comparison.csv
  - results/tables/filter_ranking_summary.csv
  - results/tables/mc_correlations.csv
  - results/tables/is_oos_correlation_by_filter.csv
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
RAW = ROOT / "results" / "raw_data"
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

ASSETS = ['BTC', 'DOGE', 'BNB', 'SOL']
ASSET_LABELS = {
    'BTC': 'BTC 30m 27W', 'DOGE': 'DOGE 30m 21W',
    'BNB': 'BNB 15m 30W', 'SOL': 'SOL 1h 7W',
}
# Technical robustness perturbations present in window_pairs files:
#   ent (entry noise), fee (transaction cost up), sli (slippage up),
#   entind (combined entry + indicator perturbation).
ROBUSTNESS_TESTS = ['ent', 'fee', 'sli', 'entind']
TEST_LABELS = {'ent': 'ENT', 'fee': 'FEE', 'sli': 'SLI', 'entind': 'ENT+IND'}


def savetable(df, name):
    """Save a DataFrame as CSV to the tables output directory."""
    path = TAB / name
    df.to_csv(path, index=False)
    print(f"  -> {path}")


def load_merged():
    """Load and merge MC rank data with window-pair data for all assets."""
    merged = {}
    for a in ASSETS:
        mc = pd.read_csv(RAW / f'{a.lower()}_mc_perwindow.csv')
        wp = pd.read_csv(RAW / f'{a.lower()}_window_pairs.csv')
        mc['window_i'] = mc['window'].str.replace('W', '').astype(int)
        m = pd.merge(
            wp,
            mc[['strategy', 'window_i', 'n_trades', 'actual_roi',
                'actual_sharpe', 'actual_pf',
                'roi_pct_rank', 'sharpe_pct_rank', 'pf_pct_rank']],
            on=['strategy', 'window_i'], how='inner',
        )
        merged[a] = m
        print(f"  {a}: merged {len(m)} rows")
    return merged


def section_mc_rank_summary(merged):
    """Table 4: summary of MC percentile ranks per asset per metric."""
    print("\n=== MC Percentile Rank Summary ===")
    rows = []
    for a in ASSETS:
        m = merged[a]
        for col, metric in [('roi_pct_rank', 'ROI'),
                            ('sharpe_pct_rank', 'Sharpe'),
                            ('pf_pct_rank', 'PF')]:
            vals = m[col].dropna()
            rows.append({
                'Asset': a, 'Metric': metric,
                'Mean Pct Rank': f'{vals.mean():.1f}',
                '% Below 50': f'{(vals < 50).mean() * 100:.1f}',
                'Median': f'{vals.median():.1f}',
                'Std': f'{vals.std():.1f}',
            })
    savetable(pd.DataFrame(rows), 'mc_pct_rank_summary.csv')


def section_mc_filters_vs_next_oos(merged):
    """Per-asset MC filter lift against next-window OOS profitability."""
    print("\n=== MC Filter vs Next-Window OOS Profitability ===")
    mc_filters = {
        'MC-ROI p50': ('roi_pct_rank', 50),
        'MC-ROI p75': ('roi_pct_rank', 75),
        'MC-ROI p90': ('roi_pct_rank', 90),
        'MC-Sharpe p50': ('sharpe_pct_rank', 50),
        'MC-Sharpe p75': ('sharpe_pct_rank', 75),
        'MC-PF p50': ('pf_pct_rank', 50),
        'MC-PF p75': ('pf_pct_rank', 75),
    }

    rows = []
    for a in ASSETS:
        m = merged[a]
        m_v = m[m['next_baseline_oos_pf'].notna()]
        next_prof = m_v['next_baseline_oos_pf'] > 1.0
        baseline_rate = next_prof.mean() * 100

        for fname, (col, thresh) in mc_filters.items():
            passing = m_v[col] >= thresh
            failing = m_v[col] < thresh
            n_pass, n_fail = passing.sum(), failing.sum()
            pass_rate = next_prof[passing].mean() * 100 if n_pass > 0 else np.nan
            fail_rate = next_prof[failing].mean() * 100 if n_fail > 0 else np.nan
            lift = pass_rate - baseline_rate if not np.isnan(pass_rate) else np.nan
            rows.append({
                'Asset': a, 'Filter': fname, 'Threshold': thresh,
                'N Pass': int(n_pass), 'N Fail': int(n_fail),
                'Pass Next OOS%': f'{pass_rate:.1f}',
                'Fail Next OOS%': f'{fail_rate:.1f}',
                'Baseline%': f'{baseline_rate:.1f}',
                'Lift (pp)': f'{lift:.1f}',
            })
    savetable(pd.DataFrame(rows), 'mc_filter_vs_next_oos.csv')


def section_all_filters(merged):
    """Full side-by-side filter comparison (MC, robustness, combinations)."""
    print("\n=== Full Filter Comparison ===")
    mc_filters = {
        'MC-ROI p50': ('roi_pct_rank', 50),
        'MC-ROI p75': ('roi_pct_rank', 75),
        'MC-ROI p90': ('roi_pct_rank', 90),
        'MC-Sharpe p50': ('sharpe_pct_rank', 50),
        'MC-Sharpe p75': ('sharpe_pct_rank', 75),
        'MC-PF p50': ('pf_pct_rank', 50),
        'MC-PF p75': ('pf_pct_rank', 75),
    }

    rows = []
    for a in ASSETS:
        m = merged[a]
        m_v = m[m['baseline_oos_pf'].notna()]
        oos_prof = m_v['baseline_oos_pf'] > 1.0
        bl = m_v['baseline_is_pf'] > 1.0
        baseline_rate = oos_prof.mean() * 100

        filters = {
            'No filter': (oos_prof.mean() * 100, len(m_v)),
            'IS PF>1': (oos_prof[bl].mean() * 100 if bl.sum() > 0 else np.nan,
                        int(bl.sum())),
        }
        for test in ROBUSTNESS_TESTS:
            joint = bl & (m_v[f'{test}_is_pf'] > 1.0)
            rate = oos_prof[joint].mean() * 100 if joint.sum() > 0 else np.nan
            filters[f'Rob: {TEST_LABELS[test]}'] = (rate, int(joint.sum()))
        all4 = bl.copy()
        for t in ROBUSTNESS_TESTS:
            all4 = all4 & (m_v[f'{t}_is_pf'] > 1.0)
        filters['Rob: All 4'] = (
            oos_prof[all4].mean() * 100 if all4.sum() > 0 else np.nan,
            int(all4.sum()))
        for fname, (col, thresh) in mc_filters.items():
            passing = m_v[col] >= thresh
            rate = oos_prof[passing].mean() * 100 if passing.sum() > 0 else np.nan
            filters[fname] = (rate, int(passing.sum()))
        for fname, (col, thresh) in mc_filters.items():
            joint = bl & (m_v[col] >= thresh)
            rate = oos_prof[joint].mean() * 100 if joint.sum() > 0 else np.nan
            filters[f'{fname} + IS PF>1'] = (rate, int(joint.sum()))

        for fname, (rate, n) in filters.items():
            rows.append({
                'Asset': a, 'Filter': fname,
                'Same-Window OOS Prof%': rate, 'Pool Size': n,
                'Lift vs Baseline (pp)':
                    rate - baseline_rate if not np.isnan(rate) else np.nan,
            })

    df = pd.DataFrame(rows)
    savetable(df, 'all_filters_comparison.csv')

    summary = df.groupby('Filter').agg({
        'Same-Window OOS Prof%': 'mean',
        'Pool Size': 'mean',
        'Lift vs Baseline (pp)': 'mean',
    }).round(1).sort_values('Same-Window OOS Prof%', ascending=False)
    savetable(summary.reset_index(), 'filter_ranking_summary.csv')
    print(summary.head(15).to_string())


def section_correlations(merged):
    """Table 7: MC rank vs same-window OOS-profitable Pearson correlations."""
    print("\n=== MC Rank vs OOS Correlations ===")
    rows = []
    for a in ASSETS:
        m = merged[a]
        m_v = m[m['baseline_oos_pf'].notna()]
        oos_prof = (m_v['baseline_oos_pf'] > 1.0).astype(float)
        for mc_col, label in [('roi_pct_rank', 'ROI'),
                              ('sharpe_pct_rank', 'Sharpe'),
                              ('pf_pct_rank', 'PF')]:
            valid = oos_prof.notna() & m_v[mc_col].notna()
            if valid.sum() < 100:
                continue
            r, p = stats.pearsonr(m_v.loc[valid, mc_col], oos_prof[valid])
            rows.append({
                'Asset': a,
                'MC Metric': label,
                'Pearson r': f'{r:.4f}',
                'p-value': f'{p:.2e}',
                'R-squared %': f'{r * r * 100:.2f}',
                'N': int(valid.sum()),
            })
    savetable(pd.DataFrame(rows), 'mc_correlations.csv')


def section_is_oos(merged):
    """IS-vs-OOS correlation conditional on each filter (inline stats)."""
    print("\n=== IS-OOS Correlation by Filter ===")
    rows = []
    for a in ASSETS:
        m = merged[a]
        m_v = m[m['baseline_oos_pf'].notna()]
        for is_col, oos_col, label in [
            ('baseline_is_pf', 'baseline_oos_pf', 'PF'),
            ('baseline_is_sharpe', 'baseline_oos_sharpe', 'Sharpe'),
            ('baseline_is_roi', 'baseline_oos_roi', 'ROI'),
        ]:
            valid = m_v[is_col].notna() & m_v[oos_col].notna()
            if valid.sum() < 100:
                continue
            r, _ = stats.pearsonr(m_v.loc[valid, is_col],
                                  m_v.loc[valid, oos_col])
            rows.append({'Asset': a, 'Metric': label, 'Filter': 'None',
                         'Pearson r': f'{r:.3f}', 'N': int(valid.sum())})

        bl = m_v['baseline_is_pf'] > 1.0
        for test in ROBUSTNESS_TESTS + ['all4']:
            if test == 'all4':
                filt = bl.copy()
                for t in ROBUSTNESS_TESTS:
                    filt = filt & (m_v[f'{t}_is_pf'] > 1.0)
                label = 'All 4 Rob'
            else:
                filt = bl & (m_v[f'{test}_is_pf'] > 1.0)
                label = TEST_LABELS[test]
            if filt.sum() < 100:
                continue
            r, _ = stats.pearsonr(m_v.loc[filt, 'baseline_is_pf'],
                                  m_v.loc[filt, 'baseline_oos_pf'])
            rows.append({'Asset': a, 'Metric': 'PF', 'Filter': label,
                         'Pearson r': f'{r:.3f}', 'N': int(filt.sum())})

    savetable(pd.DataFrame(rows), 'is_oos_correlation_by_filter.csv')


def section_fair_comparison(merged):
    """Fair comparison with block-conditional baselines (fair_comparison.csv)."""
    print("\n=== Fair Comparison (Block-Conditional) ===")
    mc_filters = [
        ('MC-ROI p50', 'roi_pct_rank', 50),
        ('MC-ROI p75', 'roi_pct_rank', 75),
        ('MC-ROI p90', 'roi_pct_rank', 90),
        ('MC-Sharpe p50', 'sharpe_pct_rank', 50),
        ('MC-Sharpe p75', 'sharpe_pct_rank', 75),
    ]
    rows = []
    for a in ASSETS:
        m = merged[a]
        m_v = m[m['baseline_oos_pf'].notna()]
        oos_prof = m_v['baseline_oos_pf'] > 1.0
        bl = m_v['baseline_is_pf'] > 1.0
        baseline_prof = oos_prof.mean() * 100
        gate_prof = oos_prof[bl].mean() * 100 if bl.sum() else np.nan
        rows.append({'Asset': a, 'Block': 'Baseline', 'Filter': 'No filter',
                     'OOS Prof%': round(baseline_prof, 2),
                     'Pool': int(len(m_v)),
                     'Lift vs Block Baseline (pp)': 0.0})
        rows.append({'Asset': a, 'Block': 'Baseline', 'Filter': 'IS PF>1',
                     'OOS Prof%': round(gate_prof, 2),
                     'Pool': int(bl.sum()),
                     'Lift vs Block Baseline (pp)': round(gate_prof - baseline_prof, 2)})
        for fname, col, thresh in mc_filters:
            passing = m_v[col] >= thresh
            rate = oos_prof[passing].mean() * 100 if passing.sum() else np.nan
            rows.append({'Asset': a, 'Block': 'A: No IS PF gate',
                         'Filter': fname,
                         'OOS Prof%': round(rate, 2),
                         'Pool': int(passing.sum()),
                         'Lift vs Block Baseline (pp)': round(rate - baseline_prof, 2)})
        for fname, col, thresh in mc_filters:
            joint = bl & (m_v[col] >= thresh)
            rate = oos_prof[joint].mean() * 100 if joint.sum() else np.nan
            rows.append({'Asset': a, 'Block': 'B: With IS PF>1 gate',
                         'Filter': f'{fname} + IS PF>1',
                         'OOS Prof%': round(rate, 2),
                         'Pool': int(joint.sum()),
                         'Lift vs Block Baseline (pp)': round(rate - gate_prof, 2)})
    savetable(pd.DataFrame(rows), 'fair_comparison.csv')


def section_mc_filter_pass_fail(merged):
    """Table: MC filter pass/fail OOS profitability breakdown."""
    print("\n=== MC Filter Pass/Fail ===")
    mc_filters = [
        ('MC-ROI p50', 'roi_pct_rank', 50),
        ('MC-ROI p75', 'roi_pct_rank', 75),
        ('MC-ROI p90', 'roi_pct_rank', 90),
        ('MC-Sharpe p50', 'sharpe_pct_rank', 50),
        ('MC-Sharpe p75', 'sharpe_pct_rank', 75),
        ('MC-PF p50', 'pf_pct_rank', 50),
        ('MC-PF p75', 'pf_pct_rank', 75),
    ]
    rows = []
    for a in ASSETS:
        m = merged[a]
        m_v = m[m['baseline_oos_pf'].notna()]
        oos_prof = m_v['baseline_oos_pf'] > 1.0
        baseline_prof = oos_prof.mean() * 100
        for fname, col, thresh in mc_filters:
            passing = m_v[col] >= thresh
            failing = ~passing
            pass_rate = oos_prof[passing].mean() * 100 if passing.sum() else np.nan
            fail_rate = oos_prof[failing].mean() * 100 if failing.sum() else np.nan
            rows.append({
                'Asset': a, 'Filter': fname,
                'N Pass': int(passing.sum()),
                'N Fail': int(failing.sum()),
                'Pass OOS%': round(pass_rate, 1),
                'Fail OOS%': round(fail_rate, 1),
                'Lift (pp)': round(pass_rate - baseline_prof, 2),
            })
    savetable(pd.DataFrame(rows), 'mc_filter_pass_fail.csv')


def section_headline(merged):
    """Print headline statistics cited inline in the paper."""
    print("\n" + "=" * 60)
    print("HEADLINE STATS (cited inline in paper)")
    print("=" * 60)
    for a in ASSETS:
        m = merged[a]
        m_v = m[m['baseline_oos_pf'].notna()]
        bl = m_v['baseline_is_pf'] > 1.0
        all4 = bl.copy()
        for t in ROBUSTNESS_TESTS:
            all4 = all4 & (m_v[f'{t}_is_pf'] > 1.0)
        mc50 = m_v['roi_pct_rank'] >= 50
        mc75 = m_v['roi_pct_rank'] >= 75
        oos_prof = m_v['baseline_oos_pf'] > 1.0

        print(f"\n{ASSET_LABELS[a]}:")
        print(f"  Baseline: {oos_prof.mean()*100:.1f}%")
        print(f"  IS PF>1: {oos_prof[bl].mean()*100:.1f}%")
        print(f"  Rob All4: {oos_prof[all4].mean()*100:.1f}%")
        print(f"  MC ROI>=50: {oos_prof[mc50].mean()*100:.1f}%")
        print(f"  MC ROI>=75: {oos_prof[mc75].mean()*100:.1f}%")
        print(f"  Rob All4 + MC>=50: {oos_prof[all4 & mc50].mean()*100:.1f}%")
        print(f"  Mean MC ROI rank: {m['roi_pct_rank'].mean():.1f}")


def main():
    print("Loading data...")
    merged = load_merged()
    section_mc_rank_summary(merged)
    section_mc_filters_vs_next_oos(merged)
    section_all_filters(merged)
    section_correlations(merged)
    section_is_oos(merged)
    section_fair_comparison(merged)
    section_mc_filter_pass_fail(merged)
    section_headline(merged)
    print("\nDone.")


if __name__ == '__main__':
    main()
