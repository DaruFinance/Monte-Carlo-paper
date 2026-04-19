"""
Reviewer-requested analyses:
  1. Matched-pool-size placebo for the IS-PF>1-gated baseline.
     (Table 10: MC-filtered vs random subsample of the same size.)
  2. Transaction cost sensitivity: MC lift at two cost levels (base + 50%)
     plus a sweep over the IS PF gating threshold.
     (Table 16: MC lift at two transaction cost levels.)

Inputs (relative to project root):
  - results/raw_data/<asset>_mc_perwindow.csv
  - results/raw_data/<asset>_window_pairs.csv  (includes fee_is_pf column)

Outputs:
  - results/tables/matched_pool_placebo.csv
  - results/tables/cost_sensitivity_two_levels.csv
  - results/tables/cost_sensitivity_pf_sweep.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
RAW = ROOT / "results" / "raw_data"
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

ASSETS = ['BTC', 'DOGE', 'BNB', 'SOL']


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
        print(f"  {a}: {len(m)} rows")
    return merged


def analysis_matched_pool_placebo(merged, n_placebo=1000, seed=42):
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Matched-pool-size placebo (IS PF > 1 gated)")
    print("=" * 70)
    rng = np.random.RandomState(seed)
    rows = []
    for a in ASSETS:
        m = merged[a]
        m_v = m[m['next_baseline_oos_pf'].notna()].copy()
        is_profitable = m_v['baseline_is_pf'] > 1.0
        is_pool = m_v[is_profitable]
        next_prof_is = is_pool['next_baseline_oos_pf'] > 1.0
        is_baseline_rate = next_prof_is.mean() * 100

        mc_and_is = is_profitable & (m_v['roi_pct_rank'] >= 50)
        mc_pool = m_v[mc_and_is]
        mc_pool_size = len(mc_pool)
        mc_rate = (mc_pool['next_baseline_oos_pf'] > 1.0).mean() * 100
        mc_lift = mc_rate - is_baseline_rate

        placebo_rates = np.empty(n_placebo)
        for i in range(n_placebo):
            idx = rng.choice(len(is_pool), size=mc_pool_size, replace=False)
            placebo_rates[i] = (
                is_pool.iloc[idx]['next_baseline_oos_pf'] > 1.0
            ).mean() * 100

        placebo_mean = placebo_rates.mean()
        placebo_ci_lo = np.percentile(placebo_rates, 2.5)
        placebo_ci_hi = np.percentile(placebo_rates, 97.5)
        placebo_lift = placebo_mean - is_baseline_rate

        rows.append({
            'Asset': a,
            'IS PF>1 pool': len(is_pool),
            'IS PF>1 OOS%': f'{is_baseline_rate:.2f}',
            'MC+IS pool': mc_pool_size,
            'MC+IS OOS%': f'{mc_rate:.2f}',
            'MC lift (pp)': f'{mc_lift:.2f}',
            'Placebo mean OOS%': f'{placebo_mean:.2f}',
            'Placebo 95% CI': f'[{placebo_ci_lo:.2f}, {placebo_ci_hi:.2f}]',
            'Placebo lift (pp)': f'{placebo_lift:.2f}',
            'MC below placebo CI': mc_rate < placebo_ci_lo,
        })
        print(f"\n{a}:  MC lift={mc_lift:.2f} pp  placebo CI "
              f"[{placebo_ci_lo:.2f}, {placebo_ci_hi:.2f}]")

    df = pd.DataFrame(rows)
    out = TAB / 'matched_pool_placebo.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


def analysis_cost_sensitivity(merged):
    """Assess MC-filter lift stability under higher transaction costs and IS PF threshold sweeps."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Transaction cost sensitivity")
    print("=" * 70)

    # Approach A: use fee_is_pf as an already-computed higher-cost scenario.
    rows_a = []
    for a in ASSETS:
        m = merged[a]
        m_v = m[m['next_baseline_oos_pf'].notna()].copy()
        for cost_label, pf_col in [
            ('Baseline (standard cost)', 'baseline_is_pf'),
            ('Cost +50% (fee-perturbed)', 'fee_is_pf'),
        ]:
            is_profitable = m_v[pf_col] > 1.0
            is_pool = m_v[is_profitable]
            is_rate = (is_pool['next_baseline_oos_pf'] > 1.0).mean() * 100
            mc_and_is = is_profitable & (m_v['roi_pct_rank'] >= 50)
            mc_pool = m_v[mc_and_is]
            mc_rate = (
                mc_pool['next_baseline_oos_pf'] > 1.0
            ).mean() * 100 if len(mc_pool) > 0 else np.nan
            mc_lift = mc_rate - is_rate if not np.isnan(mc_rate) else np.nan
            rows_a.append({
                'Asset': a, 'Cost Level': cost_label,
                'IS PF>1 pool': len(is_pool),
                'IS PF>1 OOS%': f'{is_rate:.2f}',
                'MC+IS pool': len(mc_pool),
                'MC+IS OOS%': f'{mc_rate:.2f}',
                'MC lift vs IS gate (pp)': f'{mc_lift:.2f}',
            })
    df_a = pd.DataFrame(rows_a)
    out_a = TAB / 'cost_sensitivity_two_levels.csv'
    df_a.to_csv(out_a, index=False)
    print(f"Saved {out_a}")

    # Approach B: sweep the IS PF gating threshold as a cost proxy.
    pf_thresholds = [0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
    rows_b = []
    for a in ASSETS:
        m = merged[a]
        m_v = m[m['next_baseline_oos_pf'].notna()].copy()
        baseline_all = (m_v['next_baseline_oos_pf'] > 1.0).mean() * 100
        for thresh in pf_thresholds:
            is_pass = m_v['baseline_is_pf'] > thresh
            is_pool = m_v[is_pass]
            is_rate = (
                is_pool['next_baseline_oos_pf'] > 1.0
            ).mean() * 100 if len(is_pool) > 0 else np.nan
            mc_and_is = is_pass & (m_v['roi_pct_rank'] >= 50)
            mc_pool = m_v[mc_and_is]
            mc_rate = (
                mc_pool['next_baseline_oos_pf'] > 1.0
            ).mean() * 100 if len(mc_pool) > 0 else np.nan
            mc_lift = (
                mc_rate - is_rate
                if not (np.isnan(mc_rate) or np.isnan(is_rate))
                else np.nan
            )
            mc_only = m_v['roi_pct_rank'] >= 50
            mc_only_rate = (
                m_v[mc_only]['next_baseline_oos_pf'] > 1.0
            ).mean() * 100
            rows_b.append({
                'Asset': a, 'IS PF Threshold': thresh,
                'IS Pool': len(is_pool),
                'IS OOS%': round(is_rate, 2) if not np.isnan(is_rate) else np.nan,
                'MC+IS Pool': len(mc_pool),
                'MC+IS OOS%':
                    round(mc_rate, 2) if not np.isnan(mc_rate) else np.nan,
                'MC Lift vs IS gate (pp)':
                    round(mc_lift, 2) if not np.isnan(mc_lift) else np.nan,
                'MC-only Lift (pp)':
                    round(mc_only_rate - baseline_all, 2),
            })
    df_b = pd.DataFrame(rows_b)
    out_b = TAB / 'cost_sensitivity_pf_sweep.csv'
    df_b.to_csv(out_b, index=False)
    print(f"Saved {out_b}")


def main():
    print("Loading data...")
    merged = load_merged()
    analysis_matched_pool_placebo(merged)
    analysis_cost_sensitivity(merged)
    print("\nDone.")


if __name__ == '__main__':
    main()
