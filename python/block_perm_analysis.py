"""
Block permutation MC analysis.

Merges Rust-generated block-permutation rank CSVs (one per asset) with the
per-window window-pair files and computes, for each block size b in
{1, 2, 3, 5, 10, 20}, the MC-filter "lift":

    lift = P(OOS profitable | rank >= 50) - baseline

Produces the core per-asset statistics feeding:
  - Table 5  (MC filter performance across 9 instruments)
  - Table 11 (MC filter under IS PF gating)
  - Table 12 (per-asset MC filter performance, all_filters_comparison.csv)
  - Table 13 (forex/commodity MC filter performance)
  - Table 19 (block permutation MC test, all 9 instruments)

Inputs  (relative to project root, i.e. ROOT = parents[1]):
  - results/raw_data/block_perm_<asset>.csv      (from the Rust block-perm binary)
  - results/raw_data/<asset>_window_pairs.csv    (from the backtesting pipeline)

Outputs:
  - results/tables/block_perm_per_asset.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
RAW = ROOT / "results" / "raw_data"
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

ASSETS = {
    # Crypto
    'BTC':     {'csv': 'block_perm_btc.csv',    'wp': 'btc_window_pairs.csv',    'class': 'Crypto'},
    'DOGE':    {'csv': 'block_perm_doge.csv',   'wp': 'doge_window_pairs.csv',   'class': 'Crypto'},
    'BNB':     {'csv': 'block_perm_bnb.csv',    'wp': 'bnb_window_pairs.csv',    'class': 'Crypto'},
    'SOL':     {'csv': 'block_perm_sol.csv',    'wp': 'sol_window_pairs.csv',    'class': 'Crypto'},
    # Forex
    'EUR/USD': {'csv': 'block_perm_eurusd.csv', 'wp': 'eurusd_window_pairs.csv', 'class': 'Forex'},
    'USD/JPY': {'csv': 'block_perm_usdjpy.csv', 'wp': 'usdjpy_window_pairs.csv', 'class': 'Forex'},
    'EUR/GBP': {'csv': 'block_perm_eurgbp.csv', 'wp': 'eurgbp_window_pairs.csv', 'class': 'Forex'},
    # Commodity
    'XAU/USD': {'csv': 'block_perm_xauusd.csv', 'wp': 'xauusd_window_pairs.csv', 'class': 'Commodity'},
    'WTI':     {'csv': 'block_perm_wti.csv',    'wp': 'wti_window_pairs.csv',    'class': 'Commodity'},
}

BLOCK_COLS = [
    ('i.i.d. (b=1)',  'iid_rank'),
    ('Block (b=2)',   'block2_rank'),
    ('Block (b=3)',   'block3_rank'),
    ('Block (b=5)',   'block5_rank'),
    ('Block (b=10)',  'block10_rank'),
    ('Block (b=20)',  'block20_rank'),
]


def main():
    per_asset_rows = []
    all_results = []

    for asset, info in ASSETS.items():
        bp_path = RAW / info['csv']
        wp_path = RAW / info['wp']
        if not bp_path.exists() or not wp_path.exists():
            print(f"  {asset}: missing data, skipping")
            continue

        bp = pd.read_csv(bp_path)
        wp = pd.read_csv(wp_path)

        # Convert W-prefixed window labels to integers to match window_pairs
        bp['window_i'] = bp['window'].astype(str).str.replace('W', '').astype(int)

        # Build (strategy, window_i) -> baseline OOS PF lookup
        wp_dict = {
            (row['strategy'], row['window_i']): row['baseline_oos_pf']
            for _, row in wp.iterrows()
        }

        bp['oos_profitable'] = bp.apply(
            lambda r: 1 if wp_dict.get((r['strategy'], r['window_i']), 0) > 1.0 else 0,
            axis=1,
        )
        bp_matched = bp[bp.apply(
            lambda r: (r['strategy'], r['window_i']) in wp_dict, axis=1
        )].copy()

        n_strats = bp_matched['strategy'].nunique()
        n_obs = len(bp_matched)
        baseline = bp_matched['oos_profitable'].mean() * 100

        print(f"\n{'='*70}")
        print(f"{asset} [{info['class']}]: {n_strats:,} strategies, {n_obs:,} obs")
        print(f"Baseline OOS profitability: {baseline:.1f}%")
        print('=' * 70)

        for name, col in BLOCK_COLS:
            if col not in bp_matched.columns:
                continue
            passed = bp_matched[bp_matched[col] >= 50]
            failed = bp_matched[bp_matched[col] < 50]
            n_pass = len(passed)
            pass_rate = n_pass / len(bp_matched) * 100
            pass_oos = passed['oos_profitable'].mean() * 100 if n_pass > 0 else 0
            fail_oos = failed['oos_profitable'].mean() * 100 if len(failed) > 0 else 0
            lift = pass_oos - baseline
            corr = bp_matched[col].corr(bp_matched['oos_profitable'])

            print(f"  {name:18s}: Pass={n_pass:>8,} ({pass_rate:4.1f}%)  "
                  f"Pass-OOS={pass_oos:5.1f}%  Fail-OOS={fail_oos:5.1f}%  "
                  f"Lift={lift:+.2f}pp  r={corr:+.4f}")

            per_asset_rows.append({
                'Asset': asset, 'Class': info['class'], 'Method': name,
                'N_strategies': n_strats, 'N_obs': n_obs,
                'Pass_rate': round(pass_rate, 1),
                'Pass_OOS_pct': round(pass_oos, 1),
                'Fail_OOS_pct': round(fail_oos, 1),
                'Lift_pp': round(lift, 2),
                'Corr_rank_oos': round(corr, 4),
            })

        bp_matched['asset'] = asset
        bp_matched['asset_class'] = info['class']
        all_results.append(bp_matched)

    if not all_results:
        print("No data loaded.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    per_asset_df = pd.DataFrame(per_asset_rows)

    # Paper table: lift by asset for the key block sizes
    print(f"\n{'='*70}\nPAPER TABLE: Lift (pp) by asset and block size\n{'='*70}")
    key_methods = ['i.i.d. (b=1)', 'Block (b=5)', 'Block (b=10)']
    asset_order = ['BTC', 'DOGE', 'BNB', 'SOL', 'EUR/USD', 'USD/JPY',
                   'EUR/GBP', 'XAU/USD', 'WTI']
    available = [a for a in asset_order if a in per_asset_df['Asset'].values]

    pivot_lift = per_asset_df[per_asset_df['Method'].isin(key_methods)].pivot_table(
        index='Method', columns='Asset', values='Lift_pp', aggfunc='first'
    ).reindex(columns=available)
    pivot_lift['Mean'] = pivot_lift.mean(axis=1)
    pivot_lift = pivot_lift.reindex(key_methods)
    print(pivot_lift.to_string(float_format='{:.2f}'.format))

    # Paper table: correlation(rank, OOS profitable)
    print(f"\n{'='*70}\nPAPER TABLE: r(rank, OOS) by asset and block size\n{'='*70}")
    pivot_corr = per_asset_df[per_asset_df['Method'].isin(key_methods)].pivot_table(
        index='Method', columns='Asset', values='Corr_rank_oos', aggfunc='first'
    ).reindex(columns=available)
    pivot_corr['Mean'] = pivot_corr.mean(axis=1)
    pivot_corr = pivot_corr.reindex(key_methods)
    print(pivot_corr.to_string(float_format='{:.4f}'.format))

    # Pooled and per-class block-size sweeps
    print(f"\n{'='*70}\nBLOCK SIZE SWEEP (pooled)\n{'='*70}")
    baseline_all = combined['oos_profitable'].mean() * 100
    print(f"Pooled baseline: {baseline_all:.1f}%  ({len(combined):,} obs)\n")
    for name, col in BLOCK_COLS:
        if col not in combined.columns:
            continue
        passed = combined[combined[col] >= 50]
        lift = passed['oos_profitable'].mean() * 100 - baseline_all
        corr = combined[col].corr(combined['oos_profitable'])
        print(f"  {name:18s}: Lift={lift:+.2f}pp  r={corr:+.4f}")

    for ac in ['Crypto', 'Forex', 'Commodity']:
        subset = combined[combined['asset_class'] == ac]
        if len(subset) == 0:
            continue
        bl = subset['oos_profitable'].mean() * 100
        print(f"\n  {ac} ({len(subset):,} obs, baseline {bl:.1f}%):")
        for name, col in BLOCK_COLS:
            if col not in subset.columns:
                continue
            passed = subset[subset[col] >= 50]
            lift = passed['oos_profitable'].mean() * 100 - bl
            corr = subset[col].corr(subset['oos_profitable'])
            print(f"    {name:18s}: Lift={lift:+.2f}pp  r={corr:+.4f}")

    out = TABLES / 'block_perm_per_asset.csv'
    per_asset_df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
