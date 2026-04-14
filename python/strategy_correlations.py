"""
Strategy-level OOS Profit Factor correlation analysis.

For each of the 9 instruments this script computes the pairwise Pearson
correlation of strategies' OOS Profit Factor profiles across WFO windows
and reports within-family vs cross-family correlation statistics
(by indicator family: ATR, EMA, SMA, PPO, RSI, RSI_LEVEL, STOCHK, MACD).

Outputs feed:
  - Figure 1 (fig_strategy_correlations.pdf) — cross-instrument summary panel
  - Table 3  (within-family vs cross-family bar-level PnL correlation)

Inputs (relative to project root):
  - results/raw_data/<asset>_window_pairs.csv  (all 9 assets)

Outputs:
  - results/figures/fig_strategy_correlations.pdf
  - results/tables/strategy_oos_summary.csv
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
RAW = ROOT / "results" / "raw_data"
FIGS = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

ASSETS = {
    'BTC':    {'wp': 'btc_window_pairs.csv',    'class': 'Crypto'},
    'DOGE':   {'wp': 'doge_window_pairs.csv',   'class': 'Crypto'},
    'BNB':    {'wp': 'bnb_window_pairs.csv',    'class': 'Crypto'},
    'SOL':    {'wp': 'sol_window_pairs.csv',    'class': 'Crypto'},
    'EURUSD': {'wp': 'eurusd_window_pairs.csv', 'class': 'Forex'},
    'USDJPY': {'wp': 'usdjpy_window_pairs.csv', 'class': 'Forex'},
    'EURGBP': {'wp': 'eurgbp_window_pairs.csv', 'class': 'Forex'},
    'XAUUSD': {'wp': 'xauusd_window_pairs.csv', 'class': 'Commodity'},
    'WTI':    {'wp': 'wti_window_pairs.csv',    'class': 'Commodity'},
}

# Generic technical-indicator family prefixes. Ordered longest-first so that
# e.g. RSI_LEVEL matches before plain RSI.
FAMILY_PREFIXES = ['ATR', 'EMA', 'SMA', 'PPO', 'RSI_LEVEL',
                   'RSI', 'STOCHK', 'MACD']


def get_family(name):
    for p in sorted(FAMILY_PREFIXES, key=len, reverse=True):
        if name.upper().startswith(p):
            return p
    return 'OTHER'


def process_asset(asset, info):
    wp_path = RAW / info['wp']
    if not wp_path.exists():
        print(f"  {asset}: missing {wp_path}")
        return None

    print(f"\nProcessing {asset} [{info['class']}]")
    wp = pd.read_csv(wp_path)
    wp['family'] = wp['strategy'].apply(get_family)

    n_strats = wp['strategy'].nunique()
    n_windows = wp['window_i'].nunique()
    families = sorted(wp['family'].unique())
    print(f"  {n_strats:,} strategies, {n_windows} windows, "
          f"{len(families)} families")

    # Sample strategies uniformly across families to keep correlation
    # matrices tractable; fixed seed for determinism.
    np.random.seed(42)
    max_per_family = 40
    sampled = []
    for fam in families:
        fam_strats = wp[wp['family'] == fam]['strategy'].unique()
        n_sample = min(max_per_family, len(fam_strats))
        sampled.extend(np.random.choice(fam_strats, n_sample, replace=False))

    wp_sampled = wp[wp['strategy'].isin(sampled)]
    strat_pivot = wp_sampled.pivot_table(
        index='strategy', columns='window_i', values='baseline_oos_pf', aggfunc='first'
    )
    # Drop strategies with insufficient observed windows.
    min_windows = max(3, n_windows // 2)
    strat_pivot = strat_pivot.dropna(thresh=min_windows)

    strat_corr = strat_pivot.T.corr()
    n_sampled = len(strat_corr)

    within, cross = [], []
    strat_fams = [get_family(s) for s in strat_corr.index]
    for i in range(n_sampled):
        for j in range(i + 1, n_sampled):
            val = strat_corr.iloc[i, j]
            if np.isnan(val):
                continue
            if strat_fams[i] == strat_fams[j]:
                within.append(val)
            else:
                cross.append(val)
    within = np.array(within)
    cross = np.array(cross)

    return {
        'asset': asset,
        'n_strategies': n_strats,
        'n_windows': n_windows,
        'n_families': len(families),
        'mean_within_family_corr':
            within.mean() if len(within) else np.nan,
        'median_within_family_corr':
            np.median(within) if len(within) else np.nan,
        'mean_cross_family_corr':
            cross.mean() if len(cross) else np.nan,
        'median_cross_family_corr':
            np.median(cross) if len(cross) else np.nan,
        'pct_within_above_07':
            (within > 0.7).mean() * 100 if len(within) else 0,
        'pct_cross_above_07':
            (cross > 0.7).mean() * 100 if len(cross) else 0,
    }


def main():
    print(f"Output: {FIGS}/")
    stats_rows = []
    for asset, info in ASSETS.items():
        s = process_asset(asset, info)
        if s:
            stats_rows.append(s)

    if not stats_rows:
        print("No data processed.")
        return

    df = pd.DataFrame(stats_rows)
    print("\nSUMMARY: Strategy Correlation Statistics Across All Instruments")
    print(df[['asset', 'n_strategies', 'n_windows', 'n_families',
              'mean_within_family_corr', 'mean_cross_family_corr',
              'pct_within_above_07', 'pct_cross_above_07']].to_string(
        index=False, float_format='%.3f'))

    out_csv = TABLES / 'strategy_oos_summary.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # 3x3 grid: within vs cross family correlation per instrument.
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for idx, s in enumerate(stats_rows):
        ax = axes[idx // 3][idx % 3]
        ax.bar(['Within\nFamily', 'Cross\nFamily'],
               [s['mean_within_family_corr'], s['mean_cross_family_corr']],
               color=['#e74c3c', '#3498db'], alpha=0.8)
        ax.set_ylim(-0.2, 1.0)
        ax.set_ylabel('Mean Correlation')
        ax.set_title(f"{s['asset']} "
                     f"({s['n_strategies']:,} strats, {s['n_windows']}W)")
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        ax.text(0, s['mean_within_family_corr'] + 0.03,
                f"{s['mean_within_family_corr']:.3f}",
                ha='center', fontsize=10, fontweight='bold')
        ax.text(1, s['mean_cross_family_corr'] + 0.03,
                f"{s['mean_cross_family_corr']:.3f}",
                ha='center', fontsize=10, fontweight='bold')

    # Hide unused axes if fewer than 9 instruments processed.
    for idx in range(len(stats_rows), 9):
        axes[idx // 3][idx % 3].set_visible(False)

    plt.suptitle(
        'Within-Family vs Cross-Family Strategy Correlation (OOS PF)',
        fontsize=14, y=1.01)
    plt.tight_layout()
    out_fig = FIGS / 'fig_strategy_correlations.pdf'
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_fig}")


if __name__ == '__main__':
    main()
