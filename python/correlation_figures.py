"""
Strategy correlation figures: per-asset pairwise correlation heatmaps.

For each of the 9 instruments this script builds three figures from the
OOS Profit Factor profile of every strategy across WFO windows:

  1. Family-level correlation heatmap (mean OOS PF per family per window)
  2. Sampled strategy-level correlation heatmap (~300 strategies, ordered
     by family, with family-boundary grid lines)
  3. Within-family vs cross-family correlation distribution histogram

Plus a 3x3 panel summary figure across all 9 assets.

Outputs feed:
  - Figure 1 (fig_strategy_correlations.pdf)  -- per-asset heatmaps and
    within-family vs cross-family correlation distributions are the
    source artefacts behind the cross-instrument summary panel produced
    by `strategy_correlations.py`.

This is the figure-generating counterpart to `strategy_correlations.py`
(which produces the summary statistics CSV for Table 3). Both scripts
consume the same raw window_pairs CSVs.

Inputs (relative to project root):
  - results/raw_data/<asset>_window_pairs.csv  (all 9 assets)

Outputs:
  - results/figures/<asset>_family_corr.pdf          (9 files)
  - results/figures/<asset>_strategy_corr.pdf        (9 files)
  - results/figures/<asset>_corr_distribution.pdf    (9 files)
  - results/figures/all_assets_corr_summary.pdf
  - results/tables/correlation_summary.csv
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
FIGDIR = ROOT / "results" / "figures"
TABDIR = ROOT / "results" / "tables"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    # Crypto
    'BTC':     {'wp': 'btc_window_pairs.csv',    'class': 'Crypto'},
    'DOGE':    {'wp': 'doge_window_pairs.csv',   'class': 'Crypto'},
    'BNB':     {'wp': 'bnb_window_pairs.csv',    'class': 'Crypto'},
    'SOL':     {'wp': 'sol_window_pairs.csv',    'class': 'Crypto'},
    # Forex
    'EURUSD':  {'wp': 'eurusd_window_pairs.csv', 'class': 'Forex'},
    'USDJPY':  {'wp': 'usdjpy_window_pairs.csv', 'class': 'Forex'},
    'EURGBP':  {'wp': 'eurgbp_window_pairs.csv', 'class': 'Forex'},
    # Commodity
    'XAUUSD':  {'wp': 'xauusd_window_pairs.csv', 'class': 'Commodity'},
    'WTI':     {'wp': 'wti_window_pairs.csv',    'class': 'Commodity'},
}


def get_family(name):
    """Extract indicator family from strategy name."""
    prefixes = ['ATR', 'EMA', 'SMA', 'PPO', 'RSI_LEVEL', 'RSI', 'STOCHK', 'MACD']
    for p in sorted(prefixes, key=len, reverse=True):
        if str(name).upper().startswith(p):
            return p
    return 'OTHER'


def process_asset(args):
    """Process one asset: build correlation matrices and generate plots."""
    asset, info = args
    wp_path = RAW / info['wp']
    if not wp_path.exists():
        print(f"  {asset}: MISSING {wp_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Processing {asset} [{info['class']}]")
    print(f"{'='*60}")

    wp = pd.read_csv(wp_path)
    wp['family'] = wp['strategy'].apply(get_family)

    n_strats = wp['strategy'].nunique()
    n_windows = wp['window_i'].nunique()
    families = sorted(wp['family'].unique())
    print(f"  {n_strats:,} strategies, {n_windows} windows, {len(families)} families: {families}")

    # =====================================================
    # 1. FAMILY-LEVEL CORRELATION
    # =====================================================
    fam_pivot = wp.groupby(['family', 'window_i'])['baseline_oos_pf'].mean().unstack(fill_value=np.nan)
    fam_corr = fam_pivot.T.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(fam_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(fam_corr.columns)))
    ax.set_yticks(range(len(fam_corr.index)))
    ax.set_xticklabels(fam_corr.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(fam_corr.index, fontsize=9)
    for i in range(len(fam_corr)):
        for j in range(len(fam_corr)):
            val = fam_corr.iloc[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title(f'{asset}: Family-Level OOS PF Correlation\n(mean OOS PF per family across {n_windows} windows)', fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGDIR / f'{asset.lower()}_family_corr.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {asset.lower()}_family_corr.pdf")

    # =====================================================
    # 2. SAMPLED STRATEGY-LEVEL CORRELATION
    # =====================================================
    np.random.seed(42)
    max_per_family = 40
    sampled_strats = []
    family_labels = []
    for fam in families:
        fam_strats = wp[wp['family'] == fam]['strategy'].unique()
        n_sample = min(max_per_family, len(fam_strats))
        chosen = np.random.choice(fam_strats, n_sample, replace=False)
        sampled_strats.extend(chosen)
        family_labels.extend([fam] * n_sample)

    wp_sampled = wp[wp['strategy'].isin(sampled_strats)]
    strat_pivot = wp_sampled.pivot_table(
        index='strategy', columns='window_i', values='baseline_oos_pf', aggfunc='first'
    )
    strat_order = []
    for fam in families:
        fam_strats_in_pivot = [s for s in sampled_strats if get_family(s) == fam and s in strat_pivot.index]
        strat_order.extend(fam_strats_in_pivot)
    strat_pivot = strat_pivot.reindex(strat_order)

    min_windows = max(3, n_windows // 2)
    strat_pivot = strat_pivot.dropna(thresh=min_windows)

    strat_corr = strat_pivot.T.corr()
    n_sampled = len(strat_corr)

    fam_boundaries = []
    fam_centers = []
    fam_names_for_plot = []
    current_fam = None
    start_idx = 0
    for i, s in enumerate(strat_corr.index):
        f = get_family(s)
        if f != current_fam:
            if current_fam is not None:
                fam_boundaries.append(i)
                fam_centers.append((start_idx + i) / 2)
                fam_names_for_plot.append(current_fam)
            current_fam = f
            start_idx = i
    fam_centers.append((start_idx + len(strat_corr)) / 2)
    fam_names_for_plot.append(current_fam)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(strat_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto',
                   interpolation='nearest')
    for b in fam_boundaries:
        ax.axhline(y=b - 0.5, color='black', linewidth=1.5, alpha=0.7)
        ax.axvline(x=b - 0.5, color='black', linewidth=1.5, alpha=0.7)

    ax.set_xticks(fam_centers)
    ax.set_xticklabels(fam_names_for_plot, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(fam_centers)
    ax.set_yticklabels(fam_names_for_plot, fontsize=8)
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    ax.set_title(f'{asset}: Strategy-Level OOS PF Correlation\n({n_sampled} sampled strategies, ordered by family, {n_windows} windows)',
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGDIR / f'{asset.lower()}_strategy_corr.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {asset.lower()}_strategy_corr.pdf ({n_sampled} strategies)")

    # =====================================================
    # 3. WITHIN vs CROSS-FAMILY CORRELATION DISTRIBUTIONS
    # =====================================================
    within_corrs = []
    cross_corrs = []
    strat_families = [get_family(s) for s in strat_corr.index]

    for i in range(n_sampled):
        for j in range(i + 1, n_sampled):
            val = strat_corr.iloc[i, j]
            if np.isnan(val):
                continue
            if strat_families[i] == strat_families[j]:
                within_corrs.append(val)
            else:
                cross_corrs.append(val)

    within_corrs = np.array(within_corrs)
    cross_corrs = np.array(cross_corrs)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-1, 1, 80)
    ax.hist(within_corrs, bins=bins, alpha=0.6,
            label=f'Within-family (n={len(within_corrs):,}, mean={within_corrs.mean():.3f})',
            color='#e74c3c', density=True)
    ax.hist(cross_corrs, bins=bins, alpha=0.6,
            label=f'Cross-family (n={len(cross_corrs):,}, mean={cross_corrs.mean():.3f})',
            color='#3498db', density=True)
    ax.axvline(within_corrs.mean(), color='#c0392b', linestyle='--', linewidth=2)
    ax.axvline(cross_corrs.mean(), color='#2980b9', linestyle='--', linewidth=2)
    ax.set_xlabel('Pearson Correlation', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{asset}: Within-Family vs Cross-Family Strategy Correlation\n(OOS PF across {n_windows} windows)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGDIR / f'{asset.lower()}_corr_distribution.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {asset.lower()}_corr_distribution.pdf")

    # =====================================================
    # 4. FULL-UNIVERSE CORRELATION STATISTICS (larger sample)
    # =====================================================
    np.random.seed(123)
    max_full = 200
    full_strats = []
    for fam in families:
        fam_strats = wp[wp['family'] == fam]['strategy'].unique()
        n_sample = min(max_full, len(fam_strats))
        chosen = np.random.choice(fam_strats, n_sample, replace=False)
        full_strats.extend(chosen)

    wp_full = wp[wp['strategy'].isin(full_strats)]
    full_pivot = wp_full.pivot_table(
        index='strategy', columns='window_i', values='baseline_oos_pf', aggfunc='first'
    ).dropna(thresh=min_windows)
    full_corr = full_pivot.T.corr()
    full_families = [get_family(s) for s in full_corr.index]

    # Absolute correlation statistics (|r|)
    within_abs = np.abs(within_corrs) if len(within_corrs) > 0 else np.array([])
    cross_abs = np.abs(cross_corrs) if len(cross_corrs) > 0 else np.array([])
    null_abs_r = np.sqrt(2.0 / (np.pi * n_windows)) if n_windows > 0 else np.nan

    stats = {
        'asset': asset,
        'n_strategies': n_strats,
        'n_windows': n_windows,
        'n_families': len(families),
        'mean_within_family_corr': within_corrs.mean() if len(within_corrs) > 0 else np.nan,
        'median_within_family_corr': np.median(within_corrs) if len(within_corrs) > 0 else np.nan,
        'mean_cross_family_corr': cross_corrs.mean() if len(cross_corrs) > 0 else np.nan,
        'median_cross_family_corr': np.median(cross_corrs) if len(cross_corrs) > 0 else np.nan,
        'pct_within_above_07': (within_corrs > 0.7).mean() * 100 if len(within_corrs) > 0 else 0,
        'pct_cross_above_07': (cross_corrs > 0.7).mean() * 100 if len(cross_corrs) > 0 else 0,
        'mean_within_abs_corr': within_abs.mean() if len(within_abs) > 0 else np.nan,
        'mean_cross_abs_corr': cross_abs.mean() if len(cross_abs) > 0 else np.nan,
        'null_expected_abs_r': null_abs_r,
    }
    return stats


if __name__ == '__main__':
    print("Strategy Correlation Figures")
    print(f"Figures: {FIGDIR}/")
    print(f"Tables:  {TABDIR}/\n")

    all_stats = []
    for asset, info in ASSETS.items():
        stats = process_asset((asset, info))
        if stats:
            all_stats.append(stats)

    if all_stats:
        print(f"\n{'='*80}")
        print("SUMMARY: Strategy Correlation Statistics Across All Instruments")
        print(f"{'='*80}\n")

        df = pd.DataFrame(all_stats)
        print(df[['asset', 'n_strategies', 'n_windows', 'n_families',
                   'mean_within_family_corr', 'mean_cross_family_corr',
                   'pct_within_above_07', 'pct_cross_above_07']].to_string(index=False, float_format='%.3f'))

        print(f"\nABSOLUTE CORRELATION |r| STATISTICS:")
        print(df[['asset', 'mean_within_abs_corr', 'mean_cross_abs_corr',
                   'null_expected_abs_r']].to_string(index=False, float_format='%.4f'))

        df.to_csv(TABDIR / 'correlation_summary.csv', index=False)
        print(f"\nSaved {TABDIR / 'correlation_summary.csv'}")

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        for idx, stats in enumerate(all_stats):
            ax = axes[idx // 3][idx % 3]
            asset = stats['asset']
            ax.bar(['Within\nFamily', 'Cross\nFamily'],
                   [stats['mean_within_family_corr'], stats['mean_cross_family_corr']],
                   color=['#e74c3c', '#3498db'], alpha=0.8)
            ax.set_ylim(-0.2, 1.0)
            ax.set_ylabel('Mean Correlation')
            ax.set_title(f"{asset} ({stats['n_strategies']:,} strats, {stats['n_windows']}W)")
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.grid(axis='y', alpha=0.3)
            ax.text(0, stats['mean_within_family_corr'] + 0.03,
                    f"{stats['mean_within_family_corr']:.3f}", ha='center', fontsize=10, fontweight='bold')
            ax.text(1, stats['mean_cross_family_corr'] + 0.03,
                    f"{stats['mean_cross_family_corr']:.3f}", ha='center', fontsize=10, fontweight='bold')

        plt.suptitle('Within-Family vs Cross-Family Strategy Correlation (OOS PF)', fontsize=14, y=1.01)
        plt.tight_layout()
        fig.savefig(FIGDIR / 'all_assets_corr_summary.pdf', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"-> all_assets_corr_summary.pdf")

    print("\nDone.")
