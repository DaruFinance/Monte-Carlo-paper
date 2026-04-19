#!/usr/bin/env python3
"""
Regenerate the 10 figures that appear in the paper (figures_new/*.pdf).

Figures produced:
  Fig 2.  window_level_mc_vs_oos.pdf          — forex/commodity per-window
  Fig 3.  fig_bootstrap_lift_distributions.pdf — MC-lift bootstrap (crypto)
  Fig 4.  fig_regime_robustness.pdf           — per-window MC ROI rank (crypto)
  Fig 5.  fig_synthetic_mc_ranks.pdf          — synthetic scenarios A/B/C panel 1
  Fig 6.  fig_synthetic_mc_analysis.pdf       — synthetic scenarios A/B/C panel 2
  Fig 7.  fig_synthetic_pipeline_v4.pdf       — full-pipeline synthetic overview
  Fig 8.  fig_synthetic_pipeline_detail.pdf   — pipeline tier details
  Fig 9.  mc_pct_rank_distributions.pdf       — forex/commodity MC rank dists
  Fig 10. mc_roi_vs_next_oos_binned.pdf       — forex/commodity MC vs next OOS

Figure 1 (fig_strategy_correlations.pdf) is produced by
strategy_correlations.py.

Inputs (relative to ROOT = parents[1], or $MC_PAPER_DATA):
  - results/raw_data/<asset>_window_pairs.csv        (9 assets)
  - results/raw_data/<asset>_mc_perwindow.csv        (9 assets)
  - results/tables/synthetic_v4_summaries.csv
  - results/tables/synthetic_v4_null_filters.csv
  - results/tables/synthetic_v4_edge_filters.csv
  - results/tables/synthetic_v4_adversarial_filters.csv
  - results/tables/synthetic_v4_signal_sweep.csv (optional)

Outputs:
  - results/figures/*.pdf   (9 files)

Determinism: all synthetic simulations seeded (np.random.seed(42)).
"""
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr

warnings.filterwarnings('ignore')

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
RAW = ROOT / "results" / "raw_data"
TBL = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

C_NAVY   = '#1d3557'
C_RED    = '#c1292e'
C_GRAY   = '#6c757d'
C_GREEN  = '#27ae60'
C_PURPLE = '#7b2d8e'
C_ORANGE = '#e07a2f'
C_TEAL   = '#2a9d8f'

CRYPTO_ASSETS = ['BTC', 'DOGE', 'BNB', 'SOL']
ASSET_COLORS = {'BTC': '#e6853e', 'DOGE': '#2a9d8f',
                'BNB': '#5e60ce', 'SOL': '#2c5f8a'}

FOREX_ASSETS = ['EURUSD', 'USDJPY', 'EURGBP', 'XAUUSD', 'WTI']
FOREX_LABELS = {
    'EURUSD': 'EUR/USD 1h 15W',
    'USDJPY': 'USD/JPY 1h 15W',
    'EURGBP': 'EUR/GBP 1h 15W',
    'XAUUSD': 'XAU/USD 1h 15W',
    'WTI':    'WTI 1h 15W',
}


def savefig(fig, name):
    """Save figure as 300-dpi PDF to the output directory and close it."""
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {path}')


# Forex/commodity MC CSVs use strategy names that may contain commas — the
# pipeline emits them without quoting. Parse with a window-regex split.
def read_mc_perwindow(filepath):
    """Parse a per-window MC CSV whose strategy names may contain commas."""
    rows = []
    with open(filepath) as f:
        _ = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r',W(\d+),', line)
            if not m:
                continue
            strategy = line[:m.start()]
            parts = line[m.start() + 1:].split(',')
            if len(parts) != 8:
                continue
            rows.append({
                'strategy': strategy,
                'window': parts[0],
                'n_trades': int(parts[1]),
                'actual_roi': float(parts[2]),
                'actual_sharpe': float(parts[3]),
                'actual_pf': float(parts[4]),
                'roi_pct_rank': float(parts[5]),
                'sharpe_pct_rank': float(parts[6]),
                'pf_pct_rank': float(parts[7]),
            })
    return pd.DataFrame(rows)


def load_forex_data():
    """Load and merge MC per-window and window-pair CSVs for all forex assets."""
    merged = {}
    for a in FOREX_ASSETS:
        mc_path = RAW / f'{a.lower()}_mc_perwindow.csv'
        wp_path = RAW / f'{a.lower()}_window_pairs.csv'
        if not mc_path.exists() or not wp_path.exists():
            print(f'  WARNING: missing forex data for {a}')
            continue
        mc_df = read_mc_perwindow(mc_path)
        mc_df['window_i'] = mc_df['window'].str.replace('W', '').astype(int)
        wp_df = pd.read_csv(wp_path)
        merged[a] = pd.merge(
            wp_df,
            mc_df[['strategy', 'window_i', 'n_trades', 'actual_roi',
                   'actual_sharpe', 'actual_pf',
                   'roi_pct_rank', 'sharpe_pct_rank', 'pf_pct_rank']],
            on=['strategy', 'window_i'], how='inner',
        )
    return merged


def load_crypto_merged():
    """Load and merge MC per-window and window-pair CSVs for all crypto assets."""
    merged = {}
    for a in CRYPTO_ASSETS:
        wp_path = RAW / f'{a.lower()}_window_pairs.csv'
        mc_path = RAW / f'{a.lower()}_mc_perwindow.csv'
        if not wp_path.exists() or not mc_path.exists():
            print(f'  WARNING: missing crypto data for {a}')
            continue
        wp = pd.read_csv(wp_path)
        mc = pd.read_csv(mc_path)
        if 'window' in mc.columns and mc['window'].dtype == object:
            mc['window_i'] = mc['window'].str.replace('W', '').astype(int)
        elif 'window_i' not in mc.columns:
            mc['window_i'] = mc['window']
        merged[a] = pd.merge(
            wp,
            mc[['strategy', 'window_i', 'roi_pct_rank',
                'sharpe_pct_rank', 'pf_pct_rank']],
            on=['strategy', 'window_i'], how='inner',
        )
    return merged


# ---------------------------------------------------------------------------
#  Figure 2: window_level_mc_vs_oos.pdf (forex/commodity)
# ---------------------------------------------------------------------------
def figure_2():
    """Produce Fig 2 (window_level_mc_vs_oos.pdf): per-window IS/OOS rates for forex/commodity."""
    print('\n=== Figure 2: Window-Level MC vs OOS (Forex/Commodity) ===')
    merged = load_forex_data()
    if not merged:
        print('  No forex data, skipping.')
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes_flat = axes.flatten()
    for idx, a in enumerate(FOREX_ASSETS):
        if a not in merged:
            continue
        ax = axes_flat[idx]
        m = merged[a]
        windows = sorted(m['window_i'].unique())
        bl_pass, mc_ranks, oos_rates = [], [], []
        for w in windows:
            wdf = m[m['window_i'] == w]
            bl_pass.append((wdf['baseline_is_pf'] > 1.0).mean() * 100)
            mc_ranks.append(wdf['roi_pct_rank'].mean())
            has_next = wdf['next_baseline_oos_pf'].notna()
            if has_next.sum() > 0:
                oos_rates.append(
                    (wdf.loc[has_next, 'next_baseline_oos_pf'] > 1.0).mean() * 100)
            else:
                oos_rates.append(np.nan)

        ax2 = ax.twinx()
        ax.plot(windows, bl_pass, 'o-', color='blue', alpha=0.7,
                label='IS Pass Rate (%)', markersize=3)
        ax.plot(windows, oos_rates, 's-', color='green', alpha=0.7,
                label='Next OOS Prof (%)', markersize=3)
        ax2.plot(windows, mc_ranks, '^-', color='red', alpha=0.7,
                 label='Mean MC ROI Rank', markersize=3)
        ax.set_xlabel('Window')
        ax.set_ylabel('Rate (%)', color='blue')
        ax2.set_ylabel('MC ROI Pct Rank', color='red')
        ax.set_title(FOREX_LABELS[a])
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=6)
        ax.grid(alpha=0.2)

    axes_flat[-1].set_visible(False)
    plt.suptitle(
        'IS Pass Rate, MC Rank, and Next OOS Profitability by Window '
        '(Forex/Commodity)', fontsize=13)
    plt.tight_layout()
    savefig(fig, 'window_level_mc_vs_oos.pdf')


# ---------------------------------------------------------------------------
#  Figure 3: bootstrap lift distributions (crypto, MC-only)
# ---------------------------------------------------------------------------
def figure_3():
    """Produce Fig 3 (fig_bootstrap_lift_distributions.pdf): MC-lift bootstrap for crypto."""
    print('\n=== Figure 3: Bootstrap Lift Distributions (MC Only) ===')
    crypto_merged = load_crypto_merged()
    if not crypto_merged:
        print('  No crypto data, skipping.')
        return

    all_lifts = []
    per_asset = {}
    for a in CRYPTO_ASSETS:
        if a not in crypto_merged:
            continue
        m = crypto_merged[a]
        m_v = m[m['next_baseline_oos_pf'].notna()].copy()
        asset_lifts = []
        for w in sorted(m_v['window_i'].unique()):
            wdf = m_v[m_v['window_i'] == w]
            w_next = (wdf['next_baseline_oos_pf'] > 1.0).astype(float)
            if len(w_next) < 10:
                continue
            w_base = w_next.mean() * 100
            mc_pass = wdf['roi_pct_rank'] >= 50
            if mc_pass.sum() > 5:
                lift = w_next[mc_pass].mean() * 100 - w_base
                asset_lifts.append(lift)
                all_lifts.append(lift)
        per_asset[a] = np.array(asset_lifts)

    all_lifts = np.array(all_lifts)
    if len(all_lifts) < 5:
        print('  Not enough data for bootstrap, skipping.')
        return

    n_boot = 10000
    np.random.seed(42)
    boot_means = np.array([
        np.random.choice(all_lifts, len(all_lifts), replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: window-level lift histogram
    ax = axes[0]
    ax.hist(all_lifts, bins=30, density=True, alpha=0.5, color=C_RED,
            edgecolor='white', linewidth=0.5)
    if all_lifts.std() > 0.1:
        kde = gaussian_kde(all_lifts, bw_method=0.3)
        x = np.linspace(all_lifts.min() - 2, all_lifts.max() + 2, 300)
        ax.plot(x, kde(x), color=C_RED, linewidth=2)
    ax.axvline(0, color=C_GRAY, ls='--', lw=1.5, alpha=0.7, label='Zero lift')
    ax.axvline(all_lifts.mean(), color=C_RED, lw=2,
               label=f'Mean = {all_lifts.mean():.2f} pp')
    ax.set_xlabel('MC-ROI p50 Lift (pp)')
    ax.set_ylabel('Density')
    ax.set_title('(A) Window-Level MC Lift Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    pct_neg = (all_lifts < 0).mean() * 100
    ax.annotate(f'N = {len(all_lifts)} windows\n{pct_neg:.0f}% negative lift',
                xy=(0.97, 0.95), xycoords='axes fraction',
                fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=C_GRAY, alpha=0.9))

    # Panel B: bootstrap distribution
    ax = axes[1]
    ax.hist(boot_means, bins=50, density=True, alpha=0.5, color=C_RED,
            edgecolor='white', linewidth=0.5)
    if boot_means.std() > 0.001:
        kde = gaussian_kde(boot_means, bw_method=0.3)
        x = np.linspace(boot_means.min() - 0.5, boot_means.max() + 0.5, 300)
        ax.plot(x, kde(x), color=C_RED, linewidth=2)
    ax.axvline(0, color=C_GRAY, ls='--', lw=1.5, alpha=0.7)
    ax.axvspan(ci_lo, ci_hi, alpha=0.15, color=C_RED,
               label=f'95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]')
    ax.axvline(boot_means.mean(), color=C_RED, lw=2,
               label=f'Mean = {boot_means.mean():.2f} pp')
    ax.set_xlabel('Bootstrap Mean MC Lift (pp)')
    ax.set_ylabel('Density')
    ax.set_title('(B) Bootstrap Distribution (10K resamples)',
                 fontweight='bold')
    ax.legend(fontsize=8)

    # Panel C: per-asset
    ax = axes[2]
    assets_plot = [a for a in CRYPTO_ASSETS
                   if a in per_asset and len(per_asset[a]) > 2]
    means, cis_lo, cis_hi = [], [], []
    for a in assets_plot:
        vals = per_asset[a]
        boot_a = np.array([
            np.random.choice(vals, len(vals), replace=True).mean()
            for _ in range(n_boot)
        ])
        means.append(vals.mean())
        cis_lo.append(np.percentile(boot_a, 2.5))
        cis_hi.append(np.percentile(boot_a, 97.5))

    y_pos = np.arange(len(assets_plot))
    ax.barh(y_pos, means, color=C_RED, alpha=0.8,
            edgecolor='white', height=0.6)
    for i, (mv, lo, hi) in enumerate(zip(means, cis_lo, cis_hi)):
        ax.plot([lo, hi], [i, i], color='black', lw=2, zorder=5)
        ax.plot([lo, lo], [i - 0.1, i + 0.1], color='black', lw=2, zorder=5)
        ax.plot([hi, hi], [i - 0.1, i + 0.1], color='black', lw=2, zorder=5)
        ax.text(hi + 0.1, i, f'{mv:.2f} [{lo:.2f}, {hi:.2f}]',
                va='center', fontsize=8, color=C_RED, fontweight='bold')

    ax.axvline(0, color=C_GRAY, ls='--', lw=1.5, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(assets_plot, fontsize=11, fontweight='bold')
    ax.set_xlabel('MC-ROI p50 Lift (pp)')
    ax.set_title('(C) Per-Asset MC Lift with 95% CI', fontweight='bold')

    fig.suptitle('MC-ROI p50 Filter Lift: Window-Level Evidence',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    savefig(fig, 'fig_bootstrap_lift_distributions.pdf')


# ---------------------------------------------------------------------------
#  Figure 4: regime robustness (crypto 2x2)
# ---------------------------------------------------------------------------
def figure_4():
    """Produce Fig 4 (fig_regime_robustness.pdf): per-window MC ROI rank across crypto assets."""
    print('\n=== Figure 4: Regime Robustness (Crypto) ===')
    crypto_merged = load_crypto_merged()
    if not crypto_merged:
        print('  No crypto data, skipping.')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for idx, asset in enumerate(CRYPTO_ASSETS):
        if asset not in crypto_merged:
            axes_flat[idx].set_visible(False)
            continue
        ax = axes_flat[idx]
        m = crypto_merged[asset]
        windows = sorted(m['window_i'].unique())
        mean_ranks, std_ranks = [], []
        for w in windows:
            wdf = m[m['window_i'] == w]
            mean_ranks.append(wdf['roi_pct_rank'].mean())
            std_ranks.append(wdf['roi_pct_rank'].std())
        mean_ranks = np.array(mean_ranks)
        std_ranks = np.array(std_ranks)

        ax.plot(windows, mean_ranks, marker='o', color=ASSET_COLORS[asset],
                linewidth=2, markersize=5, label='Mean MC ROI Rank', zorder=4)
        ax.fill_between(windows,
                        np.clip(mean_ranks - std_ranks, 0, 100),
                        np.clip(mean_ranks + std_ranks, 0, 100),
                        color=ASSET_COLORS[asset], alpha=0.15)
        ax.axhline(50, color=C_GRAY, ls='--', lw=1.2, alpha=0.6,
                   label='Random expectation (50%)')
        ax.set_xlabel('WFO Window')
        ax.set_ylabel('Mean MC ROI Rank')
        ax.set_title(f'{asset}', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 80)
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('MC ROI Rank Across WFO Windows: '
                 'High Variance, No Consistent Signal',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    savefig(fig, 'fig_regime_robustness.pdf')


# ---------------------------------------------------------------------------
#  Synthetic helpers used by figures 5 and 6
# ---------------------------------------------------------------------------
def compute_path_metrics(rets):
    """Compute ROI, max drawdown, Calmar, Sharpe, and profit factor from a returns matrix."""
    roi = rets.sum(axis=1)
    eq = np.cumsum(rets, axis=1)
    rm = np.maximum.accumulate(eq, axis=1)
    dd = rm - eq
    mdd = dd.max(axis=1)
    calmar = np.where(mdd > 1e-10, roi / mdd, 0.0)
    mn = rets.mean(axis=1)
    sd = rets.std(axis=1)
    sharpe = np.where(sd > 1e-10, mn / sd * np.sqrt(rets.shape[1]), 0.0)
    pos = np.where(rets > 0, rets, 0).sum(axis=1)
    neg = np.abs(np.where(rets < 0, rets, 0).sum(axis=1))
    pf = np.where(neg > 1e-10, pos / neg, 999.0)
    return {'roi': roi, 'mdd': mdd, 'calmar': calmar, 'sharpe': sharpe, 'pf': pf}


def mc_ranks_vectorized(is_rets, obs_metrics, n_mc=500, batch_size=500):
    """Compute Monte Carlo percentile ranks for ROI, MDD, and Calmar in batches."""
    n = len(is_rets)
    rank_keys = ['roi', 'mdd', 'calmar']
    counts = {k: np.zeros(n) for k in rank_keys}
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = is_rets[start:end]
        bs = end - start
        local = {k: np.zeros(bs) for k in rank_keys}
        for _ in range(n_mc):
            keys = np.random.random(batch.shape)
            idx_arr = np.argsort(keys, axis=1)
            shuffled = np.take_along_axis(batch, idx_arr, axis=1)
            sm = compute_path_metrics(shuffled)
            local['roi'] += (obs_metrics['roi'][start:end] > sm['roi'])
            local['calmar'] += (obs_metrics['calmar'][start:end] > sm['calmar'])
            local['mdd'] += (obs_metrics['mdd'][start:end] < sm['mdd'])
        for k in rank_keys:
            counts[k][start:end] = local[k] / n_mc * 100
    return counts


def rolling_mean_2d(arr, w):
    cs = np.cumsum(arr, axis=1)
    out = np.full_like(arr, np.nan)
    out[:, w - 1:] = cs[:, w - 1:] / w
    out[:, w:] = (cs[:, w:] - cs[:, :-w]) / w
    return out


def filter_stats_synth(df, mask, baseline_oos, label):
    """Return OOS profitability and lift stats for a filtered subset of synthetic data."""
    sub = df[mask]
    if len(sub) == 0:
        return None
    oos = sub['oos_profitable'].mean() * 100
    edge = sub['has_edge'].mean() * 100
    return {'Filter': label, 'Pool': len(sub),
            'OOS Prof%': round(oos, 2),
            'Lift (pp)': round(oos - baseline_oos, 2),
            '% Edge': round(edge, 1)}


def run_synthetic_v3():
    """Deterministically regenerate synthetic scenarios A/B/C used by
    figures 5 and 6. Seed fixed at 42."""
    np.random.seed(42)

    # ---- Scenario A: realistic returns (Student-t, AR(1), GARCH) ----
    N_STRAT_A, N_WIN_A, N_IS, N_OOS = 2000, 3, 200, 200
    T_DF, AR_PHI = 5, 0.05
    GARCH_A, GARCH_B, BASE_VOL_A = 0.10, 0.85, 0.02
    REGIME_BAD_PROB = 0.30
    PERTURB_A, PERTURB_B = 0.001, 0.0008
    N_MC_A = 500

    edge_alphas = np.clip(np.random.exponential(0.002, N_STRAT_A), 0, 0.01)
    noedge_alphas = np.zeros(N_STRAT_A)
    all_alphas = np.concatenate([edge_alphas, noedge_alphas])
    all_edge = np.concatenate([np.ones(N_STRAT_A, dtype=bool),
                               np.zeros(N_STRAT_A, dtype=bool)])
    n_total_a = len(all_alphas)

    sw_alpha = np.repeat(all_alphas, N_WIN_A)
    sw_edge = np.repeat(all_edge, N_WIN_A)
    sw_strat = np.repeat(np.arange(n_total_a), N_WIN_A)
    sw_win = np.tile(np.arange(N_WIN_A), n_total_a)
    n_sw = len(sw_alpha)

    regime_bad = np.random.random(n_sw) < REGIME_BAD_PROB
    effective_alpha = np.where(regime_bad, 0.0, sw_alpha)

    def gen_returns_realistic(n, n_trades, alpha_vec):
        rets = np.zeros((n, n_trades))
        sigma2 = np.full(n, BASE_VOL_A ** 2)
        prev_r = np.zeros(n)
        for t in range(n_trades):
            if t > 0:
                omega = BASE_VOL_A ** 2 * (1 - GARCH_A - GARCH_B)
                sigma2 = omega + GARCH_A * rets[:, t - 1] ** 2 + GARCH_B * sigma2
                sigma2 = np.clip(sigma2, 1e-10, 0.01)
            z = np.random.standard_t(T_DF, size=n)
            z = z / np.sqrt(T_DF / (T_DF - 2))
            rets[:, t] = AR_PHI * prev_r + alpha_vec + np.sqrt(sigma2) * z
            prev_r = rets[:, t]
        return rets

    print('  Generating Scenario A...')
    is_rets_a = gen_returns_realistic(n_sw, N_IS, effective_alpha)
    oos_rets_a = gen_returns_realistic(n_sw, N_OOS, effective_alpha)
    is_m = compute_path_metrics(is_rets_a)
    oos_m = compute_path_metrics(oos_rets_a)
    oos_profitable = oos_m['pf'] > 1.0

    print('  Running MC for Scenario A...')
    mc_ranks = mc_ranks_vectorized(is_rets_a, is_m, n_mc=N_MC_A, batch_size=500)

    rob_pass = ((compute_path_metrics(is_rets_a - PERTURB_A)['pf'] > 1.0) &
                (compute_path_metrics(is_rets_a - PERTURB_B)['pf'] > 1.0))

    df_a = pd.DataFrame({
        'strategy': sw_strat, 'window': sw_win,
        'has_edge': sw_edge, 'true_alpha': sw_alpha, 'regime_bad': regime_bad,
        'is_roi': is_m['roi'], 'is_pf': is_m['pf'], 'is_mdd': is_m['mdd'],
        'is_calmar': is_m['calmar'], 'is_sharpe': is_m['sharpe'],
        'oos_profitable': oos_profitable,
        'mc_roi_rank': mc_ranks['roi'],
        'mc_mdd_rank': mc_ranks['mdd'],
        'mc_calmar_rank': mc_ranks['calmar'],
        'rob_pass': rob_pass, 'is_pf_pass': is_m['pf'] > 1.0,
    })
    baseline_oos_a = df_a['oos_profitable'].mean() * 100

    prev_results = []
    for prev in [0.02, 0.10, 0.50]:
        n_edge = int(N_STRAT_A * prev / (1 - prev)) if prev < 0.5 else N_STRAT_A
        n_edge = min(n_edge, N_STRAT_A)
        edge_idx = df_a[df_a['has_edge']].index[:n_edge * N_WIN_A]
        noedge_idx = df_a[~df_a['has_edge']].index[:N_STRAT_A * N_WIN_A]
        sub = df_a.loc[np.concatenate([edge_idx.values, noedge_idx.values])]
        bl = sub['oos_profitable'].mean() * 100
        row = {'Prevalence': f'{prev*100:.0f}%', 'N': len(sub),
               'Baseline OOS%': round(bl, 1)}
        for fname, mask_col, thresh, better in [
            ('MC-ROI p50', 'mc_roi_rank', 50, 'ge'),
            ('MC-MDD p50', 'mc_mdd_rank', 50, 'ge'),
            ('MC-Calmar p50', 'mc_calmar_rank', 50, 'ge'),
            ('Rob: Combined', 'rob_pass', True, 'eq'),
        ]:
            mm = sub[mask_col] >= thresh if better == 'ge' else sub[mask_col] == thresh
            fsub = sub[mm]
            row[f'{fname} Lift'] = round(fsub['oos_profitable'].mean() * 100 - bl, 1) if len(fsub) else None
        prev_results.append(row)

    a_filters = []
    for fname, mask in [
        ('No filter', pd.Series(True, index=df_a.index)),
        ('IS PF>1', df_a['is_pf_pass']),
        ('Rob: Combined', df_a['rob_pass']),
        ('MC-ROI p50', df_a['mc_roi_rank'] >= 50),
        ('MC-MDD p50', df_a['mc_mdd_rank'] >= 50),
        ('MC-Calmar p50', df_a['mc_calmar_rank'] >= 50),
    ]:
        fs = filter_stats_synth(df_a, mask, baseline_oos_a, fname)
        if fs:
            a_filters.append(fs)
    a_filter_df = pd.DataFrame(a_filters)

    # ---- Scenario B: data-mined MA crossovers on momentum + RW assets ----
    print('  Generating Scenario B...')
    N_ASSETS_MOM, N_ASSETS_RW = 25, 25
    N_BARS, N_IS_B, N_OOS_B = 2000, 1000, 1000
    MOM_PHI, MOM_DRIFT, PRICE_VOL = 0.08, 0.0003, 0.015
    TX_COST, N_MC_B = 0.0005, 300

    FAST_PERIODS = [5, 10, 15, 20, 30]
    SLOW_PERIODS = [50, 100, 200]
    DIRECTIONS = ['long', 'longshort']

    n_assets = N_ASSETS_MOM + N_ASSETS_RW
    bar_returns = np.zeros((n_assets, N_BARS))
    is_momentum = np.zeros(n_assets, dtype=bool)
    is_momentum[:N_ASSETS_MOM] = True

    for i in range(n_assets):
        phi = MOM_PHI if is_momentum[i] else 0.0
        drift = MOM_DRIFT if is_momentum[i] else 0.0
        prev = 0.0
        for t in range(N_BARS):
            z = np.random.standard_t(5) / np.sqrt(5 / 3)
            bar_returns[i, t] = drift + phi * prev + PRICE_VOL * z
            prev = bar_returns[i, t]

    prices = 100 * np.exp(np.cumsum(bar_returns, axis=1))
    strat_rows_b, all_is, all_oos = [], [], []

    for fast in FAST_PERIODS:
        for slow in SLOW_PERIODS:
            if fast >= slow:
                continue
            ma_fast = rolling_mean_2d(prices, fast)
            ma_slow = rolling_mean_2d(prices, slow)
            for direction in DIRECTIONS:
                raw_signal = np.where(ma_fast > ma_slow, 1.0, -1.0)
                if direction == 'long':
                    raw_signal = np.where(raw_signal > 0, 1.0, 0.0)
                valid = ~(np.isnan(ma_fast) | np.isnan(ma_slow))
                raw_signal[~valid] = 0.0
                trade_rets = raw_signal[:, :-1] * bar_returns[:, 1:]
                sig_changes = np.abs(np.diff(raw_signal, axis=1))
                cost_adj = sig_changes[:, :-1] * TX_COST
                mlen = min(trade_rets.shape[1], cost_adj.shape[1])
                trade_robust = trade_rets[:, :mlen] - cost_adj[:, :mlen]
                is_tr = trade_rets[:, :N_IS_B]
                oos_tr = trade_rets[:, N_IS_B:N_IS_B + N_OOS_B]
                is_tr_rob = trade_robust[:, :N_IS_B]
                for ai in range(n_assets):
                    if np.abs(is_tr[ai]).sum() < 1e-10:
                        continue
                    all_is.append(is_tr[ai])
                    all_oos.append(oos_tr[ai])
                    strat_rows_b.append({
                        'asset': ai,
                        'has_momentum': is_momentum[ai],
                        'fast': fast, 'slow': slow, 'direction': direction,
                        'rob_pf': compute_path_metrics(
                            is_tr_rob[ai:ai + 1])['pf'][0],
                    })

    is_rets_b = np.array(all_is)
    oos_rets_b = np.array(all_oos)
    is_m_b = compute_path_metrics(is_rets_b)
    oos_m_b = compute_path_metrics(oos_rets_b)

    print('  Running MC for Scenario B...')
    mc_ranks_b = mc_ranks_vectorized(is_rets_b, is_m_b, n_mc=N_MC_B, batch_size=500)

    for i, row in enumerate(strat_rows_b):
        row['is_roi'] = is_m_b['roi'][i]
        row['is_pf'] = is_m_b['pf'][i]
        row['is_mdd'] = is_m_b['mdd'][i]
        row['is_calmar'] = is_m_b['calmar'][i]
        row['oos_pf'] = oos_m_b['pf'][i]
        row['oos_profitable'] = oos_m_b['pf'][i] > 1.0
        row['mc_roi_rank'] = mc_ranks_b['roi'][i]
        row['mc_mdd_rank'] = mc_ranks_b['mdd'][i]
        row['mc_calmar_rank'] = mc_ranks_b['calmar'][i]
        row['rob_pass'] = row['rob_pf'] > 1.0
        row['has_edge'] = row['has_momentum']

    df_b = pd.DataFrame(strat_rows_b)
    bl_b = df_b['oos_profitable'].mean() * 100

    b_filters = []
    for fname, mask in [
        ('No filter', pd.Series(True, index=df_b.index)),
        ('IS PF>1', df_b['is_pf'] > 1),
        ('Rob: TX cost survives', df_b['rob_pass']),
        ('MC-ROI p50', df_b['mc_roi_rank'] >= 50),
        ('MC-MDD p50', df_b['mc_mdd_rank'] >= 50),
        ('MC-Calmar p50', df_b['mc_calmar_rank'] >= 50),
    ]:
        fs = filter_stats_synth(df_b, mask, bl_b, fname)
        if fs:
            b_filters.append(fs)
    b_filter_df = pd.DataFrame(b_filters)

    # ---- Scenario C: factor-model portfolios ----
    print('  Generating Scenario C...')
    N_STRAT_C, N_TRADES_C = 500, 200
    N_PORT, PORT_SIZE = 500, 10
    FACTOR_VOL, IDIO_VOL, N_MC_C = 0.012, 0.015, 300

    factor = np.random.normal(0, FACTOR_VOL, N_TRADES_C)
    betas = np.random.uniform(0.5, 1.5, N_STRAT_C)
    alphas_c = np.zeros(N_STRAT_C)
    alphas_c[:N_STRAT_C // 2] = np.clip(
        np.random.exponential(0.002, N_STRAT_C // 2), 0, 0.01)
    has_edge_c = np.zeros(N_STRAT_C, dtype=bool)
    has_edge_c[:N_STRAT_C // 2] = True

    is_rets_c = (betas[:, None] * factor[None, :] +
                 alphas_c[:, None] +
                 np.random.normal(0, IDIO_VOL, (N_STRAT_C, N_TRADES_C)))
    factor_oos = np.random.normal(0, FACTOR_VOL, N_TRADES_C)
    oos_rets_c = (betas[:, None] * factor_oos[None, :] +
                  alphas_c[:, None] +
                  np.random.normal(0, IDIO_VOL, (N_STRAT_C, N_TRADES_C)))
    is_m_c = compute_path_metrics(is_rets_c)

    top24 = np.argsort(-is_m_c['pf'])[:24]
    port_results = []
    for p_id in range(N_PORT):
        sel = np.random.choice(top24, PORT_SIZE, replace=False)
        p_is_roi = is_rets_c[sel].mean(axis=0).sum()
        p_oos_roi = oos_rets_c[sel].mean(axis=0).sum()
        null_rois = np.zeros(N_MC_C)
        for mc_i in range(N_MC_C):
            shuffled = np.zeros(N_TRADES_C)
            for s in sel:
                perm = np.random.permutation(N_TRADES_C)
                shuffled += is_rets_c[s, perm]
            shuffled /= PORT_SIZE
            null_rois[mc_i] = shuffled.sum()
        port_results.append({
            'portfolio_id': p_id,
            'is_roi': p_is_roi, 'oos_roi': p_oos_roi,
            'oos_profitable': p_oos_roi > 0,
            'mc_rank': (p_is_roi > null_rois).mean() * 100,
            'pct_edge': has_edge_c[sel].mean() * 100,
            'mean_corr':
                np.corrcoef(is_rets_c[sel])[
                    np.triu_indices(PORT_SIZE, k=1)].mean(),
        })
    port_df = pd.DataFrame(port_results)

    return {
        'df_a': df_a, 'baseline_oos_a': baseline_oos_a,
        'a_filter_df': a_filter_df, 'prev_results': prev_results,
        'df_b': df_b, 'bl_b': bl_b, 'b_filter_df': b_filter_df,
        'port_df': port_df,
    }


def figure_5(syn):
    """Produce Fig 5 (fig_synthetic_mc_ranks.pdf): synthetic scenarios A/B/C rank distributions."""
    print('\n=== Figure 5: Synthetic MC Ranks ===')
    df_a = syn['df_a']
    baseline_oos = syn['baseline_oos_a']
    a_filter_df = syn['a_filter_df']
    prev_results = syn['prev_results']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    for edge, color, label in [(True, C_TEAL, 'WITH edge'),
                                (False, C_RED, 'WITHOUT edge')]:
        vals = df_a[df_a['has_edge'] == edge]['mc_mdd_rank'].values
        ax.hist(vals, bins=50, density=True, alpha=0.3, color=color,
                edgecolor='white', linewidth=0.3)
        if vals.std() > 0.5:
            kde = gaussian_kde(vals, bw_method=0.1)
            x = np.linspace(0, 100, 300)
            ax.plot(x, kde(x), color=color, linewidth=2,
                    label=f'{label} (mean={vals.mean():.1f}%)')
    ax.axvline(50, color=C_GRAY, ls='--', lw=1.5, alpha=0.7)
    ax.set_xlabel('MC Max-Drawdown Percentile Rank')
    ax.set_ylabel('Density')
    ax.set_title('Path-Dependent MC: Max Drawdown\n'
                 '(heavy tails, AR(1), GARCH, regime shifts)',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    r_mdd, _ = pearsonr(df_a['mc_mdd_rank'], df_a['oos_profitable'].astype(float))
    ax.annotate(f'r vs OOS = {r_mdd:.4f}\nR\u00b2 = {r_mdd**2*100:.3f}%',
                xy=(0.03, 0.95), xycoords='axes fraction', fontsize=8,
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=C_GRAY, alpha=0.9))

    ax = axes[0, 1]
    for edge, color, label in [(True, C_TEAL, 'WITH edge'),
                                (False, C_RED, 'WITHOUT edge')]:
        vals = df_a[df_a['has_edge'] == edge]['mc_calmar_rank'].values
        ax.hist(vals, bins=50, density=True, alpha=0.3, color=color,
                edgecolor='white', linewidth=0.3)
        if vals.std() > 0.5:
            kde = gaussian_kde(vals, bw_method=0.1)
            x = np.linspace(0, 100, 300)
            ax.plot(x, kde(x), color=color, linewidth=2,
                    label=f'{label} (mean={vals.mean():.1f}%)')
    ax.axvline(50, color=C_GRAY, ls='--', lw=1.5, alpha=0.7)
    ax.set_xlabel('MC Calmar Percentile Rank')
    ax.set_ylabel('Density')
    ax.set_title('Path-Dependent MC: Calmar Ratio',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)

    # (c) Prevalence sweep
    ax = axes[1, 0]
    prevs = [2, 10, 50]
    mdd_l = [r.get('MC-MDD p50 Lift', 0) or 0 for r in prev_results]
    cal_l = [r.get('MC-Calmar p50 Lift', 0) or 0 for r in prev_results]
    rob_l = [r.get('Rob: Combined Lift', 0) or 0 for r in prev_results]
    x_pos = np.arange(len(prevs))
    w = 0.25
    ax.bar(x_pos - w, mdd_l, w, color=C_RED, alpha=0.8, label='MC-MDD p50')
    ax.bar(x_pos, cal_l, w, color=C_ORANGE, alpha=0.8, label='MC-Calmar p50')
    ax.bar(x_pos + w, rob_l, w, color=C_NAVY, alpha=0.8, label='Rob: Combined')
    ax.axhline(0, color=C_GRAY, lw=1, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{p}%' for p in prevs])
    ax.set_xlabel('True Edge Prevalence')
    ax.set_ylabel('Lift vs Baseline (pp)')
    ax.set_title('Filter Lift Across Edge Prevalences',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')

    # (d) Filter bar chart
    ax = axes[1, 1]
    show = a_filter_df[a_filter_df['Filter'] != 'No filter'].copy()
    show = show.sort_values('OOS Prof%', ascending=True).reset_index(drop=True)

    def fcolor(f):
        if 'MC' in f:
            return C_RED
        if 'Rob' in f:
            return C_NAVY
        return C_GRAY

    colors = [fcolor(f) for f in show['Filter']]
    ax.barh(range(len(show)), show['OOS Prof%'], color=colors,
            edgecolor='white', height=0.65, alpha=0.9)
    ax.axvline(baseline_oos, color=C_GRAY, ls='--', lw=1.5, alpha=0.6,
               label=f'Baseline ({baseline_oos:.1f}%)')
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(
        [f"{r['Filter']}  (n={r['Pool']/1e3:.0f}K)"
         for _, r in show.iterrows()], fontsize=9)
    for i, (_, r) in enumerate(show.iterrows()):
        lift = r['Lift (pp)']
        c = C_GREEN if lift > 0 else C_RED
        ax.text(r['OOS Prof%'] + 0.15, i, f'{lift:+.1f} pp',
                va='center', fontsize=8, color=c, fontweight='bold')
    ax.set_xlabel('OOS Profitability (%)')
    ax.set_title('Filter Effectiveness (50% prevalence)',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, 'fig_synthetic_mc_ranks.pdf')


def figure_6(syn):
    """Produce Fig 6 (fig_synthetic_mc_analysis.pdf): data-mining scenario and portfolio MC ranks."""
    print('\n=== Figure 6: Synthetic MC Analysis ===')
    df_a = syn['df_a']
    df_b = syn['df_b']
    bl_b = syn['bl_b']
    b_filter_df = syn['b_filter_df']
    port_df = syn['port_df']
    r_mdd_a, _ = pearsonr(df_a['mc_mdd_rank'], df_a['oos_profitable'].astype(float))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    for mom, color, label in [(True, C_TEAL, 'Momentum asset (has signal)'),
                               (False, C_RED, 'Random walk (no signal)')]:
        vals = df_b[df_b['has_momentum'] == mom]['mc_mdd_rank'].values
        ax.hist(vals, bins=50, density=True, alpha=0.3, color=color,
                edgecolor='white', linewidth=0.3)
        if vals.std() > 0.5:
            kde = gaussian_kde(vals, bw_method=0.1)
            x = np.linspace(0, 100, 300)
            ax.plot(x, kde(x), color=color, linewidth=2,
                    label=f'{label}\n(mean={vals.mean():.1f}%)')
    ax.axvline(50, color=C_GRAY, ls='--', lw=1.5, alpha=0.7)
    ax.set_xlabel('MC MDD Percentile Rank')
    ax.set_ylabel('Density')
    ax.set_title('Data-Mining Scenario: MC MDD Rank by Asset Type',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right')
    r_b_mdd, _ = pearsonr(df_b['mc_mdd_rank'],
                          df_b['oos_profitable'].astype(float))
    ax.annotate(f'r vs OOS = {r_b_mdd:.4f}',
                xy=(0.03, 0.95), xycoords='axes fraction', fontsize=8,
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe0e0',
                          edgecolor=C_RED, alpha=0.9))

    ax = axes[0, 1]

    def fcolor(f):
        if 'MC' in f:
            return C_RED
        if 'Rob' in f:
            return C_NAVY
        return C_GRAY

    bdf = b_filter_df[b_filter_df['Filter'] != 'No filter'].copy()
    bdf = bdf.sort_values('OOS Prof%', ascending=True).reset_index(drop=True)
    colors = [fcolor(f) for f in bdf['Filter']]
    ax.barh(range(len(bdf)), bdf['OOS Prof%'], color=colors,
            edgecolor='white', height=0.65, alpha=0.9)
    ax.axvline(bl_b, color=C_GRAY, ls='--', lw=1.5, alpha=0.6)
    ax.set_yticks(range(len(bdf)))
    ylabels = [f"{r['Filter']}  (n={r['Pool']/1e3:.0f}K, {r['% Edge']:.0f}% edge)"
               for _, r in bdf.iterrows()]
    ax.set_yticklabels(ylabels, fontsize=8)
    for i, (_, r) in enumerate(bdf.iterrows()):
        lift = r['Lift (pp)']
        c = C_GREEN if lift > 0 else C_RED
        ax.text(r['OOS Prof%'] + 0.15, i, f'{lift:+.1f} pp',
                va='center', fontsize=8, color=c, fontweight='bold')
    ax.set_xlabel('OOS Profitability (%)')
    ax.set_title('Data-Mining: Filter Effectiveness',
                 fontweight='bold', fontsize=10)

    ax = axes[1, 0]
    sub = df_a.sample(min(5000, len(df_a)), random_state=42)
    colors_s = [C_TEAL if e else C_RED for e in sub['has_edge']]
    ax.scatter(sub['mc_mdd_rank'],
               sub['oos_profitable'].astype(float) + np.random.normal(0, 0.02, len(sub)),
               alpha=0.08, s=5, c=colors_s, zorder=2)
    bins_arr = np.linspace(0, 100, 21)
    sub_copy = sub.copy()
    sub_copy['mdd_bin'] = pd.cut(sub_copy['mc_mdd_rank'], bins_arr)
    binned = sub_copy.groupby('mdd_bin', observed=True)['oos_profitable'].mean()
    bc = [(b.left + b.right) / 2 for b in binned.index]
    ax.plot(bc, binned.values, 'ko-', linewidth=2, markersize=5, zorder=4,
            label='Binned OOS rate')
    ax.axhline(0.5, color=C_GRAY, ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('MC MDD Percentile Rank')
    ax.set_ylabel('OOS Profitable (0/1)')
    ax.set_title('MC MDD Rank vs OOS Profitability',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.annotate(f'r = {r_mdd_a:.4f}',
                xy=(0.97, 0.05), xycoords='axes fraction', fontsize=8,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=C_GRAY, alpha=0.9))

    ax = axes[1, 1]
    ax.hist(port_df['mc_rank'], bins=40, density=True, alpha=0.4,
            color=C_PURPLE, edgecolor='white', linewidth=0.3)
    if port_df['mc_rank'].std() > 0.5:
        kde = gaussian_kde(port_df['mc_rank'].values, bw_method=0.15)
        x = np.linspace(max(0, port_df['mc_rank'].min() - 5),
                        min(100, port_df['mc_rank'].max() + 5), 300)
        ax.plot(x, kde(x), color=C_PURPLE, linewidth=2)
    ax.axvline(50, color=C_GRAY, ls='--', lw=1.5, alpha=0.7,
               label='Random expectation')
    ax.axvline(port_df['mc_rank'].mean(), color=C_RED, lw=2,
               label=f'Mean = {port_df["mc_rank"].mean():.1f}%')
    ax.set_xlabel('Portfolio MC ROI Percentile Rank')
    ax.set_ylabel('Density')
    ax.set_title(
        f'Correlated Portfolios '
        f'(mean \u03c1 = {port_df["mean_corr"].mean():.2f})',
        fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, 'fig_synthetic_mc_analysis.pdf')


# ---------------------------------------------------------------------------
#  Figures 7 & 8: pre-saved synthetic pipeline summaries
# ---------------------------------------------------------------------------
def _load_pipeline_tables():
    # Prefer the _matched.csv files produced by the current Rust synthetic
    # pipeline; fall back to the legacy non-matched Python-era files.
    sum_matched = TBL / 'synthetic_v4_summaries_matched.csv'
    sum_legacy = TBL / 'synthetic_v4_summaries.csv'
    if sum_matched.exists():
        all_sum = pd.read_csv(sum_matched)
    elif sum_legacy.exists():
        all_sum = pd.read_csv(sum_legacy)
    else:
        return None
    tiers = {
        'null': all_sum[all_sum['tier'] == 'null'],
        'edge': all_sum[all_sum['tier'] == 'edge'],
        'adversarial': all_sum[all_sum['tier'] == 'adversarial'],
    }

    def _load_filters(tier):
        matched = TBL / f'synthetic_v4_{tier}_filters_matched.csv'
        legacy = TBL / f'synthetic_v4_{tier}_filters.csv'
        if matched.exists():
            df = pd.read_csv(matched)
            # Rename matched snake_case schema to the Title-case the
            # downstream figure code uses.
            return df.rename(columns={
                'filter': 'Filter', 'pool': 'Pool',
                'oos_prof_pct': 'OOS Prof%', 'lift_pp': 'Lift (pp)',
                'pool_pct': 'Pool %',
            })
        if legacy.exists():
            return pd.read_csv(legacy)
        return None

    filters = {}
    for tier in ('null', 'edge', 'adversarial'):
        df = _load_filters(tier)
        if df is not None:
            filters[tier] = df
    return tiers, filters


def figure_7():
    """Produce Fig 7 (fig_synthetic_pipeline_v4.pdf): full-pipeline synthetic overview."""
    print('\n=== Figure 7: Synthetic Pipeline v4 Overview ===')
    loaded = _load_pipeline_tables()
    if loaded is None:
        print('  synthetic_v4_summaries.csv missing, skipping.')
        return
    tiers, tier_filters = loaded
    t1, t2, t3 = tiers['null'], tiers['edge'], tiers['adversarial']

    sweep_matched = TBL / 'synthetic_v4_signal_sweep_matched.csv'
    sweep_legacy = TBL / 'synthetic_v4_signal_sweep.csv'
    if sweep_matched.exists():
        sweep_df = pd.read_csv(sweep_matched)
        # Matched schema encodes phi in the tier string as "sweep_phi_0.04".
        sweep_df = sweep_df[sweep_df['tier'].str.startswith('sweep_phi_')].copy()
        sweep_df['phi'] = sweep_df['tier'].str.replace('sweep_phi_', '',
                                                       regex=False).astype(float)
        sweep_df = sweep_df.rename(columns={
            'mc_roi_lift_p50': 'mc_p50_lift',
            'rob_all_lift': 'rob_lift',
        })
    elif sweep_legacy.exists():
        sweep_df = pd.read_csv(sweep_legacy)
        if 'rob_lift' not in sweep_df.columns:
            sweep_df['rob_lift'] = np.nan  # legacy file lacks the column
    else:
        sweep_df = None

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    ax = axes[0, 0]
    tier_labels = ['Tier 1\n(Null)', 'Tier 2\n(Edge)', 'Tier 3\n(Adversarial)']
    bl_vals = [t1['baseline_oos'].mean(), t2['baseline_oos'].mean(),
               t3['baseline_oos'].mean()]
    ax.bar(range(3), bl_vals, color=[C_NAVY, C_TEAL, C_RED], alpha=0.8,
           edgecolor='white', width=0.6)
    for i, v in enumerate(bl_vals):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(tier_labels, fontsize=10)
    ax.set_ylabel('Baseline OOS Profitability (%)')
    ax.set_title('Baseline OOS by Tier\n(no filtering)', fontweight='bold')
    ax.axhline(50, color=C_GRAY, ls='--', lw=1.5)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    filters_to_show = ['IS PF > 1', 'MC-ROI >= p50', 'MC-ROI >= p75',
                       'Rob: Combined']
    fcolors = [C_GRAY, C_RED, C_RED, C_NAVY]
    x_pos = np.arange(3)
    w = 0.2
    for fi, (fname, fc) in enumerate(zip(filters_to_show, fcolors)):
        vals = []
        for tname in ['null', 'edge', 'adversarial']:
            if tname in tier_filters:
                tf = tier_filters[tname]
                row = tf[tf['Filter'] == fname]
                vals.append(row['Lift (pp)'].values[0] if len(row) else 0)
            else:
                vals.append(0)
        offset = (fi - len(filters_to_show) / 2 + 0.5) * w
        ax.bar(x_pos + offset, vals, w, color=fc, alpha=0.8,
               label=fname, edgecolor='white')
    ax.axhline(0, color=C_GRAY, ls='--', lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tier_labels, fontsize=9)
    ax.set_ylabel('Lift vs Baseline (pp)')
    ax.set_title('Filter Lift by Tier', fontweight='bold')
    ax.legend(fontsize=7, loc='best')

    ax = axes[0, 2]
    if sweep_df is not None:
        sweep_avg = sweep_df.groupby('phi').agg({
            'baseline_oos': 'mean',
            'mc_p50_lift': 'mean',
            'rob_lift': 'mean',
        }).reset_index()
        ax.plot(sweep_avg['phi'], sweep_avg['mc_p50_lift'], 'o-',
                color=C_RED, linewidth=2, markersize=5, label='MC-ROI p50 Lift')
        ax.plot(sweep_avg['phi'], sweep_avg['rob_lift'], 's-',
                color=C_NAVY, linewidth=2, markersize=5, label='Rob: Combined Lift')
        ax.axhline(0, color=C_GRAY, ls='--', lw=1.5)
        ax.set_xlabel('Signal Strength (AR coefficient)')
        ax.set_ylabel('Lift vs Baseline (pp)')
        ax.set_title('Filter Lift vs Signal Strength', fontweight='bold')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Signal sweep data\nnot available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax = axes[1, 0]
    for name, sdf, color in [
        ('Tier 1: Null', t1, C_NAVY),
        ('Tier 2: Edge', t2, C_TEAL),
        ('Tier 3: Adversarial', t3, C_RED),
    ]:
        lifts = sdf['mc_roi_lift_p50'].dropna().values
        if len(lifts) > 2:
            ax.hist(lifts, bins=15, alpha=0.4, color=color,
                    edgecolor='white', density=True,
                    label=f'{name} (mean={lifts.mean():.2f})')
            if lifts.std() > 0.01:
                kde = gaussian_kde(lifts, bw_method=0.3)
                xk = np.linspace(lifts.min() - 1, lifts.max() + 1, 200)
                ax.plot(xk, kde(xk), color=color, lw=2)
    ax.axvline(0, color=C_GRAY, ls='--', lw=1.5)
    ax.set_xlabel('MC-ROI p50 Lift (pp)')
    ax.set_ylabel('Density')
    ax.set_title('MC Lift Distribution Across Runs', fontweight='bold')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for name, sdf, color in [
        ('Tier 1: Null', t1, C_NAVY),
        ('Tier 2: Edge', t2, C_TEAL),
        ('Tier 3: Adversarial', t3, C_RED),
    ]:
        lifts = sdf['rob_all_lift'].dropna().values
        if len(lifts) > 2:
            ax.hist(lifts, bins=15, alpha=0.4, color=color,
                    edgecolor='white', density=True,
                    label=f'{name} (mean={lifts.mean():.2f})')
            if lifts.std() > 0.01:
                kde = gaussian_kde(lifts, bw_method=0.3)
                xk = np.linspace(lifts.min() - 1, lifts.max() + 1, 200)
                ax.plot(xk, kde(xk), color=color, lw=2)
    ax.axvline(0, color=C_GRAY, ls='--', lw=1.5)
    ax.set_xlabel('Rob: Combined Lift (pp)')
    ax.set_ylabel('Density')
    ax.set_title('Robustness Lift Distribution Across Runs', fontweight='bold')
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    for name, sdf, color, marker in [
        ('Tier 1: Null', t1, C_NAVY, 'o'),
        ('Tier 2: Edge', t2, C_TEAL, 's'),
        ('Tier 3: Adversarial', t3, C_RED, '^'),
    ]:
        vals = sdf['port_nofilter_oos'].dropna().values
        if len(vals) > 0:
            ax.scatter(range(len(vals)), vals, color=color, marker=marker,
                       s=40, alpha=0.7,
                       label=f'{name} (mean={vals.mean():.1f}%)')
    ax.axhline(50, color=C_GRAY, ls='--', lw=1.5, label='Null expectation (50%)')
    ax.set_xlabel('Simulation Run')
    ax.set_ylabel('Portfolio OOS Profitability (%)')
    ax.set_title('Portfolio-Level Results Across Runs', fontweight='bold')
    ax.legend(fontsize=8, loc='best')

    fig.tight_layout()
    savefig(fig, 'fig_synthetic_pipeline_v4.pdf')


def figure_8():
    """Produce Fig 8 (fig_synthetic_pipeline_detail.pdf): per-tier lift details."""
    print('\n=== Figure 8: Synthetic Pipeline v4 Detail ===')
    loaded = _load_pipeline_tables()
    if loaded is None:
        print('  synthetic_v4_summaries.csv missing, skipping.')
        return
    tiers, tier_filters = loaded
    t1, t2, t3 = tiers['null'], tiers['edge'], tiers['adversarial']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    lifts_t1 = t1['mc_roi_lift_p50'].dropna().values
    if len(lifts_t1) > 0:
        ax.hist(lifts_t1, bins=12, density=True, alpha=0.5, color=C_NAVY,
                edgecolor='white')
        ax.axvline(0, color=C_RED, ls='--', lw=1.5, label='Zero lift')
        ax.axvline(lifts_t1.mean(), color=C_NAVY, lw=2,
                   label=f'Mean = {lifts_t1.mean():.2f} pp')
        ax.set_xlabel('MC ROI p50 Lift (pp)')
        ax.set_ylabel('Density')
        ax.set_title('Tier 1 (Null): MC Lift Distribution', fontweight='bold')
        ax.legend(fontsize=8)

    ax = axes[0, 1]
    lifts_t2 = t2['mc_roi_lift_p50'].dropna().values
    if len(lifts_t2) > 0:
        ax.hist(lifts_t2, bins=12, density=True, alpha=0.5, color=C_TEAL,
                edgecolor='white')
        ax.axvline(0, color=C_RED, ls='--', lw=1.5, label='Zero lift')
        ax.axvline(lifts_t2.mean(), color=C_TEAL, lw=2,
                   label=f'Mean = {lifts_t2.mean():.2f} pp')
        ax.set_xlabel('MC ROI p50 Lift (pp)')
        ax.set_ylabel('Density')
        ax.set_title('Tier 2 (Edge): MC Lift Distribution', fontweight='bold')
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    if 'adversarial' in tier_filters:
        tf = tier_filters['adversarial'].copy()
        tf['Filter'] = tf['Filter'].str.replace('Rob: All', 'Rob: Combined')
        tf = tf[tf['Filter'] != 'No filter']
        tf = tf.sort_values('Lift (pp)', ascending=True).reset_index(drop=True)

        def fcolor2(f):
            if 'MC' in f:
                return C_RED
            if 'Rob' in f:
                return C_NAVY
            return C_GRAY

        colors = [fcolor2(f) for f in tf['Filter']]
        ax.barh(range(len(tf)), tf['Lift (pp)'], color=colors,
                edgecolor='white', height=0.65, alpha=0.9)
        ax.axvline(0, color=C_GRAY, ls='--', lw=1.5)
        ax.set_yticks(range(len(tf)))
        ax.set_yticklabels(tf['Filter'], fontsize=9)
        ax.set_xlabel('Lift vs Baseline (pp)')
        ax.set_title('Tier 3 (Adversarial): Filter Lifts', fontweight='bold')

    ax = axes[1, 1]
    tier_labels = ['Tier 1\n(Null)', 'Tier 2\n(Edge)', 'Tier 3\n(Adversarial)']
    mc_means = [t1['mc_roi_lift_p50'].mean(),
                t2['mc_roi_lift_p50'].mean(),
                t3['mc_roi_lift_p50'].mean()]
    rob_means = [t1['rob_all_lift'].mean(),
                 t2['rob_all_lift'].mean(),
                 t3['rob_all_lift'].mean()]
    mc_cis = [(s['mc_roi_lift_p50'].quantile(0.025),
               s['mc_roi_lift_p50'].quantile(0.975))
              for s in [t1, t2, t3]]
    rob_cis = [(s['rob_all_lift'].quantile(0.025),
                s['rob_all_lift'].quantile(0.975))
               for s in [t1, t2, t3]]
    x_pos = np.arange(3)
    w = 0.35
    mc_err = [[m - ci[0] for m, ci in zip(mc_means, mc_cis)],
              [ci[1] - m for m, ci in zip(mc_means, mc_cis)]]
    rob_err = [[m - ci[0] for m, ci in zip(rob_means, rob_cis)],
               [ci[1] - m for m, ci in zip(rob_means, rob_cis)]]
    ax.bar(x_pos - w / 2, mc_means, w, color=C_RED, alpha=0.8,
           label='MC-ROI p50', yerr=mc_err, capsize=4, ecolor=C_GRAY)
    ax.bar(x_pos + w / 2, rob_means, w, color=C_NAVY, alpha=0.8,
           label='Rob: Combined', yerr=rob_err, capsize=4, ecolor=C_GRAY)
    ax.axhline(0, color=C_GRAY, ls='--', lw=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tier_labels, fontsize=9)
    ax.set_ylabel('Lift vs Baseline (pp)')
    ax.set_title('Filter Lift Across Tiers (95% CI)', fontweight='bold')
    ax.legend(fontsize=9)

    fig.tight_layout()
    savefig(fig, 'fig_synthetic_pipeline_detail.pdf')


# ---------------------------------------------------------------------------
#  Figures 9 & 10: forex/commodity MC distributions and binned OOS
# ---------------------------------------------------------------------------
def figure_9():
    """Produce Fig 9 (mc_pct_rank_distributions.pdf): forex/commodity MC rank distributions."""
    print('\n=== Figure 9: MC Pct Rank Distributions (Forex/Commodity) ===')
    merged = load_forex_data()
    if not merged:
        return
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes_flat = axes.flatten()
    for idx, a in enumerate(FOREX_ASSETS):
        if a not in merged:
            continue
        ax = axes_flat[idx]
        m = merged[a]
        for col, label, color in [
            ('roi_pct_rank', 'ROI', '#1f77b4'),
            ('sharpe_pct_rank', 'Sharpe', '#ff7f0e'),
            ('pf_pct_rank', 'PF', '#2ca02c'),
        ]:
            vals = m[col].dropna()
            ax.hist(vals, bins=50, alpha=0.5,
                    label=f'{label} (mean={vals.mean():.1f})',
                    color=color, density=True)
        ax.axvline(x=50, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('MC Percentile Rank')
        ax.set_ylabel('Density')
        ax.set_title(FOREX_LABELS[a])
        ax.legend(fontsize=7)
    axes_flat[-1].set_visible(False)
    plt.suptitle(
        'Distribution of MC Percentile Ranks (Per-Window IS) - Forex/Commodity',
        fontsize=13)
    plt.tight_layout()
    savefig(fig, 'mc_pct_rank_distributions.pdf')


def figure_10():
    """Produce Fig 10 (mc_roi_vs_next_oos_binned.pdf): binned MC ROI vs next-window OOS rate."""
    print('\n=== Figure 10: MC ROI vs Next OOS Binned (Forex/Commodity) ===')
    merged = load_forex_data()
    if not merged:
        return
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes_flat = axes.flatten()
    for idx, a in enumerate(FOREX_ASSETS):
        if a not in merged:
            continue
        ax = axes_flat[idx]
        m = merged[a]
        has_next = m['next_baseline_oos_pf'].notna()
        m_v = m[has_next]
        x = m_v['roi_pct_rank'].values
        y = (m_v['next_baseline_oos_pf'] > 1.0).astype(float).values
        bins_arr = np.linspace(0, 100, 21)
        bin_centers = (bins_arr[:-1] + bins_arr[1:]) / 2
        bin_rates = []
        for lo, hi in zip(bins_arr[:-1], bins_arr[1:]):
            sel = y[(x >= lo) & (x < hi)]
            bin_rates.append(sel.mean() * 100 if len(sel) > 10 else np.nan)
        ax.bar(bin_centers, bin_rates, width=4.5, color='steelblue', alpha=0.7)
        ax.axhline(y=y.mean() * 100, color='red', linestyle='--',
                   label=f'Baseline: {y.mean()*100:.1f}%')
        ax.set_xlabel('MC ROI Percentile Rank')
        ax.set_ylabel('Next OOS Profitability (%)')
        ax.set_title(FOREX_LABELS[a])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    axes_flat[-1].set_visible(False)
    plt.suptitle(
        'MC ROI Percentile Rank vs Next-Window OOS Profitability Rate '
        '(Forex/Commodity)', fontsize=13)
    plt.tight_layout()
    savefig(fig, 'mc_roi_vs_next_oos_binned.pdf')


if __name__ == '__main__':
    print('=' * 70)
    print('REGENERATING FIGURES 2-10 FOR THE PAPER')
    print('=' * 70)
    print(f'Output directory: {OUT}')
    print()

    figure_2()
    figure_3()
    figure_4()

    print('\n--- Running Synthetic v3 Experiments (Figures 5-6) ---')
    try:
        syn = run_synthetic_v3()
        figure_5(syn)
        figure_6(syn)
    except Exception as e:
        import traceback
        print(f'  ERROR in synthetic v3: {e}')
        traceback.print_exc()

    figure_7()
    figure_8()
    figure_9()
    figure_10()

    print('\n' + '=' * 70)
    print('DONE')
    print('=' * 70)
    expected = [
        'window_level_mc_vs_oos.pdf',
        'fig_bootstrap_lift_distributions.pdf',
        'fig_regime_robustness.pdf',
        'fig_synthetic_mc_ranks.pdf',
        'fig_synthetic_mc_analysis.pdf',
        'fig_synthetic_pipeline_v4.pdf',
        'fig_synthetic_pipeline_detail.pdf',
        'mc_pct_rank_distributions.pdf',
        'mc_roi_vs_next_oos_binned.pdf',
    ]
    for fname in expected:
        status = 'OK' if (OUT / fname).exists() else 'MISSING'
        print(f'  [{status}] {fname}')
