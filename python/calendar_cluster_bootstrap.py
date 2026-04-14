"""
Calendar-period clustered bootstrap for the MC-ROI p50 lift estimate.

Each (asset, window) OOS period is assigned to a calendar-quarter cluster and
the bootstrap resamples those clusters (rather than individual observations
or asset-windows) to account for cross-asset dependence: crypto assets share
overlapping calendar periods, as do forex/commodity assets.

This script generates:
  - The point estimate and 95% CI for MC-ROI p50 lift reported in Table 15.
  - The underlying distribution displayed in Figure 3
    (fig_bootstrap_lift_distributions.pdf) — note the figure itself is built
    by regenerate_all_figures.py; this script produces the CI numbers it
    annotates.

Inputs (relative to project root):
  - results/raw_data/block_perm_<asset>.csv    (Rust block-perm output)
  - results/raw_data/<asset>_window_pairs.csv  (backtest pipeline)

Outputs:
  - results/tables/empirical_bootstrap_ci.csv
"""
import os
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
RAW = ROOT / "results" / "raw_data"
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

ASSETS_INFO = {
    'BTC':     {'csv': 'block_perm_btc.csv',    'wp': 'btc_window_pairs.csv'},
    'DOGE':    {'csv': 'block_perm_doge.csv',   'wp': 'doge_window_pairs.csv'},
    'BNB':     {'csv': 'block_perm_bnb.csv',    'wp': 'bnb_window_pairs.csv'},
    'SOL':     {'csv': 'block_perm_sol.csv',    'wp': 'sol_window_pairs.csv'},
    'EUR/USD': {'csv': 'block_perm_eurusd.csv', 'wp': 'eurusd_window_pairs.csv'},
    'USD/JPY': {'csv': 'block_perm_usdjpy.csv', 'wp': 'usdjpy_window_pairs.csv'},
    'EUR/GBP': {'csv': 'block_perm_eurgbp.csv', 'wp': 'eurgbp_window_pairs.csv'},
    'XAU/USD': {'csv': 'block_perm_xauusd.csv', 'wp': 'xauusd_window_pairs.csv'},
    'WTI':     {'csv': 'block_perm_wti.csv',    'wp': 'wti_window_pairs.csv'},
}

# Only the i.i.d. (b=1) rank is needed for the main MC-ROI p50 lift result.
METHOD_COLS = ['iid_rank']
METHOD_NAMES = ['MC-ROI p50 (i.i.d.)']

N_BOOT = 10000


def compute_oos_periods():
    """Compute calendar OOS start/end for each (asset, window) from the WFO
    protocol: IS = 10,000 candles, advance = 5,000 candles. Crypto assets
    trade 24/7 (candles map directly to calendar time). Forex/commodity
    windows are approximated in clock hours."""
    periods = {}

    # (data_start, timeframe_minutes, n_windows)
    crypto_config = {
        'BTC':  (datetime(2019, 12, 31), 30, 27),
        'DOGE': (datetime(2020, 7, 10),  30, 21),
        'BNB':  (datetime(2020, 2, 10),  15, 30),
        'SOL':  (datetime(2020, 9, 14),  60, 7),
    }
    for asset, (data_start, tf_min, n_windows) in crypto_config.items():
        is_candles, advance_candles = 10000, 5000
        candle = timedelta(minutes=tf_min)
        is_duration = is_candles * candle
        oos_duration = advance_candles * candle
        for w in range(1, n_windows + 1):
            is_start = data_start + (w - 1) * advance_candles * candle
            oos_start = is_start + is_duration
            oos_end = oos_start + oos_duration
            periods[(asset, w)] = (oos_start, oos_end)

    # Forex/commodity: 15 sliding windows; approximate calendar days from
    # clock-hour-equivalent advance/IS durations (weekend closures ignored).
    fx_config = {
        'EUR/USD': (datetime(2017, 7, 3),  15),
        'USD/JPY': (datetime(2016, 3, 24), 15),
        'EUR/GBP': (datetime(2016, 3, 24), 15),
        'XAU/USD': (datetime(2016, 3, 28), 15),
        'WTI':     (datetime(2016, 3, 28), 15),
    }
    advance_days_fx = 5000 / 24
    is_days_fx = 10000 / 24
    for asset, (data_start, n_windows) in fx_config.items():
        for w in range(1, n_windows + 1):
            is_start = data_start + timedelta(days=(w - 1) * advance_days_fx)
            oos_start = is_start + timedelta(days=is_days_fx)
            oos_end = oos_start + timedelta(days=advance_days_fx)
            periods[(asset, w)] = (oos_start, oos_end)

    return periods


def assign_calendar_clusters(periods):
    """Group (asset, window) pairs by calendar-quarter of OOS midpoint."""
    clusters = {}
    for (asset, w), (oos_start, oos_end) in periods.items():
        midpoint = oos_start + (oos_end - oos_start) / 2
        quarter = (midpoint.year, (midpoint.month - 1) // 3 + 1)
        clusters.setdefault(quarter, []).append((asset, w))
    return list(clusters.values())


def load_and_aggregate(calendar_clusters):
    """Load data and aggregate counts per (asset, window), then per cluster."""
    print("Loading data...")
    aw_data = {}

    for asset, info in ASSETS_INFO.items():
        bp_path = RAW / info['csv']
        wp_path = RAW / info['wp']
        if not bp_path.exists() or not wp_path.exists():
            print(f"  {asset}: MISSING, skipping")
            continue

        bp = pd.read_csv(bp_path)
        wp = pd.read_csv(wp_path)

        wp_dict = {
            (row['strategy'], row['window_i']): row['baseline_oos_pf']
            for _, row in wp.iterrows()
        }
        bp['oos_prof'] = bp.apply(
            lambda r: 1 if wp_dict.get((r['strategy'], r['window']), 0) > 1.0 else 0,
            axis=1,
        )
        bp = bp[bp.apply(
            lambda r: (r['strategy'], r['window']) in wp_dict, axis=1
        )].copy()

        for w, grp in bp.groupby('window'):
            entry = {
                'asset': asset, 'window': w,
                'n': len(grp),
                'n_oos_prof': int(grp['oos_prof'].sum()),
            }
            for col in METHOD_COLS:
                if col in grp.columns:
                    passed = grp[grp[col] >= 50]
                    entry[f'{col}_n_pass'] = len(passed)
                    entry[f'{col}_n_pass_oos'] = int(passed['oos_prof'].sum())
                else:
                    entry[f'{col}_n_pass'] = 0
                    entry[f'{col}_n_pass_oos'] = 0
            aw_data[(asset, w)] = entry
        print(f"  {asset}: {len(bp):,} obs -> {bp['window'].nunique()} windows")

    cluster_data = []
    for cluster_members in calendar_clusters:
        combined = {'members': cluster_members, 'n': 0, 'n_oos_prof': 0}
        for col in METHOD_COLS:
            combined[f'{col}_n_pass'] = 0
            combined[f'{col}_n_pass_oos'] = 0
        for (asset, w) in cluster_members:
            if (asset, w) in aw_data:
                e = aw_data[(asset, w)]
                combined['n'] += e['n']
                combined['n_oos_prof'] += e['n_oos_prof']
                for col in METHOD_COLS:
                    combined[f'{col}_n_pass'] += e[f'{col}_n_pass']
                    combined[f'{col}_n_pass_oos'] += e[f'{col}_n_pass_oos']
        if combined['n'] > 0:
            cluster_data.append(combined)
    return cluster_data


def bootstrap_batch(args):
    cluster_data, n_clusters, batch_size, seed = args
    rng = np.random.RandomState(seed)
    results = []
    for _ in range(batch_size):
        idx = rng.randint(0, n_clusters, size=n_clusters)
        total_n = total_oos = 0
        method_pass = {c: 0 for c in METHOD_COLS}
        method_pass_oos = {c: 0 for c in METHOD_COLS}
        for i in idx:
            c = cluster_data[i]
            total_n += c['n']
            total_oos += c['n_oos_prof']
            for col in METHOD_COLS:
                method_pass[col] += c[f'{col}_n_pass']
                method_pass_oos[col] += c[f'{col}_n_pass_oos']
        baseline = total_oos / total_n if total_n > 0 else 0
        lifts = {}
        for col in METHOD_COLS:
            if method_pass[col] > 0:
                lifts[col] = (method_pass_oos[col] / method_pass[col] - baseline) * 100
            else:
                lifts[col] = np.nan
        results.append(lifts)
    return results


def main():
    print("Computing OOS calendar periods...")
    periods = compute_oos_periods()
    print(f"  Total (asset, window) pairs: {len(periods)}")

    calendar_clusters = assign_calendar_clusters(periods)
    print(f"  Calendar-period clusters: {len(calendar_clusters)}")

    cluster_data = load_and_aggregate(calendar_clusters)
    n_clusters = len(cluster_data)
    print(f"\nNon-empty calendar clusters: {n_clusters}")

    total_n = sum(c['n'] for c in cluster_data)
    total_oos = sum(c['n_oos_prof'] for c in cluster_data)
    baseline = total_oos / total_n
    print(f"Total observations: {total_n:,}")
    print(f"Pooled baseline: {baseline*100:.2f}%")

    point_lifts = {}
    for col in METHOD_COLS:
        total_pass = sum(c[f'{col}_n_pass'] for c in cluster_data)
        total_pass_oos = sum(c[f'{col}_n_pass_oos'] for c in cluster_data)
        pass_rate = total_pass_oos / total_pass if total_pass > 0 else 0
        point_lifts[col] = (pass_rate - baseline) * 100

    n_workers = min(cpu_count(), 32)
    batch_size = N_BOOT // n_workers
    remainder = N_BOOT % n_workers
    # Deterministic seeds: 42 + worker index
    args_list = [
        (cluster_data, n_clusters,
         batch_size + (1 if i < remainder else 0), 42 + i)
        for i in range(n_workers)
    ]

    print(f"\nRunning {N_BOOT} resamples on {n_workers} workers "
          f"({n_clusters} clusters)...")
    with Pool(n_workers) as pool:
        all_batches = pool.map(bootstrap_batch, args_list)

    all_results = [r for batch in all_batches for r in batch]
    print(f"Completed {len(all_results)} resamples.\n")

    print('=' * 75)
    print(f"CALENDAR-PERIOD CLUSTERED BOOTSTRAP CIs  "
          f"({N_BOOT} resamples, {n_clusters} clusters)")
    print('=' * 75)

    rows = []
    for col, name in zip(METHOD_COLS, METHOD_NAMES):
        lifts = np.array([r[col] for r in all_results if not np.isnan(r[col])])
        point = point_lifts[col]
        se = np.std(lifts)
        ci_lo = np.percentile(lifts, 2.5)
        ci_hi = np.percentile(lifts, 97.5)
        z = point / se if se > 0 else np.nan
        p_pos = np.mean(lifts >= 0)

        print(f"\n  {name}:")
        print(f"    Point estimate: {point:+.2f} pp")
        print(f"    Bootstrap SE:   {se:.4f}")
        print(f"    95% CI:         [{ci_lo:+.2f}, {ci_hi:+.2f}]")
        print(f"    z-score:        {z:.1f}")
        print(f"    P(lift >= 0):   {p_pos:.6f}")

        rows.append({
            'Filter': name,
            'Point_pp': round(point, 3),
            'SE': round(se, 4),
            'CI_lo': round(ci_lo, 3),
            'CI_hi': round(ci_hi, 3),
            'z': round(z, 2),
            'P_lift_ge_0': p_pos,
        })

    out = TABLES / 'empirical_bootstrap_ci.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
