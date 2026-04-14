"""
Window-clustered bootstrap confidence intervals for block-permutation lift.

Resamples (asset, window) clusters — treating each as one observation —
and computes 95% CIs for MC-filter lift at block sizes b = 1, 5, 10.
Used as a supporting robustness check for Table 15 alongside the
calendar-clustered variant in calendar_cluster_bootstrap.py.

Inputs (relative to project root):
  - results/raw_data/block_perm_<asset>.csv
  - results/raw_data/<asset>_window_pairs.csv

Outputs:
  - results/tables/block_perm_window_cluster_ci.csv
"""
import os
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

METHOD_COLS = ['iid_rank', 'block5_rank', 'block10_rank']
METHOD_NAMES = ['i.i.d. (b=1)', 'Block (b=5)', 'Block (b=10)']

N_BOOT = 10000


def load_and_aggregate():
    print("Loading and pre-aggregating by cluster...")
    cluster_data = []
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
            entry = {'asset': asset, 'window': w, 'n': len(grp),
                     'n_oos_prof': int(grp['oos_prof'].sum())}
            for col in METHOD_COLS:
                if col in grp.columns:
                    passed = grp[grp[col] >= 50]
                    entry[f'{col}_n_pass'] = len(passed)
                    entry[f'{col}_n_pass_oos'] = int(passed['oos_prof'].sum())
                else:
                    entry[f'{col}_n_pass'] = 0
                    entry[f'{col}_n_pass_oos'] = 0
            cluster_data.append(entry)
        print(f"  {asset}: {len(bp):,} obs -> {bp['window'].nunique()} clusters")
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
    cluster_data = load_and_aggregate()
    n_clusters = len(cluster_data)
    print(f"\nTotal clusters: {n_clusters}")

    total_n = sum(c['n'] for c in cluster_data)
    total_oos = sum(c['n_oos_prof'] for c in cluster_data)
    baseline = total_oos / total_n
    print(f"Total observations: {total_n:,}")
    print(f"Pooled baseline: {baseline*100:.1f}%")

    point_lifts = {}
    for col in METHOD_COLS:
        total_pass = sum(c[f'{col}_n_pass'] for c in cluster_data)
        total_pass_oos = sum(c[f'{col}_n_pass_oos'] for c in cluster_data)
        pass_rate = total_pass_oos / total_pass if total_pass > 0 else 0
        point_lifts[col] = (pass_rate - baseline) * 100

    n_workers = min(cpu_count(), 32)
    batch_size = N_BOOT // n_workers
    remainder = N_BOOT % n_workers
    args_list = [
        (cluster_data, n_clusters,
         batch_size + (1 if i < remainder else 0), 42 + i)
        for i in range(n_workers)
    ]

    print(f"\nRunning {N_BOOT} bootstrap resamples on {n_workers} workers...")
    with Pool(n_workers) as pool:
        all_batches = pool.map(bootstrap_batch, args_list)
    all_results = [r for batch in all_batches for r in batch]
    print(f"Completed {len(all_results)} resamples.")

    print(f"\n{'='*70}")
    print(f"WINDOW-CLUSTERED BOOTSTRAP CIs "
          f"({N_BOOT} resamples, {n_clusters} clusters)")
    print('=' * 70)

    rows = []
    for col, name in zip(METHOD_COLS, METHOD_NAMES):
        lifts = np.array([r[col] for r in all_results if not np.isnan(r[col])])
        point = point_lifts[col]
        se = np.std(lifts)
        ci_lo = np.percentile(lifts, 2.5)
        ci_hi = np.percentile(lifts, 97.5)
        z = point / se if se > 0 else np.nan
        p_pos = np.mean(lifts >= 0)
        print(f"{name:18s}: Point={point:+.2f}pp  SE={se:.3f}  "
              f"95% CI=[{ci_lo:+.2f}, {ci_hi:+.2f}]  z={z:.1f}  "
              f"P(lift>=0)={p_pos:.4f}")
        rows.append({
            'Filter': name, 'Point_pp': round(point, 3),
            'SE': round(se, 4), 'CI_lo': round(ci_lo, 3),
            'CI_hi': round(ci_hi, 3), 'z': round(z, 2),
            'P_lift_ge_0': p_pos,
        })

    out = TABLES / 'block_perm_window_cluster_ci.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
