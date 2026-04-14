"""
Portfolio-level MC permutation analysis (streaming).

Computes per-asset, per-filter MC percentile-rank statistics at the portfolio
level from large CSVs (each row is one portfolio-window-filter observation
with an MC rank triple). Feeds Table 14 (strategy vs portfolio OOS
profitability) by comparing strategy-level and portfolio-level mean ranks.

Inputs (relative to project root):
  - results/raw_data/<asset>_portfolio_mc.csv
  - results/raw_data/<asset>_mc_perwindow.csv  (strategy-level ranks
    used for the strategy-vs-portfolio comparison)

Outputs:
  - results/tables/portfolio_mc_summary.csv
  - results/tables/strat_vs_portfolio_mc.csv
"""
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
RAW = ROOT / "results" / "raw_data"
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

ASSETS = ['sol', 'btc', 'doge', 'bnb']
ASSET_LABELS = {'sol': 'SOL 1H', 'btc': 'BTC 30m',
                'doge': 'DOGE 30m', 'bnb': 'BNB 15m'}
FILTERS = ['baseline', 'rob_ent', 'rob_fee', 'rob_sli',
           'rob_entind', 'rob_all4']
FILTER_LABELS = {
    'baseline': 'Baseline (IS PF>1)',
    'rob_ent': 'Rob: ENT',
    'rob_fee': 'Rob: FEE',
    'rob_sli': 'Rob: SLI',
    'rob_entind': 'Rob: ENT+IND',
    'rob_all4': 'Rob: All-4',
}


def streaming_stats(asset):
    """Stream a portfolio_mc CSV once, accumulating per-filter running sums."""
    path = RAW / f'{asset}_portfolio_mc.csv'
    filter_stats = defaultdict(lambda: {
        'count': 0, 'roi_sum': 0.0, 'roi_sq_sum': 0.0,
        'sharpe_sum': 0.0, 'sharpe_sq_sum': 0.0, 'pf_sum': 0.0,
        'roi_above50': 0, 'sharpe_above50': 0,
    })

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filt = row['filter']
            roi_r = float(row['roi_pct_rank'])
            sharpe_r = float(row['sharpe_pct_rank'])
            pf_r = float(row['pf_pct_rank'])
            s = filter_stats[filt]
            s['count'] += 1
            s['roi_sum'] += roi_r
            s['roi_sq_sum'] += roi_r * roi_r
            s['sharpe_sum'] += sharpe_r
            s['sharpe_sq_sum'] += sharpe_r * sharpe_r
            s['pf_sum'] += pf_r
            if roi_r > 50:
                s['roi_above50'] += 1
            if sharpe_r > 50:
                s['sharpe_above50'] += 1
    return filter_stats


def mean_std(s, key):
    n = s['count']
    if n == 0:
        return 0, 0
    mean = s[f'{key}_sum'] / n
    var = s[f'{key}_sq_sum'] / n - mean * mean
    return mean, math.sqrt(max(0, var))


def main():
    all_filter_stats = {}
    for asset in ASSETS:
        print(f"Processing {ASSET_LABELS[asset]}...", flush=True)
        all_filter_stats[asset] = streaming_stats(asset)

    # Table: portfolio-level MC summary per (asset, filter).
    summary_rows = []
    for asset in ASSETS:
        for filt in FILTERS:
            s = all_filter_stats[asset].get(filt)
            if not s or s['count'] == 0:
                continue
            n = s['count']
            roi_mean, roi_std = mean_std(s, 'roi')
            sharpe_mean, sharpe_std = mean_std(s, 'sharpe')
            pf_mean = s['pf_sum'] / n
            summary_rows.append({
                'Asset': ASSET_LABELS[asset],
                'Filter': FILTER_LABELS.get(filt, filt),
                'N_Portfolios': n,
                'Mean_ROI_Rank': f'{roi_mean:.1f}',
                'Std_ROI_Rank': f'{roi_std:.1f}',
                'Mean_Sharpe_Rank': f'{sharpe_mean:.1f}',
                'Std_Sharpe_Rank': f'{sharpe_std:.1f}',
                'Mean_PF_Rank': f'{pf_mean:.2f}',
                'Pct_ROI_Above50': f'{s["roi_above50"]/n*100:.1f}',
                'Pct_Sharpe_Above50': f'{s["sharpe_above50"]/n*100:.1f}',
            })

    out1 = TAB / 'portfolio_mc_summary.csv'
    with open(out1, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Saved: {out1}")

    # Table 14 comparison: strategy-level vs portfolio-level mean MC ROI rank.
    comp_rows = []
    for asset in ASSETS:
        strat_path = RAW / f'{asset}_mc_perwindow.csv'
        strat_n = 0
        strat_roi_sum = 0.0
        strat_sharpe_sum = 0.0
        strat_roi_above50 = 0
        if strat_path.exists():
            with open(strat_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    strat_n += 1
                    r = float(row['roi_pct_rank'])
                    strat_roi_sum += r
                    strat_sharpe_sum += float(row['sharpe_pct_rank'])
                    if r > 50:
                        strat_roi_above50 += 1

        ps = all_filter_stats[asset].get('rob_all4', {'count': 0})
        port_n = ps['count']
        port_roi_mean = ps['roi_sum'] / port_n if port_n else 0
        port_sharpe_mean = ps['sharpe_sum'] / port_n if port_n else 0

        comp_rows.append({
            'Asset': ASSET_LABELS[asset],
            'Strat_N': strat_n,
            'Strat_Mean_ROI_Rank':
                f'{strat_roi_sum/strat_n:.1f}' if strat_n else 'N/A',
            'Strat_Mean_Sharpe_Rank':
                f'{strat_sharpe_sum/strat_n:.1f}' if strat_n else 'N/A',
            'Strat_Pct_ROI_Above50':
                f'{strat_roi_above50/strat_n*100:.1f}' if strat_n else 'N/A',
            'Port_N': port_n,
            'Port_Mean_ROI_Rank': f'{port_roi_mean:.1f}',
            'Port_Mean_Sharpe_Rank': f'{port_sharpe_mean:.1f}',
            'Port_Pct_ROI_Above50':
                f'{ps["roi_above50"]/port_n*100:.1f}' if port_n else 'N/A',
        })
        print(f"  {ASSET_LABELS[asset]:>10}: "
              f"Strat ROI rank={strat_roi_sum/strat_n:.1f}, "
              f"Port ROI rank={port_roi_mean:.1f}")

    out2 = TAB / 'strat_vs_portfolio_mc.csv'
    with open(out2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=comp_rows[0].keys())
        w.writeheader()
        w.writerows(comp_rows)
    print(f"  Saved: {out2}")


if __name__ == '__main__':
    main()
