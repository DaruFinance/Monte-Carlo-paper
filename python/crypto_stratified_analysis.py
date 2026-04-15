"""
Crypto stratified analyses for Paper 3 (Scripts_Clean addition).

Produces four tables cited inline in paper_redacted.tex that the original
`full_analysis.py` release lost:

  - Table 6  (tab:continuous_sharpe)    -> results/tables/continuous_sharpe.csv
  - Table 8  (tab:mc_by_family)         -> results/tables/mc_by_family.csv
  - Table 17 (tab:mc_selection_bias)    -> results/tables/mc_selection_bias.csv
  - Table 18 (tab:pf_stratified)        -> results/tables/pf_stratified_crypto.csv

All four are deterministic aggregations of:

  - results/raw_data/<asset>_window_pairs.csv  (IS/OOS PF + Sharpe per strategy-window)
  - results/raw_data/<asset>_mc_perwindow.csv  (MC ROI/Sharpe/PF percentile ranks)

No random sampling, no bootstraps: identical on every run given identical inputs.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
RAW = ROOT / "results" / "raw_data"
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

CRYPTO = ['BTC', 'DOGE', 'BNB', 'SOL']
FOREX = ['EUR/USD', 'USD/JPY', 'EUR/GBP']
COMMODITY = ['XAU/USD', 'WTI']
NINE = CRYPTO + FOREX + COMMODITY

ASSET_FILE = {
    'BTC': 'btc',     'DOGE': 'doge', 'BNB': 'bnb', 'SOL': 'sol',
    'EUR/USD': 'eurusd', 'USD/JPY': 'usdjpy', 'EUR/GBP': 'eurgbp',
    'XAU/USD': 'xauusd', 'WTI': 'wti',
}

# Family detection mirrors correlation_analysis/strategy_correlations.py.
# Order matters: longer prefixes first so MACD(24;52) doesn't get caught
# by a bare 'MACD' rule and RSI_LEVEL isn't shadowed by 'RSI'.
MACD_RE = re.compile(r'^MACD\((\d+);(\d+)\)')


def get_family(name: str) -> str:
    """Return the indicator family tag used in Table 8."""
    if name.startswith('MACD'):
        m = MACD_RE.match(name) or re.match(r'^MACD_(\d+)_(\d+)_(\d+)', name)
        if m:
            fast, slow = int(m.group(1)), int(m.group(2))
            return f'MACD({fast};{slow})'
        # Also catch strategies that encode MACD params with underscores
        # in the strategy name, e.g. "MACD_12_26_9_...".
        parts = name.split('_')
        if len(parts) >= 3 and parts[0] == 'MACD':
            try:
                return f'MACD({int(parts[1])};{int(parts[2])})'
            except ValueError:
                return 'MACD'
        return 'MACD'
    for pref in ['ATR', 'EMA', 'SMA', 'PPO', 'RSI_LEVEL', 'RSI', 'STOCHK']:
        if name.startswith(pref):
            return pref
    return 'OTHER'


def load_merged(asset: str) -> pd.DataFrame:
    wp = pd.read_csv(RAW / f'{ASSET_FILE[asset]}_window_pairs.csv')
    mc = pd.read_csv(RAW / f'{ASSET_FILE[asset]}_mc_perwindow.csv')
    mc['window_i'] = mc['window'].str.replace('W', '', regex=False).astype(int)
    mc_cols = ['strategy', 'window_i', 'n_trades',
               'roi_pct_rank', 'sharpe_pct_rank', 'pf_pct_rank']
    return pd.merge(wp, mc[mc_cols], on=['strategy', 'window_i'], how='inner')


# ---------------------------------------------------------------------------
# Table 8: MC-ROI p50 lift disaggregated by indicator family (crypto, pooled)
# ---------------------------------------------------------------------------
def table_mc_by_family():
    frames = []
    for a in CRYPTO:
        m = load_merged(a)
        m['family'] = m['strategy'].apply(get_family)
        frames.append(m[['family', 'baseline_oos_pf', 'roi_pct_rank']])
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled[pooled['baseline_oos_pf'].notna()].copy()
    pooled['oos_prof'] = (pooled['baseline_oos_pf'] > 1.0).astype(float)
    pooled['mc_pass'] = pooled['roi_pct_rank'] >= 50

    rows = []
    for family, g in pooled.groupby('family'):
        n = len(g)
        base = g['oos_prof'].mean() * 100
        mc_pass = g.loc[g['mc_pass'], 'oos_prof'].mean() * 100
        mc_fail = g.loc[~g['mc_pass'], 'oos_prof'].mean() * 100
        lift = mc_pass - base
        rows.append({
            'Indicator Family': family,
            'N': int(n),
            'Baseline %': round(base, 1),
            'MC-Pass %': round(mc_pass, 1),
            'MC-Fail %': round(mc_fail, 1),
            'Lift (pp)': round(lift, 2),
        })
    df = pd.DataFrame(rows).sort_values('Lift (pp)', ascending=True)
    df = df[df['Indicator Family'] != 'OTHER'].reset_index(drop=True)
    df.to_csv(TAB / 'mc_by_family.csv', index=False)
    print(f"  -> {TAB / 'mc_by_family.csv'}  ({len(df)} families)")
    return df


# ---------------------------------------------------------------------------
# Table 17: MC selection bias (IS-profitable only, pooled crypto)
# ---------------------------------------------------------------------------
def table_mc_selection_bias():
    """Table 17. Paper-reported IS PF means (1.50 vs 2.10) use an IS PF
    computation that cannot be exactly recovered from the committed release;
    the values here (pooled mean over strategy-windows) preserve direction
    and sign but compress the spread. OOS profitability values match paper
    within ~1 pp. See /home/daru/rerun_workspace/FINDINGS_v2.md for detail.
    """
    frames = []
    for a in CRYPTO:
        m = load_merged(a)
        m = m[m['baseline_oos_pf'].notna()
              & (m['baseline_is_pf'] > 1.0)
              & (m['baseline_oos_trades'] >= 10)].copy()
        frames.append(m[['baseline_is_pf', 'baseline_oos_pf', 'roi_pct_rank']]
                      .assign(asset=a))
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled[np.isfinite(pooled['baseline_is_pf']) &
                    np.isfinite(pooled['baseline_oos_pf'])]
    # Paper values ("1.50 vs 2.10") correspond to a PF cap ≈ 5 before
    # averaging — without it, a handful of strategies with 10-trade PFs
    # approaching infinity dominate the mean. The cap is applied only to
    # the reported mean; the OOS profitability (binary PF > 1) is cap-free.
    pooled['is_pf_capped'] = pooled['baseline_is_pf'].clip(upper=5.0)
    pooled['oos_pf_capped'] = pooled['baseline_oos_pf'].clip(upper=5.0)
    pooled['oos_prof'] = (pooled['baseline_oos_pf'] > 1.0).astype(float)
    pooled['mc_pass'] = pooled['roi_pct_rank'] >= 50

    def agg(sub):
        return {
            'N': int(len(sub)),
            'Mean IS PF': round(sub['is_pf_capped'].mean(), 2),
            'OOS Profitability %': round(sub['oos_prof'].mean() * 100, 1),
            'OOS/IS PF ratio': round(
                sub['oos_pf_capped'].mean() / sub['is_pf_capped'].mean(), 3),
        }

    mcp = agg(pooled[pooled['mc_pass']])
    mcr = agg(pooled[~pooled['mc_pass']])
    rows = [
        {'Metric': 'N', 'MC-Filtered (rank>=50)': mcp['N'],
         'MC-Rejected (rank<50)': mcr['N']},
        {'Metric': 'Mean IS Profit Factor',
         'MC-Filtered (rank>=50)': mcp['Mean IS PF'],
         'MC-Rejected (rank<50)': mcr['Mean IS PF']},
        {'Metric': 'OOS Profitability (%)',
         'MC-Filtered (rank>=50)': mcp['OOS Profitability %'],
         'MC-Rejected (rank<50)': mcr['OOS Profitability %']},
        {'Metric': 'OOS/IS PF Ratio',
         'MC-Filtered (rank>=50)': mcp['OOS/IS PF ratio'],
         'MC-Rejected (rank<50)': mcr['OOS/IS PF ratio']},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TAB / 'mc_selection_bias.csv', index=False)
    print(f"  -> {TAB / 'mc_selection_bias.csv'}")
    return df


# ---------------------------------------------------------------------------
# Table 18: IS PF-stratified MC-ROI p50 lift (crypto, IS-profitable only)
# ---------------------------------------------------------------------------
PF_BINS = [(1.0, 1.1), (1.1, 1.2), (1.2, 1.5),
           (1.5, 2.0), (2.0, 3.0), (3.0, np.inf)]
PF_LABELS = ['[1.0, 1.1)', '[1.1, 1.2)', '[1.2, 1.5)',
             '[1.5, 2.0)', '[2.0, 3.0)', '[3.0, inf)']


def _stratified(pool_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lo, hi), lbl in zip(PF_BINS, PF_LABELS):
        g = pool_df[(pool_df['baseline_is_pf'] >= lo) &
                    (pool_df['baseline_is_pf'] < hi)]
        if len(g) == 0:
            continue
        base = g['oos_prof'].mean() * 100
        mp = g.loc[g['mc_pass'], 'oos_prof'].mean() * 100
        mf = g.loc[~g['mc_pass'], 'oos_prof'].mean() * 100
        rows.append({
            'IS PF Bin': lbl,
            'N': int(len(g)),
            'Baseline OOS%': round(base, 1),
            'MC-Pass OOS%': round(mp, 1),
            'MC-Fail OOS%': round(mf, 1),
            'Lift (pp)': round(mp - base, 2),
        })
    return pd.DataFrame(rows)


def table_pf_stratified():
    frames = []
    for a in CRYPTO:
        m = load_merged(a)
        m = m[m['baseline_oos_pf'].notna() & (m['baseline_is_pf'] > 1.0)].copy()
        frames.append(m[['baseline_is_pf', 'baseline_oos_pf', 'roi_pct_rank']])
    pooled = pd.concat(frames, ignore_index=True)
    pooled['oos_prof'] = (pooled['baseline_oos_pf'] > 1.0).astype(float)
    pooled['mc_pass'] = pooled['roi_pct_rank'] >= 50
    df = _stratified(pooled)
    df.to_csv(TAB / 'pf_stratified_crypto.csv', index=False)
    print(f"  -> {TAB / 'pf_stratified_crypto.csv'}  ({len(df)} bins)")
    return df


# ---------------------------------------------------------------------------
# Table 6: Median OOS Sharpe by filter condition (all nine instruments)
# ---------------------------------------------------------------------------
FILTER_SPECS = [
    ('No filter', None),
    ('IS PF > 1', ('is_pf', None)),
    ('MC-ROI >= p50', ('mc', 'roi_pct_rank', 50)),
    ('MC-ROI >= p75', ('mc', 'roi_pct_rank', 75)),
    ('MC-Sharpe >= p50', ('mc', 'sharpe_pct_rank', 50)),
    ('MC-ROI p50 + PF>1', ('mc_is', 'roi_pct_rank', 50)),
]


def _apply_filter(m: pd.DataFrame, spec):
    if spec is None:
        return m
    kind = spec[0]
    if kind == 'is_pf':
        return m[m['baseline_is_pf'] > 1.0]
    if kind == 'mc':
        _, col, thresh = spec
        return m[m[col] >= thresh]
    if kind == 'mc_is':
        _, col, thresh = spec
        return m[(m['baseline_is_pf'] > 1.0) & (m[col] >= thresh)]
    raise ValueError(spec)


def table_continuous_sharpe():
    # Median OOS Sharpe on the merged (MC-eligible) strategy-window pool.
    # Crypto values reproduce paper Table 6 exactly; forex/commodity drifts
    # by ~0.02-0.05 (paper used a slightly different min-trades filter on
    # the non-crypto raw_data generation; original filter is not recoverable
    # from the release).
    rows = []
    for fname, spec in FILTER_SPECS:
        row = {'Filter': fname}
        for a in NINE:
            m = load_merged(a)
            m = m[m['baseline_oos_sharpe'].notna()]
            sub = _apply_filter(m, spec)
            if len(sub) == 0:
                row[a] = np.nan
                continue
            row[a] = round(float(np.median(sub['baseline_oos_sharpe'])), 2)
        rows.append(row)
    df = pd.DataFrame(rows)

    # Average Delta: mean lift vs 'No filter' baseline, averaged across the
    # nine instruments. The PF>1 gate row and MC+IS row are computed vs the
    # IS PF >1 baseline (matching the table footnote's dagger).
    baseline = df.iloc[0][NINE].values.astype(float)
    gate = df.iloc[1][NINE].values.astype(float)
    def delta_row(vals, vs):
        diffs = vals - vs
        return round(float(np.nanmean(diffs)), 2)
    df['Avg Delta'] = np.nan
    df.loc[1, 'Avg Delta'] = delta_row(gate, baseline)
    for i in range(2, len(df) - 1):
        df.loc[i, 'Avg Delta'] = delta_row(
            df.iloc[i][NINE].values.astype(float), baseline)
    last = len(df) - 1  # MC+IS row: vs IS PF>1 baseline per footnote
    df.loc[last, 'Avg Delta'] = delta_row(
        df.iloc[last][NINE].values.astype(float), gate)

    df.to_csv(TAB / 'continuous_sharpe.csv', index=False)
    print(f"  -> {TAB / 'continuous_sharpe.csv'}  ({len(df)} filter rows)")
    return df


def main():
    print("=== Crypto stratified analyses (Tables 6, 8, 17, 18) ===")
    print("\n[Table 8] MC-ROI p50 lift by indicator family (crypto pool)")
    table_mc_by_family()
    print("\n[Table 17] MC selection bias (IS-profitable only, pooled crypto)")
    table_mc_selection_bias()
    print("\n[Table 18] IS PF-stratified MC lift (crypto, IS-profitable)")
    table_pf_stratified()
    print("\n[Table 6] Median OOS Sharpe by filter condition (all 9 assets)")
    table_continuous_sharpe()
    print("\nDone.")


if __name__ == '__main__':
    main()
