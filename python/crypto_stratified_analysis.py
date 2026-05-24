"""
Rebuild stratified-analysis tables that previously consumed the broken MC rank:

  - Table 8  : MC by family (mean MC rank within each technical-indicator family)
  - Table 17 : MC selection bias (does the MC filter select for IS-PF-passers?)
  - Table 18 : PF stratified crypto (OOS PF distribution stratified by MC filter)
  - Table 16 : cost sensitivity (filter lift under increased fees/slippage)

Outputs land in results/tables/. We use the path-dependent MC-MDD and
MC-Calmar ranks throughout, with the artefactual MC-ROI* shown for transparency.

Requires results/raw_data/<asset>_corrected_ranks.csv + <asset>_window_pairs.csv.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ASSETS = [("BTC","btc"), ("DOGE","doge"), ("BNB","bnb"), ("SOL","sol"),
          ("EUR/USD","eurusd"), ("USD/JPY","usdjpy"), ("EUR/GBP","eurgbp"),
          ("XAU/USD","xauusd"), ("WTI","wti")]
FAMILIES = ["ATR", "EMA", "MACD", "PPO", "RSI", "RSI_LEVEL", "SMA", "STOCHK"]


def family_of(strategy_name: str) -> str:
    """Recover indicator family from strategy name. Strategies are typically
    named '<FAMILY>_<...>' or '<F1>_x_<F2>_<...>'; we use the leading token."""
    s = strategy_name.strip().strip('"')
    head = re.split(r"[_\-]", s)[0]
    head = head.upper()
    # special handling: RSI_LEVEL appears as 'RSI' + ... + 'LEVEL' or as a distinct family
    if head == "RSI" and "_LEVEL" in s.upper():
        return "RSI_LEVEL"
    return head if head in FAMILIES else "OTHER"


def load_merged(short: str) -> pd.DataFrame:
    rp = DATA / f"{short}_corrected_ranks.csv"
    wp = DATA / f"{short}_window_pairs.csv"
    if not rp.exists() or not wp.exists(): return pd.DataFrame()
    r = pd.read_csv(rp)
    r["window_i"] = r["window"].str.replace("W", "").astype(int)
    w = pd.read_csv(wp)
    df = pd.merge(w, r[["strategy","window_i","n_trades","actual_roi","actual_mdd",
                        "actual_calmar","actual_ulcer","roi_rank_broken","mdd_rank",
                        "calmar_rank","ulcer_rank"]], on=["strategy","window_i"])
    df["family"] = df["strategy"].apply(family_of)
    return df


# ----------------------------------------------------------------------
# Table 8: MC rank means by strategy family
# ----------------------------------------------------------------------
def table8_mc_by_family():
    rows = []
    for asset, short in ASSETS:
        m = load_merged(short)
        if len(m) == 0: continue
        for fam, g in m.groupby("family"):
            rows.append({
                "Asset": asset, "Family": fam, "N strat-windows": len(g),
                "Mean MC-MDD":     round(g["mdd_rank"].mean(), 2),
                "Mean MC-Calmar":  round(g["calmar_rank"].mean(), 2),
                "Mean MC-Ulcer":   round(g["ulcer_rank"].mean(), 2),
                "Mean MC-ROI*":    round(g["roi_rank_broken"].mean(), 2),
                "Same-win OOS Prof%": round((g["baseline_oos_pf"] > 1).mean()*100, 2),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table8_mc_by_family_corrected.csv", index=False)
    print(f"→ {OUT / 'table8_mc_by_family_corrected.csv'}  ({len(df)} rows)")
    return df


# ----------------------------------------------------------------------
# Table 17: MC selection bias
# Does the MC filter preferentially keep strategies that also pass IS PF>1?
# If MC carries information independent of IS profitability, the pass rate
# inside the IS-PF>1 bucket should equal the pass rate inside the IS-PF<=1
# bucket. Departures indicate that MC is partially a proxy for IS PF.
# ----------------------------------------------------------------------
def table17_mc_selection_bias():
    rows = []
    for asset, short in ASSETS:
        m = load_merged(short).dropna(subset=["baseline_is_pf"])
        if len(m) == 0: continue
        bl_pos = m["baseline_is_pf"] > 1.0
        bl_neg = ~bl_pos
        for col, mname in [("mdd_rank","MDD"), ("calmar_rank","Calmar"),
                           ("ulcer_rank","Ulcer"), ("roi_rank_broken","ROI*")]:
            for thr in (50, 75, 90):
                pass_in_pos = (m.loc[bl_pos, col] >= thr).mean() * 100
                pass_in_neg = (m.loc[bl_neg, col] >= thr).mean() * 100
                rows.append({
                    "Asset": asset, "MC Metric": mname, "Threshold": thr,
                    "Pass rate | IS PF>1 (%)": round(pass_in_pos, 2),
                    "Pass rate | IS PF<=1 (%)": round(pass_in_neg, 2),
                    "Diff (pp)": round(pass_in_pos - pass_in_neg, 2),
                    "Selection bias ratio": round(pass_in_pos / pass_in_neg, 3)
                                            if pass_in_neg > 1e-6 else np.nan,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table17_mc_selection_bias_corrected.csv", index=False)
    print(f"→ {OUT / 'table17_mc_selection_bias_corrected.csv'}  ({len(df)} rows)")
    return df


# ----------------------------------------------------------------------
# Table 18: OOS PF distribution stratified by MC filter
# ----------------------------------------------------------------------
def table18_pf_stratified():
    rows = []
    for asset, short in ASSETS:
        m = load_merged(short).dropna(subset=["baseline_oos_pf"])
        if len(m) == 0: continue
        for col, mname in [("mdd_rank","MDD"), ("calmar_rank","Calmar"),
                           ("ulcer_rank","Ulcer"), ("roi_rank_broken","ROI*")]:
            for cond_name, cond in [
                (f"{mname} all",   pd.Series(True, index=m.index)),
                (f"{mname} ≥ p50", m[col] >= 50),
                (f"{mname} ≥ p75", m[col] >= 75),
                (f"{mname} ≥ p90", m[col] >= 90),
                (f"{mname} < p50", m[col] < 50),
            ]:
                sub = m[cond]
                if len(sub) == 0: continue
                pf = sub["baseline_oos_pf"]
                rows.append({
                    "Asset": asset, "Filter": cond_name, "N": len(sub),
                    "Median OOS PF": round(pf.median(), 4),
                    "Mean OOS PF":   round(pf.mean(), 4),
                    "% OOS PF>1":    round((pf > 1).mean()*100, 2),
                    "Median OOS Sharpe": round(sub["baseline_oos_sharpe"].median(), 4),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table18_pf_stratified_corrected.csv", index=False)
    print(f"→ {OUT / 'table18_pf_stratified_corrected.csv'}  ({len(df)} rows)")
    return df


# ----------------------------------------------------------------------
# Table 16: cost sensitivity — DOWNGRADED to a placeholder
# Real cost sensitivity requires re-running strategies with perturbed
# fee/slippage. We DON'T have those columns in trades.bin. Producing a
# diagnostic note instead.
# ----------------------------------------------------------------------
def table16_cost_sensitivity_note():
    note = (
        "table16_cost_sensitivity: NOT REBUILT.\n\n"
        "The original Table 16 used the {ent, fee, sli, entind}_is_pf perturbation\n"
        "columns produced by re-running each strategy with elevated fee/slippage\n"
        "parameters. Those columns are NOT in the toolkit's window_pairs CSVs\n"
        "(they require the upstream backtester). To rebuild Table 16 in the\n"
        "corrected paper, regenerate <asset>_window_pairs.csv with the ent/fee/\n"
        "sli/entind perturbations included, then re-run full_analysis.py\n"
        "with the appropriate filter rows enabled (they exist in the code but\n"
        "are skipped when the perturbation columns are absent).\n"
    )
    (OUT / "table16_cost_sensitivity_NOTE.txt").write_text(note)
    print(f"→ {OUT / 'table16_cost_sensitivity_NOTE.txt'}  (note only — needs upstream backtester)")


if __name__ == "__main__":
    table8_mc_by_family()
    table17_mc_selection_bias()
    table18_pf_stratified()
    table16_cost_sensitivity_note()
