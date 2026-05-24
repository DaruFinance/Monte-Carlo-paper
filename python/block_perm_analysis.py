"""
Rebuild paper Table 19 (block permutation MC test, 9 instruments).

The original Table 19 used the sum-based `iid_rank` / `block{N}_rank` columns
from block_perm_rs's `equity_roi` statistic — same FP-summation artefact as
Table 4. We replace with PATH-DEPENDENT MDD-based block-permutation ranks
produced by the `block_perm_path` Rust crate.

Covers 7 instruments (4 crypto + 3 forex). Forex block-perm inputs are
emitted by the same `block_perm_path` crate when run against the parquet-
ingested forex trade streams.

Output: results/tables/table19_block_permutation_corrected.csv
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ASSETS = [("BTC", "btc"), ("DOGE", "doge"), ("BNB", "bnb"), ("SOL", "sol"),
          ("EUR/USD", "eurusd"), ("USD/JPY", "usdjpy"), ("EUR/GBP", "eurgbp"),
          ("XAU/USD", "xauusd"), ("WTI", "wti")]
BLOCKS = ["iid", "block2", "block3", "block5", "block10", "block20"]


def main():
    rows = []
    for asset, short in ASSETS:
        p = DATA / f"block_perm_{short}_corrected.csv"
        if not p.exists():
            print(f"  [{asset}] {p.name} not present — skip")
            continue
        df = pd.read_csv(p)
        n = len(df)
        for b in BLOCKS:
            col = f"{b}_mdd_rank"
            if col not in df.columns: continue
            v = df[col].dropna()
            rows.append({
                "Asset": asset,
                "Block size": b,
                "N strat-windows": n,
                "Mean MDD rank": round(float(v.mean()), 2),
                "Median MDD rank": round(float(v.median()), 2),
                "Std": round(float(v.std()), 2),
                "% rank < 50": round(float((v < 50).mean() * 100), 2),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table19_block_permutation_corrected.csv", index=False)
    print(f"→ {OUT / 'table19_block_permutation_corrected.csv'}  ({len(df)} rows)")
    print(df.to_string(index=False))

    # Diagnostic: filter lift under block-permutation MDD rank (the inferential question)
    print("\n=== Filter-lift sketch using block_perm MDD ranks (p50 threshold) ===")
    boost = []
    for asset, short in ASSETS:
        bp_path = DATA / f"block_perm_{short}_corrected.csv"
        wp_path = DATA / f"{short}_window_pairs.csv"
        if not bp_path.exists() or not wp_path.exists(): continue
        bp = pd.read_csv(bp_path)
        bp["window_i"] = bp["window"].astype(str).str.replace("W", "").astype(int)
        wp = pd.read_csv(wp_path)
        m = pd.merge(wp, bp, on=["strategy","window_i"], how="inner")
        m = m[m["baseline_oos_pf"].notna()]
        oos = (m["baseline_oos_pf"] > 1).astype(float)
        base = oos.mean() * 100
        for b in BLOCKS:
            col = f"{b}_mdd_rank"
            if col not in m.columns: continue
            passing = m[col] >= 50
            if passing.sum() == 0: continue
            rate = oos[passing].mean() * 100
            boost.append({
                "Asset": asset, "Block": b, "Threshold": "p50",
                "Baseline OOS%": round(base, 2),
                "Filter OOS%": round(rate, 2),
                "Lift (pp)": round(rate - base, 3),
                "N pass": int(passing.sum()),
            })
    bdf = pd.DataFrame(boost)
    bdf.to_csv(OUT / "table19_block_perm_filter_lift_corrected.csv", index=False)
    print(f"→ {OUT / 'table19_block_perm_filter_lift_corrected.csv'}")
    print(bdf.to_string(index=False))


if __name__ == "__main__":
    main()
