"""
Portfolio-level MC analysis for the paper revision.

Produces Tables 14, 14b/c/d, and the top-N portfolio MC tables.

Inputs (relative to project root, override with MC_PAPER_DATA env var):
  - results/raw_data/<asset>_portfolio_mc_path.csv  (from rust/portfolio_mc_path)
  - results/raw_data/<asset>_portfolio_mc_oos.csv   (from rust/portfolio_mc_oos)
  - results/raw_data/<asset>_window_pairs.csv

Outputs:
  - results/tables/table14_portfolio_mc_corrected.csv
  - results/tables/table14b_portfolio_oos_stratified_mc_{mdd,calmar,roistar}.csv
  - results/tables/table14c_portfolio_oos_topbottom.csv
  - results/tables/table14d_portfolio_oos_stratified_pooled.csv
  - results/tables/topn_portfolio_mc{,_summary,_floor30,_floor30_summary}.csv
"""
import os  # noqa: F401

# ======================================================================
# === Section: Portfolio MC rebuild (per-asset) ===
# ======================================================================

"""
Rebuild paper Table 14 (Portfolio MC) using corrected path-dependent ranks.

We construct random portfolios of size K from the IS-PF>1 pool inside each
window, build the additive equity curve, and compute MC ranks for:
  - mc_mdd_rank     : rank of actual portfolio MDD vs MDD of permuted-order portfolios
  - mc_calmar_rank  : analogue using portfolio Calmar
  - mc_roi_rank_broken : artefactual rank kept for transparency (FP only)

Per asset we sample N_PORTFOLIOS random portfolios in each walk-forward
window (matching the paper's protocol). MC inside each portfolio is via
permutation of the COMBINED trade stream.

Output: results/tables/table14_portfolio_mc_corrected.csv

Runtime: ~5 min on 24 cores per asset (paper uses K=10 portfolios from large pool).
"""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, random, math

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

INIT = 1000.0
N_PORT = 1000     # random portfolios per window (paper used 10k; 1k is fine for stable percentiles)
PORT_SIZE = 10    # strategies per portfolio (matches paper)
N_MC = 200        # MC permutations per portfolio
MIN_TRADES = 5    # min combined trades in a window for portfolio MC

ASSETS = [
    ("BTC",  "btc",  os.environ.get("BTC_STRATS_DIR", "")),
    ("DOGE", "doge", os.environ.get("DOGE_STRATS_DIR", "")),
    ("BNB",  "bnb",  os.environ.get("BNB_STRATS_DIR", "")),
    ("SOL",  "sol",  os.environ.get("SOL_STRATS_DIR", "")),
]


def read_is_sections(path):
    """Return dict[window_int] -> np.ndarray of IS pnls."""
    data = Path(path).read_bytes()
    pos = 0
    n = len(data)
    out = {}
    while pos + 2 <= n:
        nl = struct.unpack_from("<H", data, pos)[0]; pos += 2
        if nl == 0 or pos+nl > n: break
        pos += nl
        if pos+2 > n: break
        lbl = struct.unpack_from("<H", data, pos)[0]; pos += 2
        pos += lbl
        if pos+2 > n: break
        sl = struct.unpack_from("<H", data, pos)[0]; pos += 2
        sec = data[pos:pos+sl].decode("utf-8", errors="ignore")
        pos += sl
        if pos+4 > n: break
        cnt = struct.unpack_from("<I", data, pos)[0]; pos += 4
        pnls = np.empty(cnt, dtype=np.float64)
        ok = True
        for i in range(cnt):
            if pos+17 > n: ok = False; break
            pnls[i] = struct.unpack_from("<d", data, pos+9)[0]
            pos += 17
        if not ok: break
        if sec.endswith("-IS"):
            try: out[int(sec[1:].split("-")[0])] = pnls
            except ValueError: pass
    return out


def mdd_calmar_roi(pnls):
    eq = INIT
    peak = INIT
    max_dd = 0.0
    for p in pnls:
        eq += p
        if eq > peak: peak = eq
        dd = peak - eq
        if dd > max_dd: max_dd = dd
    roi = (eq - INIT) / INIT * 100
    cal = roi / (max_dd / INIT * 100) if max_dd > 1e-10 else 0.0
    return max_dd, cal, roi


def portfolio_mc_ranks(combined_pnls, n_mc, rng):
    a_mdd, a_cal, a_roi = mdd_calmar_roi(combined_pnls)
    n = len(combined_pnls)
    work = combined_pnls.copy()
    c_mdd = 0; c_cal = 0; c_roi = 0
    for _ in range(n_mc):
        rng.shuffle(work)
        m, c, r = mdd_calmar_roi(work)
        if a_mdd < m: c_mdd += 1   # actual is less bad
        if c < a_cal: c_cal += 1
        if r < a_roi: c_roi += 1   # the buggy one
    return c_mdd*100/n_mc, c_cal*100/n_mc, c_roi*100/n_mc


def process_window(args):
    asset, win, strat_names, paths, n_port, port_size, n_mc, seed = args
    rng = np.random.default_rng(seed)
    ports = [rng.choice(strat_names, size=port_size, replace=False) for _ in range(n_port)]
    unique = set()
    for p in ports: unique.update(p)
    cache = {}
    for s in unique:
        p = paths.get(s)
        if p is None: continue
        secs = read_is_sections(p)
        arr = secs.get(win)
        if arr is not None and len(arr) >= 5:
            cache[s] = arr
    rng_mc = np.random.default_rng(seed + 7)
    rows = []
    for pi, port in enumerate(ports):
        pnls_list = [cache[s] for s in port if s in cache]
        if len(pnls_list) < port_size - 2: continue
        combined = np.concatenate(pnls_list)
        if len(combined) < MIN_TRADES: continue
        m, c, r2 = portfolio_mc_ranks(combined, n_mc, rng_mc)
        a_mdd, a_cal, a_roi = mdd_calmar_roi(combined)
        rows.append({
            "asset": asset, "window": int(win), "port_id": pi,
            "n_trades_combined": len(combined),
            "actual_roi": float(a_roi),
            "actual_mdd": float(a_mdd),
            "actual_calmar": float(a_cal),
            "port_mdd_rank": float(m),
            "port_calmar_rank": float(c),
            "port_roi_rank_broken": float(r2),
        })
    return rows


def process_asset(args):
    asset, short, base_dir = args
    rp = DATA / f"{short}_corrected_ranks.csv"
    wp = DATA / f"{short}_window_pairs.csv"
    if not rp.exists() or not wp.exists():
        return asset, pd.DataFrame()
    r = pd.read_csv(rp); r["window_i"] = r["window"].str.replace("W","").astype(int)
    w = pd.read_csv(wp)
    merged = pd.merge(w, r[["strategy","window_i"]], on=["strategy","window_i"])
    isp = merged[merged["baseline_is_pf"] > 1.0]
    windows = sorted(isp["window_i"].unique())
    paths = {}
    for fam in Path(base_dir).iterdir():
        if not fam.is_dir(): continue
        for strat in fam.iterdir():
            tb = strat / "trades.bin"
            if tb.exists():
                paths[strat.name] = str(tb)
    work = []
    for win in windows:
        pool = isp[isp["window_i"] == win]
        if len(pool) < PORT_SIZE: continue
        strat_names = pool["strategy"].tolist()
        work.append((asset, win, strat_names, paths, N_PORT, PORT_SIZE, N_MC,
                     int(win * 9931 + (hash(asset) & 0xFFFFFF))))
    rows = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        for r in ex.map(process_window, work):
            rows.extend(r)
    return asset, pd.DataFrame(rows)


def main():
    all_rows = []
    # Run assets sequentially (each uses 24 workers; running in parallel oversubscribes)
    for arg in ASSETS:
        asset, df = process_asset(arg)
        if len(df) == 0:
            print(f"  [{asset}] no data — skip")
            continue
        all_rows.append(df)
        print(f"  [{asset}] {len(df)} portfolios MC-ranked")
    if not all_rows:
        print("No portfolios produced.")
        return
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(DATA / "portfolio_mc_corrected_full.csv", index=False)
    print(f"→ {DATA / 'portfolio_mc_corrected_full.csv'}")

    # Summary table (paper Table 14 analogue)
    summary = []
    for asset, g in out.groupby("asset"):
        for col, mname in [("port_mdd_rank","Portfolio MC-MDD"),
                           ("port_calmar_rank","Portfolio MC-Calmar"),
                           ("port_roi_rank_broken","Portfolio MC-ROI*")]:
            v = g[col].dropna()
            summary.append({
                "Asset": asset, "Metric": mname, "N portfolios": len(v),
                "Mean rank": round(float(v.mean()), 2),
                "Median rank": round(float(v.median()), 2),
                "Std": round(float(v.std()), 2),
                "% <50": round(float((v<50).mean()*100), 2),
            })
    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "table14_portfolio_mc_corrected.csv", index=False)
    print(f"→ {OUT / 'table14_portfolio_mc_corrected.csv'}")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    # Use multiprocessing inside process_asset? Currently single-process per asset.
    # For speed: parallelise across portfolios within each asset.
    # We keep single-process to avoid over-subscription with the other background jobs.
    main()

# ======================================================================
# === Section: Table 14 portfolio MC aggregation ===
# ======================================================================

"""
Build paper Table 14 (Portfolio MC) summary from the Rust portfolio_mc_path
outputs.
"""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ASSETS = [("BTC", "btc"), ("DOGE", "doge"), ("BNB", "bnb"), ("SOL", "sol"),
          ("EURUSD", "eurusd"), ("USDJPY", "usdjpy"), ("EURGBP", "eurgbp"),
          ("XAUUSD", "xauusd"), ("WTI", "wti")]


def main():
    all_df = []
    for asset, short in ASSETS:
        p = DATA / f"portfolio_mc_{short}_corrected.csv"
        if not p.exists():
            print(f"  [{asset}] {p.name} missing — skip")
            continue
        df = pd.read_csv(p)
        all_df.append(df)
    if not all_df: return
    big = pd.concat(all_df, ignore_index=True)
    big.to_csv(DATA / "portfolio_mc_corrected_all.csv", index=False)

    rows = []
    for asset, _ in ASSETS:
        g = big[big["asset"] == asset]
        if len(g) == 0: continue
        for col, mname in [("port_mdd_rank", "Portfolio MC-MDD"),
                           ("port_calmar_rank", "Portfolio MC-Calmar"),
                           ("port_roi_rank_broken", "Portfolio MC-ROI*")]:
            v = g[col].dropna()
            rows.append({
                "Asset": asset, "Metric": mname,
                "N portfolios": len(v),
                "N windows": int(g["window"].nunique()),
                "Mean rank": round(float(v.mean()), 2),
                "Median rank": round(float(v.median()), 2),
                "Std": round(float(v.std()), 2),
                "% < 50": round(float((v < 50).mean() * 100), 2),
                "% > 90 (top decile)": round(float((v > 90).mean() * 100), 2),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table14_portfolio_mc_corrected.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n→ {OUT / 'table14_portfolio_mc_corrected.csv'}")


if __name__ == "__main__":
    main()

# ======================================================================
# === Section: Table 14b/c/d portfolio OOS stratification ===
# ======================================================================

"""
Stratify portfolios by IS portfolio MC-MDD rank decile and report same-window
and next-window OOS portfolio profitability. This is the forward-OOS test
that the strategy-level Section 6 lacks (Reviewer #1's biggest ask).

Consumes the portfolio_mc_oos_<asset>.csv files produced by
`portfolio_mc_oos` Rust binary.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ASSETS = [("BTC", "btc"), ("DOGE", "doge"), ("BNB", "bnb"), ("SOL", "sol"),
          ("EURUSD", "eurusd"), ("USDJPY", "usdjpy"), ("EURGBP", "eurgbp"),
          ("XAUUSD", "xauusd"), ("WTI", "wti")]


def load_all():
    parts = []
    for asset, short in ASSETS:
        p = DATA / f"portfolio_mc_oos_{short}.csv"
        if not p.exists(): continue
        df = pd.read_csv(p)
        df["asset"] = asset
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def stratify_table(df, rank_col="port_mdd_rank", label="MC-MDD"):
    """Decile-stratify portfolios by rank, report same-window + next-window OOS
    profitability across deciles."""
    bins = np.linspace(0, 100, 11)
    labels = [f"D{i+1} ({int(bins[i])}-{int(bins[i+1])})" for i in range(10)]
    df = df.copy()
    df["decile"] = pd.cut(df[rank_col], bins=bins, labels=labels, include_lowest=True)

    rows = []
    for asset, sub in df.groupby("asset"):
        same_total = (sub["oos_same_pf"] > 1.0).mean() * 100
        next_total = (sub["oos_next_pf"] > 1.0).mean() * 100
        for d, dsub in sub.groupby("decile", observed=True):
            if len(dsub) == 0: continue
            same_profitable = (dsub["oos_same_pf"] > 1.0).mean() * 100
            next_profitable_mask = dsub["oos_next_pf"].notna()
            next_profitable = (dsub.loc[next_profitable_mask, "oos_next_pf"] > 1.0).mean() * 100 \
                              if next_profitable_mask.sum() > 0 else np.nan
            rows.append({
                "Asset": asset,
                f"{label} decile": d,
                "N portfolios": len(dsub),
                "Same-win OOS profitable %": round(same_profitable, 2),
                "Same-win lift vs asset baseline (pp)": round(same_profitable - same_total, 2),
                "Next-win OOS profitable %": round(next_profitable, 2),
                "Next-win lift vs asset baseline (pp)": round(next_profitable - next_total, 2),
                "Mean OOS same PF": round(dsub["oos_same_pf"].mean(), 3),
                "Mean OOS next PF": round(dsub.loc[next_profitable_mask, "oos_next_pf"].mean(), 3),
            })
    return pd.DataFrame(rows)


def topbottom_test(df, rank_col="port_mdd_rank"):
    """Top decile vs bottom decile next-window OOS lift, with binomial CI."""
    rows = []
    for asset, sub in df.groupby("asset"):
        sub = sub[sub["oos_next_pf"].notna()].copy()
        if len(sub) == 0: continue
        top_mask = sub[rank_col] >= 90
        bot_mask = sub[rank_col] <= 10
        top_rate = (sub.loc[top_mask, "oos_next_pf"] > 1.0).mean() * 100
        bot_rate = (sub.loc[bot_mask, "oos_next_pf"] > 1.0).mean() * 100
        n_top = int(top_mask.sum())
        n_bot = int(bot_mask.sum())
        # Wilson 95% CI on difference of proportions
        p_top = top_rate / 100
        p_bot = bot_rate / 100
        se_diff = np.sqrt(p_top*(1-p_top)/n_top + p_bot*(1-p_bot)/n_bot) if n_top and n_bot else np.nan
        diff = top_rate - bot_rate
        rows.append({
            "Asset": asset, "Rank metric": rank_col,
            "N top decile": n_top, "N bottom decile": n_bot,
            "Top decile next OOS %": round(top_rate, 2),
            "Bottom decile next OOS %": round(bot_rate, 2),
            "Top - Bottom (pp)": round(diff, 2),
            "SE (pp)": round(se_diff * 100, 2),
            "95% CI": f"[{diff - 1.96*se_diff*100:.2f}, {diff + 1.96*se_diff*100:.2f}]",
        })
    return pd.DataFrame(rows)


def main():
    df = load_all()
    print(f"Total portfolios: {len(df):,}")
    print(f"Have next-window OOS: {df['oos_next_pf'].notna().sum():,}")
    print()

    for rank_col, label in [("port_mdd_rank", "MC-MDD"), ("port_calmar_rank", "MC-Calmar"),
                            ("port_roi_rank_broken", "MC-ROI*")]:
        st = stratify_table(df, rank_col, label)
        suffix = label.replace("*", "star").replace("-", "_").lower()
        out_path = OUT / f"table14b_portfolio_oos_stratified_{suffix}.csv"
        st.to_csv(out_path, index=False)
        print(f"→ {out_path}")

    tb = pd.concat([topbottom_test(df, c) for c in
                    ["port_mdd_rank", "port_calmar_rank", "port_roi_rank_broken"]],
                   ignore_index=True)
    tb.to_csv(OUT / "table14c_portfolio_oos_topbottom.csv", index=False)
    print(f"\n=== Top-vs-bottom decile next-window OOS test ===")
    print(tb.to_string(index=False))

    # Cross-asset pooled
    print(f"\n=== Cross-asset pooled MC-MDD stratification ===")
    pooled_st = stratify_table(df.assign(asset="ALL"), "port_mdd_rank", "MC-MDD")
    pooled_st["Asset"] = "ALL (pooled)"
    print(pooled_st.to_string(index=False))
    pooled_st.to_csv(OUT / "table14d_portfolio_oos_stratified_pooled.csv", index=False)


if __name__ == "__main__":
    main()

# ======================================================================
# === Section: Top-N portfolio MC ===
# ======================================================================

"""
Top-N portfolio MC (new construction).

Construction (per WFO window):
  1. Filter strategies to IS PF > 1 within the asset-class pool.
  2. Sort by IS PF, take top N (N in {10, 15, 20}).
  3. Concatenate their OOS trade pnls into a portfolio trade stream.
  4. Run B=1000 permutation MC on the portfolio's OOS trades; rank the
     actual ordering on MDD, Calmar, Ulcer (path-dependent) and ROI
     (FP-artefactual reference column).

Asset-class configs:
  crypto:    pool = BTC + DOGE + BNB + SOL
  forex:     pool = EUR/USD + USD/JPY + EUR/GBP
  commodity: pool = XAU/USD + WTI
  mixed:     30 strategies total = top 10 from each of {crypto, forex, commodity}
             (only at windows where all three classes have data: W01-W07)

Output:
  results/tables/topn_portfolio_mc.csv  — per (config, N, window) row
  results/tables/topn_portfolio_mc_summary.csv — aggregate per (config, N)
"""
from __future__ import annotations
import os
import struct
from pathlib import Path
from collections import defaultdict
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
OUT = ROOT / "results" / "tables"
INIT = 1000.0
N_MC = 1000
N_TOP_VALUES = [10, 15, 20]

CRYPTO_BASE = {
    "btc":  os.environ.get("BTC_STRATS_DIR", ""),
    "doge": os.environ.get("DOGE_STRATS_DIR", ""),
    "bnb":  os.environ.get("BNB_STRATS_DIR", ""),
    "sol":  os.environ.get("SOL_STRATS_DIR", ""),
}
PARQUETS = {
    "eurusd": os.environ.get("EURUSD_TRADES_PARQUET", ""),
    "usdjpy": os.environ.get("USDJPY_TRADES_PARQUET", ""),
    "eurgbp": os.environ.get("EURGBP_TRADES_PARQUET", ""),
    "xauusd": os.environ.get("XAUUSD_TRADES_PARQUET", ""),
    "wti":    os.environ.get("WTI_TRADES_PARQUET", ""),
}
CLASSES = {
    "crypto":    ["btc", "doge", "bnb", "sol"],
    "forex":     ["eurusd", "usdjpy", "eurgbp"],
    "commodity": ["xauusd", "wti"],
}


# ----------- per-strategy OOS trade loaders -----------
_crypto_path_cache: dict[str, dict[str, str]] = {}

def crypto_strat_path(asset: str, strat: str) -> str | None:
    if asset not in _crypto_path_cache:
        base = Path(CRYPTO_BASE[asset])
        m: dict[str, str] = {}
        for p in base.glob("*/*/trades.bin"):
            m[p.parent.name] = str(p)
        _crypto_path_cache[asset] = m
        print(f"  [{asset}] indexed {len(m):,} strategy folders")
    return _crypto_path_cache[asset].get(strat)


def read_crypto_oos(asset: str, strat: str, window_i: int) -> np.ndarray:
    path = crypto_strat_path(asset, strat)
    if path is None:
        return np.array([], dtype=np.float64)
    data = Path(path).read_bytes()
    pos = 0
    n = len(data)
    target = f"W{window_i:02d}-OOS"
    while pos + 2 <= n:
        nl = struct.unpack_from("<H", data, pos)[0]; pos += 2
        if nl == 0 or pos + nl > n: break
        pos += nl
        if pos + 2 > n: break
        lbl = struct.unpack_from("<H", data, pos)[0]; pos += 2
        pos += lbl
        if pos + 2 > n: break
        sl = struct.unpack_from("<H", data, pos)[0]; pos += 2
        sec = data[pos:pos+sl].decode("utf-8", errors="ignore")
        pos += sl
        if pos + 4 > n: break
        cnt = struct.unpack_from("<I", data, pos)[0]; pos += 4
        if sec == target:
            pnls = np.empty(cnt, dtype=np.float64)
            for i in range(cnt):
                if pos + 17 > n: return np.array([], dtype=np.float64)
                pnls[i] = struct.unpack_from("<d", data, pos+9)[0]
                pos += 17
            return pnls
        pos += cnt * 17
    return np.array([], dtype=np.float64)


_forex_oos_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}

def load_forex_oos_window(asset: str, window_i: int) -> dict[str, np.ndarray]:
    """Load all strategies' OOS trades for one window. Cached."""
    key = (asset, window_i)
    if key in _forex_oos_cache:
        return _forex_oos_cache[key]
    con = duckdb.connect()
    con.execute("SET threads = 4")
    df = con.execute(f"""
        SELECT strategy, pnl
        FROM read_parquet('{PARQUETS[asset]}')
        WHERE sample = 'W{window_i:02d}-OOS'
        ORDER BY strategy
    """).df()
    out: dict[str, np.ndarray] = {}
    for strat, grp in df.groupby("strategy", sort=False):
        out[strat] = grp["pnl"].to_numpy(dtype=np.float64, copy=True)
    _forex_oos_cache[key] = out
    return out


def get_oos(asset: str, strat: str, window_i: int) -> np.ndarray:
    if asset in CRYPTO_BASE:
        return read_crypto_oos(asset, strat, window_i)
    else:
        return load_forex_oos_window(asset, window_i).get(strat, np.array([], dtype=np.float64))


# ----------- candidate pool from window_pairs -----------
def build_class_pool(class_name: str) -> pd.DataFrame:
    """Returns DataFrame with (asset, strategy, window_i, baseline_is_pf, baseline_oos_pf)
    where baseline_is_pf > 1. One row per (asset, strategy, window_i)."""
    frames = []
    for asset in CLASSES[class_name]:
        wp = pd.read_csv(DATA / f"{asset}_window_pairs.csv",
                         usecols=["strategy", "window_i", "baseline_is_pf", "baseline_oos_pf"])
        wp["asset"] = asset
        wp = wp[wp["baseline_is_pf"] > 1.0]
        frames.append(wp)
    return pd.concat(frames, ignore_index=True)


# ----------- MC ranks -----------
def mc_ranks(pnls: np.ndarray, n_mc: int = N_MC, seed: int = 0):
    n = len(pnls)
    if n < 5:
        return dict(n_trades=n, actual_roi=np.nan, actual_mdd=np.nan, actual_calmar=np.nan,
                    actual_ulcer=np.nan, mdd_rank=np.nan, calmar_rank=np.nan,
                    ulcer_rank=np.nan, roi_rank_broken=np.nan)
    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((n_mc, n)), axis=1)
    sh = pnls[perms]
    eq_a = INIT + np.cumsum(pnls)
    peak_a = np.maximum.accumulate(eq_a)
    dd_a = peak_a - eq_a
    a_mdd = float(dd_a.max())
    a_roi = float(pnls.sum() / INIT * 100)
    a_cal = a_roi / (a_mdd / INIT * 100) if a_mdd > 1e-10 else 0.0
    a_ulc = float(np.sqrt((dd_a ** 2).mean()))
    eq_p = INIT + np.cumsum(sh, axis=1)
    peak_p = np.maximum.accumulate(eq_p, axis=1)
    dd_p = peak_p - eq_p
    mdd_p = dd_p.max(axis=1)
    roi_p = sh.sum(axis=1) / INIT * 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        cal_p = np.where(mdd_p > 1e-10, roi_p / (mdd_p / INIT * 100), 0.0)
    ulc_p = np.sqrt((dd_p ** 2).mean(axis=1))
    f = 100.0 / n_mc
    return dict(
        n_trades=n, actual_roi=a_roi, actual_mdd=a_mdd, actual_calmar=a_cal, actual_ulcer=a_ulc,
        roi_rank_broken=float((roi_p < a_roi).sum() * f),
        mdd_rank=float((a_mdd < mdd_p).sum() * f),
        calmar_rank=float((cal_p < a_cal).sum() * f),
        ulcer_rank=float((a_ulc < ulc_p).sum() * f),
    )


# ----------- portfolio builders -----------
def topn_oos_pnls(pool_df: pd.DataFrame, window_i: int, n_top: int) -> np.ndarray:
    """Top-N strategies by IS PF at this window; concatenate their OOS trade pnls."""
    sub = pool_df[pool_df["window_i"] == window_i].copy()
    if len(sub) < n_top:
        return np.array([], dtype=np.float64)
    sub = sub.nlargest(n_top, "baseline_is_pf")
    pnls_list = [get_oos(row["asset"], row["strategy"], window_i)
                 for _, row in sub.iterrows()]
    pnls_list = [p for p in pnls_list if len(p) > 0]
    return np.concatenate(pnls_list) if pnls_list else np.array([], dtype=np.float64)


def mixed_topn_oos_pnls(pools: dict[str, pd.DataFrame], window_i: int, n_per_class: int) -> np.ndarray:
    parts = []
    for cn, pool in pools.items():
        sub = pool[pool["window_i"] == window_i]
        if len(sub) < n_per_class:
            return np.array([], dtype=np.float64)
        sub = sub.nlargest(n_per_class, "baseline_is_pf")
        for _, row in sub.iterrows():
            arr = get_oos(row["asset"], row["strategy"], window_i)
            if len(arr) > 0:
                parts.append(arr)
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


# ----------- main -----------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    pools = {cn: build_class_pool(cn) for cn in CLASSES}
    print("Pool sizes:")
    for cn, df in pools.items():
        print(f"  {cn}: {len(df):,} strategy-windows with IS PF > 1, "
              f"windows {sorted(df['window_i'].unique())[:3]}...{sorted(df['window_i'].unique())[-3:]}")

    # Per-class top-N
    for class_name, pool in pools.items():
        windows = sorted(pool["window_i"].unique())
        for n_top in N_TOP_VALUES:
            print(f"=== {class_name} N={n_top} ({len(windows)} windows) ===")
            for w in windows:
                pnls = topn_oos_pnls(pool, w, n_top)
                seed = abs(hash((class_name, n_top, w))) & 0xFFFFFFFF
                r = mc_ranks(pnls, N_MC, seed)
                rows.append({"config": class_name, "n_top": n_top, "window_i": w, **r})
            n_valid = sum(1 for r in rows if r["config"] == class_name and r["n_top"] == n_top and not np.isnan(r["mdd_rank"]))
            print(f"  -> {n_valid} valid windows")

    # Mixed: 10 of each class. Only at windows present in all 3 classes.
    common_windows = set(pools["crypto"]["window_i"].unique())
    common_windows &= set(pools["forex"]["window_i"].unique())
    common_windows &= set(pools["commodity"]["window_i"].unique())
    common_windows = sorted(common_windows)
    print(f"=== mixed (10 of each class = 30) on {len(common_windows)} common windows ===")
    for w in common_windows:
        pnls = mixed_topn_oos_pnls(pools, w, 10)
        seed = abs(hash(("mixed", 30, w))) & 0xFFFFFFFF
        r = mc_ranks(pnls, N_MC, seed)
        rows.append({"config": "mixed_30", "n_top": 30, "window_i": w, **r})
    n_valid = sum(1 for r in rows if r["config"] == "mixed_30" and not np.isnan(r["mdd_rank"]))
    print(f"  -> {n_valid} valid mixed windows")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "topn_portfolio_mc.csv", index=False)
    print(f"Wrote {OUT / 'topn_portfolio_mc.csv'}: {len(df)} rows")

    # Aggregate per (config, n_top)
    summary = df.dropna(subset=["mdd_rank"]).groupby(["config", "n_top"]).agg(
        n_windows=("window_i", "count"),
        mean_n_trades=("n_trades", "mean"),
        mdd_mean=("mdd_rank", "mean"),
        mdd_pct_below_50=("mdd_rank", lambda s: (s < 50).mean() * 100),
        mdd_pct_above_90=("mdd_rank", lambda s: (s >= 90).mean() * 100),
        calmar_mean=("calmar_rank", "mean"),
        ulcer_mean=("ulcer_rank", "mean"),
        roi_artefact_mean=("roi_rank_broken", "mean"),
        actual_oos_roi_mean=("actual_roi", "mean"),
        actual_oos_roi_median=("actual_roi", "median"),
        pct_actual_roi_positive=("actual_roi", lambda s: (s > 0).mean() * 100),
    ).round(2).reset_index()
    summary.to_csv(OUT / "topn_portfolio_mc_summary.csv", index=False)
    print()
    print("Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
