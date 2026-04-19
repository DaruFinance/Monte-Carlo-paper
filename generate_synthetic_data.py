#!/usr/bin/env python3
"""
generate_synthetic_data.py
==========================
Generate synthetic sample data that matches the exact schemas of all raw data
files consumed by the Monte-Carlo paper analysis scripts.

Files are written to ``results/raw_data/`` (or ``--output-dir``).
Existing files are backed up with a ``.real.bak`` extension before overwriting.

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --output-dir /some/other/path
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RNG_SEED = 42

# Assets
CRYPTO_ASSETS = ["btc", "bnb", "doge", "sol"]
FOREX_ASSETS = ["eurusd", "usdjpy", "eurgbp", "xauusd", "wti"]
ALL_ASSETS = CRYPTO_ASSETS + FOREX_ASSETS

# Window counts per asset (crypto get variable counts, forex get 15)
CRYPTO_WINDOWS = {"btc": 24, "bnb": 17, "doge": 18, "sol": 7}
FOREX_WINDOWS = {a: 15 for a in FOREX_ASSETS}
ASSET_WINDOWS = {**CRYPTO_WINDOWS, **FOREX_WINDOWS}

# Strategy name building blocks
FAMILIES = ["ATR", "EMA", "SMA", "PPO", "RSI", "RSI_LEVEL", "STOCHK", "MACD"]
INDICATORS = [
    "EMA50", "SMA200", "EMA21", "SMA50", "EMA100", "SMA100",
    "EMA9", "SMA20", "EMA200", "SMA10",
]
MODES = ["accel", "mom", "xover", "div", "breakout", "revert"]
PCT_VALUES = ["0.5", "1.0", "1.2", "1.5", "2.0", "0.8", "0.3"]
SL_VALUES = ["SL1", "SL2", "SL3", "SL4"]

# Row counts
ROWS_PERWINDOW = 200
ROWS_WINDOW_PAIRS = 200
ROWS_BLOCK_PERM = 200
ROWS_PORTFOLIO = 500
ROWS_OVERALL = 50


def _build_strategy_pool(rng: np.random.Generator, n: int = 30) -> list[str]:
    """Build a pool of realistic strategy names."""
    strategies = []
    seen = set()
    while len(strategies) < n:
        fam = rng.choice(FAMILIES)
        ind = rng.choice(INDICATORS)
        mode = rng.choice(MODES)
        pct = rng.choice(PCT_VALUES)
        sl = rng.choice(SL_VALUES)
        name = f"{fam}_x_{ind}_{mode}_pct{pct}_{sl}"
        if name not in seen:
            seen.add(name)
            strategies.append(name)
    return strategies


def _backup_if_exists(path: Path) -> None:
    """Move existing file to .real.bak before overwriting."""
    if path.exists():
        bak = path.with_suffix(path.suffix + ".real.bak")
        shutil.move(str(path), str(bak))
        print(f"  [backup] {path.name} -> {bak.name}")


# ---------------------------------------------------------------------------
# Generator functions
# ---------------------------------------------------------------------------

def _build_strategy_window_pairs(rng, strategies, n_windows, n_rows):
    """Build a fixed set of (strategy, window) pairs used consistently across file types."""
    pairs = []
    for _ in range(n_rows):
        strat = rng.choice(strategies)
        win = int(rng.integers(1, n_windows + 1))
        pairs.append((strat, win))
    return pairs


def generate_perwindow(rng, pairs, asset, out_dir):
    """Generate per-window MC rank CSV for one asset."""
    fname = f"{asset}_mc_perwindow.csv"
    path = out_dir / fname
    _backup_if_exists(path)

    rows = []
    for strat, win in pairs:
        rows.append({
            "strategy": strat,
            "window": f"W{win:02d}",
            "n_trades": int(rng.integers(20, 201)),
            "actual_roi": round(float(rng.uniform(-100, 200)), 4),
            "actual_sharpe": round(float(rng.uniform(-2, 2)), 4),
            "actual_pf": round(float(rng.uniform(0.3, 3.0)), 4),
            "roi_pct_rank": round(float(rng.uniform(0, 100)), 2),
            "sharpe_pct_rank": round(float(rng.uniform(0, 100)), 2),
            "pf_pct_rank": round(float(rng.uniform(0, 100)), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  wrote {fname} ({len(df)} rows)")


def generate_window_pairs(rng, pairs, asset, out_dir):
    """Generate window-pairs CSV for one asset."""
    fname = f"{asset}_window_pairs.csv"
    path = out_dir / fname
    _backup_if_exists(path)

    rows = []
    for strat, wi in pairs:
        rows.append({
            "strategy": strat,
            "window_i": wi,
            "baseline_is_pf": round(float(rng.uniform(0.3, 3.0)), 4),
            "baseline_is_sharpe": round(float(rng.uniform(-2, 2)), 4),
            "baseline_is_roi": round(float(rng.uniform(-50, 150)), 4),
            "baseline_is_trades": int(rng.integers(20, 201)),
            "ent_is_pf": round(float(rng.uniform(0.3, 3.0)), 4),
            "fee_is_pf": round(float(rng.uniform(0.3, 3.0)), 4),
            "sli_is_pf": round(float(rng.uniform(0.3, 3.0)), 4),
            "entind_is_pf": round(float(rng.uniform(0.3, 3.0)), 4),
            "baseline_oos_pf": round(float(rng.uniform(0.2, 2.5)), 4),
            "baseline_oos_sharpe": round(float(rng.uniform(-2, 2)), 4),
            "baseline_oos_roi": round(float(rng.uniform(-80, 120)), 4),
            "baseline_oos_trades": int(rng.integers(10, 150)),
            "next_baseline_oos_pf": round(float(rng.uniform(0.2, 2.5)), 4),
            "next_baseline_oos_sharpe": round(float(rng.uniform(-2, 2)), 4),
            "next_baseline_oos_roi": round(float(rng.uniform(-80, 120)), 4),
            "next_baseline_oos_trades": int(rng.integers(10, 150)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  wrote {fname} ({len(df)} rows)")


def generate_block_perm(rng, pairs, asset, out_dir):
    """Generate block-permutation CSV for one asset.

    Strategy names are double-quoted to match the real data format.
    """
    fname = f"block_perm_{asset}.csv"
    path = out_dir / fname
    _backup_if_exists(path)

    rows = []
    for strat, win in pairs:
        rows.append({
            "strategy": f'"{strat}"',
            "window": f"W{win:02d}",
            "n_trades": int(rng.integers(20, 201)),
            "iid_rank": round(float(rng.uniform(0, 100)), 2),
            "block2_rank": round(float(rng.uniform(0, 100)), 2),
            "block3_rank": round(float(rng.uniform(0, 100)), 2),
            "block5_rank": round(float(rng.uniform(0, 100)), 2),
            "block10_rank": round(float(rng.uniform(0, 100)), 2),
            "block20_rank": round(float(rng.uniform(0, 100)), 2),
        })
    df = pd.DataFrame(rows)
    # quoting=csv.QUOTE_NONE so pandas doesn't re-escape the embedded quotes
    import csv
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    print(f"  wrote {fname} ({len(df)} rows)")


def generate_portfolio_mc(rng, strategies, asset, n_windows, n_rows, out_dir):
    """Generate portfolio MC CSV (crypto assets only)."""
    fname = f"{asset}_portfolio_mc.csv"
    path = out_dir / fname
    _backup_if_exists(path)

    rows = []
    for _ in range(n_rows):
        wi = int(rng.integers(1, n_windows + 1))
        rows.append({
            "filter": "baseline",
            "window_i": wi,
            "portfolio_id": int(rng.integers(1, 10001)),
            "n_strategies": int(rng.integers(3, 20)),
            "n_trades": int(rng.integers(50, 500)),
            "actual_roi": round(float(rng.uniform(-50, 200)), 4),
            "actual_sharpe": round(float(rng.uniform(-1, 3)), 4),
            "actual_pf": round(float(rng.uniform(0.5, 3.5)), 4),
            "roi_pct_rank": round(float(rng.uniform(0, 100)), 2),
            "sharpe_pct_rank": round(float(rng.uniform(0, 100)), 2),
            "pf_pct_rank": round(float(rng.uniform(0, 100)), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  wrote {fname} ({len(df)} rows)")


def generate_overall(rng, strategies, asset, n_windows, n_rows, out_dir):
    """Generate overall summary CSV (crypto assets only)."""
    fname = f"{asset}_overall.csv"
    path = out_dir / fname
    _backup_if_exists(path)

    rows = []
    used = set()
    for _ in range(n_rows):
        strat = rng.choice(strategies)
        # Avoid exact duplicates on strategy name
        while strat in used and len(used) < len(strategies):
            strat = rng.choice(strategies)
        used.add(strat)
        rows.append({
            "strategy": strat,
            "is_opt_sharpe": round(float(rng.uniform(-1, 3)), 4),
            "is_opt_pf": round(float(rng.uniform(0.5, 3.0)), 4),
            "is_opt_roi": round(float(rng.uniform(-30, 150)), 4),
            "is_opt_trades": int(rng.integers(30, 300)),
            "oos_opt_sharpe": round(float(rng.uniform(-1, 2.5)), 4),
            "oos_opt_pf": round(float(rng.uniform(0.4, 2.5)), 4),
            "oos_opt_roi": round(float(rng.uniform(-50, 120)), 4),
            "oos_opt_trades": int(rng.integers(20, 250)),
            "num_windows": n_windows,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  wrote {fname} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic sample data for MC-paper analysis scripts."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where CSVs are written. "
            "Defaults to results/raw_data/ relative to this script."
        ),
    )
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir is None:
        script_dir = Path(__file__).resolve().parent
        out_dir = script_dir / "results" / "raw_data"
    else:
        out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # Reproducible RNG
    rng = np.random.default_rng(RNG_SEED)

    # Build shared strategy pool
    strategies = _build_strategy_pool(rng, n=30)
    print(f"Generated {len(strategies)} unique strategy names.\n")

    # Build consistent (strategy, window) pairs per asset so joins work
    asset_pairs = {}
    for asset in ALL_ASSETS:
        nw = ASSET_WINDOWS[asset]
        asset_pairs[asset] = _build_strategy_window_pairs(
            rng, strategies, nw, ROWS_PERWINDOW,
        )

    # ------------------------------------------------------------------
    # 1. Per-window MC rank files  (all 9 assets)
    # ------------------------------------------------------------------
    print("=== Per-window MC rank files ===")
    for asset in ALL_ASSETS:
        generate_perwindow(rng, asset_pairs[asset], asset, out_dir)

    # ------------------------------------------------------------------
    # 2. Window-pairs files  (all 9 assets)
    # ------------------------------------------------------------------
    print("\n=== Window-pairs files ===")
    for asset in ALL_ASSETS:
        generate_window_pairs(rng, asset_pairs[asset], asset, out_dir)

    # ------------------------------------------------------------------
    # 3. Block permutation files  (all 9 assets)
    # ------------------------------------------------------------------
    print("\n=== Block permutation files ===")
    for asset in ALL_ASSETS:
        generate_block_perm(rng, asset_pairs[asset], asset, out_dir)

    # ------------------------------------------------------------------
    # 4. Portfolio MC files  (4 crypto assets only)
    # ------------------------------------------------------------------
    print("\n=== Portfolio MC files (crypto only) ===")
    for asset in CRYPTO_ASSETS:
        nw = ASSET_WINDOWS[asset]
        generate_portfolio_mc(rng, strategies, asset, nw, ROWS_PORTFOLIO, out_dir)

    # ------------------------------------------------------------------
    # 5. Overall files  (4 crypto assets only)
    # ------------------------------------------------------------------
    print("\n=== Overall files (crypto only) ===")
    for asset in CRYPTO_ASSETS:
        nw = ASSET_WINDOWS[asset]
        generate_overall(rng, strategies, asset, nw, ROWS_OVERALL, out_dir)

    # ------------------------------------------------------------------
    print("\nSynthetic data generation complete. You can now run the analysis scripts.")


if __name__ == "__main__":
    main()
