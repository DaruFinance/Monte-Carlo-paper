"""
Cluster-bootstrap confidence intervals on the block-permutation filter lift.

Most block-permutation logic now lives in `block_perm_analysis.py` (which
consumes the path-dependent MDD-based ranks emitted by the
`rust/block_perm_path` crate). This module is the thin wrapper that
computes window-clustered bootstrap CIs on the block-perm filter lift,
mirroring the cluster-bootstrap used elsewhere in the paper.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
TABLES = ROOT / "results" / "tables"


def cluster_bootstrap_ci(lift_per_window: np.ndarray, n_boot: int = 10000,
                          seed: int = 42) -> tuple[float, float, float]:
    """Window-clustered bootstrap CI on a vector of per-window lift values."""
    if len(lift_per_window) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(lift_per_window)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = lift_per_window[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(boots.mean()), float(lo), float(hi)


def main():
    src = TABLES / "table19_block_perm_filter_lift_corrected.csv"
    if not src.exists():
        print(f"Run block_perm_analysis.py first to produce {src.name}.")
        return
    df = pd.read_csv(src)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
