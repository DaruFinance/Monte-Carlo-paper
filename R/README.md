# R Replication Scripts

This directory contains standalone R scripts that independently replicate
the statistical claims in the paper from the same raw CSV inputs used by
the Python / Rust analysis pipeline. The goal is cross-validation: a
reviewer can run R against the raw data and confirm the numbers without
trusting the Python/Rust code.

## Purpose

Each script focuses on one statistical result and is written to be as
close to base R as possible (the only non-base dependency is
`data.table` for fast CSV reading; `boot` is optional for script 02).

## Required packages

```r
install.packages(c("data.table"))
# Optional (script 02 uses a hand-rolled cluster bootstrap and does not
# strictly need `boot`, but it's handy for sanity checks):
install.packages(c("boot"))
```

No RStudio, no tidyverse, no shell-outs.

## Data layout

All scripts read raw CSVs from the directory resolved by:

1. `$MC_PAPER_DATA` environment variable, if set; else
2. `../results/raw_data/`; else
3. `../../raw_data/`; else
4. `../../../raw_data/`.

Root-level block-permutation files (`block_perm_*.csv`) are resolved via
`$MC_PAPER_ROOT` or the parent of the raw-data directory.

Expected files (9 assets: btc, doge, bnb, sol, eurusd, usdjpy, eurgbp,
xauusd, wti):

```
raw_data/<asset>_mc_perwindow.csv
raw_data/<asset>_window_pairs.csv
raw_data/<asset>_portfolio_mc.csv    (crypto only)
block_perm_<asset>.csv               (root level)
correlation_analysis/correlation_summary.csv   (optional for script 04)
```

## How to run

```bash
cd Scripts_Clean/R
export MC_PAPER_DATA=/path/to/raw_data
Rscript 01_mc_rank_means.R
Rscript 02_bootstrap_lift_ci.R
Rscript 03_block_permutation.R
Rscript 04_strategy_correlations.R
Rscript 05_portfolio_mc_ranks.R
```

Each script prints a formatted table to stdout and writes a CSV in
`Scripts_Clean/R/out/`.

## Script -> Python/Rust counterpart -> paper artifact

| R script | Python / Rust counterpart | Paper artifact reproduced |
|---|---|---|
| `01_mc_rank_means.R` | `full_analysis.py`, `block_perm_rs/src/main.rs` | Table 4, inline mean MC-ROI ranks (BTC 40.8%, EUR/USD 31.4%, ...) |
| `02_bootstrap_lift_ci.R` | `calendar_cluster_bootstrap.py`, `block_perm_analysis.py` | Table 5, Table 15 (10k bootstrap CIs on MC-filter lift) |
| `03_block_permutation.R` | `block_perm_rs/src/main.rs`, `block_perm_analysis.py` | Table 19 (block permutation sweep, b=1..20) |
| `04_strategy_correlations.R` | `correlation_analysis/strategy_correlations.py`, `corr_rs/src/main.rs` | Figure 1 summary stats, Table 3 |
| `05_portfolio_mc_ranks.R` | `portfolio_mc_analysis.py` | Table 14, Figure 10 (portfolio MC ranks) |

## Reproducibility

- All scripts call `set.seed(42)` before any resampling.
- The bootstrap in `02_bootstrap_lift_ci.R` uses 10,000 window-level
  cluster resamples with a vectorised window-tally kernel (matches the
  aggregation strategy in `calendar_cluster_bootstrap.py`).
- Outputs are deterministic given the same inputs and seed.

## Notes

- `04_strategy_correlations.R` prefers the pre-computed
  `correlation_analysis/correlation_summary.csv` (authoritative). A base-R
  fallback is provided that subsamples strategies to keep the
  correlation matrix tractable; the numbers will be approximate.
- Script 02 performs a window-cluster bootstrap per asset, which is the
  per-asset specialisation of the cross-asset calendar cluster bootstrap.
  For the pooled / cross-asset CI, see
  `calendar_cluster_bootstrap.py` (not replicated here because the
  calendar-quarter clustering relies on hard-coded WFO start dates that
  are already encoded there).
