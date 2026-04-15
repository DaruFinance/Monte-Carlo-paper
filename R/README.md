# R Replication Scripts

Standalone R scripts that independently replicate the statistical claims
in the paper from the same raw CSV inputs used by the Python / Rust
pipeline. Goal: cross-validation.

Each script focuses on one statistical result and stays close to base R
(the only non-base dependency is `data.table`; `boot` is optional for
script 02).

## Required packages

```r
install.packages(c("data.table"))
install.packages(c("boot"))   # optional
```

## Data layout

Scripts resolve raw data via:

1. `$MC_PAPER_DATA` env var; else
2. `../results/raw_data/`; else
3. `../../raw_data/`; else
4. `../../../raw_data/`.

`block_perm_*.csv` files resolve via `$MC_PAPER_ROOT` or the parent of
the raw-data directory.

## Scripts

| Script | Purpose | Inputs | Outputs (`R/out/`) | Paper targets |
|---|---|---|---|---|
| `01_mc_rank_means.R` | Mean MC percentile rank per asset (baseline audit) | `<asset>_mc_perwindow.csv` (all 9) | `01_mc_rank_means.csv` | Table 4 (`tab:mc_pct_rank_summary`), inline mean MC-ROI ranks |
| `02_bootstrap_lift_ci.R` | Per-asset window-cluster bootstrap CI of MC-filter lift (10k resamples) | `<asset>_mc_perwindow.csv`, `<asset>_window_pairs.csv` (all 9) | `02_bootstrap_lift_ci.csv` | Table 15 per-asset CIs (`tab:empirical_bootstrap_ci`) |
| `03_block_permutation.R` | Block-permutation lift sweep b = 1, 2, 3, 5, 10, 20 | `block_perm_<asset>.csv`, `<asset>_window_pairs.csv` (all 9) | `03_block_permutation.csv` | Table 19 block sizes (`tab:block_perm_per_asset`) |
| `04_strategy_correlations.R` | Within-family vs cross-family OOS PF correlations | `<asset>_window_pairs.csv` (all 9), optional `correlation_summary.csv` | `04_strategy_correlations.csv` | Figure 1 summary stats, Table 3 (`tab:corr_summary`) |
| `05_portfolio_mc_ranks.R` | Portfolio-level vs strategy-level MC rank means | `<asset>_portfolio_mc.csv`, `<asset>_mc_perwindow.csv` (crypto) | `05_portfolio_mc_ranks.csv` | Supports Table 14 (`tab:portfolio_mc_summary`), Fig 10 |

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

## Reproducibility

- All scripts call `set.seed(42)` before any resampling.
- Script 02 uses 10,000 window-cluster bootstrap resamples with a
  vectorised kernel matching `calendar_cluster_bootstrap.py` aggregation.
- Outputs are deterministic given the same inputs and seed.

## Notes

- The R scripts are **not** used to produce any artifact in the paper;
  they exist as a methodological audit.
- `04_strategy_correlations.R` prefers the authoritative
  `correlation_summary.csv` if present; otherwise falls back to a
  subsampled base-R correlation.
- Script 02 is a per-asset window-cluster specialisation. The pooled
  calendar-quarter CI is produced by the Python
  `calendar_cluster_bootstrap.py`.
