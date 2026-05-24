# R cross-validation scripts

Standalone R scripts that independently re-derive the headline statistical
claims of the paper from the same raw CSV inputs used by the Python and
Rust pipelines. The purpose is cross-language methodological audit: no R
script produces an artifact that appears in the paper.

Each script stays close to base R; the only non-base dependency is
`data.table` (`boot` is optional for script 02).

## Required packages

```r
install.packages("data.table")
install.packages("boot")   # optional
```

## Data layout

Scripts resolve raw data via:

1. `$MC_PAPER_DATA` env var; else
2. `../results/raw_data/`; else
3. `../../raw_data/`.

`block_perm_path_*.csv` and `*_portfolio_mc_path.csv` files resolve via
`$MC_PAPER_ROOT` or the parent of the raw-data directory.

## Scripts

| Script | Purpose | Inputs | Outputs (`R/out/`) | Paper target |
|---|---|---|---|---|
| `01_mc_rank_means.R` | Mean MC percentile rank per asset (baseline audit). Now reads the corrected `mdd_rank` column. | `<asset>_corrected_ranks.csv` (all 9) | `01_mc_rank_means.csv` | Table 4 (`tab:mc_rank_summary`) |
| `02_bootstrap_lift_ci.R` | Per-asset window-cluster bootstrap CI of MC-filter lift (10k resamples). | `<asset>_corrected_ranks.csv`, `<asset>_window_pairs.csv` (all 9) | `02_bootstrap_lift_ci.csv` | Table 15 per-asset CIs (`tab:bootstrap_lift_window`) |
| `03_block_permutation.R` | Block-permutation lift sweep b ∈ {1, 2, 3, 5, 10, 20} on path-dependent MDD. | `block_perm_path_<asset>.csv`, `<asset>_window_pairs.csv` (all 9) | `03_block_permutation.csv` | Table 19 (`tab:block_perm`) |
| `04_strategy_correlations.R` | Within-family vs cross-family OOS PF correlations. | `<asset>_window_pairs.csv` (all 9), optional `correlation_summary.csv` | `04_strategy_correlations.csv` | Table `family_corr` (§3.6) |
| `05_portfolio_mc_ranks.R` | Portfolio-level vs strategy-level path-dependent MC rank means. | `<asset>_portfolio_mc_path.csv`, `<asset>_corrected_ranks.csv` | `05_portfolio_mc_ranks.csv` | Supports Table 14 (`tab:portfolio_mc`) |

## How to run

```bash
cd R
export MC_PAPER_DATA=/path/to/raw_data    # or leave unset and use ../results/raw_data/
Rscript 01_mc_rank_means.R
Rscript 02_bootstrap_lift_ci.R
Rscript 03_block_permutation.R
Rscript 04_strategy_correlations.R
Rscript 05_portfolio_mc_ranks.R
```

Each script prints a formatted table to stdout and writes a CSV in
`R/out/`.

## Reproducibility

- All scripts call `set.seed(42)` before any resampling.
- Script 02 uses 10,000 window-cluster bootstrap resamples with a
  vectorised kernel matching `python/calendar_cluster_bootstrap.py`.
- Outputs are deterministic given the same inputs and seed.

## Notes

- `04_strategy_correlations.R` prefers the authoritative
  `correlation_summary.csv` (produced by `rust/corr_rs`) if present;
  otherwise falls back to a subsampled base-R correlation.
- Script 02 is a per-asset window-cluster specialisation. The pooled
  calendar-quarter CI is produced by the Python pipeline
  (`python/calendar_cluster_bootstrap.py`).
- The corrected `mdd_rank` / `calmar_rank` / `ulcer_rank` columns
  produced by `rust/mc_path_ranks` are what the R scripts now consume;
  the legacy `roi_rank_broken` column is shown only as a side-by-side
  artefact (see §8.4 of the paper).
