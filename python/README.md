# Python analysis scripts

This folder contains the cleaned Python scripts used to generate every
figure, table, and inline statistic that appears in the paper. All
scripts are fully self-contained and resolve paths relative to a common
project root.

## Layout and path conventions

Every script assumes the following directory layout under the project root
(resolved as `Path(__file__).resolve().parents[2]`, or via the
`MC_PAPER_DATA` environment variable if set):

    <ROOT>/
      Scripts_Clean/python/        (this folder)
      results/
        raw_data/     # per-asset backtest / Rust-permutation CSVs
        figures/      # PDF figure outputs
        tables/       # CSV table outputs

Before running anything, place the per-asset CSVs listed below in
`results/raw_data/` (they are produced by the backtesting pipeline and by
the Rust block-permutation binary `block_perm_rs`). Either symlink the
existing location or set `MC_PAPER_DATA` to point at the directory that
contains `results/`.

## Dependencies

Standard scientific Python stack:

    numpy
    pandas
    scipy
    matplotlib
    seaborn

No other third-party packages are required. Parallelism uses the stdlib
`multiprocessing.Pool`. All RNG-using code is seeded (`np.random.seed(42)`
or an explicit `RandomState`) for deterministic output.

## Scripts

### `regenerate_all_figures.py` — master figure script
Produces nine of the ten paper figures:

| Figure | File | Inputs |
|---|---|---|
| 2  | `window_level_mc_vs_oos.pdf`          | forex/commodity window_pairs + mc_perwindow |
| 3  | `fig_bootstrap_lift_distributions.pdf`| crypto window_pairs + mc_perwindow |
| 4  | `fig_regime_robustness.pdf`           | crypto window_pairs + mc_perwindow |
| 5  | `fig_synthetic_mc_ranks.pdf`          | internal synthetic sim (seed 42) |
| 6  | `fig_synthetic_mc_analysis.pdf`       | internal synthetic sim (seed 42) |
| 7  | `fig_synthetic_pipeline_v4.pdf`       | `results/tables/synthetic_v4_*.csv` |
| 8  | `fig_synthetic_pipeline_detail.pdf`   | `results/tables/synthetic_v4_*.csv` |
| 9  | `mc_pct_rank_distributions.pdf`       | forex/commodity mc_perwindow |
| 10 | `mc_roi_vs_next_oos_binned.pdf`       | forex/commodity window_pairs + mc_perwindow |

Run: `python regenerate_all_figures.py`
Runtime: ~5 min (synthetic scenarios dominate; single-threaded NumPy).

### `strategy_correlations.py` — Figure 1, Table 3
Pairwise OOS Profit Factor correlation analysis. Within-family vs
cross-family correlation distributions across all 9 instruments.

- Inputs: `results/raw_data/*_window_pairs.csv` (9 assets)
- Outputs: `results/figures/fig_strategy_correlations.pdf`,
  `results/tables/strategy_oos_summary.csv`
- Run: `python strategy_correlations.py`

### `block_perm_analysis.py` — Tables 5, 11, 12, 13, 19
Merges Rust block-permutation outputs with window_pairs and computes
per-asset lift (rank >= 50 vs baseline) for block sizes 1/2/3/5/10/20.

- Inputs: `results/raw_data/block_perm_<asset>.csv`,
  `results/raw_data/<asset>_window_pairs.csv`
- Outputs: `results/tables/block_perm_per_asset.csv`
- Run: `python block_perm_analysis.py`

### `calendar_cluster_bootstrap.py` — Table 15, Figure 3 annotations
Calendar-quarter clustered bootstrap (10,000 resamples) of MC-ROI p50 lift,
accounting for cross-asset calendar dependence. Multiprocessing
(up to 32 workers). Deterministic seeds.

- Inputs: same as above.
- Outputs: `results/tables/empirical_bootstrap_ci.csv`
- Run: `python calendar_cluster_bootstrap.py`
- Runtime: ~1-2 min on a typical workstation.

### `block_perm_bootstrap.py` — Supplementary CIs for Table 15
Asset-window clustered bootstrap (alternative to the calendar-quarter
clusters). Reports CIs for block sizes b = 1, 5, 10.

- Inputs: same as above.
- Outputs: `results/tables/block_perm_window_cluster_ci.csv`
- Runtime: ~1-2 min.

### `full_analysis.py` — Tables 4, 6, 7, 9
Comprehensive crypto-asset empirical analysis (no figures — those live in
`regenerate_all_figures.py`). Produces the summary CSVs feeding the MC
rank distribution table, filter ranking, MC correlations, and IS-OOS
correlation tables. Prints headline inline statistics cited in the paper.

- Inputs: `results/raw_data/<asset>_mc_perwindow.csv`,
  `results/raw_data/<asset>_window_pairs.csv` (4 crypto assets)
- Outputs: `results/tables/mc_pct_rank_summary.csv`,
  `mc_filter_vs_next_oos.csv`, `all_filters_comparison.csv`,
  `filter_ranking_summary.csv`, `mc_correlations.csv`,
  `is_oos_correlation_by_filter.csv`

### `portfolio_mc_analysis.py` — Table 14
Streaming portfolio-level MC analysis on large `*_portfolio_mc.csv`
files; compares strategy-level and portfolio-level mean MC ROI rank.

- Inputs: `results/raw_data/<asset>_portfolio_mc.csv`,
  `<asset>_mc_perwindow.csv` (4 crypto assets)
- Outputs: `results/tables/portfolio_mc_summary.csv`,
  `strat_vs_portfolio_mc.csv`

### `reviewer_analyses.py` — Tables 10, 16
Matched-pool placebo test (random subsample of the IS-gated pool, same
size as the MC-filtered pool) and two-level transaction cost sensitivity
analysis plus an IS PF threshold sweep.

- Inputs: same crypto window_pairs + mc_perwindow.
- Outputs: `results/tables/matched_pool_placebo.csv`,
  `cost_sensitivity_two_levels.csv`, `cost_sensitivity_pf_sweep.csv`

## Reproducibility notes

- All stochastic code is seeded. Bootstraps use fixed base seeds
  (`42 + worker_index`) per `multiprocessing.Pool` worker for
  deterministic output.
- Synthetic scenarios in `regenerate_all_figures.py` use
  `np.random.seed(42)` at the start of `run_synthetic_v3()`.
- Runtimes on a 16-core workstation:
  - `regenerate_all_figures.py`: ~5 min (dominated by synthetic sims)
  - `calendar_cluster_bootstrap.py` / `block_perm_bootstrap.py`: ~1-2 min each
  - all other scripts: seconds
