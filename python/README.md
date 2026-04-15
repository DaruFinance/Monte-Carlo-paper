# Python analysis scripts

Cleaned Python scripts that reproduce every figure, table, and inline
statistic in the paper.

## Layout / paths

Every script resolves paths relative to the project root:

    ROOT = Path(os.environ["MC_PAPER_DATA"])       # if set
         | Path(__file__).resolve().parents[1]     # else Scripts_Clean/

    <ROOT>/results/raw_data/   # per-asset CSVs (inputs)
    <ROOT>/results/figures/    # PDF outputs
    <ROOT>/results/tables/     # CSV outputs

Set `MC_PAPER_DATA` to the directory containing `results/` if you run
from a different location.

## Dependencies

    numpy pandas scipy matplotlib seaborn statsmodels joblib

All RNG-using code is seeded (seed 42) for deterministic output.

## Scripts

| Script | Purpose | Inputs (from `results/raw_data/`) | Outputs | Paper targets |
|---|---|---|---|---|
| `regenerate_all_figures.py` | Master figure producer | `<asset>_window_pairs.csv`, `<asset>_mc_perwindow.csv` (all 9) | `results/figures/*.pdf` (9 figs) | Figs 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `strategy_correlations.py` | Cross-asset correlation summary (Table 3 CSV) | `<asset>_window_pairs.csv` (all 9) | `strategy_oos_summary.csv`, `fig_strategy_correlations.pdf` | Fig 1 summary, Table 3 (`tab:corr_summary`) |
| `correlation_figures.py` | Per-asset heatmaps + within/cross-family distributions | `<asset>_window_pairs.csv` (all 9) | `<asset>_family_corr.pdf`, `<asset>_strategy_corr.pdf`, `<asset>_corr_distribution.pdf`, `all_assets_corr_summary.pdf`, `correlation_summary.csv` | Fig 1 per-asset panels (`fig:strategy_correlations`) |
| `full_analysis.py` | Crypto empirical analysis (rank, filters, MC correlations) | `<asset>_mc_perwindow.csv`, `<asset>_window_pairs.csv` (4 crypto) | `mc_pct_rank_summary.csv`, `all_filters_comparison.csv`, `filter_ranking_summary.csv`, `mc_correlations.csv`, `mc_filter_pass_fail.csv`, `fair_comparison.csv`, `is_oos_correlation_by_filter.csv`, `mc_filter_vs_next_oos.csv` | Table 4, Table 5, Table 6 headline, Table 7, Table 10 derivatives, pass/fail |
| `crypto_stratified_analysis.py` | Stratified per-family / per-asset MC analysis | crypto `<asset>_mc_perwindow.csv`, `<asset>_window_pairs.csv` | `continuous_sharpe.csv`, `mc_by_family.csv`, `mc_selection_bias.csv`, `pf_stratified_crypto.csv` | Table 6, Table 8, Table 17, Table 18 |
| `block_perm_analysis.py` | Per-asset block-permutation lift | `block_perm_<asset>.csv`, `<asset>_window_pairs.csv` | `block_perm_per_asset.csv` | Table 19 breakdown |
| `block_perm_bootstrap.py` | Window-cluster bootstrap CIs | same as above | `block_perm_window_cluster_ci.csv` | Table 19 (`tab:block_perm_window_cluster_ci`) |
| `calendar_cluster_bootstrap.py` | Calendar-quarter cluster bootstrap (10k resamples) | crypto + fx/commodity window_pairs + mc_perwindow | `empirical_bootstrap_ci.csv` | Table 15 calendar-cluster row, Fig 3 annotations |
| `portfolio_mc_analysis.py` | Streaming portfolio-level MC | `<asset>_portfolio_mc.csv`, `<asset>_mc_perwindow.csv` (4 crypto) | `portfolio_mc_summary.csv`, `strat_vs_portfolio_mc.csv` | Table 14 |
| `reviewer_analyses.py` | Matched-pool placebo + cost sensitivity | crypto window_pairs + mc_perwindow | `matched_pool_placebo.csv`, `cost_sensitivity_two_levels.csv`, `cost_sensitivity_pf_sweep.csv` | Table 10 placebo, Table 16 |
| `synthetic_scenarios.py` | Synthetic scenario tables (pure-Python counterpart to Rust pipeline) | self-contained (seed 42) | `synthetic_a_filters.csv`, `synthetic_b_filters.csv`, `synthetic_c_portfolios.csv`, `synthetic_prevalence_sweep.csv`, `synthetic_filter_comparison.csv`, `synthetic_portfolio_results.csv` | Table synth-A, synth-B, synth-C, prevalence sweep |

## Reproducibility

- All stochastic code is seeded (`np.random.seed(42)` or explicit
  `RandomState`). Bootstraps use fixed base seeds per worker.
- Runtimes on a 16-core workstation:
  - `regenerate_all_figures.py` ~5 min
  - `calendar_cluster_bootstrap.py`, `block_perm_bootstrap.py` ~1-2 min
  - `correlation_figures.py` ~2-3 min (9 assets, per-asset heatmaps)
  - all other scripts: seconds
