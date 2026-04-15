# results/

Curated figures, tables, and raw data for the paper. Contents match
exactly what is cited in the manuscript. `strategy` columns are
anonymous integer IDs; no proprietary names appear.

Approximate total size: **~1.6 GB** (raw_data dominates; five files
>100 MB are stubbed as `*.csv.INFO`).

## Layout

```
results/
  figures/      # 10 PDFs cited via \includegraphics in the paper
  tables/       # CSVs that feed paper tables + synthetic outputs
  raw_data/     # per-asset MC / window-pair / portfolio / block-perm CSVs
```

## figures/ -- paper figures

| File | Paper figure | Producing script |
|---|---|---|
| `fig_strategy_correlations.pdf` | Fig 1 | `python/strategy_correlations.py` (+ `correlation_figures.py` for per-asset panels) |
| `window_level_mc_vs_oos.pdf` | Fig 2 | `python/regenerate_all_figures.py` |
| `fig_bootstrap_lift_distributions.pdf` | Fig 3 | `python/calendar_cluster_bootstrap.py` (+ `regenerate_all_figures.py`) |
| `fig_regime_robustness.pdf` | Fig 4 | `python/regenerate_all_figures.py` |
| `fig_synthetic_mc_ranks.pdf` | Fig 5 | `rust/synthetic_pipeline_rust/` (+ `python/regenerate_all_figures.py` render) |
| `fig_synthetic_mc_analysis.pdf` | Fig 6 | `rust/synthetic_pipeline_rust/` (+ `regenerate_all_figures.py`) |
| `fig_synthetic_pipeline_v4.pdf` | Fig 7 | `rust/synthetic_pipeline_rust/` (+ `regenerate_all_figures.py`) |
| `fig_synthetic_pipeline_detail.pdf` | Fig 8 | `rust/synthetic_pipeline_rust/` (+ `regenerate_all_figures.py`) |
| `mc_pct_rank_distributions.pdf` | Fig 9 | `python/regenerate_all_figures.py` |
| `mc_roi_vs_next_oos_binned.pdf` | Fig 10 | `python/regenerate_all_figures.py` |

Per-asset heatmap PDFs (`<asset>_family_corr.pdf`,
`<asset>_strategy_corr.pdf`, `<asset>_corr_distribution.pdf`,
`all_assets_corr_summary.pdf`) are supplementary artefacts produced by
`correlation_figures.py`; the paper's Figure 1 is the cross-instrument
summary panel assembled from the same statistics.

## tables/ -- CSVs behind paper tables

Paper tables are inline `tabular` environments; their numbers are
produced from these CSVs.

| File | Paper table(s) | Producing script |
|---|---|---|
| `strategy_oos_summary.csv` | Table 3 (`tab:corr_summary`) | `python/strategy_correlations.py` |
| `correlation_summary.csv` | Fig 1 per-asset support | `python/correlation_figures.py` |
| `mc_pct_rank_summary.csv` | Table 4 | `python/full_analysis.py` |
| `all_filters_comparison.csv` | Table 5 | `python/full_analysis.py` |
| `filter_ranking_summary.csv` | Table 6 headline ratios | `python/full_analysis.py` |
| `mc_correlations.csv` | Table 7 | `python/full_analysis.py` |
| `fair_comparison.csv` | Table 10 derivatives | `python/full_analysis.py` |
| `mc_filter_pass_fail.csv` | Table 5/6 pass/fail breakdown | `python/full_analysis.py` |
| `mc_filter_vs_next_oos.csv` | Table 10 supporting | `python/full_analysis.py` |
| `is_oos_correlation_by_filter.csv` | Table 7 supporting | `python/full_analysis.py` |
| `continuous_sharpe.csv` | Table 6 | `python/crypto_stratified_analysis.py` |
| `mc_by_family.csv` | Table 8 | `python/crypto_stratified_analysis.py` |
| `mc_selection_bias.csv` | Table 17 | `python/crypto_stratified_analysis.py` |
| `pf_stratified_crypto.csv` | Table 18 | `python/crypto_stratified_analysis.py` |
| `portfolio_mc_summary.csv` | Table 14 | `python/portfolio_mc_analysis.py` |
| `strat_vs_portfolio_mc.csv` | Table 14 | `python/portfolio_mc_analysis.py` |
| `empirical_bootstrap_ci.csv` | Table 15 (calendar-cluster row) | `python/calendar_cluster_bootstrap.py` |
| `matched_pool_placebo.csv` | Table 10 matched-pool placebo | `python/reviewer_analyses.py` |
| `cost_sensitivity_two_levels.csv` | Table 16 | `python/reviewer_analyses.py` |
| `cost_sensitivity_pf_sweep.csv` | Table 16 | `python/reviewer_analyses.py` |
| `block_perm_per_asset.csv` | Table 19 breakdown | `python/block_perm_analysis.py` |
| `block_perm_window_cluster_ci.csv` | Table 19 window-cluster CI | `python/block_perm_bootstrap.py` |
| `synthetic_a_filters.csv` | Table synth-A | `python/synthetic_scenarios.py` |
| `synthetic_b_filters.csv` | Table synth-B | `python/synthetic_scenarios.py` |
| `synthetic_c_portfolios.csv` | Scenario C | `python/synthetic_scenarios.py` |
| `synthetic_prevalence_sweep.csv` | Prevalence sweep | `python/synthetic_scenarios.py` |
| `synthetic_filter_comparison.csv` | Synthetic filter comparison | `python/synthetic_scenarios.py` |
| `synthetic_portfolio_results.csv` | Synthetic portfolio results | `python/synthetic_scenarios.py` |
| `synthetic_v4_edge_filters.csv` + `_matched.csv` | Tier 1 (edge), Figs 5-8 | `rust/synthetic_pipeline_rust/` |
| `synthetic_v4_null_filters.csv` + `_matched.csv` | Tier 2 (null), Figs 5-8 | `rust/synthetic_pipeline_rust/` |
| `synthetic_v4_adversarial_filters.csv` + `_matched.csv` | Tier 3 (adversarial), Figs 5-8 | `rust/synthetic_pipeline_rust/` |
| `synthetic_v4_signal_sweep.csv` + `_matched.csv` | Signal sweep, Figs 5-8 | `rust/synthetic_pipeline_rust/` |
| `synthetic_v4_summaries.csv` + `_matched.csv` | Tier overview, Figs 5-8 | `rust/synthetic_pipeline_rust/` |

## raw_data/ -- per-asset CSVs

Schemas (same as before):

- `*_mc_perwindow.csv`: `strategy`, `window`, `n_trades`, `actual_roi`,
  `actual_sharpe`, `actual_pf`, `roi_pct_rank`, `sharpe_pct_rank`,
  `pf_pct_rank`. One row per (strategy, window).
- `*_window_pairs.csv`: `strategy`, `window_i`,
  `baseline_is_{pf,sharpe,roi,trades}`, `{ent,fee,sli,entind}_is_pf`,
  `baseline_oos_{pf,sharpe,roi,trades}`,
  `next_baseline_oos_{pf,sharpe,roi,trades}`.
- `*_portfolio_mc.csv`: `filter`, `window_i`, `portfolio_id`,
  `n_strategies`, `n_trades`, `actual_{roi,sharpe,pf}`,
  `{roi,sharpe,pf}_pct_rank`.
- `block_perm_<asset>.csv`: `strategy`, `window`, `n_trades`,
  `iid_rank`, `block{2,3,5,10,20}_rank`, `oos_profitable`.

### Files stubbed (`.csv.INFO`) -- over 100 MB

GitHub rejects files >100 MB and this repository does not use LFS.

| Stub | Reason |
|---|---|
| `btc_window_pairs.csv.INFO` | 135 MB |
| `bnb_window_pairs.csv.INFO` | 151 MB |
| `doge_window_pairs.csv.INFO` | 101 MB |
| `bnb_portfolio_mc.csv.INFO` | 211 MB |
| `doge_portfolio_mc.csv.INFO` | 151 MB |

Regenerate via the upstream backtesting / Rust MC pipeline
(`rust/block_perm_rs` for `*_window_pairs.csv`, `portfolio_mc_analysis.py`
for `*_portfolio_mc.csv`).

### Files intentionally excluded

- `block_bootstrap_results.csv` (5.5 GB intermediate).
- `block_perm_all_assets.csv` (147 MB concatenation of the 9 per-asset).
- All `*.log`, `*.bak*` files.
- `trades.bin` proprietary binary trade lists.
