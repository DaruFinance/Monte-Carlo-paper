# results/

Curated figures, tables, and raw data for the Monte-Carlo filter paper
(`paper_redacted/paper_redacted.tex`). Contents match exactly what is cited in
the published manuscript, nothing more. Nothing in this folder contains
proprietary strategy names; `strategy` columns are anonymous integer IDs.

Approximate total size: **~1.6 GB** (raw_data dominates; five files >100 MB are
stubbed as `*.csv.INFO`, see below).

## Layout

```
results/
├── figures/          # 10 PDFs cited via \includegraphics in the paper
├── tables/           # CSVs that feed paper tables (Tabs 3-19) + synthetic outputs
└── raw_data/
    ├── *.csv             # Per-asset MC / window-pair / portfolio / overall CSVs
    ├── *.csv.INFO        # Stubs for files >100 MB (GitHub hard limit)
    └── block_perm_*.csv  # Block-permutation output (one CSV per asset, flat layout)
```

## figures/ — paper figures

| File | Paper figure | Producing script |
|---|---|---|
| `fig_strategy_correlations.pdf` | Fig 1 | `correlation_analysis/strategy_correlations.py` |
| `window_level_mc_vs_oos.pdf` | Fig 2 | `paper_redacted/regenerate_all_figures.py` |
| `fig_bootstrap_lift_distributions.pdf` | Fig 3 | `calendar_cluster_bootstrap.py` |
| `fig_regime_robustness.pdf` | Fig 4 | `paper_redacted/regenerate_all_figures.py` |
| `fig_synthetic_mc_ranks.pdf` | Fig 5 | `synthetic_pipeline_rust/` (Rust) |
| `fig_synthetic_mc_analysis.pdf` | Fig 6 | `synthetic_pipeline_rust/` (Rust) |
| `fig_synthetic_pipeline_v4.pdf` | Fig 7 | `synthetic_pipeline_rust/` (Rust) |
| `fig_synthetic_pipeline_detail.pdf` | Fig 8 | `synthetic_pipeline_rust/` (Rust) |
| `mc_pct_rank_distributions.pdf` | Fig 9 | `paper_redacted/regenerate_all_figures.py` |
| `mc_roi_vs_next_oos_binned.pdf` | Fig 10 | `paper_redacted/regenerate_all_figures.py` |

## tables/ — CSVs behind paper tables

All paper tables are inline `tabular` environments (no `\input{}` calls), but
their numbers are produced from these CSVs.

| File | Paper table(s) | Producing script |
|---|---|---|
| `strategy_oos_summary.csv` | Tab 3 | `correlation_analysis/strategy_correlations.py` |
| `mc_pct_rank_summary.csv` | Tab 4 | `full_analysis.py` / MC rank analysis |
| `all_filters_comparison.csv` | Tabs 5, 12 | `block_perm_analysis.py` / `full_analysis.py` |
| `mc_filter_pass_fail.csv` | Tabs 5, 6 | `block_perm_analysis.py` |
| `mc_correlations.csv` | Tab 7 | Correlation analysis |
| `filter_ranking_summary.csv` | Tabs 8, 11 | `block_perm_analysis.py` |
| `fair_comparison.csv` | Tab 11 | `block_perm_analysis.py` |
| `empirical_bootstrap_ci.csv` | Tab 15 | `calendar_cluster_bootstrap.py` |
| `synthetic_a_filters.csv` | §Synthetic A (Tab around Fig 5) | `synthetic_pipeline_rust/` |
| `synthetic_b_filters.csv` | §Synthetic B | `synthetic_pipeline_rust/` |
| `synthetic_c_portfolios.csv` | §Synthetic C | `synthetic_pipeline_rust/` |
| `synthetic_filter_comparison.csv` | §Synthetic overview | `synthetic_pipeline_rust/` |
| `synthetic_portfolio_results.csv` | §Synthetic C detail | `synthetic_pipeline_rust/` |
| `synthetic_prevalence_sweep.csv` | §Robustness sweep | `synthetic_pipeline_rust/` |
| `synthetic_v4_summaries.csv` / `_matched.csv` | §Tier overview (Figs 7-8) | `synthetic_pipeline_rust/` |
| `synthetic_v4_edge_filters.csv` / `_matched.csv` | §Tier 1 (edge) | `synthetic_pipeline_rust/` |
| `synthetic_v4_null_filters.csv` / `_matched.csv` | §Tier 2 (null) | `synthetic_pipeline_rust/` |
| `synthetic_v4_adversarial_filters.csv` / `_matched.csv` | §Tier 3 (adversarial) | `synthetic_pipeline_rust/` + `run_extra_tier3.py` |
| `synthetic_v4_signal_sweep.csv` / `_matched.csv` | §Signal sweep | `run_extra_signal_sweep.py` |

`.bak_*` files from `tables_v2/` were skipped.

## raw_data/ — per-asset CSVs

### Schema: `*_mc_perwindow.csv`
`strategy` (int id), `window` (int), `n_trades`, `actual_roi`, `actual_sharpe`,
`actual_pf`, `roi_pct_rank`, `sharpe_pct_rank`, `pf_pct_rank`. One row per
(strategy, window); rank columns are the empirical percentile of the realised
metric within 1000 block-permuted trade sequences.

### Schema: `*_window_pairs.csv`
`strategy`, `window_i`, `baseline_is_{pf,sharpe,roi,trades}`,
`{ent,fee,sli,entind}_is_pf` (robustness-filter variants),
`baseline_oos_{pf,sharpe,roi,trades}`,
`next_baseline_oos_{pf,sharpe,roi,trades}` (next WFO window's OOS). Joins
cleanly to `*_mc_perwindow.csv` on (strategy, window_i).

### Schema: `*_portfolio_mc.csv`
`filter`, `window_i`, `portfolio_id`, `n_strategies`, `n_trades`,
`actual_roi`, `actual_sharpe`, `actual_pf`, `roi_pct_rank`, `sharpe_pct_rank`,
`pf_pct_rank`. Portfolio-level MC ranks; 10k random portfolios per (filter,
window).

### Schema: `*_overall.csv`
`strategy`, `is_opt_{sharpe,pf,roi,trades}`, `oos_opt_{sharpe,pf,roi,trades}`,
`num_windows`. Aggregate IS/OOS metrics across the full WFO.

### Schema: `block_perm/block_perm_<asset>.csv`
`strategy`, `window`, `n_trades`, `iid_rank`, `block2_rank`, `block3_rank`,
`block5_rank`, `block10_rank`, `block20_rank`, `oos_profitable`. Output of
`block_perm_rs/src/main.rs` at block sizes b = 1, 2, 3, 5, 10, 20. Used by
`calendar_cluster_bootstrap.py` and `block_perm_analysis.py`.

### Files included (direct copies)

MC per-window: `btc`, `bnb`, `doge`, `sol`, `eurusd`, `eurgbp`, `usdjpy`,
`xauusd`, `wti` (all 9 assets).

Window pairs: `sol`, `eurusd`, `eurgbp`, `usdjpy`, `xauusd`, `wti` (the
remaining crypto window_pairs are stubbed — see below).

Portfolio MC: `btc`, `sol`. Overall: `btc`, `bnb`, `doge`, `sol`.

Block perm: all 9 assets under `block_perm/`.

### Files stubbed (`.csv.INFO`) — over 100 MB

GitHub rejects files >100 MB and this repository does not use LFS. Each stub
records the filename, size, row count, column schema, an md5 checksum, the
first 5 rows as a sample, and regeneration instructions.

| Stub | Reason |
|---|---|
| `btc_window_pairs.csv.INFO`    | 135 MB |
| `bnb_window_pairs.csv.INFO`    | 151 MB |
| `doge_window_pairs.csv.INFO`   | 101 MB |
| `bnb_portfolio_mc.csv.INFO`    | 211 MB |
| `doge_portfolio_mc.csv.INFO`   | 151 MB |

To regenerate, run the upstream backtesting/MC pipeline described in
`MANIFEST_FOR_CLEANUP.md` section 3.1 (feeds `raw_data/`). Portfolio-level
files are produced by `portfolio_mc_analysis.py`.

### Files intentionally excluded

- `block_bootstrap_results.csv` (5.5 GB) — intermediate bootstrap artifact.
- `block_perm_all_assets.csv` (147 MB) — concatenation of the nine per-asset
  files already in `block_perm/`.
- All `*.log`, `*.bak*` files.
- trades.bin files (proprietary binary trade lists).

## Proprietary-content scrub

Every copied CSV header was inspected. All `strategy` columns are anonymous
integer IDs; no strategy-family names, filesystem paths, or user identifiers
appear in any header. No scrubbing of column values was necessary.
