# Monte Carlo Filter Evaluation — Reproducibility Package

Analysis code accompanying the paper

> **Predictive Value of Within-Strategy Permutation Tests for Forward Selection:**
> *Evidence from Over 6 Billion Strategy-Level Permutations Across Three Asset Classes*
> (Revised May 2026; SSRN [abstract_id=6636018](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6636018))

The paper evaluates 437,911 strategy configurations across nine instruments
(four crypto perpetuals, three forex pairs, two commodities) over 160
walk-forward windows, with 6.63 billion within-strategy Monte Carlo
permutations plus 19.9 billion block-permutation runs (~26.5 billion total).
Three findings:

1. The standard practitioner metrics — total ROI, trade-level Sharpe, and
   Profit Factor — are permutation-invariant by construction under fixed
   per-trade notional sizing. Their MC rank distributions are degenerate;
   any non-trivial rank reported in vectorised implementations is a
   floating-point summation-order artefact (see `python/fp_pitfall_demo.py`).
2. For the genuinely path-dependent statistics (Maximum Drawdown, Calmar,
   Ulcer index), realised trade ordering is statistically indistinguishable
   from a random reshuffle and MC filtering adds at most a fraction of a
   percentage point of out-of-sample profitability over a simple in-sample
   profitability gate.
3. At the portfolio level the MC test detects genuine path-dependence
   (rightward shift of MDD ranks), but a forward test shows the signal
   carries no positive predictive content.

## What is in this repository

```
.
├── README.md           ← this file
├── LICENSE             ← MIT
├── python/             ← orchestrates all analyses; produces every figure and table
├── rust/               ← seven Cargo crates for the parallelised stages
├── R/                  ← independent cross-validation in a second language
└── results/
    ├── figures/        ← PDF figures cited in the paper
    ├── tables/         ← CSV / JSON / TeX tables feeding paper tables
    └── raw_data/       ← left empty (.gitkeep); user-supplied (see below)
```

## What is NOT in this repository

Per the paper's Data Availability section, this package consists of
**analysis scripts only**. It does *not* contain:

- The raw bar or trade data underlying the nine instruments.
- The 437,911 strategy configurations and their parameterisations.
- Pre-aggregated walk-forward output (`results/raw_data/` ships empty).

Readers wishing to reproduce the empirical numerics need to apply the
released scripts to their own bar-level data and their own strategy
universe. The strategy backtester that produces the trade streams the MC
pipeline consumes is open source and lives at
<https://github.com/DaruFinance/quant-research-framework-rs>.

## Quick start

### Reproduce the floating-point pitfall (no external data needed)

```bash
cd python
pip install numpy pandas matplotlib scipy seaborn
python fp_pitfall_demo.py
```

This is self-contained and runs in under 30 seconds. It writes
`fp_bug_evidence.csv` (a ledger of how often vectorised summation produces
a strict `>` between two mathematically equal sums) and
`fp_bug_demonstration.pdf` (publication figure).

### Reproduce a paper table / figure on your own data

Populate `results/raw_data/` with the per-asset CSVs documented in
`python/README.md` and run the relevant producer:

```bash
export MC_PAPER_DATA=$(pwd)            # if not running from the repo root

python python/full_analysis.py                # Tables 4, 5, 6, 7, 15 (corrected, MDD-based)
python python/regenerate_all_figures.py       # All main-text figures
python python/portfolio_mc_analysis.py        # Tables 14, 14b/c/d, top-N portfolio MC
python python/block_perm_analysis.py          # Table 19 (path-dependent block permutation)
python python/calendar_cluster_bootstrap.py   # Calendar-cluster bootstrap CIs
python python/crypto_stratified_analysis.py   # Tables 8, 17, 18
python python/reviewer_analyses.py            # Table 16 (cost-sensitivity), placebo, Sharpe
python python/synthetic_scenarios.py          # Synthetic Tables 23 / 24 / 25
python python/gold_mc_analysis.py             # Gold-standard bar-permutation MC
```

### Cross-validate in R (optional)

```bash
cd R
Rscript 01_mc_rank_means.R
Rscript 02_bootstrap_lift_ci.R
Rscript 03_block_permutation.R
Rscript 04_strategy_correlations.R
Rscript 05_portfolio_mc_ranks.R
```

R scripts re-derive the headline rank means, bootstrap CIs, block-permutation
lift, and portfolio rank statistics. They produce no paper artefacts; they
exist purely as a cross-language methodological audit.

## Why three languages

- **Rust** (`rust/`) handles the computationally heavy stages: billions of
  strategy-level resamples (`block_perm_rs`, `block_perm_path`), the
  strategy-correlation tensor (`corr_rs`), the path-dependent MC ranks
  (`mc_path_ranks`), portfolio-level MC and its forward-OOS variant
  (`portfolio_mc_path`, `portfolio_mc_oos`), and the synthetic validation
  pipeline (`synthetic_pipeline_rust`). All seven crates are parallelised
  with Rayon.
- **Python** (`python/`) orchestrates the analyses and produces every
  figure and table. NumPy / pandas / Matplotlib / SciPy.
- **R** (`R/`) is the cross-validation layer.

## Table / figure → producer map

### Figures (all PDFs land in `results/figures/`)

| Figure | File | Producer |
|---|---|---|
| 3   | `fig3_bootstrap_lift_corrected.pdf`, `fig3_bootstrap_lift_corrected_forex.pdf` | `python/regenerate_all_figures.py` |
| 4   | `fig4_regime_robustness_corrected.pdf`, `fig4_regime_robustness_corrected_forex.pdf` | `python/regenerate_all_figures.py` |
| 5   | `fig5_synthetic_mc_ranks_corrected.pdf` | `python/regenerate_all_figures.py` + `rust/synthetic_pipeline_rust/` |
| 6   | `fig6_synthetic_edge_strat_corrected.pdf` | `python/regenerate_all_figures.py` |
| 7   | `fig7_synthetic_tier_lift_corrected.pdf` | `python/regenerate_all_figures.py` |
| 8   | `fig8_synthetic_signal_sweep_corrected.pdf` | `python/regenerate_all_figures.py` |
| MC-rank distributions | `fig_mc_rank_distributions_corrected.pdf`, `..._forex.pdf` | `python/regenerate_all_figures.py` |
| Portfolio MC right-shift (§6.2) | `fig_portfolio_mc_rightshift.pdf` | `python/regenerate_all_figures.py` |
| Portfolio next-OOS deciles (§6.3) | `fig_portfolio_oos_decile.pdf` | `python/regenerate_all_figures.py` |
| Cross-asset forest (§6.4) | `fig_crossasset_forest.pdf` | `python/regenerate_all_figures.py` |
| Asset × family heatmap (§4) | `fig_mc_by_family_heatmap.pdf` | `python/regenerate_all_figures.py` |
| Gold-standard MC (§7.5) | `fig_gold_mc.pdf` | `python/regenerate_all_figures.py` (+ `python/gold_mc_analysis.py`) |
| Cost sensitivity (§7.4) | `fig_cost_sensitivity.pdf` | `python/regenerate_all_figures.py` |
| Synthetic ground truth (App. A2) | `fig_synthetic_groundtruth_ranks.pdf` | `python/regenerate_all_figures.py` |
| Floating-point pitfall (§8.4) | `fp_bug_demonstration.pdf` | `python/fp_pitfall_demo.py` |

### Tables (CSV/JSON/TeX in `results/tables/`)

| Table | File(s) | Producer |
|---|---|---|
| family_corr (§3.6) | (via `corr_rs` Rust output) | `python/strategy_correlations.py` |
| 4   (MC rank summary) | `table4_corrected.csv`, `table4_corrected.tex` | `python/full_analysis.py` |
| 5   (filter ranking) | `table5_filters_comparison_corrected.csv` | `python/full_analysis.py` |
| 6   (filter ranking summary, pooled) | `filter_ranking_summary_corrected.csv` | `python/full_analysis.py` |
| 7   (MC rank ↔ OOS correlations) | `table7_correlations_corrected.csv` | `python/full_analysis.py` |
| 8   (MC by indicator family) | `table8_mc_by_family_corrected.csv` | `python/crypto_stratified_analysis.py` |
| 14  (portfolio MC) | `table14_portfolio_mc_corrected.csv` | `python/portfolio_mc_analysis.py` |
| 14b/c/d (portfolio next-OOS) | `table14b_portfolio_oos_stratified_mc_*.csv`, `table14c_portfolio_oos_topbottom.csv`, `table14d_portfolio_oos_stratified_pooled.csv` | `python/portfolio_mc_analysis.py` |
| 15  (bootstrap CIs) | `table15_bootstrap_lift_corrected.csv`, `table15_calendar_cluster_bootstrap_corrected.csv` | `python/full_analysis.py`, `python/calendar_cluster_bootstrap.py` |
| 16  (cost sensitivity) | `table16_cost_sensitivity_corrected.csv`, `table16_cost_sensitivity_corrected.tex` | `python/reviewer_analyses.py` |
| 17  (MC selection bias) | `table17_mc_selection_bias_corrected.csv` | `python/crypto_stratified_analysis.py` |
| 18  (PF-stratified MC) | `table18_pf_stratified_corrected.csv` | `python/crypto_stratified_analysis.py` |
| 19  (block-permutation) | `table19_block_permutation_corrected.csv`, `table19_block_perm_filter_lift_corrected.csv` | `python/block_perm_analysis.py`, `python/block_perm_bootstrap.py` |
| 23/24/25 (synthetic tiers, signal sweep) | `table23_..._corrected.csv`, `table24_..._corrected.csv`, `table25_..._corrected.csv` | `python/synthetic_scenarios.py` |
| Gold-standard MC | `gold_mc_<asset>_agg.json` (×9), `gold_mc_placebo.json` | `python/gold_mc_analysis.py` |
| Continuous Sharpe | `continuous_sharpe_corrected.csv` | `python/reviewer_analyses.py` |
| Matched-pool placebo | `matched_pool_placebo_corrected.csv` | `python/reviewer_analyses.py` |
| Top-N portfolio MC | `topn_portfolio_mc{,_summary,_floor30,_floor30_summary}.csv` | `python/portfolio_mc_analysis.py` |
| Synthetic A/B/CMP | `synthetic_{a,b}_filters_corrected.csv`, `synthetic_filter_comparison_corrected.csv`, `synthetic_mc_rank_stats_corrected.csv` | `python/synthetic_scenarios.py` |

R cross-validation outputs land in `R/out/` and are indexed in
`R/README.md`.

## Regenerating the per-asset CSVs from scratch (Rust stage)

The path-dependent rank files (`<asset>_corrected_ranks.csv`,
`block_perm_path_<asset>.csv`, `<asset>_portfolio_mc_path.csv`,
`<asset>_portfolio_mc_oos.csv`) are produced by four new Rust crates that
expect a backtester's `trades.bin` directory layout as input. The
synthetic pipeline is fully self-contained:

```bash
cd rust/synthetic_pipeline_rust
cargo build --release
cargo run --release -- ../../results/raw_data/synthetic_v4
```

See `rust/README.md` for per-crate usage.

## Reproducibility

All stochastic code uses a fixed seed (`42` throughout). Bootstraps use
10,000 resamples. MC permutations use independent seeds per
(strategy, window) so per-cell ranks are reproducible. The lift estimates
are stable across three independent seed sequences (mean lift varies by
less than 0.05 pp).

Tested with Python 3.12, NumPy 1.26, pandas 2.2, SciPy 1.13,
matplotlib 3.9, and Rust 1.94 (stable).

## Citation

```bibtex
@unpublished{gatto2026mc,
  author = {Gatto, Daniel V.},
  title  = {Predictive Value of Within-Strategy Permutation Tests for Forward Selection:
            Evidence from Over 6 Billion Strategy-Level Permutations Across Three Asset Classes},
  year   = {2026},
  month  = {May},
  note   = {Revised May 2026. SSRN Working Paper 6636018.},
  url    = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6636018}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
