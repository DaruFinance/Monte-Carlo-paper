# Python analysis scripts

Scripts that reproduce every figure, table, and inline statistic in the paper.

## Paths

Every script resolves locations relative to the project root:

```python
ROOT = Path(os.environ["MC_PAPER_DATA"])     # if set
     | Path(__file__).resolve().parents[1]   # else, the repo root

<ROOT>/results/raw_data/   # per-asset CSVs (inputs, user-supplied)
<ROOT>/results/figures/    # PDF outputs
<ROOT>/results/tables/     # CSV / JSON / TeX outputs
```

Set `MC_PAPER_DATA` if you run the scripts from a directory other than the
repo root, or want to point them at an out-of-tree data directory.

## Dependencies

```
numpy pandas scipy matplotlib seaborn
```

All stochastic code is seeded (`np.random.seed(42)` or explicit
`RandomState`); bootstrap workers use fixed base seeds.

## Inputs expected in `results/raw_data/`

Each script reads a subset of the following files. Each is a flat CSV
keyed by `(strategy, window)`; the strategies are user-supplied (see the
strategy backtester at
<https://github.com/DaruFinance/quant-research-framework-rs>).

| File | Produced by | Schema (key columns) |
|---|---|---|
| `<asset>_corrected_ranks.csv` | `rust/mc_path_ranks` | `strategy, window, n_trades, actual_roi, actual_mdd, actual_calmar, actual_ulcer, roi_rank_broken, mdd_rank, calmar_rank, ulcer_rank` |
| `<asset>_window_pairs.csv` | upstream backtester | `strategy, window_i, is_*, oos_*` plus robustness columns `ent_*, fee_*, sli_*, entind_*` |
| `block_perm_path_<asset>.csv` | `rust/block_perm_path` | `strategy, window, n_trades, iid_mdd_rank, block{2,3,5,10,20}_mdd_rank` |
| `<asset>_portfolio_mc_path.csv` | `rust/portfolio_mc_path` | `asset, window, port_id, n_strats, n_trades, port_mdd_rank, port_calmar_rank` |
| `<asset>_portfolio_mc_oos.csv` | `rust/portfolio_mc_oos` | as above + `oos_*` outcome columns |

Per the paper's Data Availability, none of these are bundled with the
repo; populate them from your own bar data via the public backtester or
from any pipeline conforming to the documented schema.

## Scripts

| Script | Purpose | Paper targets |
|---|---|---|
| `fp_pitfall_demo.py` | **Self-contained.** Reproduces the floating-point summation-order artefact that produces a spurious leftward shift in MC ranks for permutation-invariant statistics. | §8.4, Fig. `fp_bug_demonstration` |
| `regenerate_all_figures.py` | Master figure producer; concatenates the per-figure builders. | Figs 3, 4, 5, 6, 7, 8, MC-rank distributions, portfolio right-shift, portfolio next-OOS deciles, cross-asset forest, family heatmap, cost sensitivity, gold MC, synthetic ground truth |
| `strategy_correlations.py` | Within-family / cross-family correlation table from `corr_rs` output. | Table `family_corr` (§3.6) |
| `correlation_figures.py` | Per-asset correlation panels (supporting material). | — |
| `full_analysis.py` | Tables 4 (MC rank summary), 5 (filter ranking), 6 (filter ranking summary pooled), 7 (correlations), 15 (window-level bootstrap CI). | Tables 4, 5, 6, 7, 15 |
| `crypto_stratified_analysis.py` | Tables 8 (MC by family), 17 (MC selection bias), 18 (PF-stratified MC). | Tables 8, 17, 18 |
| `block_perm_analysis.py` | Table 19 from path-dependent block permutation. | Table 19 |
| `block_perm_bootstrap.py` | Window-cluster bootstrap CIs on the block-perm lift. | Table 19 supplement |
| `calendar_cluster_bootstrap.py` | Calendar-quarter cluster bootstrap (10k resamples). | Table 15 calendar row |
| `portfolio_mc_analysis.py` | Tables 14 (portfolio MC), 14b/c/d (portfolio next-OOS), top-N portfolio MC. | Tables 14, 14b–d, top-N portfolio MC |
| `reviewer_analyses.py` | Table 16 (cost sensitivity), matched-pool placebo, continuous Sharpe. | Table 16, placebo |
| `synthetic_scenarios.py` | Synthetic A / B / CMP scenarios and signal sweep. | Tables 23, 24, 25 |
| `gold_mc_analysis.py` | Per-asset gold-standard (bar-permutation, Aronson/Masters) MC + placebo. | §7.5, Fig `gold_mc` |

## Reproducibility

- All RNG is seeded (`42`). Bootstraps use 10,000 resamples per estimate.
- Per-strategy MC permutations use deterministic seeds derived from
  `(strategy_index, window)`, so per-cell ranks are reproducible.
- Lift estimates are stable across three independent seed sequences
  (mean lift varies by less than 0.05 pp).
- Tested with Python 3.12, NumPy 1.26, pandas 2.2, SciPy 1.13,
  matplotlib 3.9.

## Runtime guide (16-core workstation)

- `fp_pitfall_demo.py`: ~30 s
- `regenerate_all_figures.py`: ~5 min
- `calendar_cluster_bootstrap.py`, `block_perm_bootstrap.py`: ~1–2 min
- `correlation_figures.py`: ~2–3 min
- all others: seconds
