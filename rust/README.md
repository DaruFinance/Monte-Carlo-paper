# Rust crates

Three independent Cargo crates that generate numerical artifacts for the paper.
They are standalone and can be built in any order, but all three depend on
upstream raw data produced by the backtesting system (not included here).

## Toolchain

- Rust edition **2021**, stable toolchain (tested with 1.75+).
- Parallelism: all three crates use [`rayon`](https://crates.io/crates/rayon) and
  will by default saturate available CPU cores. The synthetic pipeline pins
  itself to 32 worker threads via `rayon::ThreadPoolBuilder` — edit `N_WORKERS`
  in `synthetic_pipeline_rust/src/main.rs` for smaller machines.
- No non-Rust system dependencies. `nalgebra` (used by `corr_tensor`) is pure
  Rust and compiles with the default stable toolchain.
- All seeds are fixed, so rebuilding reproduces the paper numerics bit-for-bit
  on the same architecture.

## Directory layout assumed by defaults

```
Scripts_Clean/
  rust/                   <-- you are here
    block_perm_rs/
    corr_rs/
    synthetic_pipeline_rust/
  ...
results/
  raw_data/               <-- *_window_pairs.csv, *_mc_perwindow.csv, raw OHLC CSVs
  tables_v2/              <-- synthetic_pipeline output lands here by default
```

Every crate accepts explicit paths (CLI arg or env var) so the layout is a
convenience, not a requirement.

---

## 1. `block_perm_rs/` &mdash; block permutation MC ranks

**Produces:** `block_perm_<asset>.csv` files (one per asset) that feed
**Table 19** (block permutation MC test) and **Figure 3** (bootstrap lift
distributions) via the Python scripts `block_perm_analysis.py` and
`calendar_cluster_bootstrap.py`.

**Binaries:**

- `block_perm` &mdash; block-permutation ROI ranks at block sizes
  b = 1, 2, 3, 5, 10, 20.
- `mc_sharpe_pf` &mdash; bootstrap MC ranks for ROI / Sharpe / PF in the
  `raw_data/<asset>_mc_perwindow.csv` schema used throughout the paper.

**Input:** a directory laid out as
`<base_dir>/<family>/<strategy>/trades.bin`, where `trades.bin` is the
backtester's binary trade-log format. The layout, in little-endian:

```
u16 name_len, <name bytes>,
u16 lb_len,   <lookback bytes>,
u16 sec_len,  <section bytes, e.g. "W03-IS">,
u32 count,
count * (u32 entry, u32 exit, u8 dir, f64 pnl)   [17 bytes / trade]
```

Only IS sections with at least 10 trades are analysed. The binary format is
produced by an external backtester (not part of this repository); the parsing
code is kept here so that the CSVs can be regenerated from raw trade logs.

**Build and run:**

```bash
cd block_perm_rs
cargo build --release
cargo run --release --bin block_perm -- <base_dir> <n_mc> > block_perm_btc.csv
cargo run --release --bin mc_sharpe_pf -- <base_dir> <n_mc> btc_mc_perwindow.csv
```

Suggested `n_mc` for paper reproduction: `1000`.

**Output columns (`block_perm`):**
`strategy, window, n_trades, iid_rank, block2_rank, block3_rank, block5_rank,
block10_rank, block20_rank`

**Output columns (`mc_sharpe_pf`):**
`strategy, window, n_trades, actual_roi, actual_sharpe, actual_pf,
roi_pct_rank, sharpe_pct_rank, pf_pct_rank`

---

## 2. `corr_rs/` &mdash; strategy correlation + cross-asset tensor

**Produces:** inputs for **Figure 1** (within-family vs cross-family bar-level
PnL correlation) and **Table 3**, consumed by
`correlation_analysis/strategy_correlations.py`. Also produces cross-asset
rolling correlation diagnostics.

**Binaries:**

- `strat_corr` &mdash; sparse bar-level Pearson correlation of strategies
  within one asset, grouped by indicator family
  (ATR / EMA / MACD / PPO / RSI / RSI_LEVEL / SMA / STOCHK). Reads the same
  `trades.bin` format as `block_perm_rs`.

  ```bash
  cargo run --release --bin strat_corr -- \
      <base_dir> <window_size> <instrument> <market> [output_dir]
  # market is one of: crypto | forex | commodity
  ```

  Emits five CSVs per invocation:
  `<instrument>_embed.csv`, `<instrument>_fammatrix.csv`,
  `<instrument>_summary.csv`, `<instrument>_histogram.csv`,
  `<instrument>_perfamily.csv`.

- `corr_tensor` &mdash; 7-instrument rolling correlation tensor on 1h log
  returns. Resamples mixed 15m/30m/1h OHLC CSVs to hourly close and computes
  a 720-bar rolling correlation matrix (step 24 hours).

  ```bash
  cargo run --release --bin corr_tensor -- <data_dir> <out_dir>
  # or: MC_PAPER_DATA_DIR=... MC_PAPER_OUT_DIR=... cargo run --release --bin corr_tensor
  ```

  Expected filenames under `<data_dir>`:
  `BTCUSDT_30m_3_9.csv`, `BNBUSDT_15m_3_9.csv` (time column `time`),
  `EURUSD_1h_clean.csv`, `USDJPY_1h_clean.csv`, `EURGBP_1h_clean.csv`,
  `XAUUSD_1h_clean.csv`, `WTI_1h_clean.csv` (time column `timestamp`).
  SOL and DOGE are excluded because their histories are too short for the
  common-window intersection.

  Outputs: `corr_tensor.csv`, `corr_eigenvalues.csv`, `corr_avg.csv`.

---

## 3. `synthetic_pipeline_rust/` &mdash; three-tier synthetic validation

**Produces:** CSV tables for **Scenarios A/B/C** that feed paper **Figures 5-8**
(`fig_synthetic_mc_ranks.pdf`, `fig_synthetic_mc_analysis.pdf`,
`fig_synthetic_pipeline_v4.pdf`, `fig_synthetic_pipeline_detail.pdf`) via the
Python figure regeneration script. This is the Rust rewrite of
`synthetic_pipeline_v4.py` with parameters matched to the empirical setup
(B = 1000 MC permutations, ~38K strategies, 10 WFO windows).

**Tiers:**

| Tier | Description | Grid |
|---|---|---|
| 1 &mdash; null | pure-null returns, no injected edge | standard |
| 2 &mdash; edge | weak AR(1) momentum edge (phi = 0.04) | standard |
| 3 &mdash; adversarial | adversarial data-mined grid | massive |
| signal sweep | phi in {0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15} | standard |

**Input:** none &mdash; the crate self-generates synthetic returns
(AR(1) momentum + GARCH(1,1) + Student-t innovations with df = 5, plus a
two-state volatility regime).

**Build and run:**

```bash
cd synthetic_pipeline_rust
cargo build --release
cargo run --release                      # writes to ../../results/tables_v2
cargo run --release -- /custom/out/dir   # explicit output directory
MC_PAPER_TBL_DIR=/custom/out/dir cargo run --release   # or via env var
```

**Output CSVs:**

```
synthetic_v4_summaries_matched.csv
synthetic_v4_null_filters_matched.csv
synthetic_v4_edge_filters_matched.csv
synthetic_v4_adversarial_filters_matched.csv
synthetic_v4_signal_sweep_matched.csv
```

Runtime: approximately 30-60 minutes on a 32-core machine at B = 1000.
Reduce `N_MC` or `N_SIMS` in `src/main.rs` for a quicker smoke test.

---

## Reproducibility notes

- All RNGs (`SmallRng`, `StdRng`) are seeded explicitly. `block_perm` derives
  per-window seeds from `(strategy_index, window, block_size)`; `strat_corr`
  uses fixed seed 42; `synthetic_pipeline` uses per-simulation seeds.
- No unsafe code. No FFI.
- Floating-point results are deterministic within a single machine but may
  differ at the ULP level across CPU families (as is normal for rayon +
  fused-multiply-add).
