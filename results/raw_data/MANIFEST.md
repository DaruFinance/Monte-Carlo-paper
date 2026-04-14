# raw_data/ manifest

The actual CSV bytes in this directory are **not tracked by git** (see the
root `.gitignore`). They can be regenerated from the Rust stage (see
`rust/block_perm_rs/`) or the upstream walk-forward backtest. This manifest
lists every file the Python and R scripts expect to find here.

Five of the files below also have companion `*.csv.INFO` stubs committed to
git with column schema, row count, md5, and a first-rows sample.

## Expected files (30 total, ~1.6 GB uncompressed)

### Block permutation output (from `rust/block_perm_rs/`)

| File | Size | Schema |
|---|---|---|
| `block_perm_btc.csv`    | 51 MB | strategy,window,n_trades,iid_rank,block2_rank,block3_rank,block5_rank,block10_rank,block20_rank |
| `block_perm_doge.csv`   | 49 MB | same |
| `block_perm_bnb.csv`    | 72 MB | same |
| `block_perm_sol.csv`    | 16 MB | same |
| `block_perm_eurusd.csv` | 48 MB | same |
| `block_perm_usdjpy.csv` | 52 MB | same |
| `block_perm_eurgbp.csv` | 54 MB | same |
| `block_perm_xauusd.csv` | 52 MB | same |
| `block_perm_wti.csv`    | 52 MB | same |

### Per-window MC rank CSVs

Schema: `strategy, window, roi_pct_rank, sharpe_pct_rank, pf_pct_rank, ...`
(the exact column set is determined by `block_perm_rs` / the upstream
backtester).

| File | Size |
|---|---|
| `btc_mc_perwindow.csv`    | 69 MB |
| `doge_mc_perwindow.csv`   | 53 MB |
| `bnb_mc_perwindow.csv`    | 79 MB |
| `sol_mc_perwindow.csv`    | 18 MB |
| `eurusd_mc_perwindow.csv` | 53 MB |
| `usdjpy_mc_perwindow.csv` | 56 MB |
| `eurgbp_mc_perwindow.csv` | 59 MB |
| `xauusd_mc_perwindow.csv` | 57 MB |
| `wti_mc_perwindow.csv`    | 56 MB |

### IS/OOS window pairs

Schema: `strategy, window_i, baseline_is_pf, baseline_is_sharpe,
baseline_is_roi, baseline_is_trades, ent_is_pf, fee_is_pf, sli_is_pf,
entind_is_pf, baseline_oos_pf, baseline_oos_sharpe, baseline_oos_roi,
baseline_oos_trades, next_baseline_oos_pf, next_baseline_oos_sharpe,
next_baseline_oos_roi, next_baseline_oos_trades`

| File | Size | Notes |
|---|---|---|
| `btc_window_pairs.csv`    | 135 MB | stub `*.csv.INFO` committed |
| `doge_window_pairs.csv`   | 101 MB | stub committed |
| `bnb_window_pairs.csv`    | 151 MB | stub committed |
| `sol_window_pairs.csv`    |  35 MB | — |
| `eurusd_window_pairs.csv` |  90 MB | — |
| `usdjpy_window_pairs.csv` |  91 MB | — |
| `eurgbp_window_pairs.csv` |  90 MB | — |
| `xauusd_window_pairs.csv` |  91 MB | — |
| `wti_window_pairs.csv`    |  91 MB | — |

### Portfolio-level Monte Carlo (crypto only)

| File | Size | Notes |
|---|---|---|
| `btc_portfolio_mc.csv`  |  95 MB | — |
| `sol_portfolio_mc.csv`  |  50 MB | — |
| `bnb_portfolio_mc.csv`  | 211 MB | stub committed |
| `doge_portfolio_mc.csv` | 151 MB | stub committed |

### Per-asset summary

| File | Size |
|---|---|
| `btc_overall.csv`  | 3 MB |
| `doge_overall.csv` | 3 MB |
| `bnb_overall.csv`  | 3 MB |
| `sol_overall.csv`  | 3 MB |

## How to regenerate

The walk-forward aggregates (`*_mc_perwindow.csv`, `*_window_pairs.csv`,
`*_portfolio_mc.csv`, `*_overall.csv`) come from the upstream backtest stage
which is not part of this repository. The block-permutation outputs
(`block_perm_*.csv`) can be reproduced by running `rust/block_perm_rs/`
against the raw `trades.bin` files — see `rust/README.md`.

If you have access to the original data release accompanying the paper,
drop the CSVs into this directory and every Python / R script in the
package will pick them up automatically.
