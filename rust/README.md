# Rust crates

Seven independent Cargo crates that produce the numerical artifacts the
Python pipeline consumes. All are parallelised with [rayon] and all use
fixed seeds for bit-reproducible output.

[rayon]: https://crates.io/crates/rayon

| Crate | Purpose | Outputs (under `results/raw_data/`) |
|---|---|---|
| `mc_path_ranks` | Per-window MC percentile ranks on **path-dependent** statistics (MDD / Calmar / Ulcer) and, side-by-side, the **artefactual** ROI rank documented in §8.4 of the paper. | `<asset>_corrected_ranks.csv` |
| `block_perm_path` | Block-permutation MC ranks for the same path-dependent statistics, at block sizes b ∈ {2, 3, 5, 10, 20}. | `block_perm_path_<asset>.csv` |
| `portfolio_mc_path` | Portfolio-level MC ranks under permutation, using path-dependent MDD. | `<asset>_portfolio_mc_path.csv` |
| `portfolio_mc_oos` | As above, augmented with same-window and next-window OOS outcomes per portfolio. | `<asset>_portfolio_mc_oos.csv` |
| `block_perm_rs` | Legacy block-permutation binary for the sum-based ROI/Sharpe/PF ranks. Retained because §8.4 demonstrates that the leftward shift these binaries produce on real data is a floating-point summation-order artefact, not a property of the data; the artefact is reproducible from the `block_perm` binary and verifiable against `python/fp_pitfall_demo.py`. **Do not use this binary as a filter signal.** | `block_perm_<asset>.csv` |
| `corr_rs` | Within-family and cross-family strategy-return correlation matrices. | per-asset correlation tensors |
| `synthetic_pipeline_rust` | Self-contained synthetic AR(1) + GARCH + Student-t scenarios with three signal tiers (Pure Null, Known Edge, Adversarial). | `results/raw_data/synthetic_v4/*.csv` |

## Toolchain

- Rust **2021** edition; stable toolchain, tested on 1.94.
- All crates use `rayon` and will by default saturate the available CPU
  cores. `synthetic_pipeline_rust` pins itself to 32 worker threads via
  `rayon::ThreadPoolBuilder` — edit `N_WORKERS` if your machine has fewer
  cores.
- No non-Rust system dependencies.
- Seeds are fixed; rebuilding reproduces the paper's CSVs bit-for-bit on
  the same architecture.

## Build & run

```bash
# Build everything in release mode
for c in mc_path_ranks block_perm_path portfolio_mc_path portfolio_mc_oos \
         block_perm_rs corr_rs synthetic_pipeline_rust ; do
  ( cd $c && cargo build --release )
done

# Synthetic pipeline (fully self-contained, no external data)
cd synthetic_pipeline_rust
cargo run --release -- ../../results/raw_data/synthetic_v4
```

The other six crates each accept a base directory laid out as
`<base>/<family>/<strategy>/trades.bin`, where `trades.bin` is the
backtester's binary trade-log format. Per-crate `--help` documents the
exact CLI. The backtester that produces this layout is open source at
<https://github.com/DaruFinance/quant-research-framework-rs>.

## Path-dependent statistics, in one paragraph

For an equity curve E_0, E_1, …, E_n built by accumulating realised trade
PnLs, denote drawdown `d_t = E_t - max_{s ≤ t} E_s`. Then:

- **MDD** = max_t |d_t|
- **Calmar** = ROI / |MDD|
- **Ulcer** = sqrt(mean(d_t^2))

These three are functions of the realised trade *ordering*, not just the
multiset of trade PnLs; under a permutation null they produce a
non-degenerate rank distribution. The paper restricts within-strategy MC
filtering to these three; the sum-based statistics (ROI, trade-level
Sharpe, Profit Factor) are proven multiset-invariant in Appendix A4
(Proposition `multiset_invariance`) and are not usable as MC filter
scores.
