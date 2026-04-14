//! block_perm: block-permutation Monte Carlo ranks for per-window strategy PnL.
//!
//! Produces the `block_perm_<asset>.csv` files that feed Table 19 (block
//! permutation MC test, all 9 instruments) and Figure 3 (bootstrap lift
//! distributions) of the paper. Downstream Python consumers are
//! `block_perm_analysis.py` and `calendar_cluster_bootstrap.py`.
//!
//! Input:  a directory laid out as `<base_dir>/<family>/<strategy>/trades.bin`
//!         where `trades.bin` is the backtester's binary trade-log format.
//!         Binary layout (little-endian):
//!           u16 name_len, <name bytes>,
//!           u16 lb_len,   <lookback bytes>,
//!           u16 sec_len,  <section bytes, e.g. "W03-IS">,
//!           u32 count,
//!           count * (u32 entry, u32 exit, u8 dir, f64 pnl)  [17 bytes/trade]
//!         Only IS sections with >= 10 trades are analysed.
//!
//! Output: CSV to stdout with columns
//!         strategy, window, n_trades, iid_rank, block2_rank, block3_rank,
//!         block5_rank, block10_rank, block20_rank
//!
//! Run:    cargo run --release --bin block_perm -- <base_dir> <n_mc> > out.csv
//!
//! Seeds are derived deterministically from (strategy_index, window, block_size)
//! so results are reproducible.

use rayon::prelude::*;
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Parse trades.bin binary format.
/// Returns Vec<(window_num, Vec<f64 pnl>)> for IS sections with >= min_trades.
fn read_trades_bin(path: &Path, min_trades: usize) -> Vec<(u32, Vec<f64>)> {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(_) => return vec![],
    };
    let mut pos = 0;
    let mut results = Vec::new();
    let len = data.len();

    while pos + 2 <= len {
        // name
        let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if name_len == 0 || pos + name_len > len {
            break;
        }
        pos += name_len;

        // lb
        if pos + 2 > len { break; }
        let lb_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + lb_len > len { break; }
        pos += lb_len;

        // sec
        if pos + 2 > len { break; }
        let sec_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + sec_len > len { break; }
        let sec = std::str::from_utf8(&data[pos..pos + sec_len]).unwrap_or("");
        pos += sec_len;

        // count
        if pos + 4 > len { break; }
        let count = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        // Parse trades - extract pnl values
        let mut pnls = Vec::with_capacity(count);
        for _ in 0..count {
            if pos + 17 > len { break; }
            // Layout: u32 entry (4) + u32 exit (4) + u8 dir (1) + f64 pnl (8) = 17 bytes
            let pnl = f64::from_le_bytes([
                data[pos + 9], data[pos + 10], data[pos + 11], data[pos + 12],
                data[pos + 13], data[pos + 14], data[pos + 15], data[pos + 16],
            ]);
            pnls.push(pnl);
            pos += 17;
        }

        // Only keep IS sections with enough trades
        // Section format: W01-IS, W02-IS, etc.
        if sec.ends_with("-IS") && pnls.len() >= min_trades {
            if let Some(w_str) = sec.strip_suffix("-IS").and_then(|s| s.strip_prefix('W')) {
                if let Ok(w) = w_str.parse::<u32>() {
                    results.push((w, pnls));
                }
            }
        }
    }
    results
}

#[inline]
fn equity_roi(pnls: &[f64]) -> f64 {
    let mut equity: f64 = 1000.0;
    for &p in pnls {
        equity += p;
    }
    (equity - 1000.0) / 1000.0 * 100.0
}

fn mc_iid_rank(pnls: &[f64], n_mc: u32, rng: &mut SmallRng) -> f64 {
    let actual = equity_roi(pnls);
    let mut arr: Vec<f64> = pnls.to_vec();
    let mut count: u32 = 0;
    for _ in 0..n_mc {
        arr.shuffle(rng);
        if equity_roi(&arr) < actual {
            count += 1;
        }
    }
    count as f64 / n_mc as f64 * 100.0
}

fn mc_block_rank(pnls: &[f64], block_size: usize, n_mc: u32, rng: &mut SmallRng) -> f64 {
    let n = pnls.len();
    if n < block_size * 2 {
        return mc_iid_rank(pnls, n_mc, rng);
    }
    let actual = equity_roi(pnls);
    let n_blocks = n / block_size;
    let remainder = n % block_size;

    // Pre-build block slices
    let blocks: Vec<&[f64]> = (0..n_blocks)
        .map(|i| &pnls[i * block_size..(i + 1) * block_size])
        .collect();
    let tail = if remainder > 0 {
        &pnls[n_blocks * block_size..]
    } else {
        &[] as &[f64]
    };

    let mut indices: Vec<usize> = (0..n_blocks).collect();
    let mut perm_buf: Vec<f64> = Vec::with_capacity(n);

    let mut count: u32 = 0;
    for _ in 0..n_mc {
        indices.shuffle(rng);
        perm_buf.clear();
        for &bi in &indices {
            perm_buf.extend_from_slice(blocks[bi]);
        }
        perm_buf.extend_from_slice(tail);
        if equity_roi(&perm_buf) < actual {
            count += 1;
        }
    }
    count as f64 / n_mc as f64 * 100.0
}

struct StrategyEntry {
    name: String,
    bin_path: PathBuf,
}

const BLOCK_SIZES: [usize; 5] = [2, 3, 5, 10, 20];

struct WindowResult {
    strategy: String,
    window: u32,
    n_trades: usize,
    iid_rank: f64,
    block_ranks: [f64; 5], // one per BLOCK_SIZES
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: block_perm <strategy_base_dir> <n_mc>");
        eprintln!("  Outputs CSV to stdout with columns for block sizes {:?}", BLOCK_SIZES);
        std::process::exit(1);
    }
    let base_dir = &args[1];
    let n_mc: u32 = args[2].parse().unwrap_or(500);

    eprintln!("Scanning {} for trades.bin files...", base_dir);

    // Discover all strategies: base_dir/family/strategy/trades.bin
    let mut strategies: Vec<StrategyEntry> = Vec::new();
    for entry in WalkDir::new(base_dir).min_depth(2).max_depth(2).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_dir() {
            continue;
        }
        let path = entry.path().join("trades.bin");
        if path.exists() {
            let name = entry.file_name().to_string_lossy().to_string();
            strategies.push(StrategyEntry {
                name,
                bin_path: path,
            });
        }
    }

    let n_strats = strategies.len();
    eprintln!("Found {} strategies with trades.bin", n_strats);
    eprintln!("Running MC permutations with n_mc={}, block sizes {:?}", n_mc, BLOCK_SIZES);

    // Process all strategies in parallel using rayon
    let all_results: Vec<Vec<WindowResult>> = strategies
        .par_iter()
        .enumerate()
        .map(|(si, strat)| {
            let windows = read_trades_bin(&strat.bin_path, 10);
            let mut results = Vec::new();

            for (w, pnls) in &windows {
                let base_seed = (si as u64) * 1000 + (*w as u64);

                let mut rng_iid = SmallRng::seed_from_u64(base_seed);
                let iid_rank = mc_iid_rank(pnls, n_mc, &mut rng_iid);

                let mut block_ranks = [0.0f64; 5];
                for (bi, &bs) in BLOCK_SIZES.iter().enumerate() {
                    let seed = base_seed + 100_000 * (bi as u64 + 1);
                    let mut rng = SmallRng::seed_from_u64(seed);
                    block_ranks[bi] = mc_block_rank(pnls, bs, n_mc, &mut rng);
                }

                results.push(WindowResult {
                    strategy: strat.name.clone(),
                    window: *w,
                    n_trades: pnls.len(),
                    iid_rank,
                    block_ranks,
                });
            }

            if si % 5000 == 0 && si > 0 {
                eprintln!("  Progress: {}/{} strategies processed", si, n_strats);
            }

            results
        })
        .collect();

    // Write CSV to stdout
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    write!(writer, "strategy,window,n_trades,iid_rank").unwrap();
    for bs in &BLOCK_SIZES {
        write!(writer, ",block{}_rank", bs).unwrap();
    }
    writeln!(writer).unwrap();

    let mut total = 0u64;
    for batch in &all_results {
        for r in batch {
            write!(
                writer,
                "\"{}\",{},{},{:.2}",
                r.strategy, r.window, r.n_trades, r.iid_rank
            )
            .unwrap();
            for rank in &r.block_ranks {
                write!(writer, ",{:.2}", rank).unwrap();
            }
            writeln!(writer).unwrap();
            total += 1;
        }
    }
    writer.flush().unwrap();
    eprintln!("Done. {} strategy-window observations written.", total);
}
