//! mc_sharpe_pf: per-strategy per-window bootstrap MC ranks for ROI, Sharpe, PF.
//!
//! Produces rows matching the `raw_data/<asset>_mc_perwindow.csv` schema used
//! across the paper (feeds MC-rank analyses such as Table 4 and Figures 2/4/9).
//! Reads the same trades.bin binary format as `block_perm`.
//!
//! Null hypothesis: bootstrap resampling with replacement. Pure permutation is
//! a no-op for order-invariant stats (sum, mean, std, Sharpe, PF).
//!
//! Input:  <base_dir>/<family>/<strategy>/trades.bin
//! Output: CSV at <out_csv> with columns
//!         strategy, window, n_trades, actual_roi, actual_sharpe, actual_pf,
//!         roi_pct_rank, sharpe_pct_rank, pf_pct_rank
//!
//! Run:    cargo run --release --bin mc_sharpe_pf -- <base_dir> <n_mc> <out_csv>

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Starting equity for ROI calculation (USD).
const INIT_EQUITY: f64 = 1000.0;

/// Profit-factor cap: replaces infinite PF when gross loss is near zero.
const PF_CAP: f64 = 100.0;

/// Parse trades.bin binary format.
///
/// Binary layout (little-endian):
///   u16 name_len, <name bytes>,
///   u16 lb_len,   <lookback bytes>,
///   u16 sec_len,  <section bytes, e.g. "W03-IS">,
///   u32 count,
///   count * (u32 entry, u32 exit, u8 dir, f64 pnl)  [17 bytes/trade]
///
/// Returns Vec<(window_num, Vec<f64 pnl>)> for IS sections with >= min_trades.
fn read_trades_bin(path: &Path, min_trades: usize) -> Vec<(u32, Vec<f64>)> {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(_) => return vec![],
    };
    let mut pos = 0usize;
    let mut results = Vec::new();
    let len = data.len();

    while pos + 2 <= len {
        let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if name_len == 0 || pos + name_len > len {
            break;
        }
        pos += name_len;

        if pos + 2 > len {
            break;
        }
        let lb_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + lb_len > len {
            break;
        }
        pos += lb_len;

        if pos + 2 > len {
            break;
        }
        let sec_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + sec_len > len {
            break;
        }
        let sec = std::str::from_utf8(&data[pos..pos + sec_len])
            .unwrap_or("")
            .to_string();
        pos += sec_len;

        if pos + 4 > len {
            break;
        }
        let count = u32::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
        ]) as usize;
        pos += 4;

        let mut pnls = Vec::with_capacity(count);
        for _ in 0..count {
            if pos + 17 > len {
                break;
            }
            // Layout: u32 entry (4) + u32 exit (4) + u8 dir (1) + f64 pnl (8) = 17 bytes
            let pnl = f64::from_le_bytes([
                data[pos + 9],
                data[pos + 10],
                data[pos + 11],
                data[pos + 12],
                data[pos + 13],
                data[pos + 14],
                data[pos + 15],
                data[pos + 16],
            ]);
            pnls.push(pnl);
            pos += 17;
        }

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
fn stats(pnls: &[f64]) -> (f64, f64, f64) {
    let n = pnls.len() as f64;
    let sum: f64 = pnls.iter().sum();
    let mean = sum / n;
    let var: f64 = pnls.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    let sharpe = if std > 1e-12 { mean / std } else { 0.0 };
    let roi = sum / INIT_EQUITY * 100.0;
    let mut pos = 0.0_f64;
    let mut neg = 0.0_f64;
    for &p in pnls {
        if p > 0.0 {
            pos += p;
        } else if p < 0.0 {
            neg -= p;
        }
    }
    let pf = if neg > 1e-12 {
        pos / neg
    } else if pos > 0.0 {
        PF_CAP
    } else {
        1.0
    };
    (roi, sharpe, pf)
}

/// Bootstrap MC percentile ranks for a single strategy-window.
///
/// Returns `(actual_roi, actual_sharpe, actual_pf,
///           roi_pct_rank, sharpe_pct_rank, pf_pct_rank)`.
fn mc_ranks(
    pnls: &[f64],
    n_mc: u32,
    rng: &mut SmallRng,
) -> (f64, f64, f64, f64, f64, f64) {
    let (ar, as_, ap) = stats(pnls);
    let n = pnls.len();
    let mut buf = vec![0.0_f64; n];
    let mut cr = 0u32;
    let mut cs = 0u32;
    let mut cp = 0u32;
    for _ in 0..n_mc {
        for i in 0..n {
            let k = rng.gen_range(0..n);
            buf[i] = pnls[k];
        }
        let (r, s, p) = stats(&buf);
        if r < ar {
            cr += 1;
        }
        if s < as_ {
            cs += 1;
        }
        if p < ap {
            cp += 1;
        }
    }
    let f = 100.0 / n_mc as f64;
    (
        ar,
        as_,
        ap,
        cr as f64 * f,
        cs as f64 * f,
        cp as f64 * f,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: mc_sharpe_pf <base_dir> <n_mc> <out_csv>");
        std::process::exit(1);
    }
    let base_dir = &args[1];
    let n_mc: u32 = args[2].parse().unwrap_or(1000);
    let out_path = &args[3];

    eprintln!("Scanning {} ...", base_dir);
    let mut strategies: Vec<(String, PathBuf)> = Vec::new();
    for entry in WalkDir::new(base_dir)
        .min_depth(2)
        .max_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_dir() {
            continue;
        }
        let path = entry.path().join("trades.bin");
        if path.exists() {
            let name = entry.file_name().to_string_lossy().to_string();
            strategies.push((name, path));
        }
    }
    let n_strats = strategies.len();
    eprintln!("Found {} strategies. n_mc={}", n_strats, n_mc);

    let all: Vec<Vec<(String, u32, usize, f64, f64, f64, f64, f64, f64)>> = strategies
        .par_iter()
        .enumerate()
        .map(|(si, (name, p))| {
            let windows = read_trades_bin(p, 10);
            let mut out = Vec::with_capacity(windows.len());
            for (w, pnls) in &windows {
                let seed = (si as u64).wrapping_mul(10_000) + *w as u64;
                let mut rng = SmallRng::seed_from_u64(seed);
                let (ar, as_, ap, rr, sr, pr) = mc_ranks(pnls, n_mc, &mut rng);
                out.push((name.clone(), *w, pnls.len(), ar, as_, ap, rr, sr, pr));
            }
            if si % 5000 == 0 && si > 0 {
                eprintln!("  {}/{}", si, n_strats);
            }
            out
        })
        .collect();

    let f = fs::File::create(out_path).expect("create out");
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "strategy,window,n_trades,actual_roi,actual_sharpe,actual_pf,roi_pct_rank,sharpe_pct_rank,pf_pct_rank"
    )
    .unwrap();
    let mut total = 0u64;
    for batch in &all {
        for (name, win, nt, ar, as_, ap, rr, sr, pr) in batch {
            writeln!(
                w,
                "\"{}\",W{:02},{},{:.4},{:.4},{:.4},{:.1},{:.1},{:.1}",
                name, win, nt, ar, as_, ap, rr, sr, pr
            )
            .unwrap();
            total += 1;
        }
    }
    w.flush().unwrap();
    eprintln!("Wrote {} rows to {}", total, out_path);
}
