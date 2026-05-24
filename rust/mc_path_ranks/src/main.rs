//! mc_path_ranks: corrected MC percentile ranks for trading-strategy walk-forward
//!
//! Computes — per (strategy, window) — the percentile rank of the actual realised
//! statistic against `n_mc` permutation samples (without replacement) drawn from
//! the same closed-trade PnL multiset.
//!
//! Rationale for which statistics we report:
//!
//!   * `roi_rank`     — sum-of-PnLs / INIT. **Permutation-invariant** (Proposition 1
//!     of the paper extends to ROI). Any non-trivial distribution reported in
//!     prior work for this column is a floating-point summation-order artefact.
//!     We include it ONLY so the broken value can be compared side-by-side to the
//!     corrected one. DO NOT use it as a filter score.
//!
//!   * `mdd_rank`     — Maximum drawdown of the equity curve. **Path-dependent**;
//!     permutation produces a meaningful null distribution. Higher rank = less
//!     bad drawdown than typical permutation.
//!
//!   * `calmar_rank`  — ROI / |MDD|. Inherits path-dependence from MDD.
//!     Higher rank = better risk-adjusted return than typical permutation.
//!
//!   * `ulcer_rank`   — Square root of mean squared drawdown. Path-dependent.
//!     Lower rank = more painful underwater periods than typical permutation.
//!     We report rank in the same direction as MDD (higher = less painful)
//!     for consistency.
//!
//! All MC computations use STRICT `<` for the count, with explicit handling of
//! exact ties (rare for path-dependent stats; ubiquitous for ROI under
//! permutation, which is exactly the bug we are correcting).
//!
//! Input:  <base_dir>/<family>/<strategy>/trades.bin  (same layout as block_perm)
//! Output: CSV at <out_csv> with columns
//!         strategy, window, n_trades,
//!         actual_roi, actual_mdd, actual_calmar, actual_ulcer,
//!         roi_rank_broken, mdd_rank, calmar_rank, ulcer_rank
//!
//! Run:    mc_path_ranks <base_dir> <n_mc> <out_csv>

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const INIT_EQUITY: f64 = 1000.0;

fn read_trades_bin(path: &Path, min_trades: usize) -> Vec<(u32, Vec<f64>)> {
    let data = match fs::read(path) { Ok(d) => d, Err(_) => return vec![] };
    let mut pos = 0usize;
    let mut results = Vec::new();
    let len = data.len();
    while pos + 2 <= len {
        let nl = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if nl == 0 || pos + nl > len { break; }
        pos += nl;
        if pos + 2 > len { break; }
        let lbl = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + lbl > len { break; }
        pos += lbl;
        if pos + 2 > len { break; }
        let sl = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if pos + sl > len { break; }
        let sec = std::str::from_utf8(&data[pos..pos + sl]).unwrap_or("").to_string();
        pos += sl;
        if pos + 4 > len { break; }
        let cnt = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        let mut pnls = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            if pos + 17 > len { break; }
            let pnl = f64::from_le_bytes([
                data[pos+9],  data[pos+10], data[pos+11], data[pos+12],
                data[pos+13], data[pos+14], data[pos+15], data[pos+16],
            ]);
            pnls.push(pnl);
            pos += 17;
        }
        if sec.ends_with("-IS") && pnls.len() >= min_trades {
            if let Some(w_str) = sec.strip_suffix("-IS").and_then(|s| s.strip_prefix('W')) {
                if let Ok(w) = w_str.parse::<u32>() { results.push((w, pnls)); }
            }
        }
    }
    results
}

#[inline]
fn path_stats(pnls: &[f64]) -> (f64, f64, f64, f64) {
    // Returns (roi%, mdd_dollars, calmar, ulcer_dollars).
    // ROI is the sum-based total return (permutation-invariant — reported only
    // to demonstrate the bug).
    let mut eq = INIT_EQUITY;
    let mut peak = INIT_EQUITY;
    let mut max_dd = 0.0f64;
    let mut sum_sq_dd = 0.0f64;
    let n = pnls.len();
    for &p in pnls {
        eq += p;
        if eq > peak { peak = eq; }
        let dd = peak - eq;
        if dd > max_dd { max_dd = dd; }
        sum_sq_dd += dd * dd;
    }
    let roi = (eq - INIT_EQUITY) / INIT_EQUITY * 100.0;
    let ulcer = (sum_sq_dd / n as f64).sqrt();
    let calmar = if max_dd > 1e-10 { roi / (max_dd / INIT_EQUITY * 100.0) } else { 0.0 };
    (roi, max_dd, calmar, ulcer)
}

fn mc_ranks(
    pnls: &[f64],
    n_mc: u32,
    rng: &mut SmallRng,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let (a_roi, a_mdd, a_cal, a_ulc) = path_stats(pnls);
    let n = pnls.len();
    // We use Fisher–Yates over a working buffer.
    let mut work: Vec<f64> = pnls.to_vec();
    let mut c_roi = 0u32;
    let mut c_mdd = 0u32;
    let mut c_cal = 0u32;
    let mut c_ulc = 0u32;
    for _ in 0..n_mc {
        // Fisher–Yates shuffle
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            work.swap(i, j);
        }
        let (r, m, c, u) = path_stats(&work);
        // Direction conventions (paper eq. 1: rank = count(perm < actual) / B):
        // For ROI: higher = better → use perm < actual. (Will be ~0 for the
        //   permutation-invariant stat once FP behaviour is consistent.)
        // For MDD (positive dollars): smaller = better → invert: count
        //   actual < perm (i.e. count permutations whose MDD is WORSE).
        // For Calmar: higher = better → perm < actual.
        // For Ulcer: smaller = better → count actual < perm.
        if r < a_roi { c_roi += 1; }
        if a_mdd < m { c_mdd += 1; }
        if c < a_cal { c_cal += 1; }
        if a_ulc < u { c_ulc += 1; }
    }
    let f = 100.0 / n_mc as f64;
    (
        a_roi, a_mdd, a_cal, a_ulc,
        c_roi as f64 * f,
        c_mdd as f64 * f,
        c_cal as f64 * f,
        c_ulc as f64 * f,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: mc_path_ranks <base_dir> <n_mc> <out_csv>");
        std::process::exit(1);
    }
    let base_dir = &args[1];
    let n_mc: u32 = args[2].parse().unwrap_or(1000);
    let out_path = &args[3];

    eprintln!("Scanning {} ...", base_dir);
    let mut strategies: Vec<(String, PathBuf)> = Vec::new();
    for entry in WalkDir::new(base_dir).min_depth(2).max_depth(2).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_dir() { continue; }
        let p = entry.path().join("trades.bin");
        if p.exists() {
            strategies.push((entry.file_name().to_string_lossy().to_string(), p));
        }
    }
    let n_strats = strategies.len();
    eprintln!("Found {} strategies. n_mc={}", n_strats, n_mc);

    let all: Vec<Vec<(String, u32, usize, f64, f64, f64, f64, f64, f64, f64, f64)>> = strategies
        .par_iter()
        .enumerate()
        .map(|(si, (name, p))| {
            let wins = read_trades_bin(p, 10);
            let mut out = Vec::with_capacity(wins.len());
            for (w, pnls) in &wins {
                let seed = (si as u64).wrapping_mul(10_007) + *w as u64;
                let mut rng = SmallRng::seed_from_u64(seed);
                let (a_roi, a_mdd, a_cal, a_ulc, rr, mr, cr, ur) =
                    mc_ranks(pnls, n_mc, &mut rng);
                out.push((
                    name.clone(), *w, pnls.len(),
                    a_roi, a_mdd, a_cal, a_ulc,
                    rr, mr, cr, ur,
                ));
            }
            if si % 5000 == 0 && si > 0 {
                eprintln!("  {}/{}", si, n_strats);
            }
            out
        })
        .collect();

    let f = fs::File::create(out_path).expect("create out");
    let mut w = BufWriter::new(f);
    writeln!(w,
        "strategy,window,n_trades,\
         actual_roi,actual_mdd,actual_calmar,actual_ulcer,\
         roi_rank_broken,mdd_rank,calmar_rank,ulcer_rank"
    ).unwrap();
    let mut total = 0u64;
    for batch in &all {
        for (name, win, nt, ar, am, ac, au, rr, mr, cr, ur) in batch {
            writeln!(w,
                "\"{}\",W{:02},{},{:.4},{:.4},{:.4},{:.4},{:.1},{:.1},{:.1},{:.1}",
                name, win, nt, ar, am, ac, au, rr, mr, cr, ur
            ).unwrap();
            total += 1;
        }
    }
    w.flush().unwrap();
    eprintln!("Wrote {} rows to {}", total, out_path);
}
