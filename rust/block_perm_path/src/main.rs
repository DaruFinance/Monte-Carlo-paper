//! block_perm_path: block-permutation MC ranks for PATH-DEPENDENT statistics
//!
//! Replaces the broken sum-based `iid_rank` and `block{N}_rank` columns
//! produced by `block_perm` (which use a permutation-invariant `equity_roi`
//! statistic that resolves to floating-point summation noise). This binary
//! computes ranks for path-dependent maximum-drawdown statistics instead,
//! for the same block sizes as the original pipeline.
//!
//! Convention: rank = (count(permuted_MDD > actual_MDD)) / B * 100, where
//! MDD is reported as a positive dollar drawdown. Higher rank = less-bad
//! actual drawdown than typical permutation. Under exchangeability the
//! distribution is uniform on [0,100].
//!
//! Input:  <base_dir>/<family>/<strategy>/trades.bin
//! Output: CSV with columns
//!         strategy, window, n_trades, iid_mdd_rank, block2_mdd_rank,
//!         block3_mdd_rank, block5_mdd_rank, block10_mdd_rank, block20_mdd_rank
//!
//! Run:    block_perm_path <base_dir> <n_mc> <out_csv>

use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const INIT_EQUITY: f64 = 1000.0;
const BLOCK_SIZES: [usize; 5] = [2, 3, 5, 10, 20];

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
fn mdd(pnls: &[f64]) -> f64 {
    let mut eq = INIT_EQUITY;
    let mut peak = INIT_EQUITY;
    let mut max_dd = 0.0f64;
    for &p in pnls {
        eq += p;
        if eq > peak { peak = eq; }
        let dd = peak - eq;
        if dd > max_dd { max_dd = dd; }
    }
    max_dd
}

fn mc_iid_mdd_rank(pnls: &[f64], n_mc: u32, rng: &mut SmallRng) -> f64 {
    let actual = mdd(pnls);
    let mut work: Vec<f64> = pnls.to_vec();
    let n = work.len();
    let mut count = 0u32;
    for _ in 0..n_mc {
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            work.swap(i, j);
        }
        if actual < mdd(&work) { count += 1; }
    }
    count as f64 / n_mc as f64 * 100.0
}

fn mc_block_mdd_rank(pnls: &[f64], block_size: usize, n_mc: u32, rng: &mut SmallRng) -> f64 {
    let n = pnls.len();
    if n < block_size * 2 {
        return mc_iid_mdd_rank(pnls, n_mc, rng);
    }
    let actual = mdd(pnls);
    let n_blocks = n / block_size;
    let remainder = n % block_size;
    let blocks: Vec<&[f64]> = (0..n_blocks)
        .map(|i| &pnls[i * block_size..(i + 1) * block_size])
        .collect();
    let tail: &[f64] = if remainder > 0 { &pnls[n_blocks * block_size..] } else { &[] };
    let mut indices: Vec<usize> = (0..n_blocks).collect();
    let mut perm_buf: Vec<f64> = Vec::with_capacity(n);
    let mut count = 0u32;
    for _ in 0..n_mc {
        indices.shuffle(rng);
        perm_buf.clear();
        for &bi in &indices { perm_buf.extend_from_slice(blocks[bi]); }
        perm_buf.extend_from_slice(tail);
        if actual < mdd(&perm_buf) { count += 1; }
    }
    count as f64 / n_mc as f64 * 100.0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: block_perm_path <base_dir> <n_mc> <out_csv>");
        std::process::exit(1);
    }
    let base_dir = &args[1];
    let n_mc: u32 = args[2].parse().unwrap_or(500);
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
    eprintln!("Found {} strategies. n_mc={}, block_sizes={:?}", n_strats, n_mc, BLOCK_SIZES);

    let all: Vec<Vec<(String, u32, usize, f64, [f64; 5])>> = strategies
        .par_iter()
        .enumerate()
        .map(|(si, (name, p))| {
            let wins = read_trades_bin(p, 10);
            let mut out = Vec::with_capacity(wins.len());
            for (w, pnls) in &wins {
                let base_seed = (si as u64).wrapping_mul(10_007) + *w as u64;
                let mut rng_iid = SmallRng::seed_from_u64(base_seed);
                let iid = mc_iid_mdd_rank(pnls, n_mc, &mut rng_iid);
                let mut br = [0.0f64; 5];
                for (bi, &bs) in BLOCK_SIZES.iter().enumerate() {
                    let seed = base_seed.wrapping_add(100_000u64 * (bi as u64 + 1));
                    let mut rng = SmallRng::seed_from_u64(seed);
                    br[bi] = mc_block_mdd_rank(pnls, bs, n_mc, &mut rng);
                }
                out.push((name.clone(), *w, pnls.len(), iid, br));
            }
            if si % 5000 == 0 && si > 0 {
                eprintln!("  {}/{}", si, n_strats);
            }
            out
        })
        .collect();

    let f = fs::File::create(out_path).expect("create out");
    let mut w = BufWriter::new(f);
    write!(w, "strategy,window,n_trades,iid_mdd_rank").unwrap();
    for bs in &BLOCK_SIZES {
        write!(w, ",block{}_mdd_rank", bs).unwrap();
    }
    writeln!(w).unwrap();
    let mut total = 0u64;
    for batch in &all {
        for (name, win, nt, iid, br) in batch {
            write!(w, "\"{}\",W{:02},{},{:.1}", name, win, nt, iid).unwrap();
            for v in br { write!(w, ",{:.1}", v).unwrap(); }
            writeln!(w).unwrap();
            total += 1;
        }
    }
    w.flush().unwrap();
    eprintln!("Wrote {} rows to {}", total, out_path);
}
