//! portfolio_mc_path: portfolio-level MC ranks under permutation, MDD-based.
//!
//! For each asset, sample N_PORT random portfolios of size PORT_SIZE drawn from
//! the IS-PF>1 strategy pool within each walk-forward window, combine the IS
//! trade PnLs, and compute MC ranks of the actual portfolio MDD, Calmar, and ROI
//! against permutations of the combined trade stream.
//!
//! Output: CSV with columns
//!   asset, window, port_id, n_trades_combined,
//!   actual_roi, actual_mdd, actual_calmar,
//!   port_mdd_rank, port_calmar_rank, port_roi_rank_broken
//!
//! Run: portfolio_mc_path <base_dir> <asset_label> <n_port> <port_size> <n_mc> <out_csv>

use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use walkdir::WalkDir;

const INIT_EQUITY: f64 = 1000.0;
const PF_CAP: f64 = 100.0;

#[derive(Clone)]
struct WindowPnls { window: u32, pnls: Vec<f64>, is_pf: f64 }

fn read_sections(path: &Path) -> Vec<WindowPnls> {
    let data = match fs::read(path) { Ok(d) => d, Err(_) => return vec![] };
    let mut pos = 0usize;
    let mut results = Vec::new();
    let len = data.len();
    while pos + 2 <= len {
        let nl = u16::from_le_bytes([data[pos], data[pos+1]]) as usize;
        pos += 2;
        if nl == 0 || pos + nl > len { break; }
        pos += nl;
        if pos + 2 > len { break; }
        let lbl = u16::from_le_bytes([data[pos], data[pos+1]]) as usize;
        pos += 2;
        if pos + lbl > len { break; }
        pos += lbl;
        if pos + 2 > len { break; }
        let sl = u16::from_le_bytes([data[pos], data[pos+1]]) as usize;
        pos += 2;
        if pos + sl > len { break; }
        let sec = std::str::from_utf8(&data[pos..pos+sl]).unwrap_or("").to_string();
        pos += sl;
        if pos + 4 > len { break; }
        let cnt = u32::from_le_bytes([data[pos],data[pos+1],data[pos+2],data[pos+3]]) as usize;
        pos += 4;
        let mut pnls = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            if pos + 17 > len { break; }
            let p = f64::from_le_bytes([
                data[pos+9], data[pos+10], data[pos+11], data[pos+12],
                data[pos+13], data[pos+14], data[pos+15], data[pos+16],
            ]);
            pnls.push(p);
            pos += 17;
        }
        if sec.ends_with("-IS") && pnls.len() >= 5 {
            if let Some(ws) = sec.strip_suffix("-IS").and_then(|s| s.strip_prefix('W')) {
                if let Ok(w) = ws.parse::<u32>() {
                    let mut pos_sum = 0.0_f64;
                    let mut neg_sum = 0.0_f64;
                    for &p in &pnls {
                        if p > 0.0 { pos_sum += p; } else if p < 0.0 { neg_sum -= p; }
                    }
                    let is_pf = if neg_sum > 1e-12 { pos_sum / neg_sum }
                                else if pos_sum > 0.0 { PF_CAP } else { 1.0 };
                    results.push(WindowPnls { window: w, pnls, is_pf });
                }
            }
        }
    }
    results
}

#[inline]
fn mdd_calmar_roi(pnls: &[f64]) -> (f64, f64, f64) {
    let mut eq = INIT_EQUITY;
    let mut peak = INIT_EQUITY;
    let mut max_dd = 0.0_f64;
    for &p in pnls {
        eq += p;
        if eq > peak { peak = eq; }
        let dd = peak - eq;
        if dd > max_dd { max_dd = dd; }
    }
    let roi = (eq - INIT_EQUITY) / INIT_EQUITY * 100.0;
    let cal = if max_dd > 1e-10 { roi / (max_dd / INIT_EQUITY * 100.0) } else { 0.0 };
    (max_dd, cal, roi)
}

fn mc_rank_one_portfolio(combined: &[f64], n_mc: u32, rng: &mut SmallRng) -> (f64, f64, f64, f64, f64, f64) {
    let (a_mdd, a_cal, a_roi) = mdd_calmar_roi(combined);
    let n = combined.len();
    let mut work: Vec<f64> = combined.to_vec();
    let mut c_mdd = 0u32;
    let mut c_cal = 0u32;
    let mut c_roi = 0u32;
    for _ in 0..n_mc {
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            work.swap(i, j);
        }
        let (m, c, r) = mdd_calmar_roi(&work);
        if a_mdd < m { c_mdd += 1; }
        if c < a_cal { c_cal += 1; }
        if r < a_roi { c_roi += 1; }
    }
    let f = 100.0 / n_mc as f64;
    (a_roi, a_mdd, a_cal,
     c_mdd as f64 * f, c_cal as f64 * f, c_roi as f64 * f)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!("Usage: portfolio_mc_path <base_dir> <asset_label> <n_port> <port_size> <n_mc> <out_csv>");
        std::process::exit(1);
    }
    let base_dir = &args[1];
    let asset = &args[2];
    let n_port: u32 = args[3].parse().unwrap_or(1000);
    let port_size: usize = args[4].parse().unwrap_or(10);
    let n_mc: u32 = args[5].parse().unwrap_or(200);
    let out_path = &args[6];

    eprintln!("[{}] scanning {} ...", asset, base_dir);
    let mut strategy_paths: Vec<(String, PathBuf)> = Vec::new();
    for entry in WalkDir::new(base_dir).min_depth(2).max_depth(2).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_dir() { continue; }
        let p = entry.path().join("trades.bin");
        if p.exists() {
            strategy_paths.push((entry.file_name().to_string_lossy().to_string(), p));
        }
    }
    eprintln!("[{}] {} strategies found; reading all trades.bin ...", asset, strategy_paths.len());

    // Single-pass load (parallel): for each strategy, get vec of WindowPnls.
    let loaded: Vec<(String, Vec<WindowPnls>)> = strategy_paths
        .par_iter()
        .map(|(name, path)| (name.clone(), read_sections(path)))
        .collect();
    eprintln!("[{}] all trades read; building pool index ...", asset);

    // Build per-window pool of IS-PF>1 strategy indices, plus a map (strat_idx,win)->pnls.
    let mut window_pools: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut by_key: HashMap<(usize, u32), Vec<f64>> = HashMap::new();
    for (si, (_, wins)) in loaded.iter().enumerate() {
        for wp in wins {
            if wp.is_pf > 1.0 && wp.pnls.len() >= 5 {
                window_pools.entry(wp.window).or_default().push(si);
                by_key.insert((si, wp.window), wp.pnls.clone());
            }
        }
    }
    let mut win_keys: Vec<u32> = window_pools.keys().copied().collect();
    win_keys.sort();
    eprintln!("[{}] {} windows with IS-PF>1 pool sizes: {:?}",
              asset, win_keys.len(),
              win_keys.iter().map(|w| window_pools[w].len()).collect::<Vec<_>>());

    // For each window, sample portfolios and run MC.
    let by_key = Mutex::new(by_key); // shared read-only after this
    let by_key = by_key.into_inner().unwrap();

    let rows: Vec<(u32, u32, usize, f64, f64, f64, f64, f64, f64)> = win_keys
        .par_iter()
        .flat_map(|&win| {
            let pool: &Vec<usize> = &window_pools[&win];
            if pool.len() < port_size { return Vec::new(); }
            let pool_len = pool.len();
            let seed = (win as u64) * 9931 + asset.bytes().fold(0u64, |a, b| a.wrapping_add(b as u64));
            let mut rng_sample = SmallRng::seed_from_u64(seed);
            let mut rng_mc = SmallRng::seed_from_u64(seed.wrapping_add(7));
            let mut out = Vec::with_capacity(n_port as usize);
            for pi in 0..n_port {
                // Random sample without replacement of port_size strategy indices
                let mut chosen: Vec<usize> = Vec::with_capacity(port_size);
                let mut tries = 0;
                while chosen.len() < port_size && tries < 100 * port_size {
                    let cand = pool[rng_sample.gen_range(0..pool_len)];
                    if !chosen.contains(&cand) { chosen.push(cand); }
                    tries += 1;
                }
                if chosen.len() < port_size { continue; }
                let mut combined: Vec<f64> = Vec::new();
                for &si in &chosen {
                    if let Some(p) = by_key.get(&(si, win)) {
                        combined.extend_from_slice(p);
                    }
                }
                if combined.len() < 5 { continue; }
                let (a_roi, a_mdd, a_cal, mdd_r, cal_r, roi_r) =
                    mc_rank_one_portfolio(&combined, n_mc, &mut rng_mc);
                out.push((win, pi, combined.len(), a_roi, a_mdd, a_cal, mdd_r, cal_r, roi_r));
            }
            out
        })
        .collect();

    let f = fs::File::create(out_path).expect("create out");
    let mut w = BufWriter::new(f);
    writeln!(w, "asset,window,port_id,n_trades_combined,actual_roi,actual_mdd,actual_calmar,port_mdd_rank,port_calmar_rank,port_roi_rank_broken").unwrap();
    for (win, pi, ntc, ar, am, ac, mr, cr, rr) in &rows {
        writeln!(w, "{},W{:02},{},{},{:.4},{:.4},{:.4},{:.1},{:.1},{:.1}",
                 asset, win, pi, ntc, ar, am, ac, mr, cr, rr).unwrap();
    }
    w.flush().unwrap();
    eprintln!("[{}] wrote {} portfolios to {}", asset, rows.len(), out_path);
}
