//! portfolio_mc_oos: portfolio-level MC ranks computed on IS windows, with
//! same-window and next-window OOS outcomes recorded per portfolio.
//!
//! For each (asset, window) and for each of `n_port` random portfolios drawn
//! from the IS PF>1 pool, we:
//!   1. Combine the IS trade streams of the 10 selected strategies.
//!   2. Compute MC-MDD/Calmar/ROI* ranks via permutation.
//!   3. Look up the SAME-window OOS sections of the same strategies,
//!      combine, compute OOS PF and ROI.
//!   4. Look up the NEXT-window OOS sections of the same strategies,
//!      combine, compute OOS PF and ROI.
//!
//! This adds the forward-OOS test that the strategy-level Section 6 lacks,
//! responding to peer review #1 (quant finance).
//!
//! Output: CSV with columns
//!   asset, window, port_id, n_strategies, n_trades_is, n_trades_oos_same, n_trades_oos_next,
//!   port_mdd_rank, port_calmar_rank, port_roi_rank_broken,
//!   oos_same_pf, oos_same_roi, oos_same_profitable_count,
//!   oos_next_pf, oos_next_roi, oos_next_profitable_count
//!
//! Run: portfolio_mc_oos <base_dir> <asset_label> <n_port> <port_size> <n_mc> <out_csv>

use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const INIT_EQUITY: f64 = 1000.0;
const PF_CAP: f64 = 100.0;

#[derive(Clone)]
struct WindowSection { window: u32, is_pnls: Vec<f64>, oos_pnls: Vec<f64>, is_pf: f64 }

fn read_sections(path: &Path) -> Vec<WindowSection> {
    let data = match fs::read(path) { Ok(d) => d, Err(_) => return vec![] };
    let mut pos = 0usize;
    let len = data.len();
    let mut is_map: HashMap<u32, Vec<f64>> = HashMap::new();
    let mut oos_map: HashMap<u32, Vec<f64>> = HashMap::new();
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
        if sec.ends_with("-IS") {
            if let Some(ws) = sec.strip_suffix("-IS").and_then(|s| s.strip_prefix('W')) {
                if let Ok(w) = ws.parse::<u32>() { is_map.insert(w, pnls); }
            }
        } else if sec.ends_with("-OOS") {
            if let Some(ws) = sec.strip_suffix("-OOS").and_then(|s| s.strip_prefix('W')) {
                if let Ok(w) = ws.parse::<u32>() { oos_map.insert(w, pnls); }
            }
        }
    }
    let mut out = Vec::new();
    let mut keys: Vec<u32> = is_map.keys().copied().collect();
    keys.sort();
    for w in keys {
        let is_pnls = is_map.remove(&w).unwrap_or_default();
        if is_pnls.len() < 5 { continue; }
        let oos_pnls = oos_map.remove(&w).unwrap_or_default();
        let mut pos_sum = 0.0_f64;
        let mut neg_sum = 0.0_f64;
        for &p in &is_pnls {
            if p > 0.0 { pos_sum += p; } else if p < 0.0 { neg_sum -= p; }
        }
        let is_pf = if neg_sum > 1e-12 { pos_sum / neg_sum }
                    else if pos_sum > 0.0 { PF_CAP } else { 1.0 };
        out.push(WindowSection { window: w, is_pnls, oos_pnls, is_pf });
    }
    out
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

#[inline]
fn pf_roi(pnls: &[f64]) -> (f64, f64) {
    let mut pos = 0.0_f64;
    let mut neg = 0.0_f64;
    let mut sum = 0.0_f64;
    for &p in pnls {
        sum += p;
        if p > 0.0 { pos += p; } else if p < 0.0 { neg -= p; }
    }
    let pf = if neg > 1e-12 { pos / neg }
             else if pos > 0.0 { PF_CAP } else { 1.0 };
    (pf, sum / INIT_EQUITY * 100.0)
}

fn mc_rank_one_portfolio(combined: &[f64], n_mc: u32, rng: &mut SmallRng) -> (f64, f64, f64) {
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
    (c_mdd as f64 * f, c_cal as f64 * f, c_roi as f64 * f)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!("Usage: portfolio_mc_oos <base_dir> <asset_label> <n_port> <port_size> <n_mc> <out_csv>");
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

    let loaded: Vec<(String, Vec<WindowSection>)> = strategy_paths
        .par_iter()
        .map(|(name, path)| (name.clone(), read_sections(path)))
        .collect();
    eprintln!("[{}] all trades read; indexing ...", asset);

    // Per (strat_idx, window) -> (is_pnls, oos_pnls)
    let mut by_key: HashMap<(usize, u32), (Vec<f64>, Vec<f64>)> = HashMap::new();
    let mut window_pools: HashMap<u32, Vec<usize>> = HashMap::new();
    for (si, (_, wins)) in loaded.iter().enumerate() {
        for w in wins {
            if w.is_pf > 1.0 && w.is_pnls.len() >= 5 {
                window_pools.entry(w.window).or_default().push(si);
                by_key.insert((si, w.window), (w.is_pnls.clone(), w.oos_pnls.clone()));
            } else if w.is_pnls.len() >= 5 {
                // store anyway so OOS lookup for next window works
                by_key.insert((si, w.window), (w.is_pnls.clone(), w.oos_pnls.clone()));
            }
        }
    }
    let mut win_keys: Vec<u32> = window_pools.keys().copied().collect();
    win_keys.sort();
    eprintln!("[{}] {} windows with IS-PF>1 pool", asset, win_keys.len());

    let rows: Vec<(u32, u32, usize, usize, usize, usize, f64, f64, f64, f64, f64, i32, f64, f64, i32)> = win_keys
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
                let mut chosen: Vec<usize> = Vec::with_capacity(port_size);
                let mut tries = 0;
                while chosen.len() < port_size && tries < 100 * port_size {
                    let cand = pool[rng_sample.gen_range(0..pool_len)];
                    if !chosen.contains(&cand) { chosen.push(cand); }
                    tries += 1;
                }
                if chosen.len() < port_size { continue; }

                // IS combined
                let mut is_combined: Vec<f64> = Vec::new();
                for &si in &chosen {
                    if let Some((isp, _)) = by_key.get(&(si, win)) {
                        is_combined.extend_from_slice(isp);
                    }
                }
                if is_combined.len() < 5 { continue; }
                let n_is = is_combined.len();

                // MC ranks on IS
                let (mdd_r, cal_r, roi_r) = mc_rank_one_portfolio(&is_combined, n_mc, &mut rng_mc);

                // Same-window OOS combined
                let mut oos_same: Vec<f64> = Vec::new();
                let mut n_same_profitable = 0i32;
                for &si in &chosen {
                    if let Some((_, oosp)) = by_key.get(&(si, win)) {
                        oos_same.extend_from_slice(oosp);
                        // count strategies whose individual OOS is profitable
                        let (pf, _) = pf_roi(oosp);
                        if pf > 1.0 { n_same_profitable += 1; }
                    }
                }
                let n_oos_same = oos_same.len();
                let (oos_same_pf, oos_same_roi) = if oos_same.is_empty() { (1.0, 0.0) } else { pf_roi(&oos_same) };

                // Next-window OOS combined
                let mut oos_next: Vec<f64> = Vec::new();
                let mut n_next_profitable = 0i32;
                for &si in &chosen {
                    if let Some((_, oosp)) = by_key.get(&(si, win + 1)) {
                        oos_next.extend_from_slice(oosp);
                        let (pf, _) = pf_roi(oosp);
                        if pf > 1.0 { n_next_profitable += 1; }
                    }
                }
                let n_oos_next = oos_next.len();
                let (oos_next_pf, oos_next_roi) = if oos_next.is_empty() { (f64::NAN, f64::NAN) } else { pf_roi(&oos_next) };

                out.push((
                    win, pi, port_size, n_is, n_oos_same, n_oos_next,
                    mdd_r, cal_r, roi_r,
                    oos_same_pf, oos_same_roi, n_same_profitable,
                    oos_next_pf, oos_next_roi, n_next_profitable,
                ));
            }
            out
        })
        .collect();

    let f = fs::File::create(out_path).expect("create out");
    let mut w = BufWriter::new(f);
    writeln!(w, "asset,window,port_id,n_strategies,n_trades_is,n_trades_oos_same,n_trades_oos_next,\
                port_mdd_rank,port_calmar_rank,port_roi_rank_broken,\
                oos_same_pf,oos_same_roi,oos_same_profitable_count,\
                oos_next_pf,oos_next_roi,oos_next_profitable_count").unwrap();
    for r in &rows {
        let (win, pi, nstr, nis, noss, nosn, mr, cr, rr, ospf, osroi, ospc, onpf, onroi, onpc) = r;
        writeln!(w, "{},W{:02},{},{},{},{},{},{:.1},{:.1},{:.1},{:.4},{:.4},{},{:.4},{:.4},{}",
                 asset, win, pi, nstr, nis, noss, nosn, mr, cr, rr,
                 ospf, osroi, ospc, onpf, onroi, onpc).unwrap();
    }
    w.flush().unwrap();
    eprintln!("[{}] wrote {} portfolios to {}", asset, rows.len(), out_path);
}
