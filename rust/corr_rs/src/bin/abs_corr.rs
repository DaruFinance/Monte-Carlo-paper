//! Absolute-correlation analysis for strategy signal independence.
//!
//! Fixes the LLN cancellation bias in the original `strat_corr` binary:
//! signed Pearson r ranges [-1, +1], so mean(r) → 0 by the law of large
//! numbers even when real dependencies exist.  Using |r| (range [0, 1])
//! reveals the true *strength* of dependence regardless of direction.
//!
//! Outputs:
//!   {inst}_abs_summary.csv      – mean/percentiles of |r|, plus null baseline
//!   {inst}_abs_histogram.csv    – 200-bin histogram of |r| in [0, 1]
//!   {inst}_abs_fammatrix.csv    – family × family mean |r|
//!   {inst}_abs_perfamily.csv    – per-family mean |r|
//!
//! Usage: abs_corr <base_dir> <window_size> <instrument> <market> [output_dir]
//!
//! Environment: set RAYON_NUM_THREADS=30 for 30-thread parallelism.

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{Write, BufWriter};
use std::path::Path;
use walkdir::WalkDir;
use rand::prelude::*;
use rand::rngs::SmallRng;

// ── Binary parser (same as main.rs) ────────────────────────────────────────

fn read_trades_bin(path: &Path) -> Vec<(u32, bool, Vec<(u32, u32, i8)>)> {
    let data = match fs::read(path) { Ok(d) => d, Err(_) => return vec![] };
    let mut pos = 0;
    let mut sections = Vec::new();
    let len = data.len();
    while pos + 2 <= len {
        let nl = u16::from_le_bytes([data[pos], data[pos+1]]) as usize; pos += 2;
        if nl == 0 || pos + nl > len { break; } pos += nl;
        if pos + 2 > len { break; }
        let ll = u16::from_le_bytes([data[pos], data[pos+1]]) as usize; pos += 2;
        if pos + ll > len { break; } pos += ll;
        if pos + 2 > len { break; }
        let sl = u16::from_le_bytes([data[pos], data[pos+1]]) as usize; pos += 2;
        if pos + sl > len { break; }
        let sec = std::str::from_utf8(&data[pos..pos+sl]).unwrap_or(""); pos += sl;
        if pos + 4 > len { break; }
        let cnt = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize; pos += 4;
        let mut trades = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            if pos + 17 > len { break; }
            let e1 = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
            let e2 = u32::from_le_bytes([data[pos+4], data[pos+5], data[pos+6], data[pos+7]]);
            let dir: i8 = if data[pos+8] == 1 { 1 } else { -1 };
            trades.push((e1, e2, dir)); pos += 17;
        }
        if let Some(dash) = sec.find('-') {
            let prefix = &sec[..dash]; let suffix = &sec[dash+1..];
            if prefix.starts_with('W') {
                if let Ok(w) = prefix[1..].parse::<u32>() {
                    sections.push((w, suffix.eq_ignore_ascii_case("OOS"), trades));
                }
            }
        }
    }
    sections
}

// ── Family classification ──────────────────────────────────────────────────

const FAM_NAMES: [&str; 8] = ["ATR", "EMA", "MACD", "PPO", "RSI_LEVEL", "RSI", "SMA", "STOCHK"];

fn get_family(name: &str) -> usize {
    let upper = name.to_uppercase();
    for (i, p) in FAM_NAMES.iter().enumerate() {
        if upper.starts_with(p) { return i; }
    }
    FAM_NAMES.len()
}

fn family_name(id: usize) -> &'static str {
    if id < FAM_NAMES.len() { FAM_NAMES[id] } else { "OTHER" }
}

// ── Sparse position builder ───────────────────────────────────────────────

fn build_sparse_positions(sections: &[(u32, bool, Vec<(u32, u32, i8)>)], ws: u32) -> Vec<(u32, i8)> {
    let mut pos: HashMap<u32, i8> = HashMap::new();
    for (w, is_oos, trades) in sections {
        if !is_oos { continue; }
        let off = (*w - 1) * ws;
        for &(e1, e2, d) in trades {
            for bar in (off + e1)..=(off + e2) { pos.insert(bar, d); }
        }
    }
    let mut sorted: Vec<(u32, i8)> = pos.into_iter().collect();
    sorted.sort_unstable_by_key(|&(bar, _)| bar);
    sorted
}

// ── Strategy stats ───────────────────────────────────────────────────────

struct StratStats {
    name: String,
    family: usize,
    bars: Vec<(u32, i8)>,
    sum_dirs: f64,
    norm: f64,
}

// ── Sparse correlation (returns ABSOLUTE value) ──────────────────────────

fn sparse_abs_corr(a: &StratStats, b: &StratStats, nb: f64) -> f32 {
    if a.norm < 1e-12 || b.norm < 1e-12 { return 0.0; }
    let mean_a = a.sum_dirs / nb;
    let mean_b = b.sum_dirs / nb;
    let mut ia = 0usize;
    let mut ib = 0usize;
    let mut intersect_dot: f64 = 0.0;
    while ia < a.bars.len() && ib < b.bars.len() {
        let ba = a.bars[ia].0;
        let bb = b.bars[ib].0;
        if ba == bb {
            intersect_dot += (a.bars[ia].1 as f64) * (b.bars[ib].1 as f64);
            ia += 1; ib += 1;
        } else if ba < bb { ia += 1; } else { ib += 1; }
    }
    let dot = intersect_dot - nb * mean_a * mean_b;
    let r = dot / (a.norm * b.norm);
    r.abs() as f32  // ← KEY CHANGE: absolute value
}

// ── Also compute signed r for comparison output ──────────────────────────

fn sparse_signed_corr(a: &StratStats, b: &StratStats, nb: f64) -> f32 {
    if a.norm < 1e-12 || b.norm < 1e-12 { return 0.0; }
    let mean_a = a.sum_dirs / nb;
    let mean_b = b.sum_dirs / nb;
    let mut ia = 0usize;
    let mut ib = 0usize;
    let mut intersect_dot: f64 = 0.0;
    while ia < a.bars.len() && ib < b.bars.len() {
        let ba = a.bars[ia].0;
        let bb = b.bars[ib].0;
        if ba == bb {
            intersect_dot += (a.bars[ia].1 as f64) * (b.bars[ib].1 as f64);
            ia += 1; ib += 1;
        } else if ba < bb { ia += 1; } else { ib += 1; }
    }
    let dot = intersect_dot - nb * mean_a * mean_b;
    (dot / (a.norm * b.norm)) as f32
}

// ── Percentile helper ─────────────────────────────────────────────────────

fn pct(s: &[f32], p: f64) -> f64 {
    if s.is_empty() { return 0.0; }
    let i = ((p / 100.0) * (s.len() - 1) as f64) as usize;
    s[i.min(s.len() - 1)] as f64
}

// ── Null baseline: E[|r|] for independent series of length n ─────────────
// For large n, r ~ N(0, 1/sqrt(n)), so |r| ~ half-normal with
// E[|r|] = sqrt(2 / (pi * n)).  This provides the baseline against which
// observed mean(|r|) should be compared.

fn null_expected_abs_r(n_obs: f64) -> f64 {
    (2.0 / (std::f64::consts::PI * n_obs)).sqrt()
}

// ── Main ──────────────────────────────────────────────────────────────────

/// Number of random strategy pairs to sample for within-family and cross-family
/// correlation distributions (each).
const N_SAMPLES: usize = 500_000;
/// Maximum pairs sampled per family-pair cell in the family x family correlation matrix.
const N_PERFAM: usize = 50_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("Usage: abs_corr <base_dir> <window_size> <instrument> <market> [output_dir]");
        eprintln!("  market: crypto | forex | commodity");
        eprintln!();
        eprintln!("Computes ABSOLUTE Pearson correlation |r| to avoid LLN cancellation bias.");
        eprintln!("Set RAYON_NUM_THREADS=30 for 30-thread parallelism.");
        std::process::exit(1);
    }
    let base_dir = &args[1];
    let ws: u32 = args[2].parse().unwrap_or(5000);
    let instrument = &args[3];
    let market = &args[4];
    let out_dir = if args.len() > 5 { &args[5] } else { "." };

    fs::create_dir_all(out_dir).ok();

    // ── Scan strategies ───────────────────────────────────────────────────
    eprintln!("[{}] Scanning {} ...", instrument, base_dir);
    let mut strats: Vec<(String, usize, std::path::PathBuf)> = Vec::new();
    for entry in WalkDir::new(base_dir).min_depth(2).max_depth(2).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_dir() { continue; }
        let p = entry.path().join("trades.bin");
        if p.exists() {
            let name = entry.file_name().to_string_lossy().to_string();
            strats.push((name.clone(), get_family(&name), p));
        }
    }
    strats.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    let n = strats.len();
    eprintln!("[{}] {} strategies found", instrument, n);
    if n == 0 { eprintln!("No strategies found!"); std::process::exit(1); }

    // ── Read trades → sparse positions (parallel) ─────────────────────────
    eprintln!("[{}] Reading trades (parallel)...", instrument);
    let sparse_data: Vec<(String, usize, Vec<(u32, i8)>)> = strats.par_iter()
        .map(|(name, fam, path)| {
            let secs = read_trades_bin(path);
            let bars = build_sparse_positions(&secs, ws);
            (name.clone(), *fam, bars)
        })
        .collect();

    // ── Count unique bars (for Pearson denominator + null baseline) ───────
    eprintln!("[{}] Counting unique bars...", instrument);
    let mut all_bars: Vec<u32> = Vec::new();
    for (_, _, bars) in &sparse_data {
        for &(b, _) in bars { all_bars.push(b); }
    }
    all_bars.sort_unstable();
    all_bars.dedup();
    let nb = all_bars.len() as f64;
    drop(all_bars);
    eprintln!("[{}] {} unique bars", instrument, nb as u64);

    // Null baseline for independent series
    let null_abs_r = null_expected_abs_r(nb);
    eprintln!("[{}] Null E[|r|] under independence: {:.6}", instrument, null_abs_r);

    // ── Build stats (parallel) ────────────────────────────────────────────
    eprintln!("[{}] Building strategy stats...", instrument);
    let stats: Vec<StratStats> = sparse_data.into_par_iter().map(|(name, fam, bars)| {
        let sum_dirs: f64 = bars.iter().map(|&(_, d)| d as f64).sum();
        let nnz = bars.len() as f64;
        let norm_sq = nnz - sum_dirs * sum_dirs / nb;
        let norm = if norm_sq > 0.0 { norm_sq.sqrt() } else { 0.0 };
        StratStats { name, family: fam, bars, sum_dirs, norm }
    }).collect();

    // ── Group by family ───────────────────────────────────────────────────
    let mut fam_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, s) in stats.iter().enumerate() {
        fam_groups.entry(s.family).or_default().push(i);
    }
    let fam_keys: Vec<usize> = {
        let mut k: Vec<usize> = fam_groups.keys().cloned().collect();
        k.sort(); k
    };

    eprintln!("[{}] Families:", instrument);
    for &fk in &fam_keys {
        eprintln!("  {}: {} strategies", family_name(fk), fam_groups[&fk].len());
    }

    // ── Sample within/cross pairs ─────────────────────────────────────────
    let mut rng = SmallRng::seed_from_u64(42);

    let mut fam_pair_counts: Vec<(usize, u64)> = Vec::new();
    let mut total_within_pairs: u64 = 0;
    for &fk in &fam_keys {
        let sz = fam_groups[&fk].len() as u64;
        if sz >= 2 {
            let pairs = sz * (sz - 1) / 2;
            fam_pair_counts.push((fk, pairs));
            total_within_pairs += pairs;
        }
    }

    eprintln!("[{}] Sampling {} within + {} cross pairs...", instrument, N_SAMPLES, N_SAMPLES);
    let mut within_pairs: Vec<(usize, usize)> = Vec::with_capacity(N_SAMPLES);
    for _ in 0..N_SAMPLES {
        let mut r = rng.gen_range(0..total_within_pairs);
        let mut chosen_fam = fam_pair_counts[0].0;
        for &(fk, pc) in &fam_pair_counts {
            if r < pc { chosen_fam = fk; break; }
            r -= pc;
        }
        let group = &fam_groups[&chosen_fam];
        let a = rng.gen_range(0..group.len());
        let mut b = rng.gen_range(0..group.len());
        while b == a { b = rng.gen_range(0..group.len()); }
        within_pairs.push((group[a], group[b]));
    }

    let mut cross_pairs: Vec<(usize, usize)> = Vec::with_capacity(N_SAMPLES);
    let eligible_fams: Vec<usize> = fam_keys.iter().filter(|&&fk| !fam_groups[&fk].is_empty()).cloned().collect();
    for _ in 0..N_SAMPLES {
        let fa = eligible_fams[rng.gen_range(0..eligible_fams.len())];
        let mut fb = eligible_fams[rng.gen_range(0..eligible_fams.len())];
        while fb == fa { fb = eligible_fams[rng.gen_range(0..eligible_fams.len())]; }
        let ga = &fam_groups[&fa];
        let gb = &fam_groups[&fb];
        cross_pairs.push((ga[rng.gen_range(0..ga.len())], gb[rng.gen_range(0..gb.len())]));
    }

    // ── Compute |r| for all pairs (parallel, 30 threads) ─────────────────
    eprintln!("[{}] Computing within-family |r| correlations...", instrument);
    let within_abs: Vec<f32> = within_pairs.par_iter().map(|&(a, b)| {
        sparse_abs_corr(&stats[a], &stats[b], nb)
    }).collect();

    eprintln!("[{}] Computing cross-family |r| correlations...", instrument);
    let cross_abs: Vec<f32> = cross_pairs.par_iter().map(|&(a, b)| {
        sparse_abs_corr(&stats[a], &stats[b], nb)
    }).collect();

    // Also compute signed for comparison
    eprintln!("[{}] Computing signed correlations for comparison...", instrument);
    let within_signed: Vec<f32> = within_pairs.par_iter().map(|&(a, b)| {
        sparse_signed_corr(&stats[a], &stats[b], nb)
    }).collect();
    let cross_signed: Vec<f32> = cross_pairs.par_iter().map(|&(a, b)| {
        sparse_signed_corr(&stats[a], &stats[b], nb)
    }).collect();

    // ── Compute statistics ────────────────────────────────────────────────
    let within_abs_mean: f64 = within_abs.iter().map(|&x| x as f64).sum::<f64>() / within_abs.len() as f64;
    let cross_abs_mean: f64 = cross_abs.iter().map(|&x| x as f64).sum::<f64>() / cross_abs.len() as f64;
    let within_signed_mean: f64 = within_signed.iter().map(|&x| x as f64).sum::<f64>() / within_signed.len() as f64;
    let cross_signed_mean: f64 = cross_signed.iter().map(|&x| x as f64).sum::<f64>() / cross_signed.len() as f64;

    // Fraction exceeding thresholds
    let within_above_01: f64 = within_abs.iter().filter(|&&x| x > 0.1).count() as f64 / within_abs.len() as f64;
    let within_above_02: f64 = within_abs.iter().filter(|&&x| x > 0.2).count() as f64 / within_abs.len() as f64;
    let within_above_05: f64 = within_abs.iter().filter(|&&x| x > 0.5).count() as f64 / within_abs.len() as f64;
    let cross_above_01: f64 = cross_abs.iter().filter(|&&x| x > 0.1).count() as f64 / cross_abs.len() as f64;
    let cross_above_02: f64 = cross_abs.iter().filter(|&&x| x > 0.2).count() as f64 / cross_abs.len() as f64;
    let cross_above_05: f64 = cross_abs.iter().filter(|&&x| x > 0.5).count() as f64 / cross_abs.len() as f64;

    let mut ws_sorted = within_abs.clone(); ws_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cs_sorted = cross_abs.clone(); cs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Excess over null (how much above random)
    let within_excess = within_abs_mean - null_abs_r;
    let cross_excess = cross_abs_mean - null_abs_r;

    eprintln!("[{}] ═══ ABSOLUTE CORRELATION RESULTS ═══", instrument);
    eprintln!("[{}] Null E[|r|] (independence): {:.6}", instrument, null_abs_r);
    eprintln!("[{}] Within |r|: mean={:.6} (excess={:+.6}), p50={:.6}, p90={:.6}, p95={:.6}, p99={:.6}",
        instrument, within_abs_mean, within_excess,
        pct(&ws_sorted, 50.0), pct(&ws_sorted, 90.0), pct(&ws_sorted, 95.0), pct(&ws_sorted, 99.0));
    eprintln!("[{}] Cross  |r|: mean={:.6} (excess={:+.6}), p50={:.6}, p90={:.6}, p95={:.6}, p99={:.6}",
        instrument, cross_abs_mean, cross_excess,
        pct(&cs_sorted, 50.0), pct(&cs_sorted, 90.0), pct(&cs_sorted, 95.0), pct(&cs_sorted, 99.0));
    eprintln!("[{}] Signed mean (for reference): within={:.6}, cross={:.6}", instrument, within_signed_mean, cross_signed_mean);
    eprintln!("[{}] Within |r|>0.1: {:.2}%, |r|>0.2: {:.2}%, |r|>0.5: {:.2}%",
        instrument, within_above_01 * 100.0, within_above_02 * 100.0, within_above_05 * 100.0);
    eprintln!("[{}] Cross  |r|>0.1: {:.2}%, |r|>0.2: {:.2}%, |r|>0.5: {:.2}%",
        instrument, cross_above_01 * 100.0, cross_above_02 * 100.0, cross_above_05 * 100.0);

    // ── Write summary CSV ─────────────────────────────────────────────────
    {
        let path = format!("{}/{}_abs_summary.csv", out_dir, instrument.to_lowercase());
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        writeln!(f, "instrument,market,n_strategies,n_bars,null_expected_abs_r,\
            within_abs_mean,within_excess,within_signed_mean,within_p50,within_p90,within_p95,within_p99,\
            within_pct_above_01,within_pct_above_02,within_pct_above_05,\
            cross_abs_mean,cross_excess,cross_signed_mean,cross_p50,cross_p90,cross_p95,cross_p99,\
            cross_pct_above_01,cross_pct_above_02,cross_pct_above_05").unwrap();
        writeln!(f, "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4}",
            instrument, market, n, nb as u64, null_abs_r,
            within_abs_mean, within_excess, within_signed_mean,
            pct(&ws_sorted, 50.0), pct(&ws_sorted, 90.0), pct(&ws_sorted, 95.0), pct(&ws_sorted, 99.0),
            within_above_01, within_above_02, within_above_05,
            cross_abs_mean, cross_excess, cross_signed_mean,
            pct(&cs_sorted, 50.0), pct(&cs_sorted, 90.0), pct(&cs_sorted, 95.0), pct(&cs_sorted, 99.0),
            cross_above_01, cross_above_02, cross_above_05
        ).unwrap();
        eprintln!("[{}] Wrote {}", instrument, path);
    }

    // ── Write histogram CSV (|r| in [0, 1]) ──────────────────────────────
    {
        let path = format!("{}/{}_abs_histogram.csv", out_dir, instrument.to_lowercase());
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        let n_bins: usize = 200;
        let mut within_hist = vec![0u32; n_bins];
        let mut cross_hist = vec![0u32; n_bins];
        for &c in &within_abs {
            let bin = (c as f64 * n_bins as f64) as usize;
            within_hist[bin.min(n_bins - 1)] += 1;
        }
        for &c in &cross_abs {
            let bin = (c as f64 * n_bins as f64) as usize;
            cross_hist[bin.min(n_bins - 1)] += 1;
        }
        writeln!(f, "bin_center,within_count,cross_count").unwrap();
        for i in 0..n_bins {
            let center = (i as f64 + 0.5) / n_bins as f64;
            writeln!(f, "{:.4},{},{}", center, within_hist[i], cross_hist[i]).unwrap();
        }
        eprintln!("[{}] Wrote {}", instrument, path);
    }

    // ── Family × family |r| matrix ───────────────────────────────────────
    eprintln!("[{}] Computing family×family |r| matrix...", instrument);
    let nf = fam_keys.len();
    let mut fam_matrix_pairs: Vec<(usize, usize, Vec<(usize, usize)>)> = Vec::new();
    for fi in 0..nf {
        for fj in fi..nf {
            let fa = fam_keys[fi];
            let fb = fam_keys[fj];
            let ga = &fam_groups[&fa];
            let gb = &fam_groups[&fb];
            let n_samp = if fi == fj {
                let sz = ga.len();
                if sz < 2 { 0 } else { N_PERFAM.min(sz * (sz - 1) / 2) }
            } else {
                N_PERFAM.min(ga.len() * gb.len())
            };
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            for _ in 0..n_samp {
                if fi == fj {
                    let a = rng.gen_range(0..ga.len());
                    let mut b = rng.gen_range(0..ga.len());
                    while b == a { b = rng.gen_range(0..ga.len()); }
                    pairs.push((ga[a], ga[b]));
                } else {
                    pairs.push((ga[rng.gen_range(0..ga.len())], gb[rng.gen_range(0..gb.len())]));
                }
            }
            fam_matrix_pairs.push((fi, fj, pairs));
        }
    }

    let fam_matrix_corrs: Vec<(usize, usize, f64)> = fam_matrix_pairs.par_iter().map(|(fi, fj, pairs)| {
        if pairs.is_empty() { return (*fi, *fj, 0.0); }
        let sum: f64 = pairs.iter().map(|&(a, b)| sparse_abs_corr(&stats[a], &stats[b], nb) as f64).sum();
        (*fi, *fj, sum / pairs.len() as f64)
    }).collect();

    // ── Write family matrix CSV ───────────────────────────────────────────
    {
        let path = format!("{}/{}_abs_fammatrix.csv", out_dir, instrument.to_lowercase());
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        write!(f, "family").unwrap();
        for &fk in &fam_keys { write!(f, ",{}", family_name(fk)).unwrap(); }
        writeln!(f).unwrap();
        let mut matrix = vec![vec![0.0f64; nf]; nf];
        for &(fi, fj, corr) in &fam_matrix_corrs {
            matrix[fi][fj] = corr;
            matrix[fj][fi] = corr;
        }
        for fi in 0..nf {
            write!(f, "{}", family_name(fam_keys[fi])).unwrap();
            for fj in 0..nf { write!(f, ",{:.6}", matrix[fi][fj]).unwrap(); }
            writeln!(f).unwrap();
        }
        eprintln!("[{}] Wrote {}", instrument, path);
    }

    // ── Write per-family CSV ──────────────────────────────────────────────
    {
        let path = format!("{}/{}_abs_perfamily.csv", out_dir, instrument.to_lowercase());
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        writeln!(f, "instrument,market,family,n_strategies,within_abs_mean").unwrap();
        for &(fi, fj, corr) in &fam_matrix_corrs {
            if fi == fj {
                writeln!(f, "{},{},{},{},{:.6}",
                    instrument, market, family_name(fam_keys[fi]),
                    fam_groups[&fam_keys[fi]].len(), corr).unwrap();
            }
        }
        eprintln!("[{}] Wrote {}", instrument, path);
    }

    eprintln!("[{}] ═══ DONE ═══", instrument);
}
