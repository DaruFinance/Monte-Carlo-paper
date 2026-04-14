//! strat_corr: sparse bar-level PnL correlation of strategies within an asset.
//!
//! Feeds Figure 1 (within-family vs cross-family bar-level PnL correlation) and
//! Table 3 of the paper, via downstream Python
//! `correlation_analysis/strategy_correlations.py`.
//!
//! For each strategy, OOS trades are expanded into a sparse {bar_index -> direction}
//! representation, then Pearson correlations of centred direction vectors are
//! computed over the intersection of occupied bars. Within-family and
//! cross-family pair samples are taken with a fixed seed (42).
//!
//! Family classification uses generic technical-indicator prefixes
//! (ATR / EMA / MACD / PPO / RSI / SMA / STOCHK). Strategies are discovered as
//! `<base_dir>/<family>/<strategy>/trades.bin`.
//!
//! Output CSVs (to <output_dir>, default "."):
//!   <instrument>_embed.csv       3D random-projection scatter coordinates
//!   <instrument>_fammatrix.csv   family x family mean correlation
//!   <instrument>_summary.csv     within/cross mean + percentiles
//!   <instrument>_histogram.csv   200-bin histogram of within/cross correlations
//!   <instrument>_perfamily.csv   per-family within-mean
//!
//! Run: cargo run --release --bin strat_corr -- <base_dir> <window_size> \
//!          <instrument> <market: crypto|forex|commodity> [output_dir]

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{Write, BufWriter};
use std::path::Path;
use walkdir::WalkDir;
use rand::prelude::*;
use rand::rngs::SmallRng;

// ── Binary parser ──────────────────────────────────────────────────────────

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

// ── Strategy stats (no dense vectors) ─────────────────────────────────────

struct StratStats {
    name: String,
    family: usize,
    bars: Vec<(u32, i8)>,
    sum_dirs: f64,
    norm: f64,
    embed: [f64; 3], // 3D random projection coordinates
}

// ── Sparse correlation ────────────────────────────────────────────────────

fn sparse_corr(a: &StratStats, b: &StratStats, nb: f64) -> f32 {
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

// ── Hash-based random projection (deterministic, no matrix needed) ────────

fn hash_proj(bar: u32, dim: u32) -> f64 {
    // Simple deterministic hash → Gaussian-ish via Box-Muller on two hashes
    let seed = (bar as u64).wrapping_mul(2654435761).wrapping_add(dim as u64 * 1442695040888963407);
    let mut rng = SmallRng::seed_from_u64(seed);
    rng.gen::<f64>() * 2.0 - 1.0
}

fn compute_embed(bars: &[(u32, i8)]) -> [f64; 3] {
    let mut e = [0.0f64; 3];
    for &(bar, dir) in bars {
        let d = dir as f64;
        e[0] += d * hash_proj(bar, 0);
        e[1] += d * hash_proj(bar, 1);
        e[2] += d * hash_proj(bar, 2);
    }
    // Normalize to unit sphere
    let len = (e[0]*e[0] + e[1]*e[1] + e[2]*e[2]).sqrt();
    if len > 1e-12 { e[0] /= len; e[1] /= len; e[2] /= len; }
    e
}

// ── Percentile helper ─────────────────────────────────────────────────────

fn pct(s: &[f32], p: f64) -> f64 {
    if s.is_empty() { return 0.0; }
    let i = ((p / 100.0) * (s.len() - 1) as f64) as usize;
    s[i.min(s.len() - 1)] as f64
}

// ── Main ──────────────────────────────────────────────────────────────────

const N_SAMPLES: usize = 500_000;
const N_PERFAM: usize = 50_000; // samples per family pair for the matrix

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("Usage: strat_corr <base_dir> <window_size> <instrument> <market> [output_dir]");
        eprintln!("  market: crypto | forex | commodity");
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
    eprintln!("[{}] Reading trades...", instrument);
    let sparse_data: Vec<(String, usize, Vec<(u32, i8)>)> = strats.par_iter()
        .map(|(name, fam, path)| {
            let secs = read_trades_bin(path);
            let bars = build_sparse_positions(&secs, ws);
            (name.clone(), *fam, bars)
        })
        .collect();

    // ── Count unique bars ─────────────────────────────────────────────────
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

    // ── Build stats + 3D embedding (parallel, no dense vectors) ───────────
    eprintln!("[{}] Building strategy stats + 3D embedding...", instrument);
    let stats: Vec<StratStats> = sparse_data.into_par_iter().map(|(name, fam, bars)| {
        let sum_dirs: f64 = bars.iter().map(|&(_, d)| d as f64).sum();
        let nnz = bars.len() as f64;
        let norm_sq = nnz - sum_dirs * sum_dirs / nb;
        let norm = if norm_sq > 0.0 { norm_sq.sqrt() } else { 0.0 };
        let embed = compute_embed(&bars);
        StratStats { name, family: fam, bars, sum_dirs, norm, embed }
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

    // ── Write 3D embedding CSV ────────────────────────────────────────────
    eprintln!("[{}] Writing embedding CSV...", instrument);
    {
        let path = format!("{}/{}_embed.csv", out_dir, instrument.to_lowercase());
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        writeln!(f, "strategy,family,instrument,market,x,y,z").unwrap();
        for s in &stats {
            // Quote strategy name to handle names with commas like MACD(16,42)
            writeln!(f, "\"{}\",{},{},{},{:.6},{:.6},{:.6}",
                s.name, family_name(s.family), instrument, market,
                s.embed[0], s.embed[1], s.embed[2]).unwrap();
        }
        eprintln!("[{}] Wrote {}", instrument, path);
    }

    // ── Sample within/cross pairs and compute correlations ────────────────
    let mut rng = SmallRng::seed_from_u64(42);

    // Within-family pairs (weighted by pair count)
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
    let eligible_fams: Vec<usize> = fam_keys.iter().filter(|&&fk| fam_groups[&fk].len() >= 1).cloned().collect();
    for _ in 0..N_SAMPLES {
        let fa = eligible_fams[rng.gen_range(0..eligible_fams.len())];
        let mut fb = eligible_fams[rng.gen_range(0..eligible_fams.len())];
        while fb == fa { fb = eligible_fams[rng.gen_range(0..eligible_fams.len())]; }
        let ga = &fam_groups[&fa];
        let gb = &fam_groups[&fb];
        cross_pairs.push((ga[rng.gen_range(0..ga.len())], gb[rng.gen_range(0..gb.len())]));
    }

    eprintln!("[{}] Computing within-family correlations (sparse)...", instrument);
    let within_corrs: Vec<f32> = within_pairs.par_iter().map(|&(a, b)| {
        sparse_corr(&stats[a], &stats[b], nb)
    }).collect();

    eprintln!("[{}] Computing cross-family correlations (sparse)...", instrument);
    let cross_corrs: Vec<f32> = cross_pairs.par_iter().map(|&(a, b)| {
        sparse_corr(&stats[a], &stats[b], nb)
    }).collect();

    let within_mean: f64 = within_corrs.iter().map(|&x| x as f64).sum::<f64>() / within_corrs.len() as f64;
    let cross_mean: f64 = cross_corrs.iter().map(|&x| x as f64).sum::<f64>() / cross_corrs.len() as f64;

    let mut ws_sorted = within_corrs.clone(); ws_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cs_sorted = cross_corrs.clone(); cs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    eprintln!("[{}] Within: mean={:.4}, p50={:.4}, p90={:.4}, p95={:.4}, p99={:.4}",
        instrument, within_mean, pct(&ws_sorted, 50.0), pct(&ws_sorted, 90.0), pct(&ws_sorted, 95.0), pct(&ws_sorted, 99.0));
    eprintln!("[{}] Cross:  mean={:.4}, p50={:.4}, p90={:.4}, p95={:.4}, p99={:.4}",
        instrument, cross_mean, pct(&cs_sorted, 50.0), pct(&cs_sorted, 90.0), pct(&cs_sorted, 95.0), pct(&cs_sorted, 99.0));

    // ── Family × family correlation matrix ────────────────────────────────
    eprintln!("[{}] Computing family×family correlation matrix...", instrument);
    let nf = fam_keys.len();
    // Build pair list for each (i,j) family combination
    let mut fam_matrix_pairs: Vec<(usize, usize, Vec<(usize, usize)>)> = Vec::new();
    for fi in 0..nf {
        for fj in fi..nf {
            let fa = fam_keys[fi];
            let fb = fam_keys[fj];
            let ga = &fam_groups[&fa];
            let gb = &fam_groups[&fb];
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            let n_samp = if fi == fj {
                let sz = ga.len();
                if sz < 2 { 0 } else { N_PERFAM.min(sz * (sz - 1) / 2) }
            } else {
                N_PERFAM.min(ga.len() * gb.len())
            };
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
        let sum: f64 = pairs.iter().map(|&(a, b)| sparse_corr(&stats[a], &stats[b], nb) as f64).sum();
        (*fi, *fj, sum / pairs.len() as f64)
    }).collect();

    // ── Write family matrix CSV ───────────────────────────────────────────
    {
        let path = format!("{}/{}_fammatrix.csv", out_dir, instrument.to_lowercase());
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        // Header row
        write!(f, "family").unwrap();
        for &fk in &fam_keys { write!(f, ",{}", family_name(fk)).unwrap(); }
        writeln!(f).unwrap();
        // Fill symmetric matrix
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

    // ── Write summary CSV ─────────────────────────────────────────────────
    {
        let path = format!("{}/{}_summary.csv", out_dir, instrument.to_lowercase());
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        writeln!(f, "instrument,market,n_strategies,n_bars,within_mean,within_p50,within_p90,within_p95,within_p99,cross_mean,cross_p50,cross_p90,cross_p95,cross_p99").unwrap();
        writeln!(f, "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            instrument, market, n, nb as u64,
            within_mean, pct(&ws_sorted, 50.0), pct(&ws_sorted, 90.0), pct(&ws_sorted, 95.0), pct(&ws_sorted, 99.0),
            cross_mean, pct(&cs_sorted, 50.0), pct(&cs_sorted, 90.0), pct(&cs_sorted, 95.0), pct(&cs_sorted, 99.0)
        ).unwrap();
        eprintln!("[{}] Wrote {}", instrument, path);
    }

    // ── Write histogram CSV (binned, much smaller than raw samples) ──────
    {
        let path = format!("{}/{}_histogram.csv", out_dir, instrument.to_lowercase());
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        let n_bins: usize = 200; // bins from -1.0 to 1.0
        let mut within_hist = vec![0u32; n_bins];
        let mut cross_hist = vec![0u32; n_bins];
        for &c in &within_corrs {
            let bin = ((c as f64 + 1.0) / 2.0 * n_bins as f64) as usize;
            within_hist[bin.min(n_bins - 1)] += 1;
        }
        for &c in &cross_corrs {
            let bin = ((c as f64 + 1.0) / 2.0 * n_bins as f64) as usize;
            cross_hist[bin.min(n_bins - 1)] += 1;
        }
        writeln!(f, "bin_center,within_count,cross_count").unwrap();
        for i in 0..n_bins {
            let center = -1.0 + (i as f64 + 0.5) * 2.0 / n_bins as f64;
            writeln!(f, "{:.4},{},{}", center, within_hist[i], cross_hist[i]).unwrap();
        }
        eprintln!("[{}] Wrote {}", instrument, path);
    }

    // ── Per-family within-mean (for per-family bar chart) ─────────────────
    {
        let path = format!("{}/{}_perfamily.csv", out_dir, instrument.to_lowercase());
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        writeln!(f, "instrument,market,family,n_strategies,within_mean").unwrap();
        for &(fi, fj, corr) in &fam_matrix_corrs {
            if fi == fj {
                writeln!(f, "{},{},{},{},{:.6}",
                    instrument, market, family_name(fam_keys[fi]),
                    fam_groups[&fam_keys[fi]].len(), corr).unwrap();
            }
        }
        eprintln!("[{}] Wrote {}", instrument, path);
    }

    eprintln!("[{}] Done!", instrument);
}
