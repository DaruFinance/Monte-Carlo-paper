//! corr_tensor: 7-instrument rolling correlation tensor on 1h log returns.
//!
//! Supports the cross-asset correlation diagnostics referenced alongside
//! Figure 1 / Table 3 of the paper. Reads raw OHLC CSVs (mixed 15m/30m/1h),
//! resamples to the last close within each hour bucket, intersects the hour
//! buckets across all 7 instruments, and emits three CSVs:
//!
//!   corr_tensor.csv       window_end_unix + n_inst*n_inst pair columns
//!   corr_eigenvalues.csv  window_end_unix + eig1..eigN (descending)
//!   corr_avg.csv          n_inst x n_inst mean correlation matrix
//!
//! Rolling window: 720 hourly bars (~30 days), step 24 hours.
//!
//! Instruments (SOL and DOGE excluded — start dates too short):
//!   BTC (30m), BNB (15m), EURUSD, USDJPY, EURGBP, XAUUSD, WTI (all 1h).
//!
//! Run: cargo run --release --bin corr_tensor -- <data_dir> <out_dir>
//!      Or set MC_PAPER_DATA_DIR / MC_PAPER_OUT_DIR env vars.
//!      Defaults: data_dir=../../results/raw_data, out_dir=.
//!
//! Expects the following filenames under <data_dir>:
//!   BTCUSDT_30m_3_9.csv, BNBUSDT_15m_3_9.csv (time col: "time"),
//!   EURUSD_1h_clean.csv, USDJPY_1h_clean.csv, EURGBP_1h_clean.csv,
//!   XAUUSD_1h_clean.csv, WTI_1h_clean.csv (time col: "timestamp").

use nalgebra::{DMatrix, SymmetricEigen};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

struct Series {
    name: String,
    bars: BTreeMap<i64, f64>, // hour_bucket (unix hours) -> close
}

fn parse_csv(path: &str, name: &str, time_col: &str) -> Series {
    let f = File::open(path).unwrap_or_else(|_| panic!("open {}", path));
    let r = BufReader::new(f);
    let mut lines = r.lines();
    let header = lines.next().unwrap().unwrap();
    let headers: Vec<&str> = header.split(',').collect();
    let time_idx = headers
        .iter()
        .position(|h| *h == time_col)
        .unwrap_or_else(|| panic!("time col {} not in {}", time_col, path));
    let close_idx = headers
        .iter()
        .position(|h| *h == "close")
        .expect("close col");

    let mut bars: BTreeMap<i64, f64> = BTreeMap::new();
    for line in lines {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        let ts_raw: i64 = match cols[time_idx].parse() {
            Ok(v) => v,
            Err(_) => continue, // skip non-numeric timestamps (e.g. datetime strings)
        };
        let ts_sec = if ts_raw > 10_000_000_000 {
            ts_raw / 1000
        } else {
            ts_raw
        };
        let hour_bucket = ts_sec / 3600;
        let close: f64 = match cols[close_idx].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        // last bar within the hour wins (inputs are time-ordered)
        bars.insert(hour_bucket, close);
    }
    Series {
        name: name.to_string(),
        bars,
    }
}

fn main() {
    // Resolve data_dir / out_dir from CLI args, then env vars, then defaults.
    let args: Vec<String> = std::env::args().collect();
    let data_dir: String = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("MC_PAPER_DATA_DIR").ok())
        .unwrap_or_else(|| "../../results/raw_data".to_string());
    let out_dir: String = args
        .get(2)
        .cloned()
        .or_else(|| std::env::var("MC_PAPER_OUT_DIR").ok())
        .unwrap_or_else(|| ".".to_string());
    let data_dir = data_dir.as_str();
    let out_dir = out_dir.as_str();
    std::fs::create_dir_all(out_dir).ok();

    let specs: Vec<(&str, &str, &str)> = vec![
        ("BTC", "BTCUSDT_30m_3_9.csv", "time"),
        ("BNB", "BNBUSDT_15m_3_9.csv", "time"),
        ("EURUSD", "EURUSD_1h_clean.csv", "timestamp"),
        ("USDJPY", "USDJPY_1h_clean.csv", "timestamp"),
        ("EURGBP", "EURGBP_1h_clean.csv", "timestamp"),
        ("XAUUSD", "XAUUSD_1h_clean.csv", "timestamp"),
        ("WTI", "WTI_1h_clean.csv", "timestamp"),
    ];

    let series: Vec<Series> = specs
        .iter()
        .map(|(name, file, tc)| {
            let p = format!("{}/{}", data_dir, file);
            println!("Loading {} from {}", name, p);
            parse_csv(&p, name, tc)
        })
        .collect();

    for s in &series {
        let first = s.bars.keys().next().copied().unwrap_or(0) * 3600;
        let last = s.bars.keys().last().copied().unwrap_or(0) * 3600;
        println!(
            "  {}: {} hourly buckets, {}..{}",
            s.name,
            s.bars.len(),
            first,
            last
        );
    }

    // Intersection of hour buckets across all instruments.
    let mut common: Vec<i64> = series[0].bars.keys().copied().collect();
    for s in &series[1..] {
        common.retain(|h| s.bars.contains_key(h));
    }
    common.sort();
    println!(
        "Common hours: {} (from unix {} to {})",
        common.len(),
        common.first().unwrap_or(&0) * 3600,
        common.last().unwrap_or(&0) * 3600
    );
    if common.len() < 1000 {
        panic!("too few common hours: {}", common.len());
    }

    let n_inst = series.len();
    let n_bars = common.len();

    // Aligned close matrix (n_bars x n_inst), row-major.
    let mut closes: Vec<f64> = vec![0.0; n_bars * n_inst];
    for (j, s) in series.iter().enumerate() {
        for (i, h) in common.iter().enumerate() {
            closes[i * n_inst + j] = *s.bars.get(h).unwrap();
        }
    }

    // Log returns (n_ret = n_bars - 1) x n_inst.
    let n_ret = n_bars - 1;
    let mut returns: Vec<f64> = vec![0.0; n_ret * n_inst];
    for i in 0..n_ret {
        for j in 0..n_inst {
            let c0 = closes[i * n_inst + j];
            let c1 = closes[(i + 1) * n_inst + j];
            returns[i * n_inst + j] = (c1 / c0).ln();
        }
    }

    let window: usize = 720; // 30 days of hourly bars
    let step: usize = 24; // daily stride

    if n_ret < window {
        panic!("not enough data for one window");
    }
    let n_windows = (n_ret - window) / step + 1;
    println!(
        "n_bars={}, n_ret={}, window={}, step={}, n_windows={}",
        n_bars, n_ret, window, step, n_windows
    );

    // Output writers.
    let mut tensor_f =
        BufWriter::new(File::create(format!("{}/corr_tensor.csv", out_dir)).unwrap());
    let mut eig_f =
        BufWriter::new(File::create(format!("{}/corr_eigenvalues.csv", out_dir)).unwrap());
    let mut avg_f = BufWriter::new(File::create(format!("{}/corr_avg.csv", out_dir)).unwrap());

    // tensor header: window_end_unix, then i_j pair columns
    write!(tensor_f, "window_end_unix").unwrap();
    for i in 0..n_inst {
        for j in 0..n_inst {
            write!(tensor_f, ",{}_{}", specs[i].0, specs[j].0).unwrap();
        }
    }
    writeln!(tensor_f).unwrap();

    // eig header
    write!(eig_f, "window_end_unix").unwrap();
    for k in 0..n_inst {
        write!(eig_f, ",eig{}", k + 1).unwrap();
    }
    writeln!(eig_f).unwrap();

    // accumulator for average
    let mut avg = vec![0.0_f64; n_inst * n_inst];

    let wf = window as f64;
    for w in 0..n_windows {
        let start = w * step;
        let end = start + window; // exclusive; returns[start..end]

        // means
        let mut means = vec![0.0_f64; n_inst];
        for i in start..end {
            for j in 0..n_inst {
                means[j] += returns[i * n_inst + j];
            }
        }
        for m in &mut means {
            *m /= wf;
        }

        // std devs
        let mut stds = vec![0.0_f64; n_inst];
        for i in start..end {
            for j in 0..n_inst {
                let d = returns[i * n_inst + j] - means[j];
                stds[j] += d * d;
            }
        }
        for s in &mut stds {
            *s = (*s / wf).sqrt();
        }

        // correlation matrix (symmetric)
        let mut corr = vec![0.0_f64; n_inst * n_inst];
        for a in 0..n_inst {
            corr[a * n_inst + a] = 1.0;
            for b in (a + 1)..n_inst {
                let mut cov = 0.0_f64;
                for i in start..end {
                    cov += (returns[i * n_inst + a] - means[a])
                        * (returns[i * n_inst + b] - means[b]);
                }
                cov /= wf;
                let c = if stds[a] > 0.0 && stds[b] > 0.0 {
                    cov / (stds[a] * stds[b])
                } else {
                    0.0
                };
                corr[a * n_inst + b] = c;
                corr[b * n_inst + a] = c;
            }
        }

        // window_end timestamp: hour at index `end` in common[] (the bar after the last return)
        let end_hour = common[end];
        let end_unix = end_hour * 3600;

        // write tensor row + accumulate avg
        write!(tensor_f, "{}", end_unix).unwrap();
        for a in 0..n_inst {
            for b in 0..n_inst {
                let v = corr[a * n_inst + b];
                write!(tensor_f, ",{:.6}", v).unwrap();
                avg[a * n_inst + b] += v;
            }
        }
        writeln!(tensor_f).unwrap();

        // eigenvalues (symmetric eigendecomp, sorted desc)
        let m = DMatrix::from_row_slice(n_inst, n_inst, &corr);
        let eig = SymmetricEigen::new(m);
        let mut evals: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        evals.sort_by(|x, y| y.partial_cmp(x).unwrap());
        write!(eig_f, "{}", end_unix).unwrap();
        for e in &evals {
            write!(eig_f, ",{:.6}", e).unwrap();
        }
        writeln!(eig_f).unwrap();
    }

    // average matrix
    write!(avg_f, "instrument").unwrap();
    for i in 0..n_inst {
        write!(avg_f, ",{}", specs[i].0).unwrap();
    }
    writeln!(avg_f).unwrap();
    let nw = n_windows as f64;
    for a in 0..n_inst {
        write!(avg_f, "{}", specs[a].0).unwrap();
        for b in 0..n_inst {
            write!(avg_f, ",{:.6}", avg[a * n_inst + b] / nw).unwrap();
        }
        writeln!(avg_f).unwrap();
    }

    println!("Done.");
    println!("  -> {}/corr_tensor.csv", out_dir);
    println!("  -> {}/corr_eigenvalues.csv", out_dir);
    println!("  -> {}/corr_avg.csv", out_dir);
}
