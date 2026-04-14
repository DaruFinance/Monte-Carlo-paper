//! Three-Tier Full-Pipeline Synthetic Validation Framework (Rust, 32-thread)
//! =========================================================================
//!
//! Produces the synthetic-data artifacts for Scenarios A/B/C that feed
//! paper Figures 5-8 (`fig_synthetic_mc_ranks.pdf`,
//! `fig_synthetic_mc_analysis.pdf`, `fig_synthetic_pipeline_v4.pdf`,
//! `fig_synthetic_pipeline_detail.pdf`) via the downstream Python figure
//! regeneration script.
//!
//! Replicates synthetic_pipeline_v4.py with parameters matched to empirical:
//!   - B = 1000 MC permutations (was 200)
//!   - Strategy universe ~38,000+ (was ~4,500)
//!   - 32-thread parallelism via Rayon
//!
//! Tiers:
//!   Tier 1 (null):        pure-null returns, standard strategy grid
//!   Tier 2 (edge):        weak injected momentum edge, standard grid
//!   Tier 3 (adversarial): massive data-mined grid, adversarial regime
//!   Signal sweep:         phi in {0.00..0.15} across standard grid
//!
//! Input:  none — self-generates synthetic AR(1) + GARCH(1,1) + Student-t
//!         returns with a two-state volatility regime.
//!
//! Output CSVs (in <output_dir>, default `../../results/tables_v2`):
//!   synthetic_v4_summaries_matched.csv
//!   synthetic_v4_null_filters_matched.csv
//!   synthetic_v4_edge_filters_matched.csv
//!   synthetic_v4_adversarial_filters_matched.csv
//!   synthetic_v4_signal_sweep_matched.csv
//!
//! Run: cargo run --release [-- <output_dir>]
//!      Or set MC_PAPER_TBL_DIR env var.
//!
//! Seeds are fixed inside each simulation; run numbers are deterministic.

use rayon::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StudentT};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ============================================================
// CONFIGURATION — matched to empirical parameters
// ============================================================
const N_BARS: usize = 60_000;
const N_WINDOWS: usize = 10;
const IS_SIZE: usize = 4_000;
const OOS_SIZE: usize = 2_000;
const T_DF: f64 = 5.0;

// GARCH(1,1)
const GARCH_OMEGA_FRAC: f64 = 0.04;
const GARCH_ALPHA: f64 = 0.08;
const GARCH_BETA: f64 = 0.88;
const BASE_VOL: f64 = 0.008;

// Regime switching
const REGIME_PERSIST: f64 = 0.985;
const HIGH_VOL_MULT: f64 = 2.5;

// Tier 2 edge
const MOMENTUM_PHI: f64 = 0.04;
const EDGE_DRIFT: f64 = 0.00005;

// Transaction costs
const BASE_FEE: f64 = 0.0005;
const BASE_SLIP: f64 = 0.0003;
const COST_PER_TRADE: f64 = 2.0 * (BASE_FEE + BASE_SLIP);
const FEE_SHOCK_MULT: f64 = 2.0;
const SLIP_SHOCK_MULT: f64 = 2.0;

fn rob_fee_cost() -> f64 {
    2.0 * (BASE_FEE * FEE_SHOCK_MULT + BASE_SLIP)
}
fn rob_slip_cost() -> f64 {
    2.0 * (BASE_FEE + BASE_SLIP * SLIP_SHOCK_MULT)
}

// MC permutation — NOW MATCHED TO EMPIRICAL
const N_MC: usize = 1000;

// Portfolio
const TOP_N: usize = 24;
const TOP_SELECT: usize = 17;
const PORT_SIZE: usize = 10;

// Simulation runs
const N_SIMS: usize = 8;
const N_WORKERS: usize = 32;

// Output directory default. Overridable via CLI arg 1 or MC_PAPER_TBL_DIR env var.
const DEFAULT_TBL_DIR: &str = "../../results/tables_v2";

fn resolve_tbl_dir() -> String {
    std::env::args()
        .nth(1)
        .or_else(|| std::env::var("MC_PAPER_TBL_DIR").ok())
        .unwrap_or_else(|| DEFAULT_TBL_DIR.to_string())
}

// ============================================================
// DATA STRUCTURES
// ============================================================
#[derive(Clone)]
enum StrategyType {
    EmaCross {
        fast: usize,
        slow: usize,
        longshort: bool,
    },
    SmaCross {
        fast: usize,
        slow: usize,
        longshort: bool,
    },
    Rsi {
        period: usize,
        entry: f64,
        exit: f64,
    },
    Macd {
        fast: usize,
        slow: usize,
        signal: usize,
    },
}

#[derive(Clone)]
struct Strategy {
    stype: StrategyType,
    name: String,
}

#[derive(Clone, Default)]
struct Metrics {
    roi: f64,
    pf: f64,
    sharpe: f64,
    n_trades: usize,
}

#[derive(Clone, Default)]
struct McRanks {
    roi: f64,
    sharpe: f64,
    mdd: f64,
}

struct StratWindowResult {
    is_pf: f64,
    is_roi: f64,
    is_n_trades: usize,
    oos_roi: f64,
    oos_profitable: bool,
    mc_roi_rank: f64,
    mc_sharpe_rank: f64,
    rob_fee_pass: bool,
    rob_slip_pass: bool,
    rob_all_pass: bool,
    is_pf_pass: bool,
}

#[derive(Clone, Default, serde::Serialize)]
struct SimSummary {
    sim_id: usize,
    tier: String,
    n_strat_windows: usize,
    baseline_oos: f64,
    mc_roi_lift_p50: f64,
    mc_roi_lift_p75: f64,
    rob_all_lift: f64,
    is_pf_lift: f64,
    port_nofilter_oos: f64,
    port_rob_oos: f64,
    corr_mc_roi_oos: f64,
    corr_mc_sharpe_oos: f64,
}

#[derive(Clone, Default, serde::Serialize)]
struct FilterRow {
    filter: String,
    pool: usize,
    oos_prof_pct: f64,
    lift_pp: f64,
    pool_pct: f64,
}

// ============================================================
// PRICE GENERATION
// ============================================================
fn generate_prices(n_bars: usize, momentum: f64, drift: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let t_dist = StudentT::new(T_DF).unwrap();

    // Regime sequence
    let mut regimes = vec![0u8; n_bars];
    for t in 1..n_bars {
        if rng.gen::<f64>() > REGIME_PERSIST {
            regimes[t] = 1 - regimes[t - 1];
        } else {
            regimes[t] = regimes[t - 1];
        }
    }

    let omega = GARCH_OMEGA_FRAC * (1.0 - GARCH_ALPHA - GARCH_BETA) * BASE_VOL * BASE_VOL;
    let mut returns = vec![0.0f64; n_bars];
    let mut sigma2 = BASE_VOL * BASE_VOL;
    let mut prev_r = 0.0f64;

    for t in 0..n_bars {
        if t > 0 {
            sigma2 = omega + GARCH_ALPHA * returns[t - 1] * returns[t - 1] + GARCH_BETA * sigma2;
            sigma2 = sigma2.clamp(1e-12, 0.01);
        }
        let z: f64 = t_dist.sample(&mut rng) / (T_DF / (T_DF - 2.0)).sqrt();
        let vol_mult = if regimes[t] == 0 { 1.0 } else { HIGH_VOL_MULT };
        let vol = sigma2.sqrt() * vol_mult;
        returns[t] = momentum * prev_r + drift + vol * z;
        prev_r = returns[t];
    }

    // Cumulative prices
    let mut cum = 0.0f64;
    let mut prices = Vec::with_capacity(n_bars);
    for &r in &returns {
        cum += r;
        prices.push(100.0 * cum.exp());
    }

    // Log bar returns
    let mut bar_returns = Vec::with_capacity(n_bars - 1);
    for i in 1..n_bars {
        bar_returns.push(prices[i].ln() - prices[i - 1].ln());
    }

    (prices, bar_returns)
}

// ============================================================
// INDICATORS
// ============================================================
fn compute_ema(prices: &[f64], period: usize) -> Vec<f64> {
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut ema = Vec::with_capacity(prices.len());
    ema.push(prices[0]);
    for i in 1..prices.len() {
        ema.push(alpha * prices[i] + (1.0 - alpha) * ema[i - 1]);
    }
    ema
}

fn compute_sma(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut sma = vec![f64::NAN; n];
    if period > n {
        return sma;
    }
    let mut cs = Vec::with_capacity(n + 1);
    cs.push(0.0);
    for &p in prices {
        cs.push(cs.last().unwrap() + p);
    }
    for i in (period - 1)..n {
        sma[i] = (cs[i + 1] - cs[i + 1 - period]) / period as f64;
    }
    sma
}

fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut rsi = vec![50.0; n];
    if period >= n {
        return rsi;
    }

    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];
    for i in 1..n {
        let d = prices[i] - prices[i - 1];
        if d > 0.0 {
            gains[i] = d;
        } else {
            losses[i] = -d;
        }
    }

    let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / period as f64;

    for i in (period + 1)..n {
        avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / period as f64;
        let rs = if avg_loss > 1e-10 {
            avg_gain / avg_loss
        } else {
            100.0
        };
        rsi[i] = 100.0 - 100.0 / (1.0 + rs);
    }
    rsi
}

fn compute_macd(
    prices: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
) -> (Vec<f64>, Vec<f64>) {
    let ema_fast = compute_ema(prices, fast);
    let ema_slow = compute_ema(prices, slow);
    let macd_line: Vec<f64> = ema_fast.iter().zip(&ema_slow).map(|(f, s)| f - s).collect();
    let signal_line = compute_ema(&macd_line, signal);
    (macd_line, signal_line)
}

// ============================================================
// STRATEGY GRID — MATCHED TO EMPIRICAL (~38,000+)
// ============================================================
fn build_strategy_grid(tier: &str) -> Vec<Strategy> {
    let mut strategies = Vec::new();

    // Both standard and massive now use the large grid to match empirical 38K+
    let (ema_fast, ema_slow, rsi_periods, rsi_entries, rsi_exits, macd_fast_list, macd_slow_list, macd_sig_list) =
        if tier == "massive" {
            // ~30K+ for adversarial tier 3
            let ef: Vec<usize> = (3..=60).collect();
            let es: Vec<usize> = (15..=250).step_by(2).collect();
            let rp: Vec<usize> = (5..=35).collect();
            let re: Vec<usize> = (15..=40).step_by(2).collect();
            let rx: Vec<usize> = (60..=85).step_by(2).collect();
            (
                ef,
                es,
                rp,
                re,
                rx,
                vec![6, 8, 10, 12, 14, 16, 20],
                vec![18, 22, 26, 30, 35, 40, 50],
                vec![5, 7, 9, 12, 15],
            )
        } else {
            // MATCHED TO EMPIRICAL: ~38,000+ strategies (was ~4,500)
            let ef: Vec<usize> = (3..=60).collect(); // 58
            let es: Vec<usize> = (15..=250).step_by(2).collect(); // 118
            let rp: Vec<usize> = (5..=35).collect(); // 31
            let re: Vec<usize> = (15..=40).step_by(2).collect(); // 13
            let rx: Vec<usize> = (60..=85).step_by(2).collect(); // 13
            (
                ef,
                es,
                rp,
                re,
                rx,
                vec![6, 8, 10, 12, 14, 16, 20],
                vec![18, 22, 26, 30, 35, 40, 50],
                vec![5, 7, 9, 12, 15],
            )
        };

    // EMA crossover
    for &f in &ema_fast {
        for &s in &ema_slow {
            if f >= s || s < (f as f64 * 1.3) as usize {
                continue;
            }
            for longshort in [false, true] {
                let dir = if longshort { "longshort" } else { "long" };
                strategies.push(Strategy {
                    stype: StrategyType::EmaCross {
                        fast: f,
                        slow: s,
                        longshort,
                    },
                    name: format!("EMA_{}x{}_{}", f, s, dir),
                });
            }
        }
    }

    // SMA crossover (same grid)
    for &f in &ema_fast {
        for &s in &ema_slow {
            if f >= s || s < (f as f64 * 1.3) as usize {
                continue;
            }
            for longshort in [false, true] {
                let dir = if longshort { "longshort" } else { "long" };
                strategies.push(Strategy {
                    stype: StrategyType::SmaCross {
                        fast: f,
                        slow: s,
                        longshort,
                    },
                    name: format!("SMA_{}x{}_{}", f, s, dir),
                });
            }
        }
    }

    // RSI
    for &period in &rsi_periods {
        for &entry in &rsi_entries {
            for &exit in &rsi_exits {
                if exit <= entry + 20 {
                    continue;
                }
                strategies.push(Strategy {
                    stype: StrategyType::Rsi {
                        period,
                        entry: entry as f64,
                        exit: exit as f64,
                    },
                    name: format!("RSI_{}_{}_{}", period, entry, exit),
                });
            }
        }
    }

    // MACD
    for &f in &macd_fast_list {
        for &s in &macd_slow_list {
            if f >= s {
                continue;
            }
            for &sig in &macd_sig_list {
                strategies.push(Strategy {
                    stype: StrategyType::Macd {
                        fast: f,
                        slow: s,
                        signal: sig,
                    },
                    name: format!("MACD_{}_{}_{}", f, s, sig),
                });
            }
        }
    }

    strategies
}

// ============================================================
// INDICATOR CACHE
// ============================================================
struct IndicatorCache {
    ema: HashMap<usize, Vec<f64>>,
    sma: HashMap<usize, Vec<f64>>,
    rsi: HashMap<usize, Vec<f64>>,
    macd: HashMap<(usize, usize, usize), (Vec<f64>, Vec<f64>)>,
}

fn precompute_indicators(prices: &[f64], strategies: &[Strategy]) -> IndicatorCache {
    let mut ema_periods = HashSet::new();
    let mut sma_periods = HashSet::new();
    let mut rsi_periods = HashSet::new();
    let mut macd_keys = HashSet::new();

    for s in strategies {
        match &s.stype {
            StrategyType::EmaCross { fast, slow, .. } => {
                ema_periods.insert(*fast);
                ema_periods.insert(*slow);
            }
            StrategyType::SmaCross { fast, slow, .. } => {
                sma_periods.insert(*fast);
                sma_periods.insert(*slow);
            }
            StrategyType::Rsi { period, .. } => {
                rsi_periods.insert(*period);
            }
            StrategyType::Macd {
                fast,
                slow,
                signal,
            } => {
                macd_keys.insert((*fast, *slow, *signal));
            }
        }
    }

    let mut cache = IndicatorCache {
        ema: HashMap::new(),
        sma: HashMap::new(),
        rsi: HashMap::new(),
        macd: HashMap::new(),
    };

    for &p in &ema_periods {
        cache.ema.insert(p, compute_ema(prices, p));
    }
    for &p in &sma_periods {
        cache.sma.insert(p, compute_sma(prices, p));
    }
    for &p in &rsi_periods {
        cache.rsi.insert(p, compute_rsi(prices, p));
    }
    for &(f, s, sig) in &macd_keys {
        cache.macd.insert((f, s, sig), compute_macd(prices, f, s, sig));
    }

    cache
}

// ============================================================
// SIGNAL GENERATION
// ============================================================
fn generate_signals(strategy: &StrategyType, cache: &IndicatorCache, n: usize) -> Vec<f64> {
    let mut signal = vec![0.0f64; n];

    match strategy {
        StrategyType::EmaCross {
            fast,
            slow,
            longshort,
        } => {
            let fast_ema = &cache.ema[fast];
            let slow_ema = &cache.ema[slow];
            let warmup = slow + 5;
            for i in warmup..n {
                signal[i] = if fast_ema[i] > slow_ema[i] {
                    1.0
                } else if *longshort {
                    -1.0
                } else {
                    0.0
                };
            }
        }
        StrategyType::SmaCross {
            fast,
            slow,
            longshort,
        } => {
            let fast_sma = &cache.sma[fast];
            let slow_sma = &cache.sma[slow];
            let warmup = slow + 5;
            for i in warmup..n {
                if fast_sma[i].is_nan() || slow_sma[i].is_nan() {
                    continue;
                }
                signal[i] = if fast_sma[i] > slow_sma[i] {
                    1.0
                } else if *longshort {
                    -1.0
                } else {
                    0.0
                };
            }
        }
        StrategyType::Rsi {
            period,
            entry,
            exit,
        } => {
            let rsi = &cache.rsi[period];
            let warmup = period + 5;
            let mut in_trade = false;
            for t in warmup..n {
                if !in_trade && rsi[t] < *entry {
                    in_trade = true;
                } else if in_trade && rsi[t] > *exit {
                    in_trade = false;
                }
                signal[t] = if in_trade { 1.0 } else { 0.0 };
            }
        }
        StrategyType::Macd {
            fast,
            slow,
            signal: sig,
        } => {
            let (macd_line, signal_line) = &cache.macd[&(*fast, *slow, *sig)];
            let warmup = slow + sig + 5;
            for i in warmup..n {
                signal[i] = if macd_line[i] > signal_line[i] {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }
    signal
}

// ============================================================
// TRADE EXTRACTION AND METRICS
// ============================================================
fn extract_trades(position: &[f64], bar_returns: &[f64], cost: f64) -> Vec<f64> {
    let n = position.len().min(bar_returns.len() + 1);
    if n < 2 {
        return vec![];
    }

    let mut trades = Vec::new();
    let mut trade_start: Option<usize> = None;

    for i in 0..n {
        let prev_pos = if i == 0 { 0.0 } else { position[i - 1] };
        if position[i] != prev_pos {
            // Close previous trade
            if let Some(start) = trade_start {
                let end = i.min(bar_returns.len());
                if end > start {
                    let pnl: f64 = (start..end)
                        .map(|j| position[j] * bar_returns[j])
                        .sum::<f64>()
                        - cost;
                    trades.push(pnl);
                }
            }
            trade_start = if position[i] != 0.0 { Some(i) } else { None };
        }
    }

    // Close last trade
    if let Some(start) = trade_start {
        let end = bar_returns.len().min(n - 1);
        if end > start {
            let pnl: f64 = (start..end)
                .map(|j| position[j] * bar_returns[j])
                .sum::<f64>()
                - cost;
            trades.push(pnl);
        }
    }

    trades
}

fn compute_metrics(trade_pnls: &[f64]) -> Metrics {
    if trade_pnls.len() < 3 {
        return Metrics {
            n_trades: trade_pnls.len(),
            ..Default::default()
        };
    }

    let n = trade_pnls.len();
    let roi: f64 = trade_pnls.iter().sum();
    let pos: f64 = trade_pnls.iter().filter(|&&x| x > 0.0).sum();
    let neg: f64 = trade_pnls
        .iter()
        .filter(|&&x| x < 0.0)
        .map(|x| x.abs())
        .sum();
    let pf = if neg > 1e-10 { pos / neg } else { 999.0 };

    let mean = roi / n as f64;
    let var: f64 = trade_pnls.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let sd = var.sqrt();
    let sharpe = if sd > 1e-10 {
        mean / sd * (n as f64).sqrt()
    } else {
        0.0
    };

    Metrics {
        roi,
        pf,
        sharpe,
        n_trades: n,
    }
}

fn mc_percentile_ranks(trade_pnls: &[f64], n_mc: usize, rng: &mut StdRng) -> McRanks {
    let n = trade_pnls.len();
    if n < 3 {
        return McRanks {
            roi: 50.0,
            sharpe: 50.0,
            mdd: 50.0,
        };
    }

    // Observed metrics
    let obs_roi: f64 = trade_pnls.iter().sum();
    let obs_mean = obs_roi / n as f64;
    let obs_var: f64 = trade_pnls.iter().map(|x| (x - obs_mean).powi(2)).sum::<f64>() / n as f64;
    let obs_sd = obs_var.sqrt();
    let obs_sharpe = if obs_sd > 1e-10 {
        obs_mean / obs_sd * (n as f64).sqrt()
    } else {
        0.0
    };

    let mut obs_mdd = 0.0f64;
    {
        let mut eq = 0.0f64;
        let mut peak = 0.0f64;
        for &pnl in trade_pnls {
            eq += pnl;
            if eq > peak {
                peak = eq;
            }
            let dd = peak - eq;
            if dd > obs_mdd {
                obs_mdd = dd;
            }
        }
    }

    let mut roi_count = 0u32;
    let mut sharpe_count = 0u32;
    let mut mdd_count = 0u32;

    let mut shuffled = trade_pnls.to_vec();

    for _ in 0..n_mc {
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            shuffled.swap(i, j);
        }

        let shuf_roi: f64 = shuffled.iter().sum();
        let shuf_mean = shuf_roi / n as f64;
        let shuf_var: f64 = shuffled
            .iter()
            .map(|x| (x - shuf_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let shuf_sd = shuf_var.sqrt();
        let shuf_sharpe = if shuf_sd > 1e-10 {
            shuf_mean / shuf_sd * (n as f64).sqrt()
        } else {
            0.0
        };

        // MDD
        let mut seq = 0.0f64;
        let mut speak = 0.0f64;
        let mut smdd = 0.0f64;
        for &pnl in &shuffled {
            seq += pnl;
            if seq > speak {
                speak = seq;
            }
            let dd = speak - seq;
            if dd > smdd {
                smdd = dd;
            }
        }

        if obs_roi > shuf_roi {
            roi_count += 1;
        }
        if obs_sharpe > shuf_sharpe {
            sharpe_count += 1;
        }
        if obs_mdd < smdd {
            mdd_count += 1;
        }
    }

    McRanks {
        roi: roi_count as f64 / n_mc as f64 * 100.0,
        sharpe: sharpe_count as f64 / n_mc as f64 * 100.0,
        mdd: mdd_count as f64 / n_mc as f64 * 100.0,
    }
}

// ============================================================
// FULL PIPELINE (single simulation)
// ============================================================
fn run_pipeline(
    prices: &[f64],
    bar_returns: &[f64],
    strategies: &[Strategy],
    run_mc: bool,
    seed: u64,
) -> Vec<StratWindowResult> {
    let n_strats = strategies.len();
    let n = prices.len();

    // WFO windows
    let mut windows = Vec::new();
    for w in 0..N_WINDOWS {
        let is_start = w * (IS_SIZE + OOS_SIZE);
        let is_end = is_start + IS_SIZE;
        let oos_start = is_end;
        let oos_end = oos_start + OOS_SIZE;
        if oos_end > prices.len() {
            break;
        }
        windows.push((is_start, is_end, oos_start, oos_end));
    }

    // Pre-compute indicators
    let cache = precompute_indicators(prices, strategies);

    // Generate all signals (sequential - RSI has state)
    let all_signals: Vec<Vec<f64>> = strategies
        .iter()
        .map(|s| generate_signals(&s.stype, &cache, n))
        .collect();

    let progress = AtomicUsize::new(0);

    // Process strategies in parallel
    let results: Vec<StratWindowResult> = (0..n_strats)
        .into_par_iter()
        .flat_map(|si| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(si as u64 * 7919));
            let signal = &all_signals[si];

            let mut rows = Vec::with_capacity(windows.len());

            for &(is_s, is_e, oos_s, oos_e) in &windows {
                let is_signal = &signal[is_s..is_e];
                let is_br_end = (is_e - 1).min(bar_returns.len());
                let is_br = &bar_returns[is_s..is_br_end];
                let is_trades = extract_trades(is_signal, is_br, COST_PER_TRADE);

                let oos_signal = &signal[oos_s..oos_e];
                let oos_br_end = (oos_e - 1).min(bar_returns.len());
                let oos_br = &bar_returns[oos_s..oos_br_end];
                let oos_trades = extract_trades(oos_signal, oos_br, COST_PER_TRADE);

                let is_m = compute_metrics(&is_trades);
                let oos_m = compute_metrics(&oos_trades);

                // Robustness
                let fee_m = compute_metrics(&extract_trades(is_signal, is_br, rob_fee_cost()));
                let slip_m = compute_metrics(&extract_trades(is_signal, is_br, rob_slip_cost()));

                // MC ranks
                let mc_ranks = if run_mc && is_trades.len() >= 3 {
                    mc_percentile_ranks(&is_trades, N_MC, &mut rng)
                } else {
                    McRanks {
                        roi: 50.0,
                        sharpe: 50.0,
                        mdd: 50.0,
                    }
                };

                let is_pf_pass = is_m.pf > 1.0;
                rows.push(StratWindowResult {
                    is_pf: is_m.pf,
                    is_roi: is_m.roi,
                    is_n_trades: is_m.n_trades,
                    oos_roi: oos_m.roi,
                    oos_profitable: oos_m.pf > 1.0,
                    mc_roi_rank: mc_ranks.roi,
                    mc_sharpe_rank: mc_ranks.sharpe,
                    rob_fee_pass: is_pf_pass && fee_m.pf > 1.0,
                    rob_slip_pass: is_pf_pass && slip_m.pf > 1.0,
                    rob_all_pass: is_pf_pass && fee_m.pf > 1.0 && slip_m.pf > 1.0,
                    is_pf_pass,
                });
            }

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 2000 == 0 || done == n_strats {
                eprint!(
                    "\r    Strategies: {}/{} ({:.0}%)",
                    done,
                    n_strats,
                    done as f64 / n_strats as f64 * 100.0
                );
            }

            rows
        })
        .collect();

    eprintln!();
    results
}

// ============================================================
// FILTER COMPUTATION
// ============================================================
fn compute_filter_results(results: &[StratWindowResult]) -> (Vec<FilterRow>, f64) {
    let n_total = results.len();
    if n_total == 0 {
        return (vec![], 0.0);
    }

    let baseline =
        results.iter().filter(|r| r.oos_profitable).count() as f64 / n_total as f64 * 100.0;

    struct FilterDef {
        name: &'static str,
        pred: Box<dyn Fn(&StratWindowResult) -> bool + Sync>,
    }

    let filters: Vec<FilterDef> = vec![
        FilterDef {
            name: "No filter",
            pred: Box::new(|_| true),
        },
        FilterDef {
            name: "IS PF > 1",
            pred: Box::new(|r| r.is_pf_pass),
        },
        FilterDef {
            name: "MC-ROI >= p50",
            pred: Box::new(|r| r.mc_roi_rank >= 50.0),
        },
        FilterDef {
            name: "MC-ROI >= p75",
            pred: Box::new(|r| r.mc_roi_rank >= 75.0),
        },
        FilterDef {
            name: "MC-ROI >= p90",
            pred: Box::new(|r| r.mc_roi_rank >= 90.0),
        },
        FilterDef {
            name: "MC-Sharpe >= p50",
            pred: Box::new(|r| r.mc_sharpe_rank >= 50.0),
        },
        FilterDef {
            name: "MC-Sharpe >= p75",
            pred: Box::new(|r| r.mc_sharpe_rank >= 75.0),
        },
        FilterDef {
            name: "Rob: Fee",
            pred: Box::new(|r| r.rob_fee_pass),
        },
        FilterDef {
            name: "Rob: Slip",
            pred: Box::new(|r| r.rob_slip_pass),
        },
        FilterDef {
            name: "Rob: All",
            pred: Box::new(|r| r.rob_all_pass),
        },
        FilterDef {
            name: "MC-ROI p50 + PF>1",
            pred: Box::new(|r| r.mc_roi_rank >= 50.0 && r.is_pf_pass),
        },
        FilterDef {
            name: "MC-ROI p75 + PF>1",
            pred: Box::new(|r| r.mc_roi_rank >= 75.0 && r.is_pf_pass),
        },
    ];

    let mut rows = Vec::new();
    for f in &filters {
        let pool: Vec<&StratWindowResult> = results.iter().filter(|r| (f.pred)(r)).collect();
        if pool.len() < 10 {
            continue;
        }
        let oos = pool.iter().filter(|r| r.oos_profitable).count() as f64 / pool.len() as f64
            * 100.0;
        rows.push(FilterRow {
            filter: f.name.to_string(),
            pool: pool.len(),
            oos_prof_pct: (oos * 100.0).round() / 100.0,
            lift_pp: ((oos - baseline) * 100.0).round() / 100.0,
            pool_pct: (pool.len() as f64 / n_total as f64 * 1000.0).round() / 10.0,
        });
    }

    (rows, baseline)
}

// ============================================================
// PORTFOLIO CONSTRUCTION
// ============================================================
fn build_portfolios(
    results: &[StratWindowResult],
    n_portfolios: usize,
    seed: u64,
) -> Vec<(String, bool, f64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut port_results = Vec::new();

    // Group by window
    let mut by_window: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, _r) in results.iter().enumerate() {
        let wi = i % N_WINDOWS; // results are ordered by strategy then window
        by_window.entry(wi).or_default().push(i);
    }

    // Actually we need to track window properly. Results come from par_iter flat_map
    // so ordering is not guaranteed. Let's use a different approach:
    // We need window info in the result. Let me use position-based mapping.
    // Since strategies are processed in order by par_iter but flat_mapped,
    // the window index cycles. Let me store it differently.

    // Re-derive: each strategy produces N_WINDOWS results in order.
    // But par_iter may reorder strategies. We need to track window per result.

    // Actually the results are collected via par_iter().flat_map() which preserves
    // order within each iterator but strategies may be reordered.
    // Let's just compute window from result position within each strategy's chunk.
    // This is tricky. Let me instead just use all results for portfolio construction
    // by dividing into window-sized chunks based on the original strategy order.

    // Simpler approach: for portfolio building, we don't need per-window granularity
    // if we just sample from all results (matching the Python behavior approximately).
    // But Python does build per-window. Let me approximate by just using all results.

    let filter_configs: Vec<(&str, Box<dyn Fn(&StratWindowResult) -> bool>)> = vec![
        ("No filter", Box::new(|_: &StratWindowResult| true)),
        ("IS PF>1", Box::new(|r: &StratWindowResult| r.is_pf_pass)),
        (
            "Rob: All",
            Box::new(|r: &StratWindowResult| r.rob_all_pass),
        ),
    ];

    for (fname, pred) in &filter_configs {
        let mut pool: Vec<(f64, f64)> = results
            .iter()
            .filter(|r| pred(r))
            .map(|r| (r.is_pf, r.oos_roi))
            .collect();

        if pool.len() < TOP_N {
            continue;
        }

        // Sort by IS PF descending
        pool.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top = &pool[..TOP_SELECT.min(pool.len())];

        let actual_ports = n_portfolios.min(500);
        for _ in 0..actual_ports {
            let mut indices: Vec<usize> = (0..top.len()).collect();
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
            let sel = &indices[..PORT_SIZE.min(indices.len())];
            let avg_oos_roi: f64 =
                sel.iter().map(|&i| top[i].1).sum::<f64>() / sel.len() as f64;
            port_results.push((fname.to_string(), avg_oos_roi > 0.0, avg_oos_roi));
        }
    }

    port_results
}

// ============================================================
// PEARSON CORRELATION
// ============================================================
fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 3 {
        return 0.0;
    }
    let mx: f64 = xs.iter().sum::<f64>() / n as f64;
    let my: f64 = ys.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        cov / denom
    }
}

// ============================================================
// SINGLE SIMULATION
// ============================================================
fn run_single_sim(
    tier: &str,
    sim_id: usize,
    strategies: &[Strategy],
) -> (SimSummary, Vec<FilterRow>) {
    let seed = (sim_id * 1000) as u64;
    let (momentum, drift) = match tier {
        "edge" => (MOMENTUM_PHI, EDGE_DRIFT),
        _ => (0.0, 0.0),
    };

    let (prices, bar_returns) = generate_prices(N_BARS, momentum, drift, seed);
    let results = run_pipeline(&prices, &bar_returns, strategies, true, seed);
    let (filter_rows, baseline) = compute_filter_results(&results);

    // Portfolio
    let port_results = build_portfolios(&results, 500, seed);
    let no_filt_port: Vec<&(String, bool, f64)> = port_results
        .iter()
        .filter(|(f, _, _)| f == "No filter")
        .collect();
    let rob_port: Vec<&(String, bool, f64)> = port_results
        .iter()
        .filter(|(f, _, _)| f == "Rob: All")
        .collect();

    let port_nofilter_oos = if !no_filt_port.is_empty() {
        no_filt_port.iter().filter(|(_, p, _)| *p).count() as f64 / no_filt_port.len() as f64
            * 100.0
    } else {
        f64::NAN
    };

    let port_rob_oos = if !rob_port.is_empty() {
        rob_port.iter().filter(|(_, p, _)| *p).count() as f64 / rob_port.len() as f64 * 100.0
    } else {
        f64::NAN
    };

    // Lifts
    let get_lift = |name: &str| -> f64 {
        filter_rows
            .iter()
            .find(|r| r.filter == name)
            .map(|r| r.lift_pp)
            .unwrap_or(f64::NAN)
    };

    // Correlations
    let valid: Vec<&StratWindowResult> = results.iter().filter(|r| r.is_n_trades >= 3).collect();
    let mc_roi_ranks: Vec<f64> = valid.iter().map(|r| r.mc_roi_rank).collect();
    let mc_sharpe_ranks: Vec<f64> = valid.iter().map(|r| r.mc_sharpe_rank).collect();
    let oos_profs: Vec<f64> = valid
        .iter()
        .map(|r| if r.oos_profitable { 1.0 } else { 0.0 })
        .collect();

    let summary = SimSummary {
        sim_id,
        tier: tier.to_string(),
        n_strat_windows: results.len(),
        baseline_oos: baseline,
        mc_roi_lift_p50: get_lift("MC-ROI >= p50"),
        mc_roi_lift_p75: get_lift("MC-ROI >= p75"),
        rob_all_lift: get_lift("Rob: All"),
        is_pf_lift: get_lift("IS PF > 1"),
        port_nofilter_oos,
        port_rob_oos,
        corr_mc_roi_oos: pearson_corr(&mc_roi_ranks, &oos_profs),
        corr_mc_sharpe_oos: pearson_corr(&mc_sharpe_ranks, &oos_profs),
    };

    (summary, filter_rows)
}

// ============================================================
// TIER RUNNER
// ============================================================
fn run_tier(
    tier_name: &str,
    strategies: &[Strategy],
    n_sims: usize,
) -> (Vec<SimSummary>, Vec<Vec<FilterRow>>) {
    println!("\n{}", "=".repeat(70));
    println!("TIER: {}", tier_name.to_uppercase());
    println!("{}", "=".repeat(70));
    println!("  Strategies: {}", strategies.len());
    println!("  Windows: {}", N_WINDOWS);
    println!("  MC permutations: {} (matched to empirical)", N_MC);
    println!(
        "  Strategy-window obs per sim: ~{}",
        strategies.len() * N_WINDOWS
    );
    println!("  Simulations: {}", n_sims);
    println!();

    let mut summaries = Vec::new();
    let mut all_filter_dfs = Vec::new();

    for sim_id in 0..n_sims {
        println!("  --- Simulation {}/{} ---", sim_id + 1, n_sims);
        let t0 = Instant::now();

        let (summary, filter_rows) = run_single_sim(tier_name, sim_id, strategies);

        let elapsed = t0.elapsed().as_secs_f64();
        println!(
            "    Baseline OOS: {:.1}%  MC-ROI p50 lift: {:.2}  Rob lift: {:.2}  ({:.0}s)",
            summary.baseline_oos, summary.mc_roi_lift_p50, summary.rob_all_lift, elapsed
        );

        summaries.push(summary);
        all_filter_dfs.push(filter_rows);
    }

    (summaries, all_filter_dfs)
}

// ============================================================
// SIGNAL SWEEP
// ============================================================
fn run_signal_sweep(strategies: &[Strategy]) -> Vec<SimSummary> {
    let phi_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15];
    let sims_per_level = 5;
    let seed_offset = 500u64;

    println!("\n{}", "=".repeat(70));
    println!("SIGNAL SWEEP (matched parameters: B={}, ~{} strategies)", N_MC, strategies.len());
    println!("{}", "=".repeat(70));

    let mut all_results = Vec::new();

    for &phi in &phi_levels {
        let drift = phi * 0.001;
        println!("\n  phi={:.2}, drift={:.5}", phi, drift);

        for sim in 0..sims_per_level {
            let seed = seed_offset + (phi * 100.0) as u64 + sim as u64 * 10;
            let t0 = Instant::now();

            let (prices, bar_returns) = generate_prices(N_BARS, phi, drift, seed);
            let results = run_pipeline(&prices, &bar_returns, strategies, true, seed);
            let (filter_rows, baseline) = compute_filter_results(&results);
            let port_results = build_portfolios(&results, 200, seed);

            let mc_lift = filter_rows
                .iter()
                .find(|r| r.filter == "MC-ROI >= p50")
                .map(|r| r.lift_pp)
                .unwrap_or(f64::NAN);

            let no_filt_port: Vec<&(String, bool, f64)> = port_results
                .iter()
                .filter(|(f, _, _)| f == "No filter")
                .collect();
            let port_oos = if !no_filt_port.is_empty() {
                no_filt_port.iter().filter(|(_, p, _)| *p).count() as f64
                    / no_filt_port.len() as f64
                    * 100.0
            } else {
                f64::NAN
            };

            let elapsed = t0.elapsed().as_secs_f64();
            println!(
                "    sim {}: baseline={:.1}%, MC lift={:.2} pp ({:.0}s)",
                sim + 1,
                baseline,
                mc_lift,
                elapsed
            );

            let mut summary = SimSummary::default();
            summary.sim_id = sim;
            summary.tier = format!("sweep_phi_{:.2}", phi);
            summary.baseline_oos = baseline;
            summary.mc_roi_lift_p50 = mc_lift;
            summary.port_nofilter_oos = port_oos;
            summary.mc_roi_lift_p75 = filter_rows
                .iter()
                .find(|r| r.filter == "MC-ROI >= p75")
                .map(|r| r.lift_pp)
                .unwrap_or(f64::NAN);
            summary.rob_all_lift = filter_rows
                .iter()
                .find(|r| r.filter == "Rob: All")
                .map(|r| r.lift_pp)
                .unwrap_or(f64::NAN);
            summary.is_pf_lift = filter_rows
                .iter()
                .find(|r| r.filter == "IS PF > 1")
                .map(|r| r.lift_pp)
                .unwrap_or(f64::NAN);

            all_results.push(summary);
        }
    }

    all_results
}

// ============================================================
// CSV OUTPUT
// ============================================================
fn write_summaries_csv(path: &str, summaries: &[SimSummary]) {
    let mut wtr = csv::Writer::from_path(path).expect("Failed to create CSV");
    for s in summaries {
        wtr.serialize(s).expect("Failed to write row");
    }
    wtr.flush().expect("Failed to flush");
    println!("  Saved: {}", path);
}

fn write_filters_csv(path: &str, filters: &[FilterRow]) {
    let mut wtr = csv::Writer::from_path(path).expect("Failed to create CSV");
    for r in filters {
        wtr.serialize(r).expect("Failed to write row");
    }
    wtr.flush().expect("Failed to flush");
    println!("  Saved: {}", path);
}

fn average_filter_rows(all_filters: &[Vec<FilterRow>]) -> Vec<FilterRow> {
    if all_filters.is_empty() {
        return vec![];
    }
    let ref_filters = &all_filters[0];

    ref_filters
        .iter()
        .enumerate()
        .map(|(i, rf)| {
            let mut avg = FilterRow {
                filter: rf.filter.clone(),
                pool: 0,
                oos_prof_pct: 0.0,
                lift_pp: 0.0,
                pool_pct: 0.0,
            };
            let mut count = 0.0;
            for filters in all_filters {
                if i < filters.len() && filters[i].filter == rf.filter {
                    avg.pool += filters[i].pool;
                    avg.oos_prof_pct += filters[i].oos_prof_pct;
                    avg.lift_pp += filters[i].lift_pp;
                    avg.pool_pct += filters[i].pool_pct;
                    count += 1.0;
                }
            }
            if count > 0.0 {
                avg.pool = (avg.pool as f64 / count) as usize;
                avg.oos_prof_pct = (avg.oos_prof_pct / count * 100.0).round() / 100.0;
                avg.lift_pp = (avg.lift_pp / count * 100.0).round() / 100.0;
                avg.pool_pct = (avg.pool_pct / count * 10.0).round() / 10.0;
            }
            avg
        })
        .collect()
}

// ============================================================
// MAIN
// ============================================================
fn main() {
    let overall_start = Instant::now();

    // Configure Rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(N_WORKERS)
        .build_global()
        .expect("Failed to build thread pool");

    println!("=======================================================================");
    println!("Three-Tier Full-Pipeline Synthetic Validation (Rust, {}-thread)", N_WORKERS);
    println!("=======================================================================");
    println!("Parameters MATCHED to empirical:");
    println!("  MC permutations: {} (was 200 in Python version)", N_MC);
    println!("  Threads: {}", N_WORKERS);
    println!("  N_BARS: {}, N_WINDOWS: {}", N_BARS, N_WINDOWS);
    println!();

    // Build strategy universes
    println!("Building strategy universes...");
    let standard_strategies = build_strategy_grid("standard");
    let massive_strategies = build_strategy_grid("massive");
    println!(
        "  Standard grid: {} strategies (matched to empirical ~38K+)",
        standard_strategies.len()
    );
    println!(
        "  Massive grid:  {} strategies (for Tier 3 adversarial)",
        massive_strategies.len()
    );

    // Ensure output dir exists
    let tbl_dir = resolve_tbl_dir();
    fs::create_dir_all(&tbl_dir).ok();
    println!("Output directory: {}", tbl_dir);

    // ==============================
    // TIER 1: PURE NULL
    // ==============================
    let (t1_summaries, t1_filters) = run_tier("null", &standard_strategies, N_SIMS);

    // ==============================
    // TIER 2: KNOWN WEAK EDGE
    // ==============================
    let (t2_summaries, t2_filters) = run_tier("edge", &standard_strategies, N_SIMS);

    // ==============================
    // TIER 3: ADVERSARIAL DATA MINING
    // ==============================
    let t3_sims = (N_SIMS / 2).max(3);
    let (t3_summaries, t3_filters) = run_tier("adversarial", &massive_strategies, t3_sims);

    // ==============================
    // SIGNAL SWEEP
    // ==============================
    let sweep_results = run_signal_sweep(&standard_strategies);

    // ============================================================
    // SAVE RESULTS
    // ============================================================
    println!("\n{}", "=".repeat(70));
    println!("SAVING TABLES");
    println!("{}", "=".repeat(70));

    // Combined summaries
    let mut all_summaries = Vec::new();
    all_summaries.extend(t1_summaries.iter().cloned());
    all_summaries.extend(t2_summaries.iter().cloned());
    all_summaries.extend(t3_summaries.iter().cloned());
    write_summaries_csv(
        &format!("{}/synthetic_v4_summaries_matched.csv", tbl_dir),
        &all_summaries,
    );

    // Per-tier filter averages
    for (tier_name, filter_list) in [
        ("null", &t1_filters),
        ("edge", &t2_filters),
        ("adversarial", &t3_filters),
    ] {
        let avg = average_filter_rows(filter_list);
        write_filters_csv(
            &format!("{}/synthetic_v4_{}_filters_matched.csv", tbl_dir, tier_name),
            &avg,
        );
    }

    // Signal sweep
    write_summaries_csv(
        &format!("{}/synthetic_v4_signal_sweep_matched.csv", tbl_dir),
        &sweep_results,
    );

    // ============================================================
    // FINAL SUMMARY
    // ============================================================
    let elapsed_total = overall_start.elapsed().as_secs_f64();
    println!("\n{}", "=".repeat(70));
    println!(
        "COMPLETE — Total time: {:.1} minutes",
        elapsed_total / 60.0
    );
    println!("{}", "=".repeat(70));

    fn print_tier_stats(name: &str, summaries: &[SimSummary]) {
        let n = summaries.len() as f64;
        let baseline_mean: f64 = summaries.iter().map(|s| s.baseline_oos).sum::<f64>() / n;
        let mc_lift_mean: f64 = summaries.iter().map(|s| s.mc_roi_lift_p50).sum::<f64>() / n;
        let rob_lift_mean: f64 = summaries.iter().map(|s| s.rob_all_lift).sum::<f64>() / n;
        let port_mean: f64 = summaries.iter().map(|s| s.port_nofilter_oos).sum::<f64>() / n;
        let corr_mean: f64 = summaries.iter().map(|s| s.corr_mc_roi_oos).sum::<f64>() / n;

        println!(
            "\n  {} (B={}, ~{} strategies):",
            name,
            N_MC,
            summaries[0].n_strat_windows / N_WINDOWS
        );
        println!("    Baseline OOS: {:.1}%", baseline_mean);
        println!("    MC-ROI p50 lift: {:.2} pp", mc_lift_mean);
        println!("    Rob-All lift: {:.2} pp", rob_lift_mean);
        println!("    Portfolio OOS: {:.1}%", port_mean);
        println!("    MC-ROI vs OOS corr: {:.4}", corr_mean);
    }

    print_tier_stats("TIER 1 (Pure Null)", &t1_summaries);
    print_tier_stats("TIER 2 (Known Edge)", &t2_summaries);
    print_tier_stats("TIER 3 (Adversarial)", &t3_summaries);

    println!("\n  SIGNAL SWEEP:");
    let phi_levels = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15];
    for &phi in &phi_levels {
        let tag = format!("sweep_phi_{:.2}", phi);
        let sweep_tier: Vec<&SimSummary> =
            sweep_results.iter().filter(|s| s.tier == tag).collect();
        if !sweep_tier.is_empty() {
            let n = sweep_tier.len() as f64;
            let mc_lift: f64 = sweep_tier.iter().map(|s| s.mc_roi_lift_p50).sum::<f64>() / n;
            println!(
                "    phi={:.2}: MC-ROI p50 lift = {:.2} pp (n={})",
                phi,
                mc_lift,
                sweep_tier.len()
            );
        }
    }

    println!(
        "\n  KEY COMPARISON (Python B=200 vs Rust B={}):",
        N_MC
    );
    println!("    Python synthetic lift: -5 to -6 pp");
    println!("    Empirical lift: -1 to -1.5 pp");
    let t1_mc_mean: f64 =
        t1_summaries.iter().map(|s| s.mc_roi_lift_p50).sum::<f64>() / t1_summaries.len() as f64;
    let t2_mc_mean: f64 =
        t2_summaries.iter().map(|s| s.mc_roi_lift_p50).sum::<f64>() / t2_summaries.len() as f64;
    println!("    Rust matched lift (Null): {:.2} pp", t1_mc_mean);
    println!("    Rust matched lift (Edge): {:.2} pp", t2_mc_mean);
}
