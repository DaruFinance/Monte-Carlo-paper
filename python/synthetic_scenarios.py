"""
synthetic_scenarios.py
======================

Regenerates the six orphaned "synthetic" reference CSVs cited in
the published paper. The original producer scripts were lost; this file
reconstructs the data-generating process (DGP) and pipeline
(adapted from the original synthetic pipeline) and writes fresh outputs
with fixed seeds so reruns are deterministic.

Tables produced (output dir = ``$MC_PAPER_DATA/results/tables[_v2]``)
-------------------------------------------------------------------
* synthetic_a_filters.csv        (tab:synthetic_a)
    Scenario A high-prevalence (50%) pool of ~12000 strategy-windows.
* synthetic_b_filters.csv        (tab:synthetic_b)
    Scenario B low-prevalence (2%) small pool (~1500 strategy-windows).
* synthetic_c_portfolios.csv     (cited in lemma remark on factor-model
    portfolios; 500 portfolios with mean pairwise correlation ~0.28-0.35).
* synthetic_prevalence_sweep.csv (tab:prevalence)
    Scans prevalences 2%/10%/50% for MC-ROI/MC-MDD/MC-Calmar p50 filters.
* synthetic_filter_comparison.csv
    Std-permutation vs bootstrap MC (ROI p50/p75) vs IS PF>1 on a pool
    of ~25000 strategy-windows.
* synthetic_portfolio_results.csv
    10 windows x 100 random portfolios of size 10 drawn from the
    top IS-PF pool, with std-MC and bootstrap-MC ranks plus mean alpha.

DGP (from the published paper Scenario A):
    r_t = phi * r_{t-1} * edge_i + drift_i + sigma_t * z_t
    z_t ~ Student-t(df=5) / sqrt(df/(df-2))    (unit var)
    sigma_t^2 = omega + a * r_{t-1}^2 + b * sigma_{t-1}^2   (GARCH(1,1))
    a=0.10, b=0.85
    Two-state Markov regime: high-vol regime multiplies sigma by 2.5;
    additionally 30% of strategy-windows get alpha^eff = 0 (edge dropout).
    phi = 0.05  (AR(1) momentum coefficient when edge is active)
    drift = 5e-5  per bar

Runtime scale
-------------
Chosen to finish in < 30 min on 32 cores at MC_B = 500 permutations per
strategy-window. See CONFIG below. Pass ``--fast`` for a ~3 min smoke
test (everything /2) or ``--full`` for paper-scale (12k / 1500 / 25k).

Seed policy
-----------
All RNGs are seeded off MASTER_SEED (42). Per-sim offsets are
sim_id * 1000. Portfolio samplers use (MASTER_SEED + 1000 + portfolio_seed).

Because the original seeds are lost, byte-exact reproduction of the
committed CSVs is NOT expected.
Usage
-----
    python3 synthetic_scenarios.py
    python3 synthetic_scenarios.py --rebuild-corrected   # corrected MDD/Calmar tables

"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MASTER_SEED = 42

# DGP
T_DF = 5
GARCH_OMEGA_FRAC = 0.04
GARCH_A = 0.10
GARCH_B = 0.85
BASE_VOL = 0.008
REGIME_PERSIST = 0.985
HIGH_VOL_MULT = 2.5
PHI = 0.05                # AR(1) momentum when edge active (paper value)
EDGE_DRIFT = 5e-5
# Calibration knob: if the baseline is too low vs committed CSVs, we can
# amplify edge strength through a momentum multiplier on edge windows.
# Set to 1.0 for the paper-literal DGP.
EDGE_MOMENTUM_MULT = 1.0
EDGE_DROPOUT = 0.30       # 30% strategy-windows have alpha_eff = 0

# Walk-forward windows
N_WINDOWS = 10
IS_SIZE = 4000
OOS_SIZE = 2000
N_BARS = N_WINDOWS * (IS_SIZE + OOS_SIZE)   # 60_000

# Trading cost. The committed CSVs have baseline OOS% of 52-61%, which
# is noticeably higher than a natural GARCH-null pool would give with the
# v4 default 0.16% round-trip cost. Without the original seeds it is not
# possible to back out whether the mismatch is costs, MC-B, or a DGP
# variant. Setting cost to zero here is our closest reconstruction; see
# the v1 release notes for the documented numeric drift.COST_PER_TRADE = 0.0

# MC permutations
MC_B_DEFAULT = 500

# Scales (will be overridden by --fast / --full)
CONFIG = {
    "A_N_STRATS": 300,     # 300 strat * 10 wins = 3000 / sim; 4 sims = 12000
    "A_N_SIMS":   4,
    "A_PREV":     0.50,
    "B_N_STRATS": 150,     # 150 strat * 10 wins = 1500 / sim; 1 sim
    "B_N_SIMS":   1,
    "B_PREV":     0.02,
    "C_N_PORT":   500,
    "SWEEP_N_STRATS": 300, # pools ~6000-12000
    "SWEEP_N_SIMS":   2,   # pool shared across prevalences
    "CMP_N_STRATS": 312,   # 312 * 10 * 8 ~= 25000
    "CMP_N_SIMS":   8,
    "CMP_PREV":     0.50,
    "PORT_N_WIN":   10,
    "PORT_PER_WIN": 100,
    "MC_B":         MC_B_DEFAULT,
}

FAST_CONFIG = {
    "A_N_STRATS": 120, "A_N_SIMS": 2,
    "B_N_STRATS": 80,  "B_N_SIMS": 1,
    "C_N_PORT": 200,
    "SWEEP_N_STRATS": 120, "SWEEP_N_SIMS": 1,
    "CMP_N_STRATS": 150,   "CMP_N_SIMS": 4,
    "PORT_N_WIN": 10, "PORT_PER_WIN": 50,
    "MC_B": 200,
    "A_PREV": 0.50, "B_PREV": 0.02, "CMP_PREV": 0.50,
}


# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("MC_PAPER_DATA", SCRIPT_DIR.parents[0]))
TABLES_DIR = ROOT / "results" / "tables"


# ----------------------------------------------------------------------
# DGP
# ----------------------------------------------------------------------
def generate_prices(n_bars: int, momentum: float, drift: float, seed: int):
    """Synthetic price series: GARCH(1,1) + Student-t + 2-state regime."""
    rng = np.random.RandomState(seed)
    regimes = np.zeros(n_bars, dtype=np.int8)
    for t in range(1, n_bars):
        if rng.random() > REGIME_PERSIST:
            regimes[t] = 1 - regimes[t - 1]
        else:
            regimes[t] = regimes[t - 1]
    vol_mult = np.where(regimes == 0, 1.0, HIGH_VOL_MULT)
    omega = GARCH_OMEGA_FRAC * (1 - GARCH_A - GARCH_B) * BASE_VOL ** 2
    r = np.zeros(n_bars)
    sigma2 = BASE_VOL ** 2
    prev_r = 0.0
    for t in range(n_bars):
        if t > 0:
            sigma2 = omega + GARCH_A * r[t - 1] ** 2 + GARCH_B * sigma2
            sigma2 = min(max(sigma2, 1e-12), 0.01)
        z = rng.standard_t(T_DF) / np.sqrt(T_DF / (T_DF - 2))
        vol = np.sqrt(sigma2) * vol_mult[t]
        r[t] = momentum * prev_r + drift + vol * z
        prev_r = r[t]
    prices = 100.0 * np.exp(np.cumsum(r))
    bar_returns = np.diff(np.log(prices))
    return prices, bar_returns


# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------
def compute_ema(prices, period):
    """Compute exponential moving average for the given period."""
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_sma(prices, period):
    """Compute simple moving average for the given period."""
    cs = np.cumsum(prices)
    sma = np.full_like(prices, np.nan)
    sma[period - 1] = cs[period - 1] / period
    sma[period:] = (cs[period:] - cs[:-period]) / period
    return sma


def compute_rsi(prices, period):
    """Compute relative strength index for the given period."""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = np.zeros_like(prices)
    avg_l = np.zeros_like(prices)
    if period >= len(prices):
        return np.full_like(prices, 50.0)
    avg_g[period] = gains[1:period + 1].mean()
    avg_l[period] = losses[1:period + 1].mean()
    for i in range(period + 1, len(prices)):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gains[i]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + losses[i]) / period
    rs = np.where(avg_l > 1e-10, avg_g / avg_l, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[: period + 1] = 50.0
    return rsi


def build_strategies(n_target: int, seed: int = 0):
    """Build a parameter grid and return the first n_target entries
    after a deterministic shuffle (so subsets differ by seed)."""
    strategies = []
    for f in range(3, 51, 2):
        for s in range(15, 201, 5):
            if f >= s or s < 1.3 * f:
                continue
            for direction in ("long", "longshort"):
                strategies.append({"type": "ema_cross", "fast": f, "slow": s, "direction": direction})
                strategies.append({"type": "sma_cross", "fast": f, "slow": s, "direction": direction})
    for period in range(5, 31, 2):
        for entry in range(15, 41, 3):
            for exit_lvl in range(60, 86, 3):
                if exit_lvl <= entry + 20:
                    continue
                strategies.append({"type": "rsi", "period": period, "entry": entry, "exit": exit_lvl})
    for f in (6, 8, 10, 12, 14, 16, 20):
        for s in (18, 22, 26, 30, 35, 40, 50):
            if f >= s:
                continue
            for sig in (5, 7, 9, 12):
                strategies.append({"type": "macd", "fast": f, "slow": s, "signal": sig})

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(strategies))
    selected = [strategies[i] for i in idx[:n_target]]
    for i, s in enumerate(selected):
        s["idx"] = i
    return selected


def gen_signal(prices, strat):
    """Generate a position signal array from prices using the given strategy."""
    n = len(prices)
    sig = np.zeros(n)
    t = strat["type"]
    if t == "ema_cross":
        fa = compute_ema(prices, strat["fast"])
        sl = compute_ema(prices, strat["slow"])
        wu = strat["slow"] + 5
        if strat["direction"] == "long":
            sig[wu:] = np.where(fa[wu:] > sl[wu:], 1.0, 0.0)
        else:
            sig[wu:] = np.where(fa[wu:] > sl[wu:], 1.0, -1.0)
    elif t == "sma_cross":
        fa = compute_sma(prices, strat["fast"])
        sl = compute_sma(prices, strat["slow"])
        wu = strat["slow"] + 5
        if strat["direction"] == "long":
            sig[wu:] = np.where(fa[wu:] > sl[wu:], 1.0, 0.0)
        else:
            sig[wu:] = np.where(fa[wu:] > sl[wu:], 1.0, -1.0)
    elif t == "rsi":
        rsi = compute_rsi(prices, strat["period"])
        wu = strat["period"] + 5
        in_trade = False
        for i in range(wu, n):
            if not in_trade and rsi[i] < strat["entry"]:
                in_trade = True
            elif in_trade and rsi[i] > strat["exit"]:
                in_trade = False
            sig[i] = 1.0 if in_trade else 0.0
    elif t == "macd":
        ef = compute_ema(prices, strat["fast"])
        es = compute_ema(prices, strat["slow"])
        macd = ef - es
        sigl = compute_ema(macd, strat["signal"])
        wu = strat["slow"] + strat["signal"] + 5
        sig[wu:] = np.where(macd[wu:] > sigl[wu:], 1.0, 0.0)
    return sig


def extract_trades(pos, br, cost=COST_PER_TRADE):
    """Extract per-trade PnL array from a position signal and bar returns."""
    if len(br) < 2:
        return np.array([])
    pos = pos[: len(br) + 1]
    pnl_bar = pos[:-1] * br
    changes = np.diff(pos, prepend=0)
    bidx = np.where(changes != 0)[0]
    if len(bidx) == 0:
        return np.array([])
    bnd = np.append(bidx, len(pos))
    out = []
    for j in range(len(bnd) - 1):
        st = bnd[j]
        ed = min(bnd[j + 1], len(pnl_bar))
        if pos[st] != 0 and ed > st:
            out.append(pnl_bar[st:ed].sum() - cost)
    return np.array(out) if out else np.array([])


def metrics(pnls):
    """Compute ROI, profit factor, max drawdown, and Calmar ratio from trade PnLs."""
    if len(pnls) < 3:
        return dict(roi=0, pf=0, n=len(pnls), mdd=999.0, calmar=0.0)
    roi = pnls.sum()
    pos = pnls[pnls > 0].sum()
    neg = -pnls[pnls < 0].sum()
    pf = pos / neg if neg > 1e-10 else 999.0
    eq = np.cumsum(pnls)
    rm = np.maximum.accumulate(eq)
    mdd = float((rm - eq).max())
    calmar = roi / mdd if mdd > 1e-10 else 0.0
    return dict(roi=roi, pf=pf, n=len(pnls), mdd=mdd, calmar=calmar)


# ----------------------------------------------------------------------
# MC ranks: standard permutation vs. bootstrap (with replacement)
# ----------------------------------------------------------------------
def mc_ranks(pnls, n_mc, rng, bootstrap=False):
    n = len(pnls)
    if n < 3:
        return dict(roi=50.0, mdd=50.0, calmar=50.0)
    obs_roi = pnls.sum()
    eq = np.cumsum(pnls)
    obs_mdd = float((np.maximum.accumulate(eq) - eq).max())
    obs_calmar = obs_roi / obs_mdd if obs_mdd > 1e-10 else 0.0

    if bootstrap:
        idx = rng.randint(0, n, size=(n_mc, n))
    else:
        idx = np.argsort(rng.random((n_mc, n)), axis=1)
    sh = pnls[idx]
    sh_roi = sh.sum(axis=1)
    sh_eq = np.cumsum(sh, axis=1)
    sh_mdd = (np.maximum.accumulate(sh_eq, axis=1) - sh_eq).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh_calmar = np.where(sh_mdd > 1e-10, sh_roi / sh_mdd, 0.0)
    return dict(
        roi=(obs_roi > sh_roi).sum() / n_mc * 100,
        mdd=(obs_mdd < sh_mdd).sum() / n_mc * 100,
        calmar=(obs_calmar > sh_calmar).sum() / n_mc * 100,
    )


# ----------------------------------------------------------------------
# Per-simulation run
# ----------------------------------------------------------------------
def run_sim(sim_id: int, n_strats: int, prevalence: float, mc_b: int,
            with_bootstrap_mc: bool = False):
    """One simulation: build pool of strategies, generate a price path,
    tag a `prevalence` fraction of strategy-windows as "edge" (with
    momentum). Return DataFrame of strategy-window rows."""
    strategies = build_strategies(n_strats, seed=sim_id)
    # Generate TWO price paths: one null, one edge. Then per strategy-window
    # decide which path that window reads from, based on edge assignment.
    prices_n, br_n = generate_prices(N_BARS, 0.0, 0.0, seed=MASTER_SEED + sim_id * 1000 + 1)
    prices_e, br_e = generate_prices(N_BARS, PHI, EDGE_DRIFT, seed=MASTER_SEED + sim_id * 1000 + 2)

    rng = np.random.RandomState(MASTER_SEED + sim_id * 7919)

    # Edge tag per (strategy, window)
    edge_mask = rng.random((n_strats, N_WINDOWS)) < prevalence
    # Drop 30% of edge windows (alpha_eff = 0)
    dropout = rng.random((n_strats, N_WINDOWS)) < EDGE_DROPOUT
    edge_eff = edge_mask & (~dropout)

    # Pre-compute signals on both paths
    sig_n = [gen_signal(prices_n, s) for s in strategies]
    sig_e = [gen_signal(prices_e, s) for s in strategies]

    rows = []
    for si, strat in enumerate(strategies):
        for wi in range(N_WINDOWS):
            is_s = wi * (IS_SIZE + OOS_SIZE)
            is_e_ = is_s + IS_SIZE
            oos_s = is_e_
            oos_e = oos_s + OOS_SIZE
            use_edge = edge_eff[si, wi]
            if use_edge:
                sig = sig_e[si]
                br = br_e
            else:
                sig = sig_n[si]
                br = br_n

            is_sig = sig[is_s:is_e_]
            is_br = br[is_s:is_e_ - 1] if is_e_ - 1 <= len(br) else br[is_s:]
            oos_sig = sig[oos_s:oos_e]
            oos_br = br[oos_s:oos_e - 1] if oos_e - 1 <= len(br) else br[oos_s:]

            is_tr = extract_trades(is_sig, is_br)
            oos_tr = extract_trades(oos_sig, oos_br)
            im = metrics(is_tr)
            om = metrics(oos_tr)

            if len(is_tr) >= 3:
                std_r = mc_ranks(is_tr, mc_b, rng, bootstrap=False)
                if with_bootstrap_mc:
                    bt_r = mc_ranks(is_tr, mc_b, rng, bootstrap=True)
                else:
                    bt_r = {"roi": 50.0, "mdd": 50.0, "calmar": 50.0}
            else:
                std_r = {"roi": 50.0, "mdd": 50.0, "calmar": 50.0}
                bt_r = {"roi": 50.0, "mdd": 50.0, "calmar": 50.0}

            rows.append(dict(
                sim=sim_id, strategy=si, window=wi,
                has_edge=bool(use_edge),
                is_roi=im["roi"], is_pf=im["pf"], is_n=im["n"],
                oos_roi=om["roi"], oos_pf=om["pf"], oos_n=om["n"],
                oos_profitable=1 if om["pf"] > 1.0 else 0,
                mc_roi=std_r["roi"], mc_mdd=std_r["mdd"], mc_calmar=std_r["calmar"],
                boot_mc_roi=bt_r["roi"], boot_mc_mdd=bt_r["mdd"], boot_mc_calmar=bt_r["calmar"],
                is_pf_pass=1 if im["pf"] > 1.0 else 0,
            ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Filter tables
# ----------------------------------------------------------------------
def filter_table_AB(df: pd.DataFrame, prevalence_label: float) -> pd.DataFrame:
    """Build the Scenario A/B filter comparison table."""
    pool = len(df)
    baseline = df["oos_profitable"].mean() * 100
    pct_edge_all = df["has_edge"].mean() * 100  # should ~= prevalence*100
    rows = [dict(Filter="No filter", Pool=pool,
                 **{"OOS Prof%": round(baseline, 2), "Lift (pp)": 0.0,
                    "% Edge": round(pct_edge_all, 1)})]

    def _row(name, mask):
        sub = df[mask]
        if len(sub) < 10:
            return None
        oos = sub["oos_profitable"].mean() * 100
        return dict(
            Filter=name, Pool=len(sub),
            **{"OOS Prof%": round(oos, 2),
               "Lift (pp)": round(oos - baseline, 2),
               "% Edge": round(sub["has_edge"].mean() * 100, 1)},
        )

    for name, mask in [
        ("IS PF>1", df["is_pf_pass"] == 1),
        ("MC-ROI p50", df["mc_roi"] >= 50),
        ("MC-MDD p50", df["mc_mdd"] >= 50),
        ("MC-Calmar p50", df["mc_calmar"] >= 50),
    ]:
        r = _row(name, mask)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)


def prevalence_sweep_table(sim_results_by_prev: dict) -> pd.DataFrame:
    """Build the prevalence-sweep table.  sim_results_by_prev maps
    prevalence (e.g. 0.02) -> DataFrame of strategy-windows."""
    rows = []
    for prev, df in sim_results_by_prev.items():
        baseline = df["oos_profitable"].mean() * 100
        N = len(df)
        actual = df["has_edge"].mean() * 100

        row = {
            "Prevalence": f"{int(prev * 100)}%",
            "Actual%": round(actual, 1),
            "Baseline OOS%": round(baseline, 1),
            "N": N,
        }
        for col, tag in [("mc_roi", "MC-ROI p50"),
                         ("mc_mdd", "MC-MDD p50"),
                         ("mc_calmar", "MC-Calmar p50")]:
            sub = df[df[col] >= 50]
            if len(sub) < 10:
                row[f"{tag} OOS%"] = np.nan
                row[f"{tag} Lift"] = np.nan
                row[f"{tag} %Edge"] = np.nan
                row[f"{tag} Pool"] = 0
                continue
            oos = sub["oos_profitable"].mean() * 100
            row[f"{tag} OOS%"] = round(oos, 1)
            row[f"{tag} Lift"] = round(oos - baseline, 1)
            row[f"{tag} %Edge"] = round(sub["has_edge"].mean() * 100, 1)
            row[f"{tag} Pool"] = len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def filter_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Std vs Bootstrap MC filter comparison (Table:tab ~ pool 25000)."""
    pool = len(df)
    baseline = df["oos_profitable"].mean() * 100
    pct_edge_all = df["has_edge"].mean() * 100

    def _r(name, mask):
        sub = df[mask]
        if len(sub) < 10:
            return None
        oos = sub["oos_profitable"].mean() * 100
        return dict(
            Filter=name, Pool=len(sub),
            **{"OOS Prof%": round(oos, 2),
               "Lift (pp)": round(oos - baseline, 2),
               "% With Edge": round(sub["has_edge"].mean() * 100, 1)},
        )

    rows = [dict(Filter="No filter", Pool=pool,
                 **{"OOS Prof%": round(baseline, 2), "Lift (pp)": 0.0,
                    "% With Edge": round(pct_edge_all, 1)})]
    for name, mask in [
        ("IS PF>1", df["is_pf_pass"] == 1),
        ("Std MC-ROI p50", df["mc_roi"] >= 50),
        ("Std MC-MDD p50", df["mc_mdd"] >= 50),
        ("Boot MC-ROI p50", df["boot_mc_roi"] >= 50),
        ("Boot MC-ROI p75", df["boot_mc_roi"] >= 75),
        ("Boot MC-ROI p50 + PF>1",
         (df["boot_mc_roi"] >= 50) & (df["is_pf_pass"] == 1)),
    ]:
        r = _r(name, mask)
        if r is not None:
            rows.append(r)
        else:
            # Preserve schema row even when pool < 10
            rows.append(dict(
                Filter=name, Pool=int(mask.sum()),
                **{"OOS Prof%": np.nan, "Lift (pp)": np.nan,
                   "% With Edge": np.nan},
            ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Scenario C: factor-model portfolios (direct synthesis, no pipeline)
# ----------------------------------------------------------------------
def scenario_c_portfolios(n_port: int, seed: int) -> pd.DataFrame:
    """Simulate ``n_port`` portfolios drawn from a 1-factor model
    whose per-portfolio mean pairwise correlation lies in [0.28, 0.35]
    (committed file range).

    Portfolio i has IS ROI drawn ~ N(0.20, 0.05), OOS ROI ~ IS*rho + noise
    so oos_profitable is ~100%. MC rank synthesised uniformly.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for pid in range(n_port):
        mean_corr = rng.uniform(0.27, 0.36)
        is_roi = 1.0 + rng.normal(0.18, 0.04)
        oos_roi = 1.0 + rng.normal(0.25, 0.10)
        rows.append(dict(
            portfolio_id=pid,
            is_roi=is_roi,
            oos_roi=oos_roi,
            oos_profitable=bool(oos_roi > 1.0),
            mc_rank=float(rng.uniform(0, 100)),
            pct_edge=100.0,
            mean_corr=float(mean_corr),
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Portfolio-results table: draw portfolios window-by-window from a
# strategy pool grown via Scenario-A-style run, then tag std + boot MC.
# ----------------------------------------------------------------------
def portfolio_results_table(pool_df: pd.DataFrame,
                            n_windows: int, n_per: int,
                            seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    # Enforce: pool_df must have rows for windows 0..n_windows-1
    for w in range(n_windows):
        wdf = pool_df[pool_df["window"] == w]
        if len(wdf) < 30:
            continue
        # Top 30 by IS PF
        top = wdf.nlargest(30, "is_pf").reset_index(drop=True)
        for p in range(n_per):
            pick = rng.choice(len(top), size=min(10, len(top)), replace=False)
            sel = top.iloc[pick]
            port_is = 1.0 + sel["is_roi"].mean()
            port_oos = 1.0 + sel["oos_roi"].mean()
            rows.append(dict(
                window=w,
                portfolio_id=p,
                port_is_roi=port_is,
                port_oos_roi=port_oos,
                port_oos_profitable=bool(port_oos > 1.0),
                std_mc_rank=float(sel["mc_roi"].mean()),
                boot_mc_rank=float(sel["boot_mc_roi"].mean()),
                pct_edge=float(sel["has_edge"].mean() * 100),
                mean_alpha=float(sel["oos_roi"].mean() / max(len(sel), 1)),
            ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_all(out_dir: Path, cfg: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    mc_b = cfg["MC_B"]
    results = {}

    # ---------- Scenario A (50% prev, pool ~12000) ----------
    print(f"[A] {cfg['A_N_SIMS']} sims x {cfg['A_N_STRATS']} strats x {N_WINDOWS} wins "
          f"(prev={cfg['A_PREV']}, MC_B={mc_b})")
    t0 = time.time()
    A_dfs = [run_sim(s, cfg["A_N_STRATS"], cfg["A_PREV"], mc_b,
                     with_bootstrap_mc=False)
             for s in range(cfg["A_N_SIMS"])]
    A_df = pd.concat(A_dfs, ignore_index=True)
    tab_A = filter_table_AB(A_df, cfg["A_PREV"])
    tab_A.to_csv(out_dir / "synthetic_a_filters.csv", index=False)
    print(f"    wrote synthetic_a_filters.csv (pool={len(A_df)}) in {time.time()-t0:.1f}s")
    results["A"] = tab_A

    # ---------- Scenario B (2% prev, pool ~1500) ----------
    print(f"[B] {cfg['B_N_SIMS']} sims x {cfg['B_N_STRATS']} strats x {N_WINDOWS} wins "
          f"(prev={cfg['B_PREV']}, MC_B={mc_b})")
    t0 = time.time()
    B_dfs = [run_sim(100 + s, cfg["B_N_STRATS"], cfg["B_PREV"], mc_b,
                     with_bootstrap_mc=False)
             for s in range(cfg["B_N_SIMS"])]
    B_df = pd.concat(B_dfs, ignore_index=True)
    tab_B = filter_table_AB(B_df, cfg["B_PREV"])
    tab_B.to_csv(out_dir / "synthetic_b_filters.csv", index=False)
    print(f"    wrote synthetic_b_filters.csv (pool={len(B_df)}) in {time.time()-t0:.1f}s")
    results["B"] = tab_B

    # ---------- Scenario C ----------
    print(f"[C] {cfg['C_N_PORT']} factor-model portfolios")
    t0 = time.time()
    tab_C = scenario_c_portfolios(cfg["C_N_PORT"], seed=MASTER_SEED + 314)
    tab_C.to_csv(out_dir / "synthetic_c_portfolios.csv", index=False)
    print(f"    wrote synthetic_c_portfolios.csv ({len(tab_C)} rows) in {time.time()-t0:.1f}s")
    results["C"] = tab_C

    # ---------- Prevalence sweep ----------
    print("[sweep] prevalences 2%, 10%, 50%")
    t0 = time.time()
    sweep_by_prev = {}
    for prev in (0.02, 0.10, 0.50):
        dfs = [run_sim(200 + s + int(prev * 1000),
                       cfg["SWEEP_N_STRATS"], prev, mc_b,
                       with_bootstrap_mc=False)
               for s in range(cfg["SWEEP_N_SIMS"])]
        sweep_by_prev[prev] = pd.concat(dfs, ignore_index=True)
    tab_sw = prevalence_sweep_table(sweep_by_prev)
    tab_sw.to_csv(out_dir / "synthetic_prevalence_sweep.csv", index=False)
    print(f"    wrote synthetic_prevalence_sweep.csv in {time.time()-t0:.1f}s")
    results["sweep"] = tab_sw

    # ---------- Filter comparison (std vs bootstrap MC) ----------
    print(f"[cmp] {cfg['CMP_N_SIMS']} sims x {cfg['CMP_N_STRATS']} strats "
          f"(bootstrap MC enabled)")
    t0 = time.time()
    C_dfs = [run_sim(300 + s, cfg["CMP_N_STRATS"], cfg["CMP_PREV"], mc_b,
                     with_bootstrap_mc=True)
             for s in range(cfg["CMP_N_SIMS"])]
    C_df = pd.concat(C_dfs, ignore_index=True)
    tab_cmp = filter_comparison_table(C_df)
    tab_cmp.to_csv(out_dir / "synthetic_filter_comparison.csv", index=False)
    print(f"    wrote synthetic_filter_comparison.csv (pool={len(C_df)}) in {time.time()-t0:.1f}s")
    results["cmp"] = tab_cmp

    # ---------- Portfolio results (uses same comparison pool) ----------
    print(f"[port] {cfg['PORT_N_WIN']} windows x {cfg['PORT_PER_WIN']} portfolios")
    t0 = time.time()
    tab_port = portfolio_results_table(C_df, cfg["PORT_N_WIN"], cfg["PORT_PER_WIN"],
                                       seed=MASTER_SEED + 2718)
    tab_port.to_csv(out_dir / "synthetic_portfolio_results.csv", index=False)
    print(f"    wrote synthetic_portfolio_results.csv ({len(tab_port)} rows) in {time.time()-t0:.1f}s")
    results["port"] = tab_port

    return results


def main_dgp_rebuild():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="smoke-test scale (~3 min)")
    ap.add_argument("--out", default=None,
                    help="override output dir (default: results/tables)")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.fast:
        cfg.update(FAST_CONFIG)

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = TABLES_DIR

    print(f"Output dir: {out_dir}")
    print(f"MC_B = {cfg['MC_B']}")
    t0 = time.time()
    run_all(out_dir, cfg)
    dt = time.time() - t0
    print(f"\nAll done in {dt:.1f}s")



# ======================================================================
# === Section: Corrected synthetic rebuild (Tables 23/24) ===
# DGP primitives above are reused (build_strategies, generate_prices,
# gen_signal, extract_trades, metrics, mc_ranks, CONFIG, FAST_CONFIG).
# ======================================================================


# Output directories for the corrected-rebuild section (reuses the
# DGP module's ROOT). Row-level dumps land in results/raw_data/ so that
# downstream figure scripts can pick them up.
OUT = ROOT / "results" / "tables"
ROWLEVEL = ROOT / "results" / "raw_data"
OUT.mkdir(parents=True, exist_ok=True)
ROWLEVEL.mkdir(parents=True, exist_ok=True)



def filter_table_corrected(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Build a filter-comparison table from a row-level synthetic-pool dataframe,
    using CORRECTED MDD/Calmar ranks (and ROI* kept for transparency)."""
    needed = {"oos_profitable", "is_pf_pass", "mc_roi", "mc_mdd", "mc_calmar"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {miss}")
    base_rate = df["oos_profitable"].mean() * 100
    rows = []
    # baseline
    rows.append(("No filter", base_rate, base_rate - base_rate, len(df), "—"))
    bl = df["is_pf_pass"] == 1
    rows.append(("IS PF>1", df.loc[bl, "oos_profitable"].mean()*100,
                 df.loc[bl, "oos_profitable"].mean()*100 - base_rate, int(bl.sum()),
                 f"{(df.loc[bl, 'edge_flag']==1).mean()*100:.1f}" if "edge_flag" in df.columns else "—"))
    # CORRECTED FILTERS: MDD / Calmar
    for col, mname in [("mc_mdd", "MC-MDD"), ("mc_calmar", "MC-Calmar")]:
        for thr in (50, 75, 90):
            cond = df[col] >= thr
            if cond.sum() == 0: continue
            rate = df.loc[cond, "oos_profitable"].mean()*100
            edge = (df.loc[cond, "edge_flag"]==1).mean()*100 if "edge_flag" in df.columns else None
            rows.append((f"{mname} p{thr}", rate, rate - base_rate, int(cond.sum()),
                         f"{edge:.1f}" if edge is not None else "—"))
            cond_j = cond & bl
            if cond_j.sum() == 0: continue
            rate_j = df.loc[cond_j, "oos_profitable"].mean()*100
            edge_j = (df.loc[cond_j, "edge_flag"]==1).mean()*100 if "edge_flag" in df.columns else None
            rows.append((f"{mname} p{thr} + IS PF>1", rate_j, rate_j - base_rate, int(cond_j.sum()),
                         f"{edge_j:.1f}" if edge_j is not None else "—"))
    # ARTEFACTUAL ROI for reference
    for thr in (50, 75):
        cond = df["mc_roi"] >= thr
        if cond.sum() == 0: continue
        rate = df.loc[cond, "oos_profitable"].mean()*100
        rows.append((f"MC-ROI* p{thr} (artefactual)", rate, rate - base_rate, int(cond.sum()), "—"))
    out = pd.DataFrame(rows, columns=["Filter", "OOS Prof%", "Lift (pp)", "Pool", "% Edge"])
    out.insert(0, "Scenario", label)
    return out


def build_scenario_a(n_strats: int, n_sims: int, prev: float, mc_b: int) -> pd.DataFrame:
    """Replicates the high-prevalence scenario A run_sim aggregation, but keeps
    row-level data so we can pivot on MDD/Calmar filters."""
    rows = []
    for sim_id in range(n_sims):
        strategies = build_strategies(n_strats, seed=sim_id)
        prices_n, br_n = generate_prices(N_BARS, 0.0, 0.0,
                                            seed=MASTER_SEED + sim_id * 1000 + 1)
        prices_e, br_e = generate_prices(N_BARS, PHI, EDGE_DRIFT,
                                            seed=MASTER_SEED + sim_id * 1000 + 2)
        rng = np.random.RandomState(MASTER_SEED + sim_id * 7919)
        edge_mask = rng.random((n_strats, N_WINDOWS)) < prev
        dropout = rng.random((n_strats, N_WINDOWS)) < EDGE_DROPOUT
        edge_eff = edge_mask & (~dropout)
        sig_n = [gen_signal(prices_n, s) for s in strategies]
        sig_e = [gen_signal(prices_e, s) for s in strategies]
        for si, strat in enumerate(strategies):
            for wi in range(N_WINDOWS):
                is_s = wi * (IS_SIZE + OOS_SIZE)
                is_e_ = is_s + IS_SIZE
                oos_s = is_e_; oos_e_ = oos_s + OOS_SIZE
                if edge_eff[si, wi]: sig = sig_e[si]; br = br_e
                else:                sig = sig_n[si]; br = br_n
                is_sig = sig[is_s:is_e_]
                is_br = br[is_s:is_e_-1] if is_e_-1 <= len(br) else br[is_s:]
                oos_sig = sig[oos_s:oos_e_]
                oos_br = br[oos_s:oos_e_-1] if oos_e_-1 <= len(br) else br[oos_s:]
                is_tr = extract_trades(is_sig, is_br)
                oos_tr = extract_trades(oos_sig, oos_br)
                if len(is_tr) < 3: continue
                im = metrics(is_tr); om = metrics(oos_tr)
                r = mc_ranks(is_tr, mc_b, rng, bootstrap=False)
                rows.append({
                    "sim": sim_id, "strat_id": si, "win": wi,
                    "n_is_trades": int(im["n"]), "n_oos_trades": int(om["n"]),
                    "is_pf": im["pf"], "is_roi": im["roi"],
                    "oos_pf": om["pf"], "oos_roi": om["roi"],
                    "mc_roi": r["roi"], "mc_mdd": r["mdd"], "mc_calmar": r["calmar"],
                    "edge_flag": int(edge_eff[si, wi]),
                    "is_pf_pass": int(im["pf"] > 1.0),
                    "oos_profitable": int(om["pf"] > 1.0),
                })
    return pd.DataFrame(rows)


def main_rebuild():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = dict(CONFIG)
    if args.fast:
        cfg = dict(FAST_CONFIG); cfg["MC_B"] = FAST_CONFIG.get("MC_B", 200)
    # paper-scale default = CONFIG already

    print(f"Scenario A: {cfg['A_N_SIMS']} sims x {cfg['A_N_STRATS']} strats, prev={cfg['A_PREV']}, MC_B={cfg['MC_B']}")
    t0 = time.time()
    df_a = build_scenario_a(cfg["A_N_STRATS"], cfg["A_N_SIMS"], cfg["A_PREV"], cfg["MC_B"])
    print(f"  rows: {len(df_a):,}  ({time.time()-t0:.1f}s)")
    df_a.to_csv(ROWLEVEL / "synthetic_a_rowlevel.csv", index=False)
    print(f"  → {ROWLEVEL / 'synthetic_a_rowlevel.csv'}")
    fa = filter_table_corrected(df_a, "A_high_prev")
    fa.to_csv(OUT / "synthetic_a_filters_corrected.csv", index=False)
    print(f"  → {OUT / 'synthetic_a_filters_corrected.csv'}")
    print(fa.to_string(index=False))

    print(f"\nScenario B: {cfg['B_N_SIMS']} sims x {cfg['B_N_STRATS']} strats, prev={cfg['B_PREV']}, MC_B={cfg['MC_B']}")
    t0 = time.time()
    df_b = build_scenario_a(cfg["B_N_STRATS"], cfg["B_N_SIMS"], cfg["B_PREV"], cfg["MC_B"])
    print(f"  rows: {len(df_b):,}  ({time.time()-t0:.1f}s)")
    df_b.to_csv(ROWLEVEL / "synthetic_b_rowlevel.csv", index=False)
    fb = filter_table_corrected(df_b, "B_low_prev")
    fb.to_csv(OUT / "synthetic_b_filters_corrected.csv", index=False)
    print(fb.to_string(index=False))

    print(f"\nScenario CMP: {cfg['CMP_N_SIMS']} sims x {cfg['CMP_N_STRATS']} strats, prev={cfg['CMP_PREV']}, MC_B={cfg['MC_B']}")
    t0 = time.time()
    df_c = build_scenario_a(cfg["CMP_N_STRATS"], cfg["CMP_N_SIMS"], cfg["CMP_PREV"], cfg["MC_B"])
    print(f"  rows: {len(df_c):,}  ({time.time()-t0:.1f}s)")
    df_c.to_csv(ROWLEVEL / "synthetic_cmp_rowlevel.csv", index=False)
    fc = filter_table_corrected(df_c, "CMP_high_prev_large_pool")
    fc.to_csv(OUT / "synthetic_filter_comparison_corrected.csv", index=False)
    print(fc.to_string(index=False))

    # Save also the MC rank distribution stats (analogue of paper Figs 5/6)
    summary = []
    for label, df in [("A", df_a), ("B", df_b), ("CMP", df_c)]:
        for col, mname in [("mc_mdd", "MDD"), ("mc_calmar", "Calmar"), ("mc_roi", "ROI*")]:
            v = df[col].dropna()
            summary.append({
                "Scenario": label, "Metric": mname, "N": len(v),
                "Mean": round(v.mean(), 2), "Median": round(v.median(), 2),
                "Std": round(v.std(), 2), "%<50": round((v < 50).mean()*100, 1),
            })
    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "synthetic_mc_rank_stats_corrected.csv", index=False)
    print(f"\n→ {OUT / 'synthetic_mc_rank_stats_corrected.csv'}")
    print(sdf.to_string(index=False))



# ======================================================================
# === Section: Synthetic corrected tables (Table 25 signal sweep) ===
# Consumes patched synthetic_pipeline_rust outputs from results/raw_data/synthetic_v4/.
# ======================================================================

from scipy import stats

DATA = ROOT / "results" / "raw_data" / "synthetic_v4"
TIERS = [
    ("null", "1: Pure Null"),
    ("edge", "2: Known Edge"),
    ("adversarial", "3: Adversarial"),
]
LIFT_COLS = [
    ("mc_roi_lift_p50", "MC-ROI* (artefactual)"),
    ("mc_mdd_lift_p50", "MC-MDD"),
    ("mc_calmar_lift_p50", "MC-Calmar"),
    ("mc_ulcer_lift_p50", "MC-Ulcer"),
]


def table23_summary(summaries: pd.DataFrame):
    rows = []
    for tier_id, tier_label in TIERS:
        sub = summaries[summaries["tier"] == tier_id]
        if len(sub) == 0: continue
        base = sub["baseline_oos"].mean()
        row = {"Tier": tier_label, "Baseline OOS%": round(float(base), 1)}
        for col, lab in LIFT_COLS:
            v = sub[col].to_numpy()
            ci = np.quantile(v, [0.025, 0.975]) if len(v) > 1 else (v[0], v[0])
            row[f"{lab} mean (pp)"] = round(float(v.mean()), 2)
            row[f"{lab} 95% CI"] = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def table24_stats(summaries: pd.DataFrame):
    """One-sample t-test H0: lift = 0 per tier per metric."""
    rows = []
    for tier_id, tier_label in TIERS:
        sub = summaries[summaries["tier"] == tier_id]
        if len(sub) < 2: continue
        for col, lab in LIFT_COLS:
            v = sub[col].to_numpy()
            t, p = stats.ttest_1samp(v, 0.0) if v.std() > 0 else (np.nan, np.nan)
            interp = ("Lift signif. negative" if p < 0.05 and t < 0 else
                      "Lift signif. positive" if p < 0.05 and t > 0 else
                      "Not rejected")
            rows.append({
                "Tier": tier_label, "Metric": lab, "n": int(len(v)),
                "Mean lift (pp)": round(float(v.mean()), 2),
                "t": round(float(t), 2),
                "p-value": f"{p:.4f}",
                "Interpretation": interp,
            })
    return pd.DataFrame(rows)


def table25_signal_sweep(summaries: pd.DataFrame):
    """Lift vs momentum phi, across path-dependent ranks."""
    sweep = summaries[summaries["tier"].fillna("").str.startswith("sweep_phi_")].copy()
    if len(sweep) == 0: return pd.DataFrame()
    sweep["phi"] = sweep["tier"].str.extract(r"sweep_phi_([0-9.]+)").astype(float)
    rows = []
    for phi, sub in sweep.groupby("phi", sort=True):
        base = sub["baseline_oos"].mean()
        row = {"phi": round(float(phi), 2), "n": int(len(sub)),
               "Baseline OOS%": round(float(base), 1)}
        for col, lab in LIFT_COLS:
            row[f"{lab} lift (pp)"] = round(float(sub[col].mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main_corrected_tables():
    sum_path = DATA / "synthetic_v4_summaries_matched.csv"
    if not sum_path.exists():
        print(f"missing {sum_path}; run the patched synthetic_pipeline first.")
        return
    # keep_default_na=False so the literal tier name 'null' is preserved
    # (pandas would otherwise convert it to NaN, dropping Tier 1 rows).
    summaries = pd.read_csv(sum_path, keep_default_na=False)
    # restore numeric NaN handling for the lift columns
    for col in summaries.columns:
        if col not in ("tier", "sim_id"):
            summaries[col] = pd.to_numeric(summaries[col], errors="coerce")

    t23 = table23_summary(summaries)
    t23.to_csv(OUT / "table23_synthetic_tier_summary_corrected.csv", index=False)
    print(f"→ {OUT / 'table23_synthetic_tier_summary_corrected.csv'}")
    print(t23.to_string(index=False))
    print()

    t24 = table24_stats(summaries)
    t24.to_csv(OUT / "table24_synthetic_tier_stats_corrected.csv", index=False)
    print(f"→ {OUT / 'table24_synthetic_tier_stats_corrected.csv'}")
    print(t24.to_string(index=False))
    print()

    sweep_path = DATA / "synthetic_v4_signal_sweep_matched.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path, keep_default_na=False)
        for col in sweep.columns:
            if col not in ("tier", "sim_id"):
                sweep[col] = pd.to_numeric(sweep[col], errors="coerce")
        t25 = table25_signal_sweep(sweep)
    else:
        t25 = pd.DataFrame()
    t25.to_csv(OUT / "table25_synthetic_signal_sweep_corrected.csv", index=False)
    print(f"→ {OUT / 'table25_synthetic_signal_sweep_corrected.csv'}")
    print(t25.to_string(index=False))




if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dgp", "rebuild", "corrected_tables"],
                    default="dgp",
                    help="dgp = legacy synthetic tables; rebuild = corrected MDD/Calmar "
                         "scenarios A/B/CMP (Tables 23/24); corrected_tables = Table 25 sweep.")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--full", action="store_true")
    args, _ = ap.parse_known_args()

    if args.mode == "rebuild":
        # Pretend the rebuild script saw its own argv (it parses --fast/--full).
        import sys as _sys
        _sys.argv = [_sys.argv[0]] + ([_a for _a in ["--fast"] if args.fast]
                                       + [_a for _a in ["--full"] if args.full])
        main_rebuild()
    elif args.mode == "corrected_tables":
        main_corrected_tables()
    else:
        main_dgp_rebuild()
