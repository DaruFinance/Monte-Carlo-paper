"""
Master figure-generation driver for the MC paper revision.

Produces every figure cited by the paper from the corrected Rust outputs and
the upstream window_pairs CSVs. Each banner below corresponds to one figure
or one figure family (crypto + forex panels).

Inputs (relative to project root, override with MC_PAPER_DATA env var):
  - results/raw_data/<asset>_corrected_ranks.csv
  - results/raw_data/<asset>_window_pairs.csv
  - results/raw_data/<asset>_block_perm_path.csv     (block-perm figures)
  - results/raw_data/<asset>_portfolio_mc_path.csv   (portfolio figures)

Outputs:
  - results/figures/fig3_bootstrap_lift_corrected{,_forex}.pdf
  - results/figures/fig4_regime_robustness_corrected{,_forex}.pdf
  - results/figures/fig5_synthetic_mc_ranks_corrected.pdf
  - results/figures/fig6_synthetic_edge_strat_corrected.pdf
  - results/figures/fig7_synthetic_tier_lift_corrected.pdf
  - results/figures/fig8_synthetic_signal_sweep_corrected.pdf
  - results/figures/fig_mc_rank_distributions_corrected{,_forex}.pdf
  - results/figures/fig_portfolio_mc_rightshift.pdf
  - results/figures/fig_gold_mc.pdf
  - results/figures/fig_portfolio_oos_decile.pdf
  - results/figures/fig_mc_by_family_heatmap.pdf
  - results/figures/fig_crossasset_forest.pdf
  - results/figures/fig_cost_sensitivity.pdf
  - results/figures/fig_synthetic_groundtruth_ranks.pdf
"""
import os  # noqa: F401

# ======================================================================
# === Section: Fig 3/4/9/10 + corrected rank distributions (crypto+forex) ===
# ======================================================================

"""
Rebuild paper figures whose construction depends on the FP-buggy MC rank.

Affected figures (crypto only — forex/commodity excluded per scope):
  - Fig 3: fig_bootstrap_lift_distributions.pdf  (panels A,B: window-level + bootstrap lift)
  - Fig 4: fig_regime_robustness.pdf             (per-window MC ROI rank vs OOS — crypto)
  - Fig 9: mc_pct_rank_distributions.pdf         (MC rank distributions — was forex/commodity)
  - Fig 10: mc_roi_vs_next_oos_binned.pdf        (was forex/commodity)
  - Plus a NEW figure: fig_mc_rank_distributions_corrected.pdf
    showing MDD/Calmar/Ulcer + ROI* (artefactual) distributions side-by-side.

Outputs land in results/figures/. Caption text suggested in the rewrite
guide.

Requires: results/raw_data/<asset>_corrected_ranks.csv  (from rust/mc_path_ranks)
          results/raw_data/<asset>_window_pairs.csv     (from upstream backtester)
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

ASSETS = [("BTC", "btc"), ("DOGE", "doge"), ("BNB", "bnb"), ("SOL", "sol")]
FOREX_ASSETS = [("EUR/USD", "eurusd"), ("USD/JPY", "usdjpy"), ("EUR/GBP", "eurgbp"),
                ("XAU/USD", "xauusd"), ("WTI", "wti")]

C_RED = "#C0392B"; C_NAVY = "#1F3864"; C_TEAL = "#2E8B57"
C_PURPLE = "#7E57C2"; C_GRAY = "#777"

# Filters defined on CORRECTED metrics
FILTERS = {
    'MC-MDD p50':    ('mdd_rank', 50),
    'MC-MDD p75':    ('mdd_rank', 75),
    'MC-Calmar p50': ('calmar_rank', 50),
    'MC-Calmar p75': ('calmar_rank', 75),
    'MC-Ulcer p50':  ('ulcer_rank', 50),
}


def load_merged(short: str) -> pd.DataFrame:
    rp = DATA / f"{short}_corrected_ranks.csv"
    wp = DATA / f"{short}_window_pairs.csv"
    if not rp.exists() or not wp.exists():
        return pd.DataFrame()
    r = pd.read_csv(rp); r['window_i'] = r['window'].str.replace('W', '').astype(int)
    w = pd.read_csv(wp)
    return pd.merge(w, r[['strategy','window_i','n_trades','actual_roi','actual_mdd',
                          'actual_calmar','actual_ulcer','roi_rank_broken','mdd_rank',
                          'calmar_rank','ulcer_rank']], on=['strategy','window_i'])


# ----------------------------------------------------------------------
# NEW: corrected MC rank distributions across crypto assets
# ----------------------------------------------------------------------
def fig_corrected_rank_distributions():
    fig, axes = plt.subplots(4, 4, figsize=(15, 13), sharex=True, sharey='row')
    bins = np.linspace(0, 100, 41)
    cols = [
        ("mdd_rank", "MDD (path-dependent)"),
        ("calmar_rank", "Calmar (path-dependent)"),
        ("ulcer_rank", "Ulcer (path-dependent)"),
        ("roi_rank_broken", "ROI* (FP artefact)"),
    ]
    asset_data = {}
    for asset, short in ASSETS:
        p = DATA / f"{short}_corrected_ranks.csv"
        if p.exists(): asset_data[asset] = pd.read_csv(p)

    for j, (col, label) in enumerate(cols):
        for i, (asset, _) in enumerate(ASSETS):
            ax = axes[i, j]
            if asset not in asset_data:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            v = asset_data[asset][col].dropna()
            color = C_RED if "broken" in col else (C_NAVY if "mdd" in col else C_TEAL if "calmar" in col else C_PURPLE)
            ax.hist(v, bins=bins, density=True, color=color, alpha=0.7)
            ax.axvline(50, color="gray", ls="--", lw=1)
            ax.set_xlim(0, 100)
            ax.set_title(
                f"{asset} — {label}\nmean={v.mean():.1f}  std={v.std():.1f}  %<50={(v<50).mean()*100:.1f}",
                fontsize=9,
            )
            if i == 3: ax.set_xlabel("MC percentile rank")
            if j == 0: ax.set_ylabel("Density")
    fig.suptitle("MC rank distributions across crypto assets — "
                 "MDD/Calmar/Ulcer center at 50 (no leftshift); "
                 "ROI* (FP artefact)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    out = FIGS / "fig_mc_rank_distributions_corrected.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 3: bootstrap lift distribution (MC-MDD p50, crypto pooled)
# ----------------------------------------------------------------------
def fig3_bootstrap_lift_corrected(n_boot=10000):
    rng = np.random.default_rng(42)
    # collect per-window lift estimates pooled across assets, with cluster=window
    all_lift = []
    cluster_ids = []
    for asset, short in ASSETS:
        m = load_merged(short)
        if len(m) == 0: continue
        m = m[m['baseline_oos_pf'].notna()].copy()
        m['oos_prof'] = (m['baseline_oos_pf'] > 1.0).astype(float)
        m['passing'] = (m['mdd_rank'] >= 50).astype(int)
        for w, g in m.groupby('window_i'):
            base = g['oos_prof'].mean()
            passing = g[g['passing'] == 1]
            if len(passing) == 0: continue
            lift = (passing['oos_prof'].mean() - base) * 100
            all_lift.append(lift)
            cluster_ids.append(f"{asset}_W{w:02d}")
    all_lift = np.array(all_lift)
    if len(all_lift) == 0:
        print("  [skip] fig3 — no merged data")
        return
    print(f"  Fig3: {len(all_lift)} per-window MC-MDD-p50 lift estimates pooled across crypto")

    # cluster bootstrap
    boots = np.empty(n_boot)
    idx_n = len(all_lift)
    for b in range(n_boot):
        idx = rng.integers(0, idx_n, size=idx_n)
        boots[b] = all_lift[idx].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.hist(all_lift, bins=30, density=True, alpha=0.5, color=C_RED, edgecolor="white")
    try:
        kde = gaussian_kde(all_lift, bw_method=0.3)
        x = np.linspace(all_lift.min()-2, all_lift.max()+2, 300)
        ax.plot(x, kde(x), color=C_RED, lw=2)
    except Exception: pass
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.axvline(all_lift.mean(), color=C_NAVY, lw=2, label=f"mean = {all_lift.mean():.2f} pp")
    ax.set_xlabel("Per-window MC-MDD p50 filter lift (pp)")
    ax.set_ylabel("Density")
    ax.set_title("(A) Window-level MC-MDD filter lift distribution\n(crypto pooled, cluster = window)")
    ax.legend()

    ax = axes[1]
    ax.hist(boots, bins=50, density=True, alpha=0.5, color=C_RED, edgecolor="white")
    try:
        kde = gaussian_kde(boots, bw_method=0.3)
        x = np.linspace(boots.min(), boots.max(), 300)
        ax.plot(x, kde(x), color=C_RED, lw=2)
    except Exception: pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.axvline(boots.mean(), color=C_NAVY, lw=2, label=f"boot mean = {boots.mean():.2f} pp")
    ax.axvspan(lo, hi, alpha=0.15, color=C_NAVY, label=f"95% CI [{lo:.2f}, {hi:.2f}]")
    ax.set_xlabel("Bootstrapped mean MC-MDD lift (pp)")
    ax.set_title(f"(B) Cluster-bootstrap distribution ({n_boot:,} resamples)")
    ax.legend()

    fig.suptitle("Fig 3: MC-MDD p50 filter lift on same-window OOS profitability — "
                 "informative MC rank, crypto pooled", fontsize=11, y=1.02)
    plt.tight_layout()
    out = FIGS / "fig3_bootstrap_lift_corrected.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Fig 4 (analogue): per-window MC-MDD rank vs OOS profitability — crypto
# ----------------------------------------------------------------------
def fig4_window_mc_vs_oos_corrected():
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for i, (asset, short) in enumerate(ASSETS):
        ax = axes[i]
        m = load_merged(short)
        if len(m) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        m = m[m['baseline_oos_pf'].notna()].copy()
        m['oos_prof'] = (m['baseline_oos_pf'] > 1.0).astype(float)
        per_w = m.groupby('window_i').agg(
            mc=('mdd_rank','mean'),
            oos=('oos_prof','mean'),
            n=('strategy','count'),
        ).reset_index()
        ax.scatter(per_w['mc'], per_w['oos']*100, s=30, color=C_NAVY, alpha=0.7)
        ax.axvline(50, color="gray", ls="--", lw=1)
        ax.axhline(per_w['oos'].mean()*100, color="gray", ls=":", lw=1,
                   label=f"baseline = {per_w['oos'].mean()*100:.1f}%")
        ax.set_xlabel("Mean MC-MDD rank in window")
        ax.set_ylabel("OOS profitable rate (%)")
        ax.set_title(f"{asset}: per-window MC-MDD rank vs OOS profitability\n({len(per_w)} windows)")
        ax.legend(fontsize=8)
    fig.suptitle("Fig 4: per-window MC-MDD rank vs OOS profitability — crypto",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = FIGS / "fig4_regime_robustness_corrected.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Forex versions of the rank-distribution and per-window MC-vs-OOS figures.
# Mirror the crypto layout but with 3 forex pairs in place of 4 crypto.
# These fill the placeholders flagged in sections/A5_appendix_forex.tex.
# ----------------------------------------------------------------------
def fig_corrected_rank_distributions_forex():
    n_assets = len(FOREX_ASSETS)
    fig, axes = plt.subplots(n_assets, 4, figsize=(15, 3.3 * n_assets), sharex=True, sharey='row')
    bins = np.linspace(0, 100, 41)
    cols = [
        ("mdd_rank", "MDD (path-dependent)"),
        ("calmar_rank", "Calmar (path-dependent)"),
        ("ulcer_rank", "Ulcer (path-dependent)"),
        ("roi_rank_broken", "ROI* (FP artefact)"),
    ]
    asset_data = {}
    for asset, short in FOREX_ASSETS:
        p = DATA / f"{short}_corrected_ranks.csv"
        if p.exists(): asset_data[asset] = pd.read_csv(p)

    for j, (col, label) in enumerate(cols):
        for i, (asset, _) in enumerate(FOREX_ASSETS):
            ax = axes[i, j]
            if asset not in asset_data:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            v = asset_data[asset][col].dropna()
            color = C_RED if "broken" in col else (C_NAVY if "mdd" in col else C_TEAL if "calmar" in col else C_PURPLE)
            ax.hist(v, bins=bins, density=True, color=color, alpha=0.7)
            ax.axvline(50, color="gray", ls="--", lw=1)
            ax.set_xlim(0, 100)
            ax.set_title(
                f"{asset} — {label}\nmean={v.mean():.1f}  std={v.std():.1f}  %<50={(v<50).mean()*100:.1f}",
                fontsize=9,
            )
            if i == n_assets - 1: ax.set_xlabel("MC percentile rank")
            if j == 0: ax.set_ylabel("Density")
    fig.suptitle("MC rank distributions across forex pairs — "
                 "MDD/Calmar/Ulcer center near 50 (no leftshift); "
                 "ROI* artefact more extreme than crypto (smaller R-unit pnls amplify FP noise)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    out = FIGS / "fig_mc_rank_distributions_corrected_forex.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


def fig4_window_mc_vs_oos_corrected_forex():
    n = len(FOREX_ASSETS)
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 4.5))
    if n == 1:
        axes = [axes]
    for i, (asset, short) in enumerate(FOREX_ASSETS):
        ax = axes[i]
        m = load_merged(short)
        if len(m) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        m = m[m['baseline_oos_pf'].notna()].copy()
        m['oos_prof'] = (m['baseline_oos_pf'] > 1.0).astype(float)
        per_w = m.groupby('window_i').agg(
            mc=('mdd_rank','mean'),
            oos=('oos_prof','mean'),
            n=('strategy','count'),
        ).reset_index()
        ax.scatter(per_w['mc'], per_w['oos']*100, s=30, color=C_NAVY, alpha=0.7)
        ax.axvline(50, color="gray", ls="--", lw=1)
        ax.axhline(per_w['oos'].mean()*100, color="gray", ls=":", lw=1,
                   label=f"baseline = {per_w['oos'].mean()*100:.1f}%")
        ax.set_xlabel("Mean MC-MDD rank in window")
        ax.set_ylabel("OOS profitable rate (%)")
        ax.set_title(f"{asset}: per-window MC-MDD rank vs OOS profitability\n({len(per_w)} windows)")
        ax.legend(fontsize=8)
    fig.suptitle("Fig 4 (forex): per-window MC-MDD rank vs OOS profitability",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = FIGS / "fig4_regime_robustness_corrected_forex.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


def fig3_bootstrap_lift_corrected_forex(n_boot=10000):
    """Forex pooled equivalent of fig3 — same cluster-bootstrap design."""
    rng = np.random.default_rng(43)
    all_lift = []
    for asset, short in FOREX_ASSETS:
        m = load_merged(short)
        if len(m) == 0: continue
        m = m[m['baseline_oos_pf'].notna()].copy()
        m['oos_prof'] = (m['baseline_oos_pf'] > 1.0).astype(float)
        m['passing'] = (m['mdd_rank'] >= 50).astype(int)
        for w, g in m.groupby('window_i'):
            base = g['oos_prof'].mean()
            passing = g[g['passing'] == 1]
            if len(passing) == 0: continue
            lift = (passing['oos_prof'].mean() - base) * 100
            all_lift.append(lift)
    all_lift = np.array(all_lift)
    if len(all_lift) == 0:
        print("  [skip] fig3_forex — no merged data")
        return
    print(f"  Fig3-forex: {len(all_lift)} per-window MC-MDD-p50 lift estimates pooled across forex")

    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(all_lift), size=len(all_lift))
        boots[b] = all_lift[idx].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.hist(all_lift, bins=30, density=True, alpha=0.5, color=C_RED, edgecolor="white")
    try:
        kde = gaussian_kde(all_lift, bw_method=0.3)
        x = np.linspace(all_lift.min()-2, all_lift.max()+2, 300)
        ax.plot(x, kde(x), color=C_RED, lw=2)
    except Exception: pass
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.axvline(all_lift.mean(), color=C_NAVY, lw=2, label=f"mean = {all_lift.mean():.2f} pp")
    ax.set_xlabel("Per-window MC-MDD p50 filter lift (pp)")
    ax.set_ylabel("Density")
    ax.set_title("(A) Window-level MC-MDD filter lift distribution\n(forex pooled, cluster = window)")
    ax.legend()

    ax = axes[1]
    ax.hist(boots, bins=50, density=True, alpha=0.5, color=C_RED, edgecolor="white")
    try:
        kde = gaussian_kde(boots, bw_method=0.3)
        x = np.linspace(boots.min(), boots.max(), 300)
        ax.plot(x, kde(x), color=C_RED, lw=2)
    except Exception: pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.axvline(boots.mean(), color=C_NAVY, lw=2, label=f"boot mean = {boots.mean():.2f} pp")
    ax.axvspan(lo, hi, alpha=0.15, color=C_NAVY, label=f"95% CI [{lo:.2f}, {hi:.2f}]")
    ax.set_xlabel("Bootstrapped mean MC-MDD lift (pp)")
    ax.set_title(f"(B) Cluster-bootstrap distribution ({n_boot:,} resamples)")
    ax.legend()

    fig.suptitle("Fig 3 (forex): MC-MDD p50 filter lift on same-window OOS profitability — "
                 "forex pooled", fontsize=11, y=1.02)
    plt.tight_layout()
    out = FIGS / "fig3_bootstrap_lift_corrected_forex.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig_corrected_rank_distributions()
    fig3_bootstrap_lift_corrected()
    fig4_window_mc_vs_oos_corrected()
    fig_corrected_rank_distributions_forex()
    fig3_bootstrap_lift_corrected_forex()
    fig4_window_mc_vs_oos_corrected_forex()

# ======================================================================
# === Section: Fig 5/6 synthetic MC ranks + edge-strat panels ===
# ======================================================================

"""
Rebuild paper Figures 5 and 6 (synthetic MC rank distributions) using corrected
MDD/Calmar ranks. Uses synthetic_*_rowlevel.csv from synthetic_scenarios.py.

Fig 5 (corrected): synthetic A/B MC rank distributions, MDD + Calmar panels
Fig 6 (corrected): edge-vs-null stratified MC rank distributions
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
DATA = ROOT / "results" / "raw_data"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

C_NAVY = "#1F3864"; C_TEAL = "#2E8B57"; C_RED = "#C0392B"


def fig5_corrected():
    df_a = pd.read_csv(DATA / "synthetic_a_rowlevel.csv")
    df_b = pd.read_csv(DATA / "synthetic_b_rowlevel.csv")
    df_c = pd.read_csv(DATA / "synthetic_cmp_rowlevel.csv")

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    bins = np.linspace(0, 100, 41)
    rowdata = [("A (high prev, n=300)", df_a), ("B (low prev 2%, n=150)", df_b),
               ("CMP (8 sims, n=312)", df_c)]
    cols = [("mc_mdd","MDD (path-dep)",C_NAVY), ("mc_calmar","Calmar (path-dep)",C_TEAL),
            ("mc_roi","ROI* (FP artefact)",C_RED)]
    for r, (label, df) in enumerate(rowdata):
        for c, (col, mname, color) in enumerate(cols):
            ax = axes[r, c]
            v = df[col].dropna()
            ax.hist(v, bins=bins, density=True, color=color, alpha=0.7)
            ax.axvline(50, color="gray", ls="--", lw=1)
            ax.set_title(f"{label} — {mname}\nmean={v.mean():.1f} std={v.std():.1f} %<50={(v<50).mean()*100:.1f}",
                         fontsize=9)
            ax.set_xlim(0, 100)
            if r == 2: ax.set_xlabel("MC percentile rank")
            if c == 0: ax.set_ylabel("Density")
    fig.suptitle("Fig 5: synthetic-pipeline MC rank distributions — "
                 "MDD/Calmar (informative) vs ROI* (FP artefact)",
                 fontsize=11, y=0.995)
    plt.tight_layout()
    plt.savefig(FIGS / "fig5_synthetic_mc_ranks_corrected.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "fig5_synthetic_mc_ranks_corrected.png", dpi=130, bbox_inches="tight")
    print(f"Wrote {FIGS / 'fig5_synthetic_mc_ranks_corrected.pdf'}")
    plt.close(fig)


def fig6_corrected():
    # Edge-vs-null stratified distributions on the largest pool (CMP)
    df = pd.read_csv(DATA / "synthetic_cmp_rowlevel.csv")
    edge = df[df["edge_flag"] == 1]
    null = df[df["edge_flag"] == 0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    bins = np.linspace(0, 100, 41)
    cols = [("mc_mdd","MDD (path-dep)",C_NAVY), ("mc_calmar","Calmar (path-dep)",C_TEAL),
            ("mc_roi","ROI* (FP artefact)",C_RED)]
    for r, (sub, label) in enumerate([(null, "Null windows (no edge)"),
                                       (edge, "Edge windows (alpha>0)")]):
        for c, (col, mname, color) in enumerate(cols):
            ax = axes[r, c]
            v = sub[col].dropna()
            ax.hist(v, bins=bins, density=True, color=color, alpha=0.7)
            ax.axvline(50, color="gray", ls="--", lw=1)
            ax.set_title(f"{label} — {mname}\n"
                         f"N={len(v)}  mean={v.mean():.1f} std={v.std():.1f}",
                         fontsize=9)
            ax.set_xlim(0, 100)
            if r == 1: ax.set_xlabel("MC percentile rank")
            if c == 0: ax.set_ylabel("Density")
    fig.suptitle("Fig 6: MC rank by edge condition — "
                 "edge windows shift MDD/Calmar ranks consistent with stronger trading patterns",
                 fontsize=11, y=1.00)
    plt.tight_layout()
    plt.savefig(FIGS / "fig6_synthetic_edge_strat_corrected.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "fig6_synthetic_edge_strat_corrected.png", dpi=130, bbox_inches="tight")
    print(f"Wrote {FIGS / 'fig6_synthetic_edge_strat_corrected.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    fig5_corrected()
    fig6_corrected()

# ======================================================================
# === Section: Fig 7/8 synthetic tier-lift + signal-sweep panels ===
# ======================================================================

"""
Render the two figures placeholdered as Fig 7 / Fig 8 in §A3:
  Fig 7: per-tier MC-filter-lift bar chart (4 filters x 3 tiers).
         Path-dependent MC filters (MDD/Calmar/Ulcer) and the artefactual
         MC-ROI* shown side-by-side; error bars from the 95% CI columns.
  Fig 8: signal-strength sweep — lift vs phi for the 4 filter columns,
         showing all path-dependent filters stay negative (or near zero)
         across signal strengths.

Outputs:
  results/figures/fig7_synthetic_tier_lift_corrected.pdf (+ .png)
  results/figures/fig8_synthetic_signal_sweep_corrected.pdf (+ .png)
"""
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
TBL = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def parse_ci(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    m = re.match(r"\[\s*(-?[0-9.]+),\s*(-?[0-9.]+)\s*\]", str(s))
    if not m:
        return (np.nan, np.nan)
    return float(m.group(1)), float(m.group(2))


def fig7_tier_lift():
    df = pd.read_csv(TBL / "table23_synthetic_tier_summary_corrected.csv")
    tiers = df["Tier"].tolist()
    filters = [
        ("MC-MDD", "MC-MDD mean (pp)", "MC-MDD 95% CI"),
        ("MC-Calmar", "MC-Calmar mean (pp)", "MC-Calmar 95% CI"),
        ("MC-Ulcer", "MC-Ulcer mean (pp)", "MC-Ulcer 95% CI"),
        ("MC-ROI* (artefact)", "MC-ROI* (artefactual) mean (pp)", "MC-ROI* (artefactual) 95% CI"),
    ]
    colors = ["#1F3864", "#2E8B57", "#7E57C2", "#C0392B"]

    x = np.arange(len(tiers))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (label, mcol, cicol) in enumerate(filters):
        means = df[mcol].astype(float).values
        cis = df[cicol].apply(parse_ci)
        err_lo = means - cis.apply(lambda t: t[0]).values
        err_hi = cis.apply(lambda t: t[1]).values - means
        ax.bar(x + (i - 1.5) * width, means, width, label=label, color=colors[i],
               yerr=[err_lo, err_hi], capsize=4, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=0, fontsize=10)
    ax.set_ylabel("MC filter lift on OOS profitability (pp)")
    ax.set_title("Synthetic full-pipeline MC lift by tier (path-dependent)\n"
                 "All three path-dependent filters produce significant negative lift on iid synthetic; "
                 "the artefactual MC-ROI* filter matches.")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIG / "fig7_synthetic_tier_lift_corrected.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


def fig8_signal_sweep():
    df = pd.read_csv(TBL / "table25_synthetic_signal_sweep_corrected.csv")
    phi = df["phi"].astype(float).values
    filters = [
        ("MC-MDD", "MC-MDD lift (pp)", "#1F3864"),
        ("MC-Calmar", "MC-Calmar lift (pp)", "#2E8B57"),
        ("MC-Ulcer", "MC-Ulcer lift (pp)", "#7E57C2"),
        ("MC-ROI* (artefact)", "MC-ROI* (artefactual) lift (pp)", "#C0392B"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, col, color in filters:
        ax.plot(phi, df[col].astype(float).values, "o-", color=color, label=label, linewidth=2, markersize=7)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel(r"Signal strength $\varphi$ (synthetic momentum parameter)")
    ax.set_ylabel("MC filter lift on OOS profitability (pp)")
    ax.set_title("Signal-strength sweep: MC lift vs. embedded signal\n"
                 "Path-dependent filters stay non-positive across all signal levels; "
                 "the artefactual filter is more variable but never positive.")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG / "fig8_synthetic_signal_sweep_corrected.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"Wrote {out}")
    plt.close(fig)


def main():
    fig7_tier_lift()
    fig8_signal_sweep()


if __name__ == "__main__":
    main()

# ======================================================================
# === Section: Fig portfolio MC right-shift ===
# ======================================================================

#!/usr/bin/env python3
"""Portfolio MC-MDD percentile-rank distributions across the nine instruments.
Visualises the right-shift documented in Section 6.2 / Table tab:portfolio_mc."""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
SRC  = ROOT / "results" / "raw_data" / "portfolio_mc_corrected_all.csv"
OUTS = [ROOT / "results" / "figures" / "fig_portfolio_mc_rightshift.pdf"]

C_NAVY = "#1F3864"; C_GRAY = "#777"
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

df = pd.read_csv(SRC)
# (asset key, display label)
order = [("BTC","BTC"),("DOGE","DOGE"),("BNB","BNB"),("SOL","SOL"),
         ("EURUSD","EUR/USD"),("USDJPY","USD/JPY"),("EURGBP","EUR/GBP"),
         ("XAUUSD","XAU/USD"),("WTI","WTI")]
bins = np.linspace(0, 100, 41)            # 40 bins, width 2.5
uniform_density = 1.0 / 100.0             # exchangeable null is ~uniform on [0,100]

C_RED = "#C0392B"
fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
for ax, (key, label) in zip(axes.ravel(), order):
    s = df[df.asset == key]
    mdd = s["port_mdd_rank"].dropna().values
    roi = s["port_roi_rank_broken"].dropna().values        # sum-based, artefact-prone
    m_mdd, m_roi = mdd.mean(), roi.mean()
    # path-dependent MDD: filled navy
    ax.hist(mdd, bins=bins, density=True, color=C_NAVY, alpha=0.75,
            label=f"MC-MDD (path-dependent), mean {m_mdd:.0f}")
    # sum-based ROI*: red outline (artefact-prone control)
    ax.hist(roi, bins=bins, density=True, histtype="step", color=C_RED, lw=1.6,
            label=f"MC-ROI* (sum-based), mean {m_roi:.0f}")
    ax.axhline(uniform_density, color=C_GRAY, ls=":", lw=1)      # exchangeable null
    ax.axvline(50, color=C_GRAY, ls="--", lw=1)                  # 50% benchmark
    ax.set_title(f"{label}", fontsize=10)
    ax.set_xlim(0, 100); ax.set_ylim(0, 0.05)
    ax.legend(fontsize=6.6, loc="upper center", framealpha=0.9)
for ax in axes[-1]:
    ax.set_xlabel("portfolio percentile rank")
for ax in axes[:, 0]:
    ax.set_ylabel("density")

fig.suptitle("Portfolio MC rank distributions ($K=10$ IS-PF$>$1 portfolios): the path-dependent MDD "
             "right-shifts above 50,\nbut the artefact-prone sum-based ROI* does not "
             "(left of 50 on crypto/commodity; degenerate at 0 on forex)",
             fontsize=10.5, y=0.998)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
print("saved:", [str(o) for o in OUTS])

# ======================================================================
# === Section: Fig gold MC (cross-asset) ===
# ======================================================================

#!/usr/bin/env python3
"""Gold-standard (bar-permutation) MC summary figure across the nine instruments.
Panel A: in-sample edge (PF and MDD percentile ranks vs the 50% benchmark).
Panel B: forward-OOS lift from filtering on the gold-standard MC-MDD rank (vs 0).
Sources the per-instrument aggregate JSONs that also produce Table tab:gold_mc."""
import os
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
AGG  = ROOT / "results" / "tables"
OUTS = [ROOT / "results" / "figures" / "fig_gold_mc.pdf"]

C_NAVY="#1F3864"; C_TEAL="#2E8B57"; C_RED="#C0392B"; C_GRAY="#777"
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

# (json key, display label, asset class)
rows = [("btc","BTC","crypto"),("doge","DOGE","crypto"),("bnb","BNB","crypto"),
        ("sol","SOL","crypto"),("eurusd","EUR/USD","forex"),("usdjpy","USD/JPY","forex"),
        ("eurgbp","EUR/GBP","forex"),("xauusd","XAU/USD","commodity"),("wti","WTI","commodity")]
labels, pf, mdd, lift = [], [], [], []
for key,label,_ in rows:
    d = json.load(open(AGG / f"gold_mc_{key}_agg.json"))
    labels.append(label)
    pf.append(d["rank_dist"]["PF"]["mean"])
    mdd.append(d["rank_dist"]["MDD"]["mean"])
    lift.append(d["lifts"]["gold_mdd"]["lift"])

y = np.arange(len(rows))[::-1]      # top-to-bottom in listed order
h = 0.38

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6))

# Panel A: in-sample ranks
axA.barh(y+h/2, pf,  height=h, color=C_TEAL, label="PF rank (higher = beats replays)")
axA.barh(y-h/2, mdd, height=h, color=C_NAVY, label="MDD rank (lower = better drawdowns)")
axA.axvline(50, color=C_GRAY, ls="--", lw=1.2)
axA.text(50, len(rows)-0.3, " 50% (no edge)", color=C_GRAY, fontsize=8, va="bottom")
axA.set_yticks(y); axA.set_yticklabels(labels)
axA.set_xlim(0,100); axA.set_xlabel("mean in-sample percentile rank vs bar-permutation replays")
axA.set_title("(A) In-sample edge over stationary-bar replays")
axA.legend(fontsize=8, loc="lower right")

# Panel B: forward-OOS lift
cls_color = {"crypto":C_NAVY, "forex":C_RED, "commodity":C_TEAL}
bar_colors = [cls_color[c] for _,_,c in rows]
axB.barh(y, lift, height=0.6, color=bar_colors)
axB.axvline(0, color=C_GRAY, ls="-", lw=1)
axB.set_yticks(y); axB.set_yticklabels(labels)
axB.set_xlabel("forward-OOS profitability lift (pp), gold-standard MC-MDD p50 filter")
axB.set_title("(B) Forward-OOS lift from acting on the gold-standard rank")
for yi, v in zip(y, lift):
    axB.text(v + (0.08 if v>=0 else -0.08), yi, f"{v:+.2f}",
             va="center", ha="left" if v>=0 else "right", fontsize=8)
axB.set_xlim(min(lift)-1.2, max(lift)+1.2)
from matplotlib.patches import Patch
axB.legend(handles=[Patch(color=C_NAVY,label="crypto"),Patch(color=C_RED,label="forex"),
                    Patch(color=C_TEAL,label="commodity")], fontsize=8, loc="lower right")

fig.suptitle("Gold-standard bar-permutation MC: genuine in-sample edge (A) but near-zero "
             "forward selection power (B)", fontsize=11.5, y=0.99)
fig.tight_layout(rect=[0,0,1,0.96])
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
print("saved gold mc figure")

# ======================================================================
# === Section: Fig portfolio OOS decile ===
# ======================================================================

#!/usr/bin/env python3
"""Forward-OOS test of the portfolio MC-MDD signal (Section 6.3).
Pooled OOS profitability by IS MC-MDD-rank decile -- values from
Table tab:portfolio_oos_pooled_decile."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
OUTS = [ROOT / "results" / "figures" / "fig_portfolio_oos_decile.pdf"]
C_NAVY="#1F3864"; C_TEAL="#2E8B57"; C_RED="#C0392B"; C_GRAY="#777"
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

deciles = ["D1\n(0-10)","D2","D3","D4","D5","D6","D7","D8","D9","D10\n(90-100)"]
same = [21.85,21.83,20.37,20.84,19.79,20.06,18.72,18.88,19.12,17.33]
nxt  = [20.41,20.68,19.81,19.39,19.07,18.97,18.11,18.70,18.05,16.93]
x = np.arange(10); w = 0.4

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.bar(x-w/2, same, width=w, color=C_TEAL, alpha=0.85, label="same-window OOS")
ax.bar(x+w/2, nxt,  width=w, color=C_NAVY, alpha=0.95, label="next-window OOS")
# trend guide on next-window
z = np.polyfit(x, nxt, 1); ax.plot(x, np.polyval(z, x), color=C_RED, ls="--", lw=1.5,
        label=f"next-window trend ({z[0]:+.2f} pp/decile)")
ax.set_xticks(x); ax.set_xticklabels(deciles, fontsize=8)
ax.set_ylim(15, 23)
ax.set_xlabel("Portfolio IS MC-MDD rank decile  (D1 = roughest IS equity  $\\rightarrow$  D10 = smoothest)")
ax.set_ylabel("OOS profitability rate (\\%)")
ax.set_title("Forward-OOS test: smoothest-IS portfolios (high MC-MDD rank) underperform out-of-sample")
ax.legend(fontsize=9, loc="lower left")
# annotate the top-bottom gap (upper-right area, clear of the legend and the tall left bars)
ax.annotate(f"D1 $-$ D10 next-window gap: ${nxt[0]-nxt[-1]:+.2f}$ pp\n(top decile is lower)",
            xy=(9, nxt[-1]+0.05), xytext=(5.0, 22.2),
            fontsize=9, color=C_RED, ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
fig.tight_layout()
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
print("saved portfolio OOS decile figure")

# ======================================================================
# === Section: Fig MC by family heatmap ===
# ======================================================================

#!/usr/bin/env python3
"""MC-MDD rank means by asset x indicator family (Section 4, by-family).
Heatmap of the 63 asset-family cells, centred on the 50% benchmark."""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
SRC  = ROOT / "results" / "tables" / "table8_mc_by_family_corrected.csv"
OUTS = [ROOT / "results" / "figures" / "fig_mc_by_family_heatmap.pdf"]
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

df = pd.read_csv(SRC)
assets = ["BTC","DOGE","BNB","SOL","EUR/USD","USD/JPY","EUR/GBP","XAU/USD","WTI"]
fams   = ["ATR","EMA","PPO","RSI","SMA","STOCHK","OTHER"]
M = df.pivot(index="Asset", columns="Family", values="Mean MC-MDD").reindex(index=assets, columns=fams)

fig, ax = plt.subplots(figsize=(9, 7))
norm = TwoSlopeNorm(vmin=44, vcenter=50, vmax=56)
im = ax.imshow(M.values, cmap="RdBu_r", norm=norm, aspect="auto")
ax.set_xticks(range(len(fams)));   ax.set_xticklabels(fams, rotation=30, ha="right")
ax.set_yticks(range(len(assets))); ax.set_yticklabels(assets)
for i in range(len(assets)):
    for j in range(len(fams)):
        v = M.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="black")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Mean MC-MDD percentile rank")
ax.set_title("MC-MDD rank means by asset $\\times$ indicator family (63 cells):\n"
             "all cluster near the 50% exchangeability benchmark", fontsize=11)
fig.tight_layout()
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
print("range of cell means:", np.nanmin(M.values), np.nanmax(M.values))
print("saved family heatmap")

# ======================================================================
# === Section: Fig cross-asset forest plot ===
# ======================================================================

#!/usr/bin/env python3
"""Cross-asset forward-OOS confirmation (Section 6.4): per-instrument
top-minus-bottom decile MC-MDD lift with 95% CIs (Table tab:portfolio_oos_topbottom)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
OUTS = [ROOT / "results" / "figures" / "fig_crossasset_forest.pdf"]
C_NAVY="#1F3864"; C_RED="#C0392B"; C_TEAL="#2E8B57"; C_GRAY="#777"
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

# (label, class, lift, ci_lo, ci_hi)  -- top-decile minus bottom-decile, MC-MDD
rows = [
 ("BTC","crypto",-3.49,-6.02,-0.95),("DOGE","crypto",-3.74,-6.79,-0.69),
 ("BNB","crypto",+1.85,-0.12,+3.81),("SOL","crypto",+1.67,-3.07,+6.40),
 ("EUR/USD","forex",-0.21,-1.28,+0.86),("USD/JPY","forex",-0.88,-2.08,+0.33),
 ("EUR/GBP","forex",-3.14,-6.66,+0.39),("XAU/USD","commodity",+5.26,+0.35,+10.16),
 ("WTI","commodity",+1.48,-2.39,+5.36)]
pooled = -3.48
cls_color={"crypto":C_NAVY,"forex":C_RED,"commodity":C_TEAL}

y = np.arange(len(rows))[::-1]
fig, ax = plt.subplots(figsize=(9, 5.6))
for yi,(lab,cl,v,lo,hi) in zip(y,rows):
    ax.plot([lo,hi],[yi,yi], color=cls_color[cl], lw=2)
    ax.plot(v, yi, "o", color=cls_color[cl], ms=7)
ax.axvline(0, color="black", lw=1)
ax.axvline(pooled, color=C_GRAY, ls="--", lw=1.3)
ax.text(pooled, len(rows)-0.4, f" pooled {pooled:+.2f}", color=C_GRAY, fontsize=8, va="bottom")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("Top-decile $-$ bottom-decile next-window OOS profitability (pp), MC-MDD rank")
ax.set_title("Cross-asset confirmation: portfolio MC-MDD rank has no consistent positive\n"
             "forward-OOS lift (CIs straddle zero on 6 of 9; BTC/DOGE negative, XAU/USD positive)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_NAVY,label="crypto"),Patch(color=C_RED,label="forex"),
                   Patch(color=C_TEAL,label="commodity")], fontsize=8, loc="lower right")
ax.margins(y=0.08)
fig.tight_layout()
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
print("saved cross-asset forest figure")

# ======================================================================
# === Section: Fig cost sensitivity ===
# ======================================================================

#!/usr/bin/env python3
"""Transaction-cost sensitivity (Section 7.4): MC filter lift across cost levels,
per instrument. Replaces the dense 144-row table. Source table16 CSV."""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
SRC  = ROOT / "results" / "tables" / "table16_cost_sensitivity_corrected.csv"
OUTS = [ROOT / "results" / "figures" / "fig_cost_sensitivity.pdf"]
plt.rcParams.update({"font.size": 9.5, "font.family": "serif"})
C = {"MC-MDD p50":"#1F3864","MC-Calmar p50":"#2E8B57","MC-Ulcer p50":"#7E57C2",
     "MC-ROI* p50 (art.)":"#C0392B"}
CL_SHORT = {"Baseline (1x fee, 1x slip)":"Baseline","Fee+100% (2x fee)":"Fee+100%",
            "Slip+200% (3x slip)":"Slip+200%","Entry+Indicator drift":"Ent+drift"}

df = pd.read_csv(SRC)
df["clab"] = df["Cost Level"].map(CL_SHORT)
costs = ["Baseline","Fee+100%","Slip+200%","Ent+drift"]
filters = ["MC-MDD p50","MC-Calmar p50","MC-Ulcer p50","MC-ROI* p50 (art.)"]
assets = [("BTC","BTC"),("DOGE","DOGE"),("BNB","BNB"),("SOL","SOL"),
          ("EURUSD","EUR/USD"),("USDJPY","USD/JPY"),("EURGBP","EUR/GBP"),
          ("XAUUSD","XAU/USD"),("WTI","WTI")]
x = np.arange(len(costs)); w = 0.2

fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharex=True, sharey=True)
for ax,(key,label) in zip(axes.ravel(), assets):
    sub = df[df.Asset==key]
    for i,filt in enumerate(filters):
        vals = [sub[(sub.clab==c)&(sub["MC Filter"]==filt)]["Lift (pp)"].mean() for c in costs]
        ax.bar(x + (i-1.5)*w, vals, width=w, color=C[filt],
               label=filt.replace(" p50","").replace(" (art.)","*"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(label, fontsize=10); ax.set_xticks(x); ax.set_xticklabels(costs, fontsize=7.5, rotation=15)
    ax.set_ylim(-4.6, 1.2)
for ax in axes[:,0]: ax.set_ylabel("MC+IS lift (pp)")
axes[0,0].legend(fontsize=7, loc="lower left", ncol=1, framealpha=0.9)
fig.suptitle("Transaction-cost sensitivity: the MC filter's incremental lift over the IS-PF gate is essentially flat "
             "across cost levels\non every instrument (near zero on crypto/commodity; a few pp negative on forex from "
             "IS-PF dilution) --- cost is not a confound",
             fontsize=10.5, y=0.995)
fig.tight_layout(rect=[0,0,1,0.96])
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
print("saved cost sensitivity figure")

# ======================================================================
# === Section: Fig synthetic ground-truth ranks ===
# ======================================================================

#!/usr/bin/env python3
"""Synthetic ground-truth MC rank behaviour (Appendix: synthetic validation).
Mean MC percentile rank by scenario x metric -- source synthetic_mc_rank_stats_corrected.csv."""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(os.environ.get("MC_PAPER_DATA", Path(__file__).resolve().parents[1]))
SRC  = ROOT / "results" / "tables" / "synthetic_mc_rank_stats_corrected.csv"
OUTS = [ROOT / "results" / "figures" / "fig_synthetic_groundtruth_ranks.pdf"]
plt.rcParams.update({"font.size": 10, "font.family": "serif"})
C = {"MDD":"#1F3864","Calmar":"#2E8B57","ROI*":"#C0392B"}
SC = {"A":"A: realistic returns\n(heavy tails, GARCH, regimes)",
      "B":"B: data-mining\n(MA-crossover scan)",
      "CMP":"C: correlated\nportfolios"}

df = pd.read_csv(SRC)
scen = ["A","B","CMP"]; metrics = ["MDD","Calmar","ROI*"]
x = np.arange(len(scen)); w = 0.26
fig, ax = plt.subplots(figsize=(9.5, 5.2))
for i,met in enumerate(metrics):
    vals = [df[(df.Scenario==s)&(df.Metric==met)]["Mean"].iloc[0] for s in scen]
    bars = ax.bar(x + (i-1)*w, vals, width=w, color=C[met],
                  label=("MC-"+met) if met!="ROI*" else "MC-ROI* (sum-based, artefactual)")
    for b,v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.6, f"{v:.0f}", ha="center", fontsize=8)
ax.axhline(50, color="#777", ls="--", lw=1.2)
ax.text(x[-1]+0.45, 50.5, "50% benchmark", color="#777", fontsize=8, va="bottom", ha="right")
ax.set_xticks(x); ax.set_xticklabels([SC[s] for s in scen], fontsize=8.5)
ax.set_ylabel("mean MC percentile rank")
ax.set_ylim(0, 60)
ax.set_title("Synthetic ground truth: path-dependent MC ranks (MDD, Calmar) sit near the 50% benchmark\n"
             "with only a small edge-driven leftshift; the sum-based ROI* is artefactually pinned far left",
             fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper right")
fig.tight_layout()
for out in OUTS:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
print("saved synthetic ground-truth figure")
