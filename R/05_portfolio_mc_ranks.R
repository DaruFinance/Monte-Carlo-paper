#' 05_portfolio_mc_ranks.R
#'
#' Purpose:
#'   Replicate the portfolio-level MC rank statistics (corrected Table 14 +
#'   tab:portfolio_mc) for the 4 crypto assets with portfolio_mc_path CSVs.
#'
#' Cross-validates (Python/Rust source):
#'   - portfolio_mc_analysis.py (aggregation over portfolios)
#'   - rust/portfolio_mc_path (path-dependent MDD/Calmar portfolio ranks)
#'
#' Paper artifact reproduced:
#'   - Corrected Table 14 (portfolio MC, path-dependent MDD/Calmar)
#'   - Portfolio MC rank distributions (fig_portfolio_mc_rightshift.pdf)
#'
#' Method:
#'   For each crypto asset read raw_data/<asset>_portfolio_mc_path.csv.
#'   Columns: asset, window, port_id, n_trades_combined,
#'            actual_roi, actual_mdd, actual_calmar,
#'            port_mdd_rank, port_calmar_rank, port_roi_rank_broken.
#'   Aggregated per asset:
#'     - n portfolios
#'     - mean/median/SD of port_mdd_rank, port_calmar_rank
#'     - share with port_mdd_rank > 50 (portfolio-level "beats MC null")
#'     - mean actual_roi
#'   Corrected: switched from {roi,sharpe,pf}_pct_rank to {port_mdd,port_calmar}_rank
#'   (paper Section sec:fp-pitfall).
#'
#' Input:
#'   $MC_PAPER_DATA/results/raw_data/<asset>_portfolio_mc_path.csv  (btc, doge, bnb, sol)
#'
#' Output:
#'   out/05_portfolio_mc_ranks.csv (asset x filter)
#'
#' Expected runtime: <30 seconds.
#'
#' Usage:
#'   Rscript 05_portfolio_mc_ranks.R

source("_helpers.R")
set.seed(42)
out_dir <- ensure_out()

# Corrected: switched from roi_pct_rank/sharpe_pct_rank/pf_pct_rank to
# port_mdd_rank/port_calmar_rank (paper Section sec:fp-pitfall). The new
# portfolio_mc_path schema no longer carries an explicit `filter` column —
# aggregation is per-asset.

rows <- list()
for (asset in CRYPTO_ASSETS) {
  path <- portfolio_mc_path(asset)
  dt <- read_csv_fast(path)
  if (is.null(dt)) next

  by_asset <- dt[, .(
    n_portfolios          = .N,
    mean_mdd_rank         = round(mean(port_mdd_rank, na.rm = TRUE), 2),
    median_mdd_rank       = round(median(port_mdd_rank, na.rm = TRUE), 2),
    sd_mdd_rank           = round(sd(port_mdd_rank, na.rm = TRUE), 2),
    mean_calmar_rank      = round(mean(port_calmar_rank, na.rm = TRUE), 2),
    mean_roi_rank_broken  = round(mean(port_roi_rank_broken, na.rm = TRUE), 2),
    pct_mdd_above_50      = round(mean(port_mdd_rank > 50, na.rm = TRUE) * 100, 2),
    pct_calmar_above_50   = round(mean(port_calmar_rank > 50, na.rm = TRUE) * 100, 2),
    mean_actual_roi       = round(mean(actual_roi, na.rm = TRUE), 3),
    mean_actual_mdd       = round(mean(actual_mdd, na.rm = TRUE), 3),
    mean_actual_calmar    = round(mean(actual_calmar, na.rm = TRUE), 3)
  )]

  by_asset[, asset := ASSET_LABEL[[asset]]]
  setcolorder(by_asset, "asset")
  rows[[asset]] <- by_asset
  cat(sprintf("\n[%s]\n", ASSET_LABEL[[asset]]))
  print(by_asset)
}

result <- rbindlist(rows, fill = TRUE)
cat("\n=== Portfolio-level MC rank summary (crypto) ===\n")
print(result)

out_path <- file.path(out_dir, "05_portfolio_mc_ranks.csv")
fwrite(result, out_path)
cat("\nWrote:", out_path, "\n")
