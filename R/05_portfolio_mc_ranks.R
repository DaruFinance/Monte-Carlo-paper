#' 05_portfolio_mc_ranks.R
#'
#' Purpose:
#'   Replicate the portfolio-level MC rank statistics (Figure 10 / Table 14)
#'   for the 4 crypto assets with portfolio_mc CSVs.
#'
#' Cross-validates (Python/Rust source):
#'   - portfolio_mc_analysis.py (streaming aggregation over filters)
#'
#' Paper artifact reproduced:
#'   - Table 14 (strategy vs portfolio OOS profitability, 53-62 pp gap)
#'   - Figure 10 data (portfolio MC rank distributions)
#'
#' Method:
#'   For each crypto asset read raw_data/<asset>_portfolio_mc.csv.
#'   Columns: filter, window_i, portfolio_id, n_strategies, n_trades,
#'            actual_roi, actual_sharpe, actual_pf,
#'            roi_pct_rank, sharpe_pct_rank, pf_pct_rank.
#'   For each filter compute:
#'     - n portfolios
#'     - mean/median/SD of roi_pct_rank, sharpe_pct_rank, pf_pct_rank
#'     - share with roi_pct_rank > 50 (portfolio-level "beats MC null")
#'     - OOS profitability proxy = share with actual_pf > 1.0
#'       (excluding sentinel values >= 900 which mean "no losing trades")
#'     - mean actual_roi
#'
#' Input:
#'   $MC_PAPER_DATA/<asset>_portfolio_mc.csv  (btc, doge, bnb, sol)
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

PF_SENTINEL <- 900  # matches portfolio_mc_analysis.py

rows <- list()
for (asset in CRYPTO_ASSETS) {
  path <- portfolio_mc_path(asset)
  dt <- read_csv_fast(path)
  if (is.null(dt)) next

  dt[, pf_valid := actual_pf < PF_SENTINEL]

  by_filter <- dt[, .(
    n_portfolios      = .N,
    mean_roi_rank     = round(mean(roi_pct_rank, na.rm = TRUE), 2),
    median_roi_rank   = round(median(roi_pct_rank, na.rm = TRUE), 2),
    sd_roi_rank       = round(sd(roi_pct_rank, na.rm = TRUE), 2),
    mean_sharpe_rank  = round(mean(sharpe_pct_rank, na.rm = TRUE), 2),
    mean_pf_rank      = round(mean(pf_pct_rank, na.rm = TRUE), 2),
    pct_roi_above_50  = round(mean(roi_pct_rank > 50, na.rm = TRUE) * 100, 2),
    pct_sharpe_above_50 = round(mean(sharpe_pct_rank > 50, na.rm = TRUE) * 100, 2),
    mean_actual_roi   = round(mean(actual_roi, na.rm = TRUE), 3),
    mean_actual_sharpe= round(mean(actual_sharpe, na.rm = TRUE), 3),
    mean_actual_pf    = round(mean(actual_pf[pf_valid], na.rm = TRUE), 3),
    pct_oos_profitable= round(mean(actual_pf[pf_valid] > 1.0, na.rm = TRUE) * 100, 2)
  ), by = filter]

  by_filter[, asset := ASSET_LABEL[[asset]]]
  setcolorder(by_filter, c("asset", "filter"))
  rows[[asset]] <- by_filter
  cat(sprintf("\n[%s]\n", ASSET_LABEL[[asset]]))
  print(by_filter)
}

result <- rbindlist(rows, fill = TRUE)
cat("\n=== Portfolio-level MC rank summary (crypto) ===\n")
print(result)

out_path <- file.path(out_dir, "05_portfolio_mc_ranks.csv")
fwrite(result, out_path)
cat("\nWrote:", out_path, "\n")
