#' 01_mc_rank_means.R
#'
#' Purpose:
#'   Independently replicate the per-asset mean MC ROI percentile rank
#'   statistics and test each against a 50% null (the null hypothesis
#'   under exchangeability / Monte-Carlo symmetry).
#'
#' Cross-validates (Python/Rust source):
#'   - full_analysis.py (mean-rank summary)
#'   - block_perm_rs/src/main.rs (produces the underlying roi_pct_rank)
#'
#' Paper artifact reproduced:
#'   - Table 4 (per-asset mean MC ROI percentile rank)
#'   - Inline stats in Section 5 (BTC 40.8%, EUR/USD 31.4%, DOGE 40.9%, etc.)
#'   - "62.6%-70.4% of observations below the 50th percentile" inline claim
#'
#' Method:
#'   For each asset, read raw_data/<asset>_mc_perwindow.csv.
#'   Column roi_pct_rank is a 0-100 percentile rank produced by the Rust
#'   MC permutation test (1000 permutations per strategy-window).
#'   Under the exchangeability null each rank is Uniform[0,100] so E[rank]=50.
#'   We compute mean, median, SD, share <50, and a one-sample t-test vs 50.
#'
#' Input:
#'   $MC_PAPER_DATA/<asset>_mc_perwindow.csv (9 assets)
#'   Columns: strategy, window, n_trades, actual_roi, actual_sharpe, actual_pf,
#'            roi_pct_rank, sharpe_pct_rank, pf_pct_rank
#'
#' Output:
#'   out/01_mc_rank_means.csv (per-asset summary)
#'   stdout: formatted table
#'
#' Expected runtime: 30-90 seconds (fread over ~6M rows total)
#'
#' Usage:
#'   Rscript 01_mc_rank_means.R

source("_helpers.R")
set.seed(42)

out_dir <- ensure_out()

rows <- list()
for (asset in ALL_ASSETS) {
  path <- mc_perwindow_path(asset)
  dt <- read_csv_fast(path)
  if (is.null(dt)) next
  if (!"roi_pct_rank" %in% names(dt)) next

  r <- dt$roi_pct_rank
  r <- r[is.finite(r)]
  n <- length(r)
  if (n == 0) next

  mu <- mean(r)
  md <- median(r)
  sdv <- sd(r)
  pct_below_50 <- mean(r < 50) * 100

  # One-sample t-test vs 50 (Uniform[0,100] null mean)
  tt <- t.test(r, mu = 50)

  rows[[asset]] <- data.table(
    asset         = ASSET_LABEL[[asset]],
    n_obs         = n,
    mean_rank     = round(mu, 2),
    median_rank   = round(md, 2),
    sd_rank       = round(sdv, 2),
    pct_below_50  = round(pct_below_50, 2),
    t_stat        = round(unname(tt$statistic), 2),
    df            = unname(tt$parameter),
    p_value       = format.pval(tt$p.value, digits = 3, eps = 1e-300),
    ci_lo         = round(tt$conf.int[1], 2),
    ci_hi         = round(tt$conf.int[2], 2)
  )
  cat(sprintf("  %-8s  mean=%5.2f  median=%5.2f  %%<50=%5.2f  t=%8.2f  p=%s\n",
              ASSET_LABEL[[asset]], mu, md, pct_below_50,
              unname(tt$statistic),
              format.pval(tt$p.value, digits = 3, eps = 1e-300)))
}

result <- rbindlist(rows)
cat("\n=== MC ROI Rank Mean Test vs 50 (exchangeability null) ===\n")
print(result)

out_path <- file.path(out_dir, "01_mc_rank_means.csv")
fwrite(result, out_path)
cat("\nWrote:", out_path, "\n")
