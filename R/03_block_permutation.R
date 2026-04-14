#' 03_block_permutation.R
#'
#' Purpose:
#'   Independently recompute the block-permutation MC filter lift and
#'   approximate permutation p-value for block sizes b=1,2,3,5,10,20,
#'   pooled across all 9 assets and per asset class.
#'
#' Cross-validates (Python/Rust source):
#'   - block_perm_rs/src/main.rs (produces the *_rank columns)
#'   - block_perm_analysis.py (pooled sweep over block sizes)
#'
#' Paper artifact reproduced:
#'   - Table 19 (block permutation MC test, all 9 instruments, b=1..20)
#'
#' Method:
#'   For each asset read block_perm_<asset>.csv with columns
#'     iid_rank, block2_rank, block3_rank, block5_rank, block10_rank, block20_rank.
#'   Merge with raw_data/<asset>_window_pairs.csv for OOS profitability
#'   (baseline_oos_pf > 1.0).
#'   For each block size b, compute:
#'     baseline = P(OOS profitable)
#'     filter   = P(OOS profitable | rank_b >= 50)
#'     lift_pp  = (filter - baseline) * 100
#'     pearson r between rank_b and oos_profitable
#'     one-sided permutation p-value: share of strategy-window obs whose
#'       rank_b >= 50 reflects how often the permuted PnL beats actual;
#'       we report the pooled share and Fisher-combined p for the test
#'       of "mean rank > 50" using a one-sample t-test on rank_b vs 50.
#'
#' Input:
#'   $MC_PAPER_ROOT/block_perm_<asset>.csv (9 files)
#'   $MC_PAPER_DATA/<asset>_window_pairs.csv (9 files)
#'
#' Output:
#'   out/03_block_permutation.csv  (long format: asset x block_size x metric)
#'   out/03_block_permutation_pooled.csv (pooled across all / by class)
#'
#' Expected runtime: 1-3 minutes.
#'
#' Usage:
#'   Rscript 03_block_permutation.R

source("_helpers.R")
set.seed(42)

out_dir <- ensure_out()

BLOCK_COLS <- c(
  "iid (b=1)"  = "iid_rank",
  "block b=2"  = "block2_rank",
  "block b=3"  = "block3_rank",
  "block b=5"  = "block5_rank",
  "block b=10" = "block10_rank",
  "block b=20" = "block20_rank"
)

ASSET_CLASS <- c(
  btc = "Crypto", doge = "Crypto", bnb = "Crypto", sol = "Crypto",
  eurusd = "Forex", usdjpy = "Forex", eurgbp = "Forex",
  xauusd = "Commodity", wti = "Commodity"
)

per_asset <- list()
pooled_rows <- list()
combined <- list()

for (asset in ALL_ASSETS) {
  bp <- read_csv_fast(block_perm_path(asset))
  wp <- read_csv_fast(window_pairs_path(asset))
  if (is.null(bp) || is.null(wp)) next

  wp_small <- wp[, .(strategy, window_i, baseline_oos_pf)]
  setnames(wp_small, "window_i", "window")
  m <- merge(bp, wp_small, by = c("strategy", "window"))
  if (nrow(m) == 0L) next
  m[, oos_prof := as.integer(baseline_oos_pf > 1.0)]
  m[, asset_class := ASSET_CLASS[[asset]]]
  m[, asset_label := ASSET_LABEL[[asset]]]

  baseline <- mean(m$oos_prof) * 100
  cat(sprintf("\n[%s] n=%d baseline=%.2f%%\n",
              ASSET_LABEL[[asset]], nrow(m), baseline))

  for (nm in names(BLOCK_COLS)) {
    col <- BLOCK_COLS[[nm]]
    if (!col %in% names(m)) next
    r <- m[[col]]
    pass <- r >= 50
    pass_oos <- mean(m$oos_prof[pass]) * 100
    lift <- pass_oos - baseline
    cor_ro <- suppressWarnings(cor(r, m$oos_prof))
    mean_r <- mean(r)
    # one-sample t-test for mean-rank vs 50 (null = exchangeability)
    tt <- t.test(r, mu = 50)

    per_asset[[length(per_asset) + 1L]] <- data.table(
      asset = ASSET_LABEL[[asset]],
      asset_class = ASSET_CLASS[[asset]],
      method = nm,
      n_obs = nrow(m),
      n_pass = sum(pass),
      pass_rate_pct = round(mean(pass) * 100, 2),
      baseline_pct = round(baseline, 2),
      pass_oos_pct = round(pass_oos, 2),
      lift_pp = round(lift, 3),
      corr_rank_oos = round(cor_ro, 4),
      mean_rank = round(mean_r, 2),
      t_stat_vs50 = round(unname(tt$statistic), 2),
      p_value_vs50 = format.pval(tt$p.value, digits = 3, eps = 1e-300)
    )
    cat(sprintf("  %-10s  lift=%+6.2fpp  r=%+7.4f  meanRank=%.2f\n",
                nm, lift, cor_ro, mean_r))
  }

  combined[[asset]] <- m[, .(asset_class, oos_prof,
                              iid_rank, block2_rank, block3_rank,
                              block5_rank, block10_rank, block20_rank)]
}

per_asset_dt <- rbindlist(per_asset, fill = TRUE)
fwrite(per_asset_dt, file.path(out_dir, "03_block_permutation.csv"))
cat("\nWrote: 03_block_permutation.csv (", nrow(per_asset_dt), "rows)\n")

# Pooled + by-class sweep
if (length(combined) > 0) {
  big <- rbindlist(combined, fill = TRUE)
  baseline_all <- mean(big$oos_prof) * 100
  cat(sprintf("\n=== Pooled sweep (n=%d, baseline=%.2f%%) ===\n",
              nrow(big), baseline_all))

  groups <- list(All = big)
  for (cls in unique(big$asset_class)) {
    groups[[cls]] <- big[asset_class == cls]
  }

  for (gn in names(groups)) {
    g <- groups[[gn]]
    bl <- mean(g$oos_prof) * 100
    for (nm in names(BLOCK_COLS)) {
      col <- BLOCK_COLS[[nm]]
      if (!col %in% names(g)) next
      r <- g[[col]]
      pass_oos <- mean(g$oos_prof[r >= 50]) * 100
      lift <- pass_oos - bl
      cor_ro <- suppressWarnings(cor(r, g$oos_prof))
      pooled_rows[[length(pooled_rows) + 1L]] <- data.table(
        group = gn, method = nm,
        n_obs = nrow(g),
        baseline_pct = round(bl, 2),
        pass_oos_pct = round(pass_oos, 2),
        lift_pp = round(lift, 3),
        corr_rank_oos = round(cor_ro, 4)
      )
      cat(sprintf("  [%s] %-10s lift=%+6.2fpp r=%+7.4f\n",
                  gn, nm, lift, cor_ro))
    }
  }

  pooled_dt <- rbindlist(pooled_rows)
  fwrite(pooled_dt, file.path(out_dir, "03_block_permutation_pooled.csv"))
  cat("\nWrote: 03_block_permutation_pooled.csv\n")
}
