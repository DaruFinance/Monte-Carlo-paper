#' 02_bootstrap_lift_ci.R
#'
#' Purpose:
#'   Independently replicate the 10,000-resample bootstrap confidence
#'   intervals for the MC-filter lift in OOS profitability on each asset.
#'
#' Cross-validates (Python/Rust source):
#'   - calendar_cluster_bootstrap.py (clustered bootstrap, paper's primary CI)
#'   - block_perm_analysis.py (point estimate of lift)
#'
#' Paper artifact reproduced:
#'   - Table 5 (central empirical finding: lift -1.1 to -1.2 pp, 9 assets)
#'   - Table 15 (clustered bootstrap CIs, 10,000 resamples)
#'
#' Method:
#'   For each asset:
#'     (a) Merge block_perm_<asset>.csv (iid_rank) with
#'         raw_data/<asset>_window_pairs.csv on (strategy, window).
#'     (b) Define oos_profitable = 1 if baseline_oos_pf > 1.0.
#'     (c) baseline = mean(oos_profitable)
#'         filtered = mean(oos_profitable | iid_rank >= 50)
#'         lift     = (filtered - baseline) * 100    (percentage points)
#'     (d) Bootstrap the lift by clustering on (asset, window). We draw
#'         N_boot=10,000 window-level resamples with replacement and
#'         recompute the lift in each resample. We report point estimate,
#'         bootstrap SE, 95% percentile CI, and p(lift >= 0).
#'
#'   The window-clustered bootstrap is the same design used by
#'   calendar_cluster_bootstrap.py (which additionally groups windows into
#'   calendar quarters across assets). Per-asset we only have one asset, so
#'   the natural cluster is the window.
#'
#' Input:
#'   $MC_PAPER_ROOT/block_perm_<asset>.csv
#'   $MC_PAPER_DATA/<asset>_window_pairs.csv
#'
#' Output:
#'   out/02_bootstrap_lift_ci.csv
#'   stdout: formatted table with CI for each asset
#'
#' Expected runtime: 2-8 minutes across all 9 assets (dominated by the
#'   merge; the bootstrap itself is vectorized on window-level tallies).
#'
#' Usage:
#'   Rscript 02_bootstrap_lift_ci.R

source("_helpers.R")
set.seed(42)

N_BOOT <- 10000L
out_dir <- ensure_out()

bootstrap_lift_for_asset <- function(asset) {
  bp <- read_csv_fast(block_perm_path(asset))
  wp <- read_csv_fast(window_pairs_path(asset))
  if (is.null(bp) || is.null(wp)) return(NULL)

  # Window column in bp is integer-like; window_i in wp is integer.
  wp_small <- wp[, .(strategy, window_i, baseline_oos_pf)]
  setnames(wp_small, "window_i", "window")
  bp_small <- bp[, .(strategy, window, iid_rank)]

  m <- merge(bp_small, wp_small, by = c("strategy", "window"))
  if (nrow(m) == 0L) return(NULL)
  m <- m[is.finite(baseline_oos_pf) & is.finite(iid_rank)]
  m[, oos_prof := as.integer(baseline_oos_pf > 1.0)]

  # Window-level tallies used for both point estimate and cluster bootstrap
  agg <- m[, .(
    n         = .N,
    n_oos     = sum(oos_prof),
    n_pass    = sum(iid_rank >= 50),
    n_pass_oos= sum((iid_rank >= 50) * oos_prof)
  ), by = window]

  total_n      <- sum(agg$n)
  total_oos    <- sum(agg$n_oos)
  total_pass   <- sum(agg$n_pass)
  total_passos <- sum(agg$n_pass_oos)

  baseline <- total_oos / total_n
  filtered <- if (total_pass > 0) total_passos / total_pass else NA_real_
  point_lift <- (filtered - baseline) * 100

  # Window-cluster bootstrap
  K <- nrow(agg)
  n_v     <- agg$n
  oos_v   <- agg$n_oos
  pass_v  <- agg$n_pass
  passos_v<- agg$n_pass_oos

  lifts <- numeric(N_BOOT)
  for (b in seq_len(N_BOOT)) {
    idx <- sample.int(K, K, replace = TRUE)
    tn  <- sum(n_v[idx])
    to  <- sum(oos_v[idx])
    tp  <- sum(pass_v[idx])
    tpo <- sum(passos_v[idx])
    bl  <- to / tn
    fl  <- if (tp > 0) tpo / tp else NA_real_
    lifts[b] <- (fl - bl) * 100
  }
  lifts <- lifts[is.finite(lifts)]

  data.table(
    asset     = ASSET_LABEL[[asset]],
    n_obs     = total_n,
    n_windows = K,
    baseline_pct = round(baseline * 100, 2),
    filter_pct   = round(filtered * 100, 2),
    lift_pp      = round(point_lift, 3),
    boot_se      = round(sd(lifts), 4),
    ci_lo_95     = round(quantile(lifts, 0.025), 3),
    ci_hi_95     = round(quantile(lifts, 0.975), 3),
    p_lift_ge_0  = round(mean(lifts >= 0), 5)
  )
}

rows <- list()
for (asset in ALL_ASSETS) {
  cat(sprintf("[%s] bootstrapping ...\n", ASSET_LABEL[[asset]]))
  r <- bootstrap_lift_for_asset(asset)
  if (!is.null(r)) {
    rows[[asset]] <- r
    print(r)
  }
}

result <- rbindlist(rows)
cat("\n=== Bootstrap MC-Filter Lift (10,000 window-cluster resamples) ===\n")
print(result)

out_path <- file.path(out_dir, "02_bootstrap_lift_ci.csv")
fwrite(result, out_path)
cat("\nWrote:", out_path, "\n")
