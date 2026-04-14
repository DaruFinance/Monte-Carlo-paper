#' 04_strategy_correlations.R
#'
#' Purpose:
#'   Recompute the summary statistics behind Figure 1 / Table 3:
#'   within-family vs cross-family OOS Profit Factor correlations per asset.
#'
#' Cross-validates (Python/Rust source):
#'   - correlation_analysis/strategy_correlations.py
#'   - correlation_analysis/corr_rs/src/main.rs (Rust correlation crate)
#'
#' Paper artifact reproduced:
#'   - Figure 1 summary stats
#'   - Table 3 (within-family vs cross-family bar-level PnL correlation)
#'
#' Method:
#'   Two paths are supported:
#'
#'   (A) If correlation_analysis/correlation_summary.csv already exists
#'       (produced by the Rust/Python pipeline), load it and restate the
#'       mean/median within- and cross-family correlations per asset. This
#'       is the canonical numbers table and matches what enters the paper.
#'
#'   (B) Fallback: if the summary is absent, recompute a lightweight
#'       approximation from raw_data/<asset>_window_pairs.csv:
#'       - Parse the "family" from the strategy name (first underscore token).
#'       - For each (window_i) compute the Pearson correlation matrix of
#'         baseline_oos_pf across strategies.  (For large N this is memory
#'         bound; we cap at 1,500 strategies random-sample per asset for the
#'         fallback approximation only.)
#'       - Classify each off-diagonal pair as within-family or cross-family
#'         and report mean/median.
#'
#'   Path (A) is the authoritative one.
#'
#' Input:
#'   correlation_analysis/correlation_summary.csv (preferred)
#'   OR raw_data/<asset>_window_pairs.csv (fallback)
#'
#' Output:
#'   out/04_strategy_correlations.csv
#'
#' Expected runtime: <5 seconds for (A); 2-10 minutes for (B).
#'
#' Usage:
#'   Rscript 04_strategy_correlations.R

source("_helpers.R")
set.seed(42)
out_dir <- ensure_out()

# -- Path A: use published summary if present ---------------------------------
summary_path <- file.path(mc_root_dir(), "correlation_analysis", "correlation_summary.csv")

result <- NULL
if (file.exists(summary_path)) {
  cat("Using published correlation_summary.csv\n")
  raw <- read_csv_fast(summary_path)
  # Standardise column names
  keep <- c("asset", "n_strategies", "n_windows", "n_families",
            "mean_within_family_corr", "median_within_family_corr",
            "mean_cross_family_corr", "median_cross_family_corr",
            "pct_within_above_07", "pct_cross_above_07")
  keep <- intersect(keep, names(raw))
  result <- raw[, ..keep]

  # Round for display
  num_cols <- setdiff(keep, c("asset", "n_strategies", "n_windows", "n_families"))
  for (cc in num_cols) set(result, j = cc, value = round(result[[cc]], 4))

  cat("\n=== Within-family vs Cross-family OOS PF Correlation Summary ===\n")
  print(result)
} else {
  # -- Path B: fallback approximation --------------------------------------
  cat("correlation_summary.csv not found; running fallback approximation.\n")
  MAX_STRATS <- 1500L

  fam_of <- function(s) sub("_.*$", "", s)

  rows <- list()
  for (asset in ALL_ASSETS) {
    wp <- read_csv_fast(window_pairs_path(asset))
    if (is.null(wp)) next
    wp <- wp[is.finite(baseline_oos_pf)]
    wp[, family := fam_of(strategy)]

    # Sample strategies down for tractability
    strats <- unique(wp$strategy)
    if (length(strats) > MAX_STRATS) {
      strats <- sample(strats, MAX_STRATS)
      wp <- wp[strategy %in% strats]
    }

    # Wide matrix: rows=window_i, cols=strategy
    wide <- dcast(wp, window_i ~ strategy,
                  value.var = "baseline_oos_pf", fun.aggregate = mean)
    mat <- as.matrix(wide[, -1, with = FALSE])
    if (nrow(mat) < 3) next
    # Pearson correlation across windows
    M <- suppressWarnings(cor(mat, use = "pairwise.complete.obs"))
    strat_names <- colnames(M)
    fams <- fam_of(strat_names)

    # Off-diagonal pairs
    idx <- which(upper.tri(M), arr.ind = TRUE)
    vals <- M[idx]
    same_fam <- fams[idx[, 1]] == fams[idx[, 2]]
    vals_w <- vals[same_fam]
    vals_c <- vals[!same_fam]

    rows[[asset]] <- data.table(
      asset                      = ASSET_LABEL[[asset]],
      n_strategies_sampled       = length(strat_names),
      n_windows                  = nrow(mat),
      n_families                 = length(unique(fams)),
      mean_within_family_corr    = round(mean(vals_w, na.rm = TRUE), 4),
      median_within_family_corr  = round(median(vals_w, na.rm = TRUE), 4),
      mean_cross_family_corr     = round(mean(vals_c, na.rm = TRUE), 4),
      median_cross_family_corr   = round(median(vals_c, na.rm = TRUE), 4),
      pct_within_above_07        = round(mean(vals_w > 0.7, na.rm = TRUE) * 100, 3),
      pct_cross_above_07         = round(mean(vals_c > 0.7, na.rm = TRUE) * 100, 3)
    )
    cat(sprintf("  %-8s  within mean=%.4f  cross mean=%.4f\n",
                ASSET_LABEL[[asset]],
                rows[[asset]]$mean_within_family_corr,
                rows[[asset]]$mean_cross_family_corr))
  }
  result <- rbindlist(rows)
  cat("\n=== Fallback within/cross family correlation summary ===\n")
  print(result)
}

out_path <- file.path(out_dir, "04_strategy_correlations.csv")
fwrite(result, out_path)
cat("\nWrote:", out_path, "\n")
