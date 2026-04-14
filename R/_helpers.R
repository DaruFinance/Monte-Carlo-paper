# _helpers.R
# Shared I/O and path helpers for the R cross-validation scripts.
# Sourced by 01_..05_ scripts. Keep dependencies to base R + data.table.

suppressPackageStartupMessages({
  library(data.table)
})

# Resolve the directory containing the raw CSVs.
# Priority: env var MC_PAPER_DATA > ../results/raw_data/ > ../../raw_data/
mc_data_dir <- function() {
  env <- Sys.getenv("MC_PAPER_DATA", unset = "")
  if (nzchar(env) && dir.exists(env)) return(normalizePath(env))
  candidates <- c(
    file.path("..", "results", "raw_data"),
    file.path("..", "..", "raw_data"),
    file.path("..", "..", "..", "raw_data")
  )
  for (p in candidates) {
    if (dir.exists(p)) return(normalizePath(p))
  }
  stop("Could not locate raw_data directory. Set MC_PAPER_DATA env var.")
}

# Directory with root-level block_perm_*.csv files (one level up from raw_data)
mc_root_dir <- function() {
  env <- Sys.getenv("MC_PAPER_ROOT", unset = "")
  if (nzchar(env) && dir.exists(env)) return(normalizePath(env))
  # Default: parent of raw_data
  rd <- mc_data_dir()
  parent <- dirname(rd)
  if (dir.exists(parent)) return(parent)
  stop("Could not locate project root for block_perm_*.csv")
}

# Null-coalescing operator
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

# Canonical asset lists (match Python scripts)
CRYPTO_ASSETS <- c("btc", "doge", "bnb", "sol")
FX_ASSETS     <- c("eurusd", "usdjpy", "eurgbp", "xauusd", "wti")
ALL_ASSETS    <- c(CRYPTO_ASSETS, FX_ASSETS)

ASSET_LABEL <- c(
  btc = "BTC", doge = "DOGE", bnb = "BNB", sol = "SOL",
  eurusd = "EUR/USD", usdjpy = "USD/JPY", eurgbp = "EUR/GBP",
  xauusd = "XAU/USD", wti = "WTI"
)

# Fast CSV read with data.table fread
read_csv_fast <- function(path) {
  if (!file.exists(path)) {
    message("Missing file: ", path)
    return(NULL)
  }
  fread(path, showProgress = FALSE)
}

# Path helpers
mc_perwindow_path  <- function(asset) file.path(mc_data_dir(), paste0(asset, "_mc_perwindow.csv"))
window_pairs_path  <- function(asset) file.path(mc_data_dir(), paste0(asset, "_window_pairs.csv"))
portfolio_mc_path  <- function(asset) file.path(mc_data_dir(), paste0(asset, "_portfolio_mc.csv"))
block_perm_path    <- function(asset) file.path(mc_root_dir(), paste0("block_perm_", asset, ".csv"))

# Ensure out dir exists (relative to current working directory, which is
# expected to be Scripts_Clean/R/ when the scripts are run).
ensure_out <- function() {
  out <- "out"
  if (!dir.exists(out)) dir.create(out, showWarnings = FALSE, recursive = TRUE)
  normalizePath(out)
}
