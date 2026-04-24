# Paper-snapshot figures

These three PDFs are the **exact frozen images** that appear in
`paper_redacted.pdf` (as shipped with the original submission) for the
forex / commodity section:

- `mc_pct_rank_distributions.pdf`   (Fig. 9,  Mar 27 2026 vintage)
- `window_level_mc_vs_oos.pdf`      (Fig. 2,  Mar 27 2026 vintage)
- `mc_roi_vs_next_oos_binned.pdf`   (Fig. 10, Apr 20 2026 vintage)

## Why they are here

The sibling `results/figures/*.pdf` are regenerated on demand by
`python/regenerate_all_figures.py` from whatever CSVs are in
`results/raw_data/`. Two of the forex/commodity inputs
(`*_mc_perwindow.csv` for the forex pairs and WTI) were rerun on
**Apr 13 2026** after the Mar 27 figures had already been baked into the
paper; numerical values in the regenerated versions of Figs 2 and 9
will therefore differ slightly from the text of the paper (e.g. the
paper reports EUR/USD mean MC-ROI rank ≈ 31.4, the current data
produces ≈ 46.9). Fig 10 was re-rendered against the Apr 13 data on
Apr 20 and is already consistent with the current pipeline.

The qualitative finding of the paper (MC ranks cluster below 50 for
the forex/commodity pool; no positive per-window MC-rank → OOS
relationship) is unchanged in both vintages.

## How to verify you are looking at the paper's figures

Compare these PDFs to the images in `paper_redacted.pdf` §
"MC Rank Distribution Visualizations" and § "Cross-Asset Extension".
The rendered bitmaps should match exactly.

## How to regenerate the paper's exact values

The Mar 27 2026 MC-per-window CSVs for the five non-crypto instruments
are **not distributed** with this repository (they are not in
`results/raw_data/` and have no `.csv.INFO` stub). The Apr 13 2026
snapshot currently shipped has these md5 fingerprints — if your
`results/raw_data/` hashes to the same values, you are on the current
vintage and `regenerate_all_figures.py` will reproduce the sibling
figures (not the Mar 27 snapshot figures kept here):

| File | md5 (Apr 13 2026 vintage) |
|---|---|
| `eurusd_mc_perwindow.csv` | `7a5435b300fe7c3b6c76ff0613c5dfff` |
| `usdjpy_mc_perwindow.csv` | `f082a37d6be2a87b94b27a4e26e60de9` |
| `eurgbp_mc_perwindow.csv` | `3f3c441ca31af32cf40b09500601ea79` |
| `xauusd_mc_perwindow.csv` | `ab54839f4756e4f5755fc5323affbf40` |
| `wti_mc_perwindow.csv`    | `0f2b71bb5a429a3c6d5d125b308c43e9` |

To reproduce the Mar 27 snapshot numerically, the upstream trades
binaries that produced that snapshot are required; contact the author.
