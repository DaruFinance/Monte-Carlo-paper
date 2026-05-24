# results/

Outputs produced by the analysis pipeline, plus an (empty by default)
`raw_data/` slot for the user-supplied inputs.

```
results/
  figures/      ← PDF figures cited via \includegraphics in the paper
  tables/       ← CSV / JSON / TeX outputs that feed paper tables
  raw_data/     ← user-supplied per-asset CSVs (left empty: .gitkeep only)
```

## What lives where

- **`figures/`** ships the 18 PDFs cited in the paper. The producing
  scripts live under `python/` and `rust/`; see the top-level `README.md`
  for the figure-to-script map.
- **`tables/`** ships the CSV / JSON / TeX aggregates that the paper's
  inline `tabular` environments and figures are built from. None of
  these files contain strategy names; rows are keyed by anonymous
  integer IDs or by `(asset, window)`.
- **`raw_data/`** is empty in the published release. Populate it from
  your own per-asset walk-forward output following the schemas in
  `python/README.md`. The strategy backtester that produces the inputs
  is open source at
  <https://github.com/DaruFinance/quant-research-framework-rs>.

## Why `raw_data/` is empty

Per the paper's Data Availability section, this package ships analysis
scripts only — not the raw bar or trade data, and not the 437,911
strategy configurations whose parameterisations are proprietary.
Readers can either re-run the backtester on their own bar data and
strategy parameterisations, or apply the analysis scripts to any
strategy-window output conforming to the documented schema.
