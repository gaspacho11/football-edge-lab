# Codex Phase 1 — turn the seed repo into a historical prediction laboratory

You are working on a football probability research project. Do not optimize historical ROI and do not produce betting recommendations. The objective is to establish whether any feature set adds stable predictive information beyond the de-vigged bookmaker market.

## Non-negotiable scientific constraints

1. Chronological only: every model is trained exclusively on data timestamped before its test matches.
2. No result leakage: outcome/final-score/post-match fields must never enter feature construction or prediction code.
3. Every prediction row must have `snapshot_utc < kickoff_utc`.
4. Market baseline must be evaluated on exactly the same rows as every challenger model.
5. Never tune on a fold after reporting it as holdout evidence.
6. Do not replace poor results with a new metric. Report LogLoss, Brier, accuracy, calibration and accuracy-at-coverage together.
7. Any source derived from bookmaker odds must be tagged as market-dependent; do not count it as an independent vote.
8. Preserve the legacy notebook but do not make it canonical.

## Phase 1 implementation task

Build a reproducible historical pipeline with DuckDB and a CLI.

### A. Database
Create migrations / initialization for these tables:

- `matches`
- `results`
- `market_snapshots`
- `source_probability_snapshots`
- `team_ratings`
- `feature_snapshots`
- `model_predictions`
- `prospective_freezes`

Use unique keys that prevent duplicate source snapshots for the same match/source/timestamp.

### B. Importers
Implement CSV/parquet importers for:

- historical fixtures/results;
- bookmaker 1X2 snapshots;
- arbitrary external source probability snapshots.

Every importer must validate timezone-aware timestamps and reject impossible post-kickoff snapshots when importing prediction-time data.

### C. Market feature builder
For configurable horizons (T-72h, 48h, 24h, 12h, 6h, 3h, 1h, 30m, 10m):

- choose the latest snapshot at or before each horizon;
- compute de-vigged H/X/A;
- opening-to-current movement;
- horizon-to-horizon movement;
- log price movement;
- market entropy and top-two gap;
- number of books if available;
- cross-book dispersion if available.

No future snapshot may leak backward into an earlier horizon.

### D. Historical benchmark
Implement walk-forward evaluation for:

1. `MARKET_ONLY`
2. `MARKET_RESIDUAL_MARKET_FEATURES`
3. `MARKET_RESIDUAL_EXTERNAL_SOURCES`
4. combined model

Use expanding training windows and untouched future test blocks.

### E. Source-value analysis
For each external probability source report:

- coverage;
- standalone LogLoss/Brier/accuracy;
- correlation with market probabilities;
- incremental delta when added to a market residual model;
- leave-one-source-out effect;
- performance by probability confidence decile;
- performance by market favorite probability bucket.

### F. Calibration + selective prediction
Add:

- reliability tables for H/X/A;
- expected calibration error;
- accuracy-at-coverage curves;
- a selector that may abstain, but whose threshold is fitted on training/validation only and then frozen before the test block.

### G. Tests
Add tests specifically proving:

- no future snapshot selection;
- no post-kickoff snapshot accepted;
- market de-vig sums to 1;
- train timestamps precede test timestamps;
- model and market are scored on identical row IDs;
- feature columns contain no prohibited result fields;
- prospective freeze hash changes after tampering.

### H. Output
Create one command:

```bash
football-edge research run --config config/default.yaml
```

that writes a versioned report directory containing:

- `fold_metrics.csv`
- `aggregate_metrics.csv`
- `coverage_curve.csv`
- `source_ablation.csv`
- `calibration.csv`
- `predictions.parquet`
- `run_manifest.json`

`run_manifest.json` must contain git commit SHA, config hash, dataset hashes and all date boundaries.

## Acceptance criteria

Do not claim success merely because code runs. Phase 1 is complete only when:

- all tests pass;
- a small synthetic dataset can exercise the full pipeline end-to-end;
- the report explicitly shows market vs every model on identical future folds;
- leakage checks are automated;
- no API keys are committed;
- README contains exact local and Colab instructions.

When finished, summarize all changed files, tests run, any methodological assumptions, and anything that is still not statistically validated.
