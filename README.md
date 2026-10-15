# Football Edge Lab

A research-first football probability laboratory. The project is designed to answer one question rigorously:

> **Can any information source improve a bookmaker market baseline out of sample, prospectively, without leakage?**

The repository deliberately separates *probability estimation* from *betting decisions*. Market prices are a benchmark/anchor, not truth. No historical holdout may be reused indefinitely for model invention.

## Core principles

1. **Past → future only**: all evaluation is chronological walk-forward.
2. **Immutable pre-match snapshots**: every prospective prediction is hashed and frozen before kickoff.
3. **Market baseline first**: every model must beat or add information beyond de-vigged market probabilities.
4. **Residual modeling**: prefer learning a correction to the market instead of predicting 1/X/2 from zero.
5. **Ablation before complexity**: every feature group must justify itself out of sample.
6. **Calibration matters**: accuracy alone is insufficient; LogLoss, Brier, calibration and coverage curves are mandatory.
7. **No hindsight threshold tuning**: new thresholds create a new prospective protocol version.
8. **PAPER / research first**: profitability is not assumed from historical ROI alone.

## Architecture

```text
src/football_edge/
  data/          schemas + dataset validation
  features/      market movement + future feature groups
  models/        market baseline + residual model
  evaluation/    metrics, walk-forward, coverage, ablation
  prospective/   immutable freeze + settlement helpers
  cli/           command line interface
```

The original Colab notebook is preserved under `legacy/` as a behavioral reference only. The new code should become the canonical implementation.

## Quick start in Colab

```bash
!pip install -q -e ".[dev]"
!pytest -q
```

Then from Python:

```python
from football_edge.data.schema import ProbabilityVector
from football_edge.models.market_baseline import devig_decimal_odds

p = devig_decimal_odds(2.10, 3.50, 3.80)
print(p)
```

## Required dataset shape

Minimum model table:

- `match_id`
- `kickoff_utc`
- `snapshot_utc`
- `home_team`, `away_team`
- `odds_home`, `odds_draw`, `odds_away`
- optional model/source probabilities
- feature columns built **only from information known at `snapshot_utc`**
- after settlement only: `actual` (`HOME`, `DRAW`, `AWAY`)

For market-movement studies, store multiple rows per match at target horizons such as T-72h, T-48h, T-24h, T-12h, T-6h, T-3h, T-1h, T-30m, T-10m and close.

## First research program

The first model is `MarketResidualClassifier`:

- baseline = market probabilities;
- inputs = transformed market probabilities + incremental features;
- output = an adjusted 1/X/2 distribution;
- evaluation = chronological walk-forward vs market baseline on exactly the same rows.

The first pass should test feature groups independently:

1. market movement;
2. external probability sources;
3. Elo/team strength;
4. xG/form;
5. rest/schedule;
6. squad/lineup context.

A feature group survives only if the improvement is repeated across multiple future folds and is not driven by a tiny subgroup.

## Metrics

Mandatory:

- multiclass LogLoss;
- multiclass Brier;
- top-1 accuracy;
- calibration / reliability;
- accuracy-at-coverage;
- model-vs-market delta by fold;
- sample size.

If odds are available, ROI/CLV can be reported separately, but predictive quality and price quality must not be conflated.

## Codex workflow

Open `prompts/CODEX_PHASE_1.md` and give it to Codex with the repository attached. It contains the first implementation task and acceptance criteria.
