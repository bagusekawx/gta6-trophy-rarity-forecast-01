# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bagusekawx/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bagusekawx/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

Entry for the [gta6-trophy-rarity-forecast-01](https://make.fun/c/gta6-trophy-rarity-forecast-01) bounty on make.fun: forecast every GTA VI PS5 trophy's Exophase earn rate at the case's target, as five quantiles (5/25/50/75/95) per trophy, scored by mean pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only) reads a case JSON from `/input` and writes `<id>.out` to `/output`. The forecast blends three signals in logit space:

1. **Rate trend (0.8).** Per trophy, `logit(rate)` is fit on log-time since release over the last 24 hours of reads — the maturation leg. Launch-week rates are not one straight line: they fall hard while the tracked base explodes, then rise as the base matures, so the full history would average the fall into the rise. The fit is recency-weighted (12h half-life) and each trophy's slope is shrunk toward its tier's mean slope. Quantile spreads use the fit's prediction interval, including the slope's extrapolation uncertainty.
2. **Counts decomposition (0.1).** Earners are fit on log players-tracked and log-time (shrunk toward "earners track the base"), and the tracked base itself is forecast by blending a straight log-time line with a saturating concave curve. Dividing the two separates dilution (base growth) from progression (players earning more), which matters when base growth bends over the horizon.
3. **Persistence (0.1).** The last observed rate, as an anchor against extrapolation running away.

Special cases: a history of identical reads means Exophase has not refreshed yet, so the entry assumes a pending refresh (median shifted down by the expected dilution, wide band); a trophy everyone already has is forecast from the lag of new players behind the tracked base.

## Run locally

```sh
MAKEFUN_INPUT_DIR=../input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs are byte-identical. Runs in well under a second per case.

## Validation

On synthetic backtests (a launch-week simulator with refresh-stepped reads, log-time player adoption, and per-kind time-to-earn curves), 20 draws across two seeds:

| | mean pinball loss |
| --- | --- |
| this entry | 0.26 |
| persistence baseline | 0.76 |
| curve baseline (last-24h trend) | 0.32 |

The entry beats the curve baseline in 17 of 20 draws and the persistence baseline in all 20. On the practice cases (stale pre-refresh history, approximated refresh answer), loss is ~0.33 vs ~1.69 for persistence.

## License

MIT — see `LICENSE`.
