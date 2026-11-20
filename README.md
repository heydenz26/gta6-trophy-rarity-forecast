[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/heydenz26/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/heydenz26/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

Forecasts each of Grand Theft Auto VI's 55 PlayStation 5 trophies' Exophase
earn rate at 2026-11-26 00:00 UTC as five quantiles (5/25/50/75/95%), from
the house's recording of Exophase's trophy page up to the case checkpoint.

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
"Forecast how many GTA VI players earn each trophy in week one". Scored by
pinball loss on the logit scale; a prize needs a loss strictly below the best
house entry.

## How it works

`main.py` (Python 3.13, standard library only) reads every `<id>.in` case in
`/input` and writes `<id>.out` to `/output`.

A trophy's rate is `r = 100 * E / N` — earners over players tracked — and the
program forecasts E and N separately, then recombines them. At launch N
explodes while E grows steadily, so rates first fall (dilution) and only later
rise; extrapolating the rate itself cannot see that turn, the E/N split can.

- **N(t), E(t):** logistic adoption curves fitted in log space by grid search
  over (k, t0) with the saturation level solved in closed form. A logistic
  saturates, so it cannot run away over the 4.5-day horizon.
- **Median blend:** the fitted trend's median is shrunk 40/60 toward
  persistence in logit space — a 4.5-day extrapolation from 2.5 days of
  history is weakly identified for late-blooming trophies.
- **Launch prior:** on launch night the house reads are flat, yet rates fall
  ~1.5 logits within half a day as N explodes. The program carries the
  influx prior fitted on the real launch-night → 2026-11-19 11:39 UTC
  Exophase snapshot (N 638 → ~10,000): N ~ (1+t/2h)^2.74,
  E_i ~ (1+t/2h)^b_i with b_i = 1.240 + 0.723·ln(r_i). It fires only on a
  (near-)flat history within 24 h of release.
- **Constraints:** earners are monotone along the story chain
  (Ch1 ≥ Ch2 ≥ Ch3 ≥ Ch4 ≥ final mission); the platinum is capped by the
  smallest other trophy's earners; tier saturation caps bound every forecast.
- **Spread:** fit residuals + extrapolation distance + half the starter's
  random-walk floor, calibrated on backtests.

## Backtests (pinball loss, logit scale — lower is better)

| Test | Persistence (house) | Curve (house) | This model |
| --- | --- | --- | --- |
| Real: launch-night history → live Exophase snapshot 2026-11-19 11:39 UTC (55 trophies) | 1.3899 | 1.3899 | **0.1411** |
| Synthetic final regime, 20 sims (seed 7): cut day 2.5 → target day 7 | 0.7706 | 1.4060 | **0.6820** (beats persistence 18/20) |
| Synthetic final regime, 20 sims (seed 8, fresh): | 0.7503 | 1.7868 | **0.6872** (beats persistence 15/20) |

The launch prior was also split-half validated: fitted on odd trophy ids
only, it scores 0.1443 on the held-out even ids (vs 0.1399 on its training
ids) — the 10× practice win is not a fitting artifact. Synthetic sims are
labelled as such in the build notes; their parameters are sampled from ranges
fitted to the real launch points plus domain ranges for week-one saturation.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic, offline, standard library only; ~0.1 s per case.

## License

MIT — see [LICENSE](LICENSE).
