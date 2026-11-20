# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tianzhaoming28-maker/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tianzhaoming28-maker/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI's PS5 trophy earn rates on Exophase. Because the page only changes when Exophase refreshes it, the recorded history is first compressed into its distinct states; then the tracked player base and each trophy's earner count are forecast separately — damped power-law growth fits on the recent regime, per-trophy slopes shrunk toward the pooled cross-trophy slope, with a deceleration check so a saturating trophy is not extrapolated like a surging one — and composed back into rates, blended in logit space with a direct trend extrapolation. When the history holds a single frozen state, the program instead prices the overdue refresh: the tracked base multiplies while earner counts barely move, so rates dilute, with story trophies gaining earners fastest. Medians respect the game's structure (the platinum cannot be commoner than the rarest other trophy; story chapters finish in order), and the five quantiles are normal on the logit scale with a spread that grows with the horizon, the fit residuals and the size of the projected move. Pure Python 3.13 standard library, offline and deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=qn2pqd71&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
