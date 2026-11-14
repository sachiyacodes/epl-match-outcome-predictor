# Historical shot-quality experiment

This is an isolated research result, not a serving-model upgrade.

Data source: Pappalardo et al. (2019), *A public data set of spatio-temporal match events in soccer competitions*, Wyscout, [doi:10.1038/s41597-019-0247-7](https://doi.org/10.1038/s41597-019-0247-7). Original data and these data derivatives are provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: shot geometry, fitted probabilities, lagged aggregates and model evaluation. The estimated xG values are this project's model outputs, not Wyscout-provided xG.

## Fixed chronological protocol

- Source: 380 EPL matches, 643,150 events from 2017/18.
- Extracted 8,801 non-penalty shots; excluded 80 penalties.
- Fit a logistic shot-goal model from distance, goal angle and body-part context. The goal tag supplies the label, never an input feature.
- August–October provides shot-model warmup. Each November–May prediction uses a model fitted only on earlier calendar months: 6,427 out-of-time shot estimates.
- Aggregate those estimates into 280 team-match xG pairs, then use only prior completed matches' estimates for lagged form.
- November–January provides match-classifier warmup. Compare fixed logistic classifiers on the same 130 February–May fixtures, with monthly refits.
- No parameter tuning, test-set selection, confidence filtering or serving-model promotion occurs.

## Observed result

| Features | Accuracy | Log loss |
| --- | ---: | ---: |
| Goals-based baseline | 43.85% | 1.104376 |
| Baseline plus lagged estimated xG | 45.38% | 1.093868 |

Difference: **two additional correct predictions out of 130**, or +1.54 percentage points. This is a small, one-season exploratory result. It does not prove a persistent improvement, modern-season validity or betting profitability. The active audited model is unchanged.

## Reproduce

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe -m epl.shot_quality
```

First acquisition downloads the original public archives into ignored `data/external/wyscout/`, including the 77 MB events archive. Every source has a fixed SHA256. Cached data permits offline replay. No account or new dependency is needed.

`protocol.json` records the source checksums, code hash, split dates and features before evaluation. `report.json` contains metrics and SHA256 values for each derivative CSV. Re-running verifies cached sources and reuses the saved report. Changed source/code requires a separate output directory; this does not create a new untouched evaluation period.

`out-of-time-shots.csv` includes each shot's fit cutoff and estimated probability. `match-estimated-xg.csv` contains non-penalty aggregates. `outcome-predictions.csv` includes both models' probabilities and training cutoffs. No same-match xG value is a pre-match input.

Limits: nominal 105×68 metre pitch geometry, no penalty xG, limited warmup, no external-season confirmation, and no contemporary event feed. More recent coverage is needed before this feature can support current predictions.
