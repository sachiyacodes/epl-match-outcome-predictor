# Audited model comparison

The original 53.42% score is superseded because its source chronology was incorrect. After source repair, the original frozen procedure scores 53.29% on the same 760 historical fixtures. The new comparison below uses different periods and regular coefficient refitting; do not compare its percentages directly with the old score.

## Data and protocol

- 6,080 real EPL matches, 2010/11–2025/26, through May 24, 2026.
- All 66 incorrect dates reconciled against immutable source text; original bytes and artifacts preserved.
- Development: seven seasons, 2018/19–2024/25; 2,660 out-of-time predictions per complete configuration.
- Confirmation: 2025/26, 380 matches; configuration chosen and saved before confirmation scores were computed.
- Models refit before each prediction month; features update only from strictly earlier match dates.
- Temperature frozen from preceding-season out-of-time predictions during confirmation.

## Development comparisons

These scores were used for selection and are not independent confirmation estimates.

| Family | Best development accuracy | Raw log loss |
| --- | ---: | ---: |
| Corrected baseline | 53.01% | 0.997505 |
| Elo / rich logistic | 54.55% | 0.971940 |
| Shallow boosting | 53.05% | 0.993904 |
| Independent Poisson | 54.59% | 0.966018 |
| Dixon–Coles | 54.36% | 0.966512 |

Selected configuration: `{"corrected": false, "half_life": 180, "kind": "score"}`.
Development qualification: **passed**.

## Locked confirmation results

| Model | Accuracy | Raw log loss | Calibrated log loss | Brier | ECE | Draw recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Corrected monthly baseline | 47.63% | 1.050462 | 1.048861 | 0.631972 | 0.023707 | 0.00% |
| Development-selected challenger | 47.11% | 1.031608 | 1.030215 | 0.619025 | 0.073822 | 0.00% |

Accuracy difference: **-0.53 percentage points**; paired 95% weekly-block interval **[-4.44, +3.58] points**.
Calibrated log-loss difference: **-0.018646** (95% interval [-0.043361902708831085, 0.0033884873800921547]). Lower is better.

**Decision: corrected baseline retained.** Improvement not confirmed by every gate; corrected baseline retained.

Promotion required all development gates, a positive confirmation accuracy gain, a 95% accuracy interval entirely above zero, and calibrated log loss no more than 0.01 worse. The criteria were not relaxed after observing confirmation.

## Reproducibility and interpretation

- Protocol digest: `975dad874f24072a4307356606b16c588275a2a1c930c6de7d1640736e4f094f`.
- protocol.json, selection.json, confirmation.json and prediction CSVs are included under artifacts/research-v2.
- Original data/raw files, model.joblib and metrics.json remain byte-identical to the preserved v1 release.
- Final serving artifacts refit after evaluation; dashboard confirmation metrics remain out-of-time scores, not final-fit training scores.
- Per-season, promoted-team and class-level diagnostics appear in the report JSON and dashboard.
- Weekly bootstrap intervals are approximate; dependence and season-specific effects remain.
- The confirmation season is now inspected. Future tuning needs another truly later evaluation period.
- No live feed, xG, injuries, lineups, odds, confidence filtering, XGBoost or betting-profitability test was added.

## Subsequent independent-data work

The statement above describes the locked v2 model experiment. A later, separate research step adds independent verification and historical shot-quality estimates without changing that experiment or its serving decision.

- Reproducible Wyscout comparison: all **380** 2017/18 fixture identities, dates and full-time scores agree. This covers **6.25%** of the 6,080-match dataset; the other 5,700 records are not independently confirmed.
- Wyscout's CC BY 4.0 event release supplies 643,150 EPL events for 2017/18. A fixed chronological logistic shot model generates 6,427 out-of-time non-penalty shot estimates. Only previous-match aggregates enter outcome features.
- On the same **130** February–May 2018 fixtures, goals-only features score **43.85% accuracy / 1.104376 log loss**; adding lagged estimated xG scores **45.38% / 1.093868**.
- That is **two extra correct predictions**, not evidence of persistent or modern-season improvement. No parameter search or main-model promotion occurs.
- Derived estimates are this project's model outputs, not Wyscout-provided xG. Source attribution and transformation details: [DATA_SOURCES.md](DATA_SOURCES.md). Protocol and outputs: [shot-quality research](artifacts/shot-quality/README.md).
