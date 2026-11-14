# Audited v2 verification

Verified on Windows, November 14, 2026, using Python 3.12.14, Node.js 24.15.0 and the pinned dependency locks.

## Data and preservation

- Loaded 6,080 real EPL results across 16 complete seasons, 2010/11–2025/26.
- Verified raw SHA256 checksums and complete-season structural checks.
- Reconciled all 66 incorrect July 2019 dates against the immutable source text's July 2020 fixtures. Audit passed with no quarantine.
- Rechecked all 14 original raw/manifest/model/metrics files against the preserved v1 release: byte-identical.
- The superseded original benchmark is not presented as evidence of current accuracy. Its corrected frozen reproduction scores 53.29% over the old 760 inspected fixtures.

## Model experiments

- Completed 69 configurations: one baseline, 54 Elo/rich logistic challengers, eight boosting challengers and six Poisson/Dixon–Coles challengers.
- Every successful configuration produced 2,660 monthly out-of-time development predictions over seven seasons.
- Selected independent Poisson with a 180-day decay half-life on development data: 54.59% accuracy versus baseline 53.01%.
- Frozen selection was written before evaluating confirmation outcomes.
- Confirmation 2025/26: baseline **47.63%**, challenger **47.11%**, 380 matches each.
- Challenger accuracy difference **−0.53 percentage points**, approximate paired weekly-block 95% interval **[−4.44, +3.58] points**.
- Calibrated log loss: baseline **1.048861**, challenger **1.030215**. Better log loss did not satisfy the accuracy promotion requirement.
- Challenger was **not promoted**. Active serving artifact is the audited, refitted baseline.
- Replayed the completed workflow. It reused sealed predictions; the confirmation JSON checksum remained unchanged.
- See `MODEL_REPORT.md` for complete interpretation. The confirmation season is now inspected and cannot become an untouched test again.

## Automated checks for the audited v2 implementation

- **72 backend tests passed.** Coverage includes source-date reconciliation, both score schemas, quarantine and checksum failures, same-day/future leakage, Elo season handling, venue shrinkage, offline/inference parity, score likelihood gradients, positive score probabilities, optimizer failures, chronological fitting, promotion gates, deterministic bootstrap, sealed-ledger rejection, artifact path containment and API compatibility.
- Python compilation passed; `pip check` found no broken requirements.
- **Four frontend tests passed.** Coverage includes snapshot dates, legacy compatibility, audited experiment mapping and separation of development/confirmation metrics.
- TypeScript check passed. Next.js production build passed.
- No dedicated lint configuration was added. Compilation, type checking, tests, independent read-only review and runtime checks are the executed validation.
- One upstream Starlette/httpx test-client deprecation warning remains. A Windows physical-core discovery warning occurred during restricted development execution; experiments completed successfully.

## Real browser checks

Used a fresh headless Microsoft Edge session with the local API and production dashboard. No personal browser profile or fake API responses were used for the successful prediction path.

- `/teams`, `/metrics` and `/predict` returned HTTP 200 with the verified v2 artifact.
- Displayed the retained-baseline decision, model audit metadata, confirmation comparison, weekly intervals, seven development seasons and both confusion matrices.
- Arsenal FC vs Chelsea FC on hypothetical date 2026-08-01 produced home **0.579590**, draw **0.230172**, away **0.190238**; probabilities sum to one. This is a model output, not an observed label or live fixture.
- Desktop **1280×900** and mobile **390×844**: no horizontal overflow. Screenshots visually inspected.
- Keyboard navigation reached the brand link and team selection. Native labels and form controls worked.
- Simulated an unavailable API by aborting browser requests: clear alert appeared and prediction submission was disabled.
- No browser page errors occurred during the successful interaction.
- Temporary verification servers were stopped after checks. Nothing was deployed publicly.

## Remaining limits

No verified accuracy improvement was established. The corrected pipeline and conservative promotion decision improve evaluation integrity, not a guaranteed forecast score. No live data, injuries, lineups, odds, xG, XGBoost, neural networks or profitability testing were added. The historical snapshot ends May 24, 2026. Source auditing cannot prove every upstream result correct, and weekly bootstrap intervals remain approximate.

## GitHub release and later historical-data experiment

The following work is separate from the locked model comparison above:

- Added pinned, reproducible Wyscout source verification. Real downloads and offline replay agree on 380/380 2017/18 fixtures, with zero conflicts and 5,700 explicitly unchecked records.
- Added the isolated shot-quality experiment and source attribution. Source events and raw archives remain outside Git. Derived research files are CC BY 4.0 with attribution.
- The fixed 130-match historical comparison adds two correct predictions (43.85% to 45.38%); it does not establish modern performance or alter the active model.
- Added model restoration from sealed reports, with protocol/data/dependency/ledger validation. This rebuilds serving binaries without reselecting models or rescoring confirmation.
- Release backend suite: **98 tests passed**; compilation and dependency consistency passed. Four frontend tests, TypeScript and production build also pass.
- Added SHA-pinned GitHub Actions for Windows/Linux backend tests and frontend checks. Hosted workflow results must be checked after publication; local success is not a claim about hosted CI.
- Added MIT licensing for software, separate data-license attribution, documentation screenshots, no-reply commit identity and ignores for credentials, raw downloads and model binaries.
- Exported the staged Git tree into a separate clean checkout with no model binaries or dependency folders. Model restoration passed and the full 98-test backend suite passed there.
- Rebuilt the npm lockfile from the same pinned direct dependencies to remove incomplete optional-platform records. A fresh `npm ci` then passed, with zero reported vulnerabilities, followed by four frontend tests, TypeScript and production build. Platform packages have explicit versions for Windows/Linux/macOS installs.
- Source bytes and sealed CSV line endings are preserved by `.gitattributes`, preventing Git checkout normalization from invalidating provenance checksums.
