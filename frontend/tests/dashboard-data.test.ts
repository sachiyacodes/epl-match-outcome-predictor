import assert from "node:assert/strict";
import test from "node:test";
import { experimentReport, firstPredictionDate, metricDisplay, percentage, testMetricCards } from "../app/dashboard-data.ts";

test("prediction dates start strictly after the snapshot, including leap and year boundaries", () => {
  assert.equal(firstPredictionDate(""), undefined);
  assert.equal(firstPredictionDate("2024-02-28"), "2024-02-29");
  assert.equal(firstPredictionDate("2024-02-29T23:00:00"), "2024-03-01");
  assert.equal(firstPredictionDate("2024-12-31"), "2025-01-01");
});

test("legacy reports keep working without an experiment decision", () => {
  assert.equal(experimentReport(null), null);
  assert.equal(experimentReport({ splits: { test: { calibrated: { accuracy: .53 } } } }), null);
  assert.equal(metricDisplay("Test calibrated · accuracy", .5342), "53.42%");
  assert.equal(metricDisplay("Test calibrated · log loss", .9933), "0.9933");
  assert.equal(percentage(.0123, true), "+1.23 pp");
});

test("audited reports preserve failed promotion, confirmation and development separation", () => {
  const result = experimentReport({
    model_version: "audited-v2", data_through: "2026-05-24", trained_through: "2026-05-24",
    calibration_period: { start: "2024-08-16", end: "2025-05-25" }, audit_status: "passed",
    experiment: {
      promoted: false, development_qualified: true, active: "baseline", reason: "accuracy_interval_crosses_zero",
      baseline: { calibrated: { accuracy: .54, log_loss: 1.0, per_class: { D: { recall: .1 } }, confusion_matrix: [[1, 2, 3], [4, 5, 6], [7, 8, 9]] } },
      challenger: { calibrated: { accuracy: .55, log_loss: .99, per_class: { D: { recall: .2 } } } },
      development_baseline: { pooled: { accuracy: .52 }, seasons: { "2023-24": { accuracy: .51 } } },
      development_selected: { pooled: { accuracy: .56 }, seasons: { "2023-24": { accuracy: .57 } } },
      uncertainty: { accuracy_difference: .01, accuracy_ci95: [-.02, .04], log_loss_difference: -.01, log_loss_ci95: [-.04, .02] },
    },
  });
  assert.ok(result);
  assert.equal(result.promoted, false);
  assert.equal(result.active, "baseline");
  assert.equal(result.reason, "accuracy interval crosses zero");
  assert.equal(result.calibrationPeriod, "2024-08-16 to 2025-05-25");
  assert.equal(result.baseline.accuracy, .54);
  assert.equal(result.challenger.drawRecall, .2);
  assert.deepEqual(result.baseline.confusion, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  assert.equal(result.developmentChallenger.accuracy, .56);
  assert.equal(result.seasons[0].challenger.accuracy, .57);
  assert.deepEqual(result.accuracyInterval, [-.02, .04]);
});

test("headline evaluation shows test results and baseline, never training scores", () => {
  assert.deepEqual(testMetricCards(null), []);
  const cards = testMetricCards({ splits: {
    train: { calibrated: { log_loss: 0.01 } },
    validation: { calibrated: { log_loss: 0.02 } },
    test: { calibrated: { log_loss: 1.02, brier: 0.6, expected_calibration_error: 0.04, calibration: [] }, class_prior: { log_loss: 1.1 } },
  } });
  assert.deepEqual(cards, [
    { label: "Test calibrated · log loss", value: 1.02 },
    { label: "Test calibrated · brier", value: 0.6 },
    { label: "Test calibrated · expected calibration error", value: 0.04 },
    { label: "Test class prior · log loss", value: 1.1 },
  ]);
});
