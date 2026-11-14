function metricValues(value: unknown, prefix: string): { label: string; value: number }[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  return Object.entries(value).flatMap(([key, item]) => {
    const label = `${prefix}${key.replaceAll("_", " ")}`;
    if (typeof item === "number") return [{ label, value: item }];
    return metricValues(item, `${label} · `);
  });
}

export function testMetricCards(metrics: unknown) {
  const report = metrics as { splits?: { test?: { calibrated?: unknown; class_prior?: unknown } } } | null;
  return [
    ...metricValues(report?.splits?.test?.calibrated, "Test calibrated · "),
    ...metricValues(report?.splits?.test?.class_prior, "Test class prior · "),
  ].filter((item) => /log loss|brier|accuracy|ece|expected calibration error/i.test(item.label));
}

export function firstPredictionDate(dataThrough: string): string | undefined {
  if (!dataThrough) return undefined;
  return new Date(new Date(dataThrough.slice(0, 10)).getTime() + 86400000).toISOString().slice(0, 10);
}

type Values = Record<string, unknown>;

function record(value: unknown): Values {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Values : {};
}

function numeric(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function percentage(value: number | undefined, signed = false): string {
  if (value === undefined) return "—";
  return `${signed && value > 0 ? "+" : ""}${(value * 100).toFixed(2)}${signed ? " pp" : "%"}`;
}

export function metricDisplay(label: string, value: number): string {
  return /accuracy/i.test(label) ? percentage(value) : value.toFixed(4);
}

function summary(value: unknown) {
  const item = record(value);
  const matrix = item.confusion_matrix;
  return {
    accuracy: numeric(item.accuracy),
    logLoss: numeric(item.log_loss),
    drawRecall: numeric(record(record(item.per_class).D).recall),
    confusion: Array.isArray(matrix) && matrix.length === 3 && matrix.every((row) =>
      Array.isArray(row) && row.length === 3 && row.every((cell) => numeric(cell) !== undefined)) ? matrix as number[][] : undefined,
  };
}

function comparison(value: unknown) {
  const item = record(value);
  return summary(item.calibrated ?? item.raw);
}

function calibrationPeriod(value: unknown): string {
  if (typeof value === "string") return value;
  const item = record(value);
  return typeof item.start === "string" && typeof item.end === "string" ? `${item.start} to ${item.end}` : "Not supplied";
}

export function experimentReport(value: unknown) {
  const report = record(value);
  const experiment = record(report.experiment);
  if (!Object.keys(experiment).length) return null;
  const baselineDevelopment = record(experiment.development_baseline);
  const challengerDevelopment = record(experiment.development_selected);
  const baselineSeasons = record(baselineDevelopment.seasons);
  const challengerSeasons = record(challengerDevelopment.seasons);
  const uncertainty = record(experiment.uncertainty);
  const interval = (value: unknown): [number, number] | undefined =>
    Array.isArray(value) && value.length === 2 && value.every((item) => numeric(item) !== undefined) ? value as [number, number] : undefined;
  return {
    modelVersion: typeof report.model_version === "string" ? report.model_version : "Not supplied",
    trainedThrough: typeof report.trained_through === "string" ? report.trained_through : "Not supplied",
    dataThrough: typeof report.data_through === "string" ? report.data_through : "Not supplied",
    calibrationPeriod: calibrationPeriod(report.calibration_period),
    auditStatus: typeof report.audit_status === "string" ? report.audit_status : "Not supplied",
    active: typeof experiment.active === "string" ? experiment.active : "Not supplied",
    promoted: experiment.promoted === true,
    qualified: experiment.development_qualified === true,
    reason: typeof experiment.reason === "string" ? experiment.reason.replaceAll("_", " ") : "No promotion reason supplied.",
    baseline: comparison(experiment.baseline),
    challenger: comparison(experiment.challenger),
    developmentBaseline: summary(baselineDevelopment.pooled),
    developmentChallenger: summary(challengerDevelopment.pooled),
    seasons: [...new Set([...Object.keys(baselineSeasons), ...Object.keys(challengerSeasons)])].sort().map((season) => ({
      season, baseline: summary(baselineSeasons[season]), challenger: summary(challengerSeasons[season]),
    })),
    accuracyDifference: numeric(uncertainty.accuracy_difference),
    accuracyInterval: interval(uncertainty.accuracy_ci95),
    logLossDifference: numeric(uncertainty.log_loss_difference),
    logLossInterval: interval(uncertainty.log_loss_ci95),
  };
}
