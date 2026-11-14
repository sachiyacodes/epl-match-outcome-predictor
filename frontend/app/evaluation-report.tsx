import { experimentReport, percentage } from "./dashboard-data";

const outcomes = ["Away", "Draw", "Home"];

function Confusion({ title, matrix }: { title: string; matrix: number[][] | undefined }) {
  if (!matrix) return null;
  return <div className="report-table"><table><caption>{title}: actual versus predicted results</caption><thead><tr><th scope="col">Actual</th>{outcomes.map((outcome) => <th key={outcome} scope="col">Pred. {outcome}</th>)}</tr></thead><tbody>{matrix.map((row, index) => <tr key={index}><th scope="row">{outcomes[index]}</th>{row.map((count, column) => <td key={column}>{count}</td>)}</tr>)}</tbody></table></div>;
}

export default function EvaluationReport({ metrics }: { metrics: unknown }) {
  const report = experimentReport(metrics);
  if (!report) return null;
  const accuracyInterval = report.accuracyInterval?.map((value) => percentage(value, true)).join(" to ");
  const logLossInterval = report.logLossInterval?.map((value) => value.toFixed(4)).join(" to ");
  return <div className="experiment-report">
    <section className="decision" aria-labelledby="decision-title">
      <p className="eyebrow">MODEL DECISION</p>
      <h3 id="decision-title">{report.promoted ? "Challenger promoted" : "Baseline retained"}</h3>
      <p>{report.reason}</p>
      <p className="note">Development gate: {report.qualified ? "passed" : "not passed"}. Serving: {report.active}. An experiment does not automatically replace the active model.</p>
    </section>
    <dl className="report-metadata">
      {[["Model version", report.modelVersion], ["Dataset audit", report.auditStatus], ["Training through", report.trainedThrough], ["Results through", report.dataThrough], ["Calibration period", report.calibrationPeriod]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
    </dl>
    <p className="note">Metadata describes the refitted serving model. Confirmation scores use the preceding season's frozen calibration, not the serving model's in-sample performance.</p>
    <section aria-labelledby="confirmation-title">
      <h3 id="confirmation-title">Final confirmation</h3>
      <p className="note">The later season checks the development-selected challenger against the same audited baseline. These results are separate from model selection.</p>
      <div className="report-table"><table><caption>All-match confirmation results</caption><thead><tr><th scope="col">Model</th><th scope="col">Accuracy</th><th scope="col">Log loss ↓</th><th scope="col">Draw recall</th></tr></thead><tbody>{([["Baseline", report.baseline], ["Challenger", report.challenger]] as const).map(([label, result]) => <tr key={label}><th scope="row">{label}</th><td>{percentage(result.accuracy)}</td><td>{result.logLoss?.toFixed(4) ?? "—"}</td><td>{percentage(result.drawRecall)}</td></tr>)}</tbody></table></div>
      <div className="intervals"><p><strong>Accuracy difference:</strong> {percentage(report.accuracyDifference, true)}<span>95% weekly-block interval: {accuracyInterval ?? "not supplied"}</span></p><p><strong>Log-loss difference:</strong> {report.logLossDifference?.toFixed(4) ?? "—"}<span>95% weekly-block interval: {logLossInterval ?? "not supplied"}</span></p></div>
      <p className="note">Differences are challenger minus baseline. Positive accuracy and negative log loss favor the challenger. An accuracy interval crossing zero is inconclusive.</p>
    </section>
    <details><summary>Development seasons and draw diagnostics</summary>
      <p className="note">Development scores guide feature and model selection; they are not an independent confirmation.</p>
      <div className="report-table"><table><caption>Development accuracy by season</caption><thead><tr><th scope="col">Season</th><th scope="col">Baseline</th><th scope="col">Challenger</th></tr></thead><tbody><tr><th scope="row">Pooled development</th><td>{percentage(report.developmentBaseline.accuracy)}</td><td>{percentage(report.developmentChallenger.accuracy)}</td></tr>{report.seasons.map((row) => <tr key={row.season}><th scope="row">{row.season}</th><td>{percentage(row.baseline.accuracy)}</td><td>{percentage(row.challenger.accuracy)}</td></tr>)}</tbody></table></div>
      <p className="note">Draw recall measures how many actual draws the model selects. Higher draw recall alone does not guarantee better total accuracy or probabilities.</p>
      <Confusion title="Baseline confirmation" matrix={report.baseline.confusion} />
      <Confusion title="Challenger confirmation" matrix={report.challenger.confusion} />
    </details>
  </div>;
}
