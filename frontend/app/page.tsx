"use client";

import { FormEvent, useEffect, useState } from "react";
import { firstPredictionDate, metricDisplay, testMetricCards } from "./dashboard-data";
import EvaluationReport from "./evaluation-report";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
type Prediction = { probabilities: { H: number; D: number; A: number }; [key: string]: unknown };

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, { ...options, signal: AbortSignal.timeout(30000) });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status}). Check the API logs.`);
  }
  return body as T;
}

export default function Home() {
  const [teams, setTeams] = useState<string[]>([]);
  const [dataThrough, setDataThrough] = useState("");
  const [metrics, setMetrics] = useState<unknown>(null);
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const [date, setDate] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [metricsError, setMetricsError] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [fixture, setFixture] = useState("");

  useEffect(() => {
    let active = true;
    request<{ teams: string[]; data_through: string }>("/teams")
      .then((data) => { if (active) { setTeams(data.teams); setDataThrough(data.data_through); } })
      .catch(() => { if (active) setError("Cannot reach the prediction API. Start the backend and train a model using the README instructions."); })
      .finally(() => { if (active) setLoading(false); });
    request<unknown>("/metrics")
      .then((data) => { if (active) setMetrics(data); })
      .catch(() => { if (active) setMetricsError("Evaluation is unavailable. Run the training pipeline, then reload this page."); });
    return () => { active = false; };
  }, []);

  async function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(""); setPrediction(null);
    if (home === away) { setError("Choose two different teams."); return; }
    setBusy(true);
    try {
      const result = await request<Prediction>("/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ home_team: home, away_team: away, match_date: date }),
      });
      setPrediction(result); setFixture(`${home} vs ${away} · ${date}`);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Prediction failed. Check the API connection.");
    } finally { setBusy(false); }
  }

  const cards = testMetricCards(metrics);

  return (
    <main>
      <header className="topbar"><a className="brand" href="/" aria-label="Touchline home"><span className="brand-mark">T</span> TOUCHLINE</a><span className="status">Historical EPL · Research MVP</span></header>
      <section className="intro"><p className="eyebrow">PREMIER LEAGUE / MATCH INTELLIGENCE</p><h1>The next match.<br /><span>In probabilities.</span></h1><p className="lede">Explore home, draw and away chances from historical results. Pre-match features. Chronological evaluation. No certainty promised.</p></section>
      <div className="workspace">
        <section className="panel" aria-labelledby="fixture-title">
          <div className="section-heading"><span className="step">01</span><div><h2 id="fixture-title">Choose a fixture</h2><p>Use the team names available in the dataset.</p></div></div>
          <form onSubmit={predict}>
            <div className="team-fields"><label>Home team<select required value={home} disabled={loading || !teams.length || busy} onChange={(event) => { setHome(event.target.value); setPrediction(null); }}><option value="">{loading ? "Loading teams…" : "Select home team"}</option>{teams.map((team) => <option key={team}>{team}</option>)}</select></label><span className="versus" aria-hidden="true">VS</span><label>Away team<select required value={away} disabled={loading || !teams.length || busy} onChange={(event) => { setAway(event.target.value); setPrediction(null); }}><option value="">Select away team</option>{teams.map((team) => <option key={team}>{team}</option>)}</select></label></div>
            <label className="date-field">Match date<input type="date" required min={firstPredictionDate(dataThrough)} value={date} disabled={busy} onChange={(event) => { setDate(event.target.value); setPrediction(null); }} /></label>
            <button type="submit" disabled={loading || busy || !teams.length}>{busy ? "Calculating…" : "Predict match outcome"}<span aria-hidden="true">↗</span></button>
          </form>
          <div aria-live="polite">{error && <p role="alert" className="error">{error}</p>}{!loading && !teams.length && !error && <p className="note">No teams available. Ingest historical results and train the model first.</p>}</div>
          <p className="data-note">{dataThrough ? `Results available through ${dataThrough}. Choose a date after this snapshot.` : "Waiting for historical data."} Predictions use only eligible past matches.</p>
        </section>
        <section className="panel result-panel" aria-labelledby="result-title" aria-live="polite" aria-busy={busy}>
          <div className="section-heading"><span className="step">02</span><div><h2 id="result-title">Outcome probabilities</h2><p>{prediction ? fixture : "Your prediction will appear here."}</p></div></div>
          {prediction ? <div className="probabilities">{([['H', 'Home win'], ['D', 'Draw'], ['A', 'Away win']] as const).map(([key, label]) => <div className="probability" key={key}><div className="probability-label"><span>{label}</span><strong>{(prediction.probabilities[key] * 100).toFixed(1)}<small>%</small></strong></div><div className="track"><div className={`fill fill-${key}`} style={{ width: `${prediction.probabilities[key] * 100}%` }} /></div></div>)}</div> : <div className="empty-state"><div className="pitch" aria-hidden="true"><span /></div><p>{busy ? "Reading the match history…" : "A fixture, three possible outcomes."}</p><span>Choose teams and a date to get started.</span></div>}
          {prediction && typeof prediction.warning === "string" && <p className="error" role="status">{prediction.warning}</p>}
          <p className="note">Probabilities describe uncertainty. A 60% outcome can still lose. This model does not use live lineups, injuries or odds.</p>
        </section>
      </div>
      <section className="evaluation" aria-labelledby="evaluation-title"><div className="section-heading"><span className="step">03</span><div><h2 id="evaluation-title">Measured on the past</h2><p>Time-based holdouts keep future results out of training.</p></div></div>{metricsError && <p className="error">{metricsError}</p>}{cards.length > 0 && <div className="metrics-grid">{cards.slice(0, 12).map((metric) => <article className="metric" key={metric.label}><p>{metric.label}</p><strong>{metricDisplay(metric.label, metric.value)}</strong></article>)}</div>}{metrics !== null && <><p className="audit-caution">The original 53.42% benchmark is superseded: incorrect upstream fixture dates compromised its chronology. Use audited results for comparisons.</p><EvaluationReport metrics={metrics} /><details><summary>View full evaluation and calibration report</summary><pre>{JSON.stringify(metrics, null, 2)}</pre></details></>}{!metrics && !metricsError && <p className="note" role="status">Loading evaluation…</p>}<p className="note">Lower log loss and Brier score are better. Calibration compares predicted probability with observed frequency.</p></section>
      <footer>Built from real historical results.<span>Experimental model · Not betting advice</span></footer>
    </main>
  );
}
