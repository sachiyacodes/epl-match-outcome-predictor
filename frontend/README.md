# Touchline dashboard

Requires Node.js 20.9 or later and the backend running on `http://127.0.0.1:8000`.

In PowerShell, from this directory:

```powershell
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open http://127.0.0.1:3000. Ingest data and train the backend model before requesting predictions. An unavailable backend produces an error state; the app does not substitute sample predictions.

For a different backend address, edit `NEXT_PUBLIC_API_URL` in `.env.local` before starting or building. The backend must allow this dashboard's origin with CORS. No credentials belong in this public variable.

```powershell
npm run typecheck
npm test
npm run build
npm start
```

The dashboard uses `/teams`, `/metrics`, and `/predict`. The report displays numeric log loss, Brier score, accuracy, and ECE values when supplied, plus the complete evaluation JSON for calibration and split metadata.

The focused test command requires Node.js 22.18 or later for native TypeScript execution. Node.js 24 LTS is recommended for development.
