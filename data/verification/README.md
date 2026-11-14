# Independent source verification

This report is separate from model training and serving. It never repairs or replaces a record automatically.

## Reproduce

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'backend'
.\.venv\Scripts\python.exe -m epl.verify_sources
.\.venv\Scripts\python.exe -m epl.verify_sources --offline
```

The first run downloads the original match ZIP and team metadata into ignored `data/external/wyscout/`. The offline run requires those cached bytes. Both commands verify pinned SHA256 values, exact fixture identities, calendar dates and full-time scores, then write `data/verification/report.json`. A conflict returns a failing exit status; inspect the report rather than silently modifying data. Tests use synthetic source fixtures and need no network.

## Source and attribution

Data source: Wyscout, released by Luca Pappalardo, Paolo Cintia, Alessio Rossi, Emanuele Massucco, Paolo Ferragina, Dino Pedreschi and Fosca Giannotti (2019), **A public data set of spatio-temporal match events in soccer competitions**, Scientific Data 6, 236. DOI: [10.1038/s41597-019-0247-7](https://doi.org/10.1038/s41597-019-0247-7). [Original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6817871/).

The dataset itself is licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Original records: [Matches](https://figshare.com/articles/dataset/Matches/7770422), [Teams](https://figshare.com/articles/dataset/Teams/7765310). We derive a comparison report; raw source bytes remain unchanged. Changes for comparison: remove terminal ` FC` from club names, use UTC calendar dates, and select EPL 2017/18. No fuzzy name matching or result-dependent date correction is used.

## Coverage and limits

The independent Wyscout source covers one EPL season, 2017/18. Agreement verifies only the listed fixture identities, dates and full-time scores. It does not prove every provider record correct, independently validate other seasons, or establish higher prediction accuracy.

StatsBomb/Hudl 2015/16 is not included in this permissively licensed release. Its separate agreement permits research conclusions but restricts commercial use and raw redistribution, requests registration and requires branding on published analysis. No StatsBomb raw data or derived verification is bundled. [Source agreement](https://github.com/hudl/open-data/blob/master/LICENSE.pdf).
