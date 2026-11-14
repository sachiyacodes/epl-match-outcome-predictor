# Data sources, attribution and scope

The MIT license applies to project software. It does not replace third-party data licenses. No API credentials or restricted commercial data are distributed.

## OpenFootball: historical results

- Publisher: [OpenFootball football.json](https://github.com/openfootball/football.json).
- Data license: [CC0-1.0](data/SOURCE-LICENSE.md).
- Snapshot revision: `e6744429ee395bc86f247348c6184bb08d4eb361`.
- Included coverage: 6,080 EPL results, 2010/11–2025/26.
- Provenance: [source manifest](data/audited/manifest.json), [audit report](data/audited/audit.json).
- Changes: explicit team aliases and 66 date corrections applied during loading, with original bytes retained. [Correction ledger](data/audited/corrections.json).
- Correction evidence: [immutable season text](https://github.com/openfootball/eng-england/blob/9ee7a0c351805c99cc334478a9bf04b7d0432fec/2019-20/1-premierleague.txt). Official corroboration: [Arsenal–Norwich, July 1, 2020](https://www.premierleague.com/en/news/1697162).

Completeness and chronology checks are structural safeguards. They do not independently certify every match record.

## Wyscout / Pappalardo et al.: independent reference and events

Luca Pappalardo, Paolo Cintia, Alessio Rossi, Emanuele Massucco, Paolo Ferragina, Dino Pedreschi and Fosca Giannotti (2019), **A public data set of spatio-temporal match events in soccer competitions**, Scientific Data 6, 236. [DOI: 10.1038/s41597-019-0247-7](https://doi.org/10.1038/s41597-019-0247-7). Data collected by Wyscout. [Original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6817871/).

The authors release the dataset itself under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Original records: [Matches](https://figshare.com/articles/dataset/Matches/7770422), [Teams](https://figshare.com/articles/dataset/Teams/7765310), [Events](https://figshare.com/articles/dataset/Events/7770599).

- EPL coverage: 2017/18 only, 380 matches and 643,150 events.
- Match verification: fixture identities, UTC calendar dates and full-time scores match our 380 local records. No fuzzy joins. Only terminal ` FC` name normalization is used.
- Independent coverage: **380 of 6,080 records (6.25%)**. The remaining 5,700 records are not independently confirmed by this comparison.
- Research changes: select EPL events, exclude penalties, convert shot coordinates to a nominal 105×68 metre pitch, estimate non-penalty shot goal probabilities with a chronological project model, aggregate by team/match, and construct strictly lagged features.
- Derived probabilities are project estimates, **not Wyscout's proprietary xG**. Included Wyscout-derived CSVs/reports retain CC BY 4.0 attribution under this notice; they are not relicensed as MIT software.
- Raw downloads remain unchanged in ignored local caches. Original URLs, checksums, code hashes and experiment protocol appear in the generated manifests and reports.
- No provider endorsement or present-day performance guarantee is implied.

Research artifacts: [verification report](data/verification/report.json), [shot-quality protocol](artifacts/shot-quality/protocol.json), [shot-quality report](artifacts/shot-quality/report.json).

## Sources not included

- **StatsBomb / Hudl:** separate research agreement includes commercial-use, redistribution and branding conditions. Neither raw data nor StatsBomb-derived analysis is included in this release. [Agreement](https://github.com/hudl/open-data/blob/master/LICENSE.pdf).
- **Premier League / Opta website:** used for documented reference and correction corroboration. Public website access is not an open bulk-data license. [Terms](https://www.premierleague.com/en/terms-and-conditions).
- **FPL:** public endpoints do not establish automated extraction or training permission. No FPL data is included.
- **Football-Data.co.uk:** no automated collection because its [usage statement](https://football-data.co.uk/data.php) restricts automated training uses.
- **Commercial Opta or other paid feeds:** no account, purchase, credentials or licensed export is included.

No continuous contemporary EPL xG, injury or lineup history has been verified and added. All outcome forecasts use information available strictly before the fixture date. Better data provenance does not establish higher prediction accuracy.
