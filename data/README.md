# Data

Source: [CFPB HMDA Platform](https://ffiec.cfpb.gov/data-browser/) loan/application
records (LAR), filtered to Truist, Wells Fargo, and Bank of America via their
lender LEI codes.

- `raw/` — unmodified HMDA extracts (CSV/Parquet), not committed (see `.gitignore`).
- `processed/` — cleaned, feature-engineered tables produced by
  `loan_approval.data.preprocess`, also not committed.

Re-download raw data with `loan_approval.data.ingest` before running
`src/loan_approval/models/train.ipynb`.
