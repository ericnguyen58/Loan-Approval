# Let Me Have a Loan

Estimates your likelihood of mortgage approval at specific banks (Truist,
Wells Fargo, Bank of America), trained on real historical approval data
(HMDA), with per-prediction explainability and fair-lending awareness baked
into the design.

## Structure

```
src/loan_approval/
  data/            ingest + preprocess HMDA records
  models/          train / predict per-lender approval models
  explainability/  SHAP-based, plain-language reasons for a prediction
  api/             FastAPI service
data/              raw/processed HMDA data (gitignored, see data/README.md)
models/            trained model artifacts (gitignored)
deployment/        Dockerfile for the API
docs/              model card (regulatory/fair-lending documentation)
tests/
```

## Setup

```
uv sync --extra dev
uv run uvicorn loan_approval.api.main:app --reload
```

Health check: `GET /health`.
