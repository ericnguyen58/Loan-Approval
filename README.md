# Let Me Have a Loan

[![CI](https://github.com/ericnguyen58/Loan-Approval/actions/workflows/ci.yml/badge.svg)](https://github.com/ericnguyen58/Loan-Approval/actions/workflows/ci.yml)

Predicts mortgage approval odds at Truist, Wells Fargo, and Bank of America from real historical lending data (HMDA), and explains every prediction in plain language. Built to cover what a bank needs from an applied ML system beyond an accuracy number: a calibrated model, per prediction explainability, and a documented fair lending audit.

Educational project. Not used for any real credit decision. See `docs/model_card.md`.

## What it does

- Trains a classifier on public HMDA mortgage application records (63,410 held out test applications, 3 lenders) and serves it through a FastAPI endpoint.
- Calibrates a separate approval threshold per bank, so a prediction reflects that bank's own historical approval rate instead of one shared cutoff across all three.
- Explains each prediction with SHAP, formatted as the "principal reasons" language ECOA/Regulation B requires on an adverse action notice.
- Audits the trained model for disparate impact across race, ethnicity, and sex using the EEOC four fifths rule, even though none of those fields are used to train it.
- Validates every request against the value ranges the model was trained on, and returns a field level error message instead of a generic failure.

## Tech stack

Python, scikit-learn, pandas, SHAP, FastAPI, pytest, Docker, uv, GitHub Actions.

## Results

Random forest, selected over logistic regression, tuned with cross validated ROC AUC rather than accuracy: a model that approved every applicant would still score about 70% accuracy on this data, so accuracy alone doesn't say much.

| Bank | Test AUC | Actual approval rate | Predicted approval rate (calibrated) |
|---|---|---|---|
| Truist | 0.855 | 74.6% | 74.8% |
| Wells Fargo | 0.964 | 71.1% | 71.0% |
| Bank of America | 0.906 | 62.7% | 62.7% |

Pooled test accuracy: 83.5%, ROC AUC: 0.918. Full model comparison, per bank breakdown, and why a single global threshold doesn't work for this product are in `src/loan_approval/models/README.md`.

## Example: prediction API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_type": 1, "loan_purpose": 1, "loan_amount": 250000, "loan_term": 360,
    "occupancy_type": 1, "income": 90, "debt_to_income_ratio": 33,
    "applicant_credit_score_type": 3, "co-applicant_credit_score_type": 10,
    "interest_only_payment": 2, "negative_amortization": 2, "balloon_payment": 2,
    "aus-1": 1, "aus-2": -1, "aus-3": -1, "num_aus_used": 1
  }'
```

```json
[
  { "bank": "Wells Fargo Bank, N.A.", "probability": 0.99, "threshold": 0.33, "approved": true },
  { "bank": "Truist Bank", "probability": 0.87, "threshold": 0.50, "approved": true },
  { "bank": "Bank of America, N.A.", "probability": 0.84, "threshold": 0.35, "approved": true }
]
```

## Example: explainability

`src/loan_approval/explainability/explain.py` runs SHAP on the trained model, then maps the top contributing features back to plain language:

```
$ uv run python -m loan_approval.explainability.explain

Probability of approval: 99.3% (bank's threshold: 33.3%)
Predicted: APPROVED

Principal factors:
  - Desktop Underwriter (DU) increased the odds of approval (+0.173)
  - a debt-to-income ratio of 33% increased the odds of approval (+0.089)
  - a loan amount of $250,000 increased the odds of approval (+0.071)
  - a home purchase increased the odds of approval (+0.047)
  - FICO Risk Score Classic 04 increased the odds of approval (+0.038)
```

## Fair lending audit

Race, ethnicity, and sex are dropped before training and reattached only afterward, to check whether predictions land unevenly across groups. Full methodology and findings are in `docs/model_card.md`; summary below.

| Group | Actual approval | Predicted approval | Adverse impact ratio |
|---|---|---|---|
| Black or African American | 52.7% | 54.8% | 0.715 (flagged) |
| Female | 63.3% | 59.6% | 0.745 (flagged) |
| Hispanic or Latino | 54.5% | 57.7% | 0.779 (flagged) |

The model tracks each group's historical approval rate closely, so it isn't introducing new disparity. It also isn't correcting for the disparity already present in the historical data, and it shows a higher false negative rate for Black and Female applicants than for White and Male applicants. Neither finding blocks a demo project, but both are the kind of gap that would need real mitigation before a model like this could support an actual credit decision.

## Project structure

```
src/loan_approval/
  data/            ingest and preprocess HMDA records
  models/          train and score per lender approval models
  explainability/  SHAP explanations per prediction, plus the fair lending audit
  api/             FastAPI service
data/              raw and processed HMDA data (gitignored, see data/README.md)
models/            trained model artifacts (gitignored)
deployment/        Dockerfile for the API
docs/              model card (regulatory and fair lending documentation)
tests/
```

## Running it

```bash
uv sync --extra dev
uv run uvicorn loan_approval.api.main:app --reload
```

Health check: `GET /health`.

Docker, built and run from the repo root:

```bash
docker build -t loan-approval -f deployment/Dockerfile .
docker run -p 8000:8000 loan-approval
```

## Testing

```bash
uv run pytest
```

The test suite mocks the trained model, so it runs without the model artifacts on disk (see `tests/conftest.py`). The same suite runs in CI on every push.
