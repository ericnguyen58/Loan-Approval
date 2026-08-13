"""Load the trained model and score applicant profiles against each bank's calibrated threshold."""
import joblib
import pandas as pd

from loan_approval.config import BANK_LEIS, MODEL_DIR

model = joblib.load(MODEL_DIR / "best_model.joblib")
thresholds = joblib.load(MODEL_DIR / "thresholds.joblib")

# IF the bank isn't in the dataset
DEFAULT_THRESHOLD = 0.5


def predict(applicant: dict, lei: str) -> dict:
    """Score one applicant at one bank, using that bank's calibrated threshold.

    `applicant` holds every feature column except `lei` (see
    data/processed/hmda_features.csv for the full set).
    """
    row = pd.DataFrame([{**applicant, "lei": lei}])
    probability = float(model.predict_proba(row)[:, 1][0])
    threshold = float(thresholds.get(lei, DEFAULT_THRESHOLD))
    return {
        "lei": lei,
        "probability": probability,
        "threshold": threshold,
        "approved": bool(probability >= threshold),
    }


def predict_all_banks(applicant: dict) -> list[dict]:
    """Score one applicant at all 3 target banks, ranked most to least likely to approve."""
    results = [{"bank": bank, **predict(applicant, lei)} for bank, lei in BANK_LEIS.items()]
    return sorted(results, key=lambda r: r["probability"], reverse=True)
