"""Per-prediction SHAP explanations, in plain-language adverse-action-style reasons.

ECOA / Regulation B requires lenders to give applicants the *specific* reasons behind
a credit denial, not just a score. `explain()` gives the same treatment to a model
prediction: it runs SHAP's TreeExplainer on the trained RandomForest, collapses the
one-hot-encoded columns back to the original HMDA fields the applicant actually
filled in, and returns the top factors in plain English, ranked by how much each one
pushed the prediction up or down.

`lei` (which bank) is excluded from the returned reasons on purpose: it's not
something about the applicant, so it isn't a "reason" in the adverse-action sense --
see docs/model_card.md for why a bank's own historical pattern is handled separately
(disparate_impact.py), not surfaced as an applicant-facing factor.
"""
from functools import lru_cache

import pandas as pd
import shap

from loan_approval.models.predict import _load

CREDIT_SCORE_TYPES = {
    1: "Equifax Beacon 5.0", 2: "Experian Fair Isaac", 3: "FICO Risk Score Classic 04",
    4: "FICO Risk Score Classic 98", 5: "VantageScore 2.0", 6: "VantageScore 3.0",
    7: "more than one credit scoring model", 8: "another credit scoring model",
    9: "an unknown credit scoring model", 10: "no co-applicant",
}
AUS_SYSTEMS = {
    -1: "not used", 1: "Desktop Underwriter (DU)", 2: "Loan Prospector / Loan Product Advisor (LPA)",
    3: "the TOTAL Scorecard", 4: "the Guaranteed Underwriting System (GUS)", 5: "another AUS",
    6: "no automated underwriting", 7: "an internal proprietary system",
}
CATEGORY_LABELS = {
    "loan_type": {1: "a conventional loan", 2: "an FHA-insured loan", 3: "a VA-guaranteed loan", 4: "a USDA/RHS loan"},
    "loan_purpose": {
        1: "a home purchase", 2: "home improvement", 31: "a refinance", 32: "a cash-out refinance",
        4: "another loan purpose", 5: "an unspecified loan purpose",
    },
    "occupancy_type": {1: "a principal residence", 2: "a second residence", 3: "an investment property"},
    "applicant_credit_score_type": CREDIT_SCORE_TYPES,
    "co-applicant_credit_score_type": CREDIT_SCORE_TYPES,
    "interest_only_payment": {1: "interest-only payments", 2: "no interest-only payments"},
    "balloon_payment": {1: "a balloon payment", 2: "no balloon payment"},
    "aus-1": AUS_SYSTEMS, "aus-2": AUS_SYSTEMS, "aus-3": AUS_SYSTEMS,
}
NUMERIC_LABELS = {
    "loan_amount": lambda v: f"a loan amount of ${v:,.0f}",
    "loan_term": lambda v: f"a {v:.0f}-month loan term",
    "income": lambda v: f"income of ${v:,.0f}k/yr",
    "debt_to_income_ratio": lambda v: f"a debt-to-income ratio of {v:.0f}%",
    "num_aus_used": lambda v: f"{v:.0f} automated underwriting system(s) used",
}


def _describe(feature: str, value) -> str:
    """Turn a raw HMDA field + code into the plain-language phrase a reason is built from."""
    if feature in NUMERIC_LABELS:
        return NUMERIC_LABELS[feature](value)
    return CATEGORY_LABELS.get(feature, {}).get(value, f"{feature} = {value}")


def _group_by_field(feature_names, shap_row) -> dict[str, float]:
    """Collapse one-hot dummy columns back to the original HMDA field, summing SHAP
    contributions -- a categorical field's total effect is the sum across its dummies.
    'ohe__loan_type_2' -> 'loan_type'; 'scaler__loan_amount' -> 'loan_amount'."""
    grouped: dict[str, float] = {}
    for name, value in zip(feature_names, shap_row):
        prefix, rest = name.split("__", 1)
        field = rest.rsplit("_", 1)[0] if prefix == "ohe" else rest
        grouped[field] = grouped.get(field, 0.0) + value
    return grouped


@lru_cache(maxsize=1)
def _explainer(forest) -> shap.TreeExplainer:
    """Building a TreeExplainer walks all 500 trees once; cache it like _load() caches
    the model itself, so repeated calls in the same process only pay that cost once."""
    return shap.TreeExplainer(forest)


def explain(applicant: dict, lei: str, top_n: int = 5) -> dict:
    """Score one applicant at one bank and return the top `top_n` factors behind it.

    Mirrors `predict.predict()`'s scoring, plus a plain-language breakdown of why.
    """
    model, thresholds = _load()
    preprocessor, forest = model.named_steps["preprocess"], model.named_steps["model"]

    row = pd.DataFrame([{**applicant, "lei": lei}])
    transformed = preprocessor.transform(row)
    probability = float(forest.predict_proba(transformed)[:, 1][0])
    threshold = float(thresholds.get(lei, 0.5))

    shap_values = _explainer(forest).shap_values(transformed)[0, :, 1]
    contributions = _group_by_field(preprocessor.get_feature_names_out(), shap_values)
    contributions.pop("lei", None)  # which bank isn't a reason about the applicant

    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    reasons = [
        {
            "field": field,
            "description": _describe(field, applicant.get(field)),
            "effect": "increased" if impact > 0 else "decreased",
            "shap_value": round(float(impact), 4),
        }
        for field, impact in ranked
    ]
    return {
        "lei": lei,
        "probability": probability,
        "threshold": threshold,
        "approved": bool(probability >= threshold),
        "reasons": reasons,
    }


if __name__ == "__main__":
    from loan_approval.config import BANK_LEIS

    sample_applicant = {
        "loan_type": 1, "loan_purpose": 1, "loan_amount": 250_000.0, "loan_term": 360.0,
        "occupancy_type": 1, "income": 90.0, "debt_to_income_ratio": 33.0,
        "applicant_credit_score_type": 3, "co-applicant_credit_score_type": 10,
        "interest_only_payment": 2, "negative_amortization": 2, "balloon_payment": 2,
        "aus-1": 1, "aus-2": -1, "aus-3": -1, "num_aus_used": 1,
    }
    lei = BANK_LEIS["Wells Fargo Bank, N.A."]
    result = explain(sample_applicant, lei)

    print(f"Probability of approval: {result['probability']:.1%} (bank's threshold: {result['threshold']:.1%})")
    print(f"Predicted: {'APPROVED' if result['approved'] else 'DENIED'}\n")
    print("Principal factors:")
    for reason in result["reasons"]:
        print(f"  - {reason['description']} {reason['effect']} the odds of approval ({reason['shap_value']:+.3f})")
