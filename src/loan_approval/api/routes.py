"""Prediction and explanation endpoints."""
from fastapi import APIRouter

from loan_approval.api.schemas import ApplicantProfile, BankPrediction
from loan_approval.models.predict import predict_all_banks

router = APIRouter()


@router.post("/predict", response_model=list[BankPrediction])
def predict(applicant: ApplicantProfile) -> list[BankPrediction]:
    """Score one applicant at all 3 target banks, ranked most to least likely to approve."""
    return predict_all_banks(applicant.to_features())
