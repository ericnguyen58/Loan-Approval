"""Pydantic request/response models for the prediction API."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ApplicantProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    loan_type: Literal[1, 2, 3, 4]  # 1=Conventional, 2=FHA, 3=VA, 4=RHS/FSA
    loan_purpose: Literal[1, 2, 4, 5, 31, 32]
    loan_amount: float = Field(gt=0, le=5_000_000)
    loan_term: float = Field(ge=1, le=600)  # months
    occupancy_type: Literal[1, 2, 3]  # 1=Principal, 2=Second, 3=Investment
    income: float = Field(ge=0, le=10_000)  # thousands/yr
    debt_to_income_ratio: float = Field(ge=0, le=100)  # percent
    applicant_credit_score_type: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9]
    co_applicant_credit_score_type: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = Field(
        alias="co-applicant_credit_score_type"
    )  # 10 = no co-applicant
    interest_only_payment: Literal[1, 2]  # 1=Yes, 2=No
    negative_amortization: Literal[1, 2]
    balloon_payment: Literal[1, 2]
    aus_1: Literal[-1, 1, 2, 3, 4, 5, 6, 7] = Field(alias="aus-1")  # -1 = unused
    aus_2: Literal[-1, 1, 2, 3, 4, 5, 6, 7] = Field(alias="aus-2")
    aus_3: Literal[-1, 1, 2, 3, 4, 5, 6, 7] = Field(alias="aus-3")
    num_aus_used: int = Field(ge=0, le=3)

    def to_features(self) -> dict:
        """Dict keyed exactly as the trained pipeline expects (hyphenated column names)."""
        return self.model_dump(by_alias=True)


class BankPrediction(BaseModel):
    bank: str
    lei: str
    probability: float
    threshold: float
    approved: bool
