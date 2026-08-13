from fastapi.testclient import TestClient

from loan_approval.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict(mock_model):
    applicant = {
        "loan_type": 1, "loan_purpose": 1, "loan_amount": 250000, "loan_term": 360,
        "occupancy_type": 1, "income": 90, "debt_to_income_ratio": 33,
        "applicant_credit_score_type": 3, "co-applicant_credit_score_type": 10,
        "interest_only_payment": 2, "negative_amortization": 2, "balloon_payment": 2,
        "aus-1": 1, "aus-2": -1, "aus-3": -1, "num_aus_used": 1,
    }
    response = client.post("/predict", json=applicant)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 3
    assert all(r["probability"] == 0.7 for r in results)
    assert all(r["approved"] is True for r in results)  # 0.7 clears the 0.5 fallback threshold


def test_predict_rejects_out_of_range_field():
    applicant = {
        "loan_type": 9, "loan_purpose": 1, "loan_amount": 250000, "loan_term": 360,
        "occupancy_type": 1, "income": 90, "debt_to_income_ratio": 33,
        "applicant_credit_score_type": 3, "co-applicant_credit_score_type": 10,
        "interest_only_payment": 2, "negative_amortization": 2, "balloon_payment": 2,
        "aus-1": 1, "aus-2": -1, "aus-3": -1, "num_aus_used": 1,
    }
    response = client.post("/predict", json=applicant)
    assert response.status_code == 422
