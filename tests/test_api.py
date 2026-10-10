import pytest
from fastapi.testclient import TestClient

import src.api as api_module

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.1, "TotalCharges": 95.1,
}

LOW_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year",
    "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 60.0, "TotalCharges": 3600.0,
}


@pytest.fixture
def client(monkeypatch):
    """A test client whose database calls are replaced by harmless fakes."""
    logged = []
    monkeypatch.setattr(api_module, "init_db", lambda: None)
    monkeypatch.setattr(api_module, "log_prediction", lambda **kw: logged.append(kw))
    with TestClient(api_module.app) as c:  # "with" runs the startup code
        c.logged = logged
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_expected_shape(client):
    response = client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["flag_for_offer"], bool)
    assert len(body["top_reasons"]) == 3


def test_high_risk_customer_scores_above_low_risk(client):
    high = client.post("/predict", json=HIGH_RISK).json()["risk_score"]
    low = client.post("/predict", json=LOW_RISK).json()["risk_score"]
    assert high > low
    assert high >= 0.34 > low  # flagged vs not flagged


def test_invalid_contract_is_rejected(client):
    bad = {**HIGH_RISK, "Contract": "Monthly"}
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_is_rejected(client):
    bad = {k: v for k, v in HIGH_RISK.items() if k != "tenure"}
    assert client.post("/predict", json=bad).status_code == 422


def test_each_prediction_is_logged_once(client):
    client.post("/predict", json=HIGH_RISK)
    assert len(client.logged) == 1
    assert client.logged[0]["features"]["Contract"] == "Month-to-month"