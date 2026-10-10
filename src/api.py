from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI

from src.db import init_db, log_prediction
from src.model import BEST_THRESHOLD, MODEL_VERSION
from src.schemas import CustomerInput, PredictionOut, Reason

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"

state = {}  # holds the model and explainer while the app runs


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts
    pipeline = joblib.load(MODEL_PATH)
    prep = pipeline.named_steps["prep"]
    state["pipeline"] = pipeline
    state["explainer"] = shap.TreeExplainer(pipeline.named_steps["clf"])
    state["feature_names"] = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    init_db()
    yield
    state.clear()  # runs when the server stops


app = FastAPI(title="Churn Prediction API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in state}


@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerInput):
    pipeline = state["pipeline"]
    row = pd.DataFrame([customer.model_dump()])

    risk_score = float(pipeline.predict_proba(row)[0, 1])

    X_t = pipeline.named_steps["prep"].transform(row)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    contributions = state["explainer"].shap_values(X_t)[0]

    # Add up SHAP values per ORIGINAL column (all Contract_* columns together)
    grouped = {}
    for name, value in zip(state["feature_names"], contributions):
        source_col = name.split("_", 1)[0]
        grouped[source_col] = grouped.get(source_col, 0.0) + float(value)

    top = sorted(grouped.items(), key=lambda pair: abs(pair[1]), reverse=True)[:3]

    reasons = [
        Reason(
            feature=col,
            value=str(row.iloc[0][col]),
            impact=round(impact, 3),
            direction="increases churn risk" if impact > 0 else "decreases churn risk",
        )
        for col, impact in top
    ]

    response = PredictionOut(
        risk_score=round(risk_score, 3),
        flag_for_offer=risk_score >= BEST_THRESHOLD,
        threshold=BEST_THRESHOLD,
        top_reasons=reasons,
    )

    log_prediction(
        model_version=MODEL_VERSION,
        risk_score=response.risk_score,
        flag_for_offer=response.flag_for_offer,
        threshold=response.threshold,
        features=customer.model_dump(),
        top_reasons=[r.model_dump() for r in response.top_reasons],
    )
    return response