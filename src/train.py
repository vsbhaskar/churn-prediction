from pathlib import Path

import joblib
import mlflow
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score, roc_auc_score,
)

from src.data import load_data, clean_data, split_data
from src.evaluate import profit_at
from src.model import build_model, BEST_THRESHOLD

DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = Path("models/churn_model.joblib")


def main():
    df = clean_data(load_data(DATA_PATH))
    X_train, X_test, y_train, y_test = split_data(df)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name="xgboost-baseline"):
        model = build_model(X_train, y_train)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= BEST_THRESHOLD).astype(int)

        # What settings did we use?
        clf_params = model.named_steps["clf"].get_params()
        mlflow.log_params({
            "n_estimators": clf_params["n_estimators"],
            "learning_rate": clf_params["learning_rate"],
            "max_depth": clf_params["max_depth"],
            "threshold": BEST_THRESHOLD,
        })

        # How well did it do?
        metrics = {
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "recall": recall_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "profit": profit_at(proba, y_test, BEST_THRESHOLD),
        }
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        # Save the trained pipeline and attach it to the run
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))

        for name, value in metrics.items():
            print(f"{name:10s} {value:.3f}")


if __name__ == "__main__":
    main()