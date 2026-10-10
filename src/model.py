from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import build_preprocessor

# Chosen in notebook 05 by maximizing expected profit on the training data
BEST_THRESHOLD = 0.34


def build_model(X_train, y_train):
    """Create the full pipeline (preprocessing + XGBoost), not yet trained."""
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    return Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            scale_pos_weight=ratio, random_state=42, eval_metric="logloss",
        )),
    ])