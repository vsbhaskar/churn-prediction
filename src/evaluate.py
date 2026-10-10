import numpy as np

# Business assumptions (made-up numbers, stated in the README)
COST_OFFER = 20      # $ per retention offer sent
VALUE_SAVED = 400    # $ value of a customer we keep
SUCCESS_RATE = 0.30  # share of would-be churners an offer actually saves


def profit_at(proba, y_true, threshold):
    """Expected profit if we send offers to everyone scoring >= threshold."""
    y = np.asarray(y_true)
    flagged = np.asarray(proba) >= threshold
    caught = (flagged & (y == 1)).sum()
    return caught * SUCCESS_RATE * VALUE_SAVED - flagged.sum() * COST_OFFER