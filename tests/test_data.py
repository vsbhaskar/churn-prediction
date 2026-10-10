import pandas as pd

from src.data import clean_data, split_data


def make_raw_df(n=100, n_churn=20):
    """A tiny fake dataset shaped like the raw Telco CSV."""
    df = pd.DataFrame({
        "customerID": [f"id-{i}" for i in range(n)],
        "tenure": range(n),
        "MonthlyCharges": [50.0] * n,
        "TotalCharges": [str(50.0 * i) for i in range(n)],
        "Contract": ["Month-to-month"] * n,
        "Churn": ["Yes"] * n_churn + ["No"] * (n - n_churn),
    })
    df.loc[0, "TotalCharges"] = " "  # the blank string from the real data
    return df


def test_blank_total_charges_become_zero():
    cleaned = clean_data(make_raw_df())
    assert cleaned["TotalCharges"].dtype == "float64"
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert cleaned.loc[0, "TotalCharges"] == 0


def test_churn_is_converted_to_0_and_1():
    cleaned = clean_data(make_raw_df())
    assert set(cleaned["Churn"].unique()) == {0, 1}
    assert cleaned["Churn"].sum() == 20


def test_customer_id_is_dropped():
    assert "customerID" not in clean_data(make_raw_df()).columns


def test_clean_data_does_not_modify_the_input():
    raw = make_raw_df()
    clean_data(raw)
    assert raw.loc[0, "TotalCharges"] == " "
    assert "customerID" in raw.columns


def test_split_keeps_churn_ratio_and_sizes():
    X_train, X_test, y_train, y_test = split_data(clean_data(make_raw_df()))
    assert len(X_train) == 80 and len(X_test) == 20
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    assert "Churn" not in X_train.columns