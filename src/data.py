import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) ->pd.DataFrame:
    """Read the raw csv file into a dataframe"""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known problems in the raw data"""
    df = df.copy() # never modify the original dataframe

    # TotalCharges has blank strings for customers with tenure == 0.
    # They have been billed nothing so far, so 0 is the correct value.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors = "coerce").fillna(0)

    # The target column: Yes/No -> 1/0 (models need numbers)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # The target column: Yes/No -> 1/0 (models need numbers)
    df = df.drop(columns = ["customerID"])
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split into train and test sets, keeping the churn ratio equal in both."""
    X = df.drop(columns = ["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size = test_size, random_state = seed, stratify = y)