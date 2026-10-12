# Customer Churn Prediction

Predicts which telecom customers are likely to cancel, explains why, and recommends
whether a retention offer is worth sending. Built end to end: data analysis, modeling,
a FastAPI prediction service with MySQL logging, and a Streamlit demo app.

![CI](https://github.com/vsbhaskar/churn-prediction/actions/workflows/ci.yml/badge.svg)


## Demo

Enter a customer's details, or start from an example customer:

![Input form](docs/app.png)

The app returns a risk score, an offer recommendation, and the main reasons:

![Prediction result](docs/prediction.png)


## What it does

1. You enter a customer's details (contract, tenure, services, billing).
2. The API returns a **risk score**, a **send offer / don't send offer** recommendation,
   and the **top 3 reasons** behind the score (from SHAP).
3. Every prediction is saved to MySQL with the model version, so the history can be
   used later to monitor data drift.


## Results

**Dataset:** IBM Telco Customer Churn sample data, 7,043 customers, 26.5% churned.

**Model comparison** (5-fold stratified cross-validation on the training set):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.846 ± 0.012 | 0.660 ± 0.019 |
| Random Forest | 0.845 ± 0.009 | 0.653 ± 0.020 |
| XGBoost | 0.846 ± 0.011 | 0.667 ± 0.022 |

The three models are statistically tied: the gaps are smaller than the fold-to-fold
spread. This is common on small tabular datasets with simple signal. I kept XGBoost
for its tuning headroom and kept logistic regression as the documented baseline.

**Why not accuracy?** A model that always predicts "no churn" scores 73.5% accuracy
and catches zero churners. I used recall, precision, ROC-AUC and PR-AUC instead.

**Held-out test set** (1,409 customers, touched once, at the end), threshold 0.34:

| Metric | Value |
|---|---|
| ROC-AUC | 0.842 |
| Recall (churners) | 0.88 (330 of 374 caught) |
| Precision (churners) | 0.46 |

**Choosing the decision threshold with money, not 0.5.** I picked the threshold that
maximizes expected retention profit on out-of-fold training predictions, under
*assumed* numbers: a $20 offer, a $400 value per saved customer, and a 30% save rate.

| Strategy (test set, simulated) | Profit |
|---|---|
| Offer to nobody | $0 |
| Offer to everyone | $16,700 |
| Threshold 0.50 | $23,800 |
| **Threshold 0.34 (chosen)** | **$25,160** |

The model earns about 50% more than offering a discount to everyone. The gain of 0.34
over 0.50 is small and within noise: the profit curve has a wide plateau, so the
decision is not very sensitive to the exact threshold.

![Profit vs threshold](docs/profit_curve.png)


## What drives churn

Exploratory analysis and SHAP agree:

- **Contract type:** month-to-month customers churned at 42.7%, versus 11.3% (one year)
  and 2.8% (two year).
- **Tenure:** churners had an average tenure of 18.0 months versus 37.6 for stayers.
  Risk is highest in the first months.
- **Monthly charges:** churners paid more on average ($74.4 vs $61.3).

![SHAP summary](docs/shap_summary.png)

Each dot is one customer. Dots to the right push the prediction toward churn,
dots to the left toward staying; red means a high feature value, blue a low one.
For example, short tenure (blue dots on the right) and month-to-month contracts
raise the risk, while a two-year contract lowers it.

SHAP values show what the model uses, not what causes churn. Correlated features
(for example `tenure` and `TotalCharges`) share credit.


## Tech stack

Python 3.12, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Pydantic, SQLAlchemy,
MySQL, MLflow, Streamlit, pytest, GitHub Actions.


## Project structure

```
churn-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions: install, train, run tests
├── data/
│   └── telco_churn.csv         # IBM Telco Customer Churn dataset
├── docs/
│   ├── app.png                 # input form screenshot
│   ├── prediction.png          # prediction result screenshot
│   ├── profit_curve.png        # profit vs decision threshold
│   └── shap_summary.png        # SHAP dot plot
├── frontend/
│   └── app.py                  # Streamlit demo app
├── models/
│   └── churn_model.joblib      # trained pipeline (created by src.train, not committed)
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory analysis
│   ├── 02_cleaning.ipynb       # cleaning and train/test split
│   ├── 03_baseline.ipynb       # baseline models
│   ├── 04_model_comparison.ipynb  # cross-validated model comparison
│   ├── 05_threshold.ipynb      # profit-based threshold
│   └── 06_shap.ipynb           # SHAP explanations
├── src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI service (/health, /predict)
│   ├── data.py                 # load, clean, train/test split
│   ├── db.py                   # MySQL prediction logging
│   ├── evaluate.py             # profit calculation
│   ├── features.py             # preprocessing (scaling + one-hot encoding)
│   ├── model.py                # model pipeline, threshold, model version
│   ├── schemas.py              # Pydantic request/response models
│   └── train.py                # training script with MLflow tracking
├── tests/
│   ├── test_api.py             # API tests
│   └── test_data.py            # data cleaning and split tests
├── .env.example                # template for database settings
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```


## Run it locally

Requires Python 3.12. MySQL is optional (the API runs without it and simply
skips logging).

```powershell
# 1. Clone and create an environment
git clone https://github.com/vsbhaskar/churn-prediction.git
cd churn-prediction
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Train the model (writes models/churn_model.joblib and logs to MLflow)
python -m src.train

# 3. Start the API (terminal 1)
uvicorn src.api:app --reload

# 4. Start the demo app (terminal 2)
streamlit run frontend/app.py
```

- API docs: http://localhost:8000/docs
- Demo app: http://localhost:8501

The dataset is in `data/telco_churn.csv` (it is also available on Kaggle as
"Telco Customer Churn").


### Optional: log predictions to MySQL

Create a database and a limited user, then copy `.env.example` to `.env` and put your
password in it:

```sql
CREATE DATABASE churn_db;
CREATE USER 'churn_user'@'localhost' IDENTIFIED BY 'your-password';
GRANT ALL PRIVILEGES ON churn_db.* TO 'churn_user'@'localhost';
```

```
DATABASE_URL=mysql+pymysql://churn_user:your-password@127.0.0.1:3306/churn_db
```

The `predictions` table is created automatically when the API starts.


### Experiment tracking and tests

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
python -m pytest -q                                  # 11 tests
```


## API example

`POST /predict` takes the 19 customer fields (see the example in `/docs`) and returns:

- `risk_score`: the model's score (a ranking score, not a literal probability,
  because class weighting inflates it)
- `flag_for_offer`: whether the score is at or above the 0.34 threshold
- `top_reasons`: the three biggest drivers, each with the customer's value, its
  SHAP impact, and the direction

Invalid input (for example `"Contract": "Monthly"`) is rejected with HTTP 422
before it reaches the model.


## Limitations

- One public dataset from a single company. Results may not transfer elsewhere.
- The profit figures rely on made-up business numbers, so they illustrate the method
  and are not a real business result.
- The three models perform about the same, so the model choice is a judgment call.
- Drift monitoring is not implemented yet. Predictions are logged to support it.
- SHAP shows association, not causation.
- Not containerized or deployed yet.


## Next steps

- Docker packaging and cloud deployment
- Drift monitoring from the logged predictions
- Hyperparameter tuning tracked in MLflow


## Data and credits

IBM Telco Customer Churn sample dataset (publicly available on Kaggle).