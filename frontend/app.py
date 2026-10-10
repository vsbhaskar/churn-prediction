import os

import requests
import streamlit as st

# Where the FastAPI service lives. Overridable so Docker/deployment can change it.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

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
PRESETS = {
    "New fiber customer (high risk)": HIGH_RISK,
    "Loyal two-year customer (low risk)": LOW_RISK,
}

YES_NO = ["Yes", "No"]
YES_NO_NOINT = ["Yes", "No", "No internet service"]

st.title("📉 Customer Churn Predictor")
st.caption(
    "Enter a telecom customer's details to get a churn risk score, "
    "a retention-offer recommendation, and the main reasons behind it."
)

preset_name = st.selectbox("Start from an example customer", list(PRESETS))
p = PRESETS[preset_name]


def pick(label, options, field):
    # The preset name is part of the key, so switching presets resets the widgets
    return st.selectbox(
        label, options, index=options.index(p[field]), key=f"{preset_name}_{field}"
    )


col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer")
    gender = pick("Gender", ["Female", "Male"], "gender")
    senior = st.selectbox(
        "Senior citizen", [0, 1], index=p["SeniorCitizen"],
        format_func=lambda x: "Yes" if x else "No",
        key=f"{preset_name}_senior",
    )
    partner = pick("Partner", YES_NO, "Partner")
    dependents = pick("Dependents", YES_NO, "Dependents")
    tenure = st.number_input(
        "Tenure (months)", 0, 100, p["tenure"], key=f"{preset_name}_tenure"
    )

with col2:
    st.subheader("Services")
    phone = pick("Phone service", YES_NO, "PhoneService")
    lines = pick("Multiple lines", ["Yes", "No", "No phone service"], "MultipleLines")
    internet = pick("Internet service", ["DSL", "Fiber optic", "No"], "InternetService")
    security = pick("Online security", YES_NO_NOINT, "OnlineSecurity")
    backup = pick("Online backup", YES_NO_NOINT, "OnlineBackup")
    protection = pick("Device protection", YES_NO_NOINT, "DeviceProtection")
    support = pick("Tech support", YES_NO_NOINT, "TechSupport")
    tv = pick("Streaming TV", YES_NO_NOINT, "StreamingTV")
    movies = pick("Streaming movies", YES_NO_NOINT, "StreamingMovies")

with col3:
    st.subheader("Billing")
    contract = pick("Contract", ["Month-to-month", "One year", "Two year"], "Contract")
    paperless = pick("Paperless billing", YES_NO, "PaperlessBilling")
    payment = pick(
        "Payment method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        "PaymentMethod",
    )
    monthly = st.number_input(
        "Monthly charges ($)", 0.0, 500.0, float(p["MonthlyCharges"]),
        step=1.0, key=f"{preset_name}_monthly",
    )
    total = st.number_input(
        "Total charges ($)", 0.0, 100000.0, float(p["TotalCharges"]),
        step=10.0, key=f"{preset_name}_total",
    )

payload = {
    "gender": gender, "SeniorCitizen": senior, "Partner": partner,
    "Dependents": dependents, "tenure": int(tenure), "PhoneService": phone,
    "MultipleLines": lines, "InternetService": internet, "OnlineSecurity": security,
    "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": support,
    "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": float(monthly), "TotalCharges": float(total),
}

st.divider()

if st.button("Predict churn risk", type="primary"):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    except requests.exceptions.RequestException:
        st.error(f"Could not reach the prediction API at {API_URL}. Is it running?")
    else:
        if response.status_code != 200:
            st.error(f"The API rejected the request (HTTP {response.status_code}).")
            st.code(response.text)
        else:
            result = response.json()

            left, right = st.columns(2)
            left.metric("Risk score", f"{result['risk_score']:.2f}")
            right.metric("Decision threshold", f"{result['threshold']:.2f}")

            if result["flag_for_offer"]:
                st.warning("Recommendation: send a retention offer to this customer.")
            else:
                st.success("Recommendation: no retention offer needed.")

            st.progress(min(max(result["risk_score"], 0.0), 1.0))

            st.subheader("Why the model says this")
            for reason in result["top_reasons"]:
                verb = "raises" if reason["impact"] > 0 else "lowers"
                st.write(
                    f"- **{reason['feature']} = {reason['value']}** {verb} "
                    f"the risk ({reason['impact']:+.2f})"
                )

            st.caption(
                "The risk score is a model score, not a literal probability. "
                "Reasons come from SHAP and show association, not proof of cause."
            )