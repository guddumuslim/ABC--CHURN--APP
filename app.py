import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="ABC Ltd Churn Tool", layout="wide")

@st.cache_resource
def load_models():
    churn = joblib.load("churn_model.joblib")
    rev   = joblib.load("revenue_model.joblib")
    return churn, rev

churn, rev = load_models()
clf, THRESH = churn["model"], churn["threshold"]
cat_cols    = churn["cat_cols"]
num_cols    = churn["num_cols"]
rev_model   = rev["model"]
rev_cat     = rev["cat_cols"]
rev_num     = rev["num_cols"]

def yn(x):
    return "Yes" if x else "No"

tab1, tab2, tab3 = st.tabs(["Single Customer", "Batch Upload", "How It Works"])

with tab1:
    st.header("Single Customer Churn Check")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender     = st.selectbox("Gender", ["Male","Female"])
        senior     = st.selectbox("Senior Citizen", ["No","Yes"])
        partner    = st.selectbox("Partner", ["Yes","No"])
        dependents = st.selectbox("Dependents", ["Yes","No"])
        tenure     = st.slider("Tenure (months)", 0, 72, 12)
    with col2:
        contract   = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        internet   = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        payment    = st.selectbox("Payment Method", ["Electronic check",
                                   "Mailed check","Bank transfer (automatic)",
                                   "Credit card (automatic)"])
        paperless  = st.selectbox("Paperless Billing", ["Yes","No"])
        monthly    = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    with col3:
        phone      = st.selectbox("Phone Service", ["Yes","No"])
        multi      = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
        security   = st.selectbox("Online Security", ["Yes","No","No internet service"])
        backup     = st.selectbox("Online Backup", ["Yes","No","No internet service"])
        device     = st.selectbox("Device Protection", ["Yes","No","No internet service"])
        tech       = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        tv         = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
        movies     = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
        total      = monthly * tenure

    if st.button("Check This Customer"):
        row = pd.DataFrame([{
            "gender": gender, "Partner": partner, "Dependents": dependents,
            "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": device,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "SeniorCitizen": 1 if senior=="Yes" else 0,
            "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total
        }])

        prob  = clf.predict_proba(row[cat_cols + num_cols])[0,1]
        churn_flag = prob >= THRESH

        rev_row = row[rev_cat + rev_num]
        pred_bill = rev_model.predict(rev_row)[0]

        if churn_flag:
            st.error(f"HIGH CHURN RISK: {prob*100:.0f}% probability")
        else:
            st.success(f"LOW CHURN RISK: {prob*100:.0f}% probability")

        st.metric("Predicted Monthly Bill", f"${pred_bill:.2f}")
        st.metric("Revenue at Risk", f"${pred_bill * 12:.0f}/year")

        if churn_flag:
            st.subheader("Recommended Actions")
            if contract == "Month-to-month":
                st.write("Offer an annual contract with 10 percent discount")
            if internet == "Fiber optic":
                st.write("Check service quality complaints for this customer")
            if tenure < 12:
                st.write("Assign a dedicated onboarding support agent")

with tab2:
    st.header("Batch Customer Risk Check")
    st.write("Upload a CSV with the same columns as the dataset")
    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        batch = pd.read_csv(uploaded)
        batch["TotalCharges"] = pd.to_numeric(batch["TotalCharges"], errors="coerce")
        batch.dropna(inplace=True)
        if "Churn" in batch.columns:
            batch.drop(columns=["Churn"], inplace=True)
        if "customerID" in batch.columns:
            batch.rename(columns={"customerID":"CustomerID"}, inplace=True)

        probs = clf.predict_proba(batch[cat_cols + num_cols])[:,1]
        batch["Churn_Probability"] = (probs * 100).round(1)
        batch["Risk_Level"] = pd.cut(probs,
                                     bins=[0, 0.35, 0.65, 1.0],
                                     labels=["Low","Medium","High"])
        batch.sort_values("Churn_Probability", ascending=False, inplace=True)

        st.dataframe(batch[["CustomerID","Churn_Probability","Risk_Level",
                             "Contract","tenure","MonthlyCharges"]].head(20))

        csv_out = batch.to_csv(index=False).encode("utf-8")
        st.download_button("Download Full Results", csv_out,
                           "churn_results.csv", "text/csv")

with tab3:
    st.header("How This Tool Works")
    st.write("""
    This tool uses two machine learning models trained on 7,000 plus telecom customers.

    The Churn Model uses Logistic Regression to predict the probability that a customer
    will cancel their service in the next period. A threshold of 0.55 is used to flag
    high risk customers.

    The Revenue Model uses Linear Regression to estimate the monthly bill for a customer
    based on their service profile. This helps managers understand the financial impact
    of losing a customer.

    Both models were built using only standard data science libraries and trained on the
    IBM Telco Customer Churn dataset.
    """)
