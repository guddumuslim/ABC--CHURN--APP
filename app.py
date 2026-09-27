import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression

st.set_page_config(page_title="ABC Ltd Churn Tool", layout="wide")

@st.cache_resource
def load_and_train():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df.drop(columns=["customerID"], inplace=True)

    cat_cols = ["gender","Partner","Dependents","PhoneService","MultipleLines",
                "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies","Contract",
                "PaperlessBilling","PaymentMethod"]
    num_cols = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]

    X = df[cat_cols + num_cols]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])
    clf_pipe = Pipeline([("pre", pre),
                         ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))])
    clf_pipe.fit(X_train, y_train)

    rev_cat = ["gender","Partner","Dependents","PhoneService","MultipleLines",
               "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
               "TechSupport","StreamingTV","StreamingMovies","Contract",
               "PaperlessBilling","PaymentMethod"]
    rev_num = ["SeniorCitizen","tenure","TotalCharges"]

    Xr = df[rev_cat + rev_num]
    yr = df["MonthlyCharges"]
    Xr_train, _, yr_train, _ = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    pre_r = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), rev_cat),
        ("num", StandardScaler(), rev_num)
    ])
    rev_pipe = Pipeline([("pre", pre_r), ("reg", LinearRegression())])
    rev_pipe.fit(Xr_train, yr_train)

    return clf_pipe, 0.55, cat_cols, num_cols, rev_pipe, rev_cat, rev_num

with st.spinner("Loading models... please wait 30 seconds on first load"):
    clf, THRESH, cat_cols, num_cols, rev_model, rev_cat, rev_num = load_and_train()

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
        phone    = st.selectbox("Phone Service", ["Yes","No"])
        multi    = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
        security = st.selectbox("Online Security", ["Yes","No","No internet service"])
        backup   = st.selectbox("Online Backup", ["Yes","No","No internet service"])
        device   = st.selectbox("Device Protection", ["Yes","No","No internet service"])
        tech     = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        tv       = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
        movies   = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
        total    = monthly * tenure

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
        prob = clf.predict_proba(row[cat_cols + num_cols])[0,1]
        pred_bill = rev_model.predict(row[rev_cat + rev_num])[0]
        if prob >= THRESH:
            st.error(f"HIGH CHURN RISK: {prob*100:.0f}% probability")
        else:
            st.success(f"LOW CHURN RISK: {prob*100:.0f}% probability")
        st.metric("Predicted Monthly Bill", f"${pred_bill:.2f}")
        st.metric("Revenue at Risk", f"${pred_bill * 12:.0f}/year")
        if prob >= THRESH:
            st.subheader("Recommended Actions")
            if contract == "Month-to-month":
                st.write("Offer an annual contract with 10 percent discount")
            if internet == "Fiber optic":
                st.write("Check service quality complaints for this customer")
            if tenure < 12:
                st.write("Assign a dedicated onboarding support agent")

with tab2:
    st.header("Batch Customer Risk Check")
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
        batch["Risk_Level"] = pd.cut(probs, bins=[0,0.35,0.65,1.0], labels=["Low","Medium","High"])
        batch.sort_values("Churn_Probability", ascending=False, inplace=True)
        st.dataframe(batch[["CustomerID","Churn_Probability","Risk_Level","Contract","tenure","MonthlyCharges"]].head(20))
        st.download_button("Download Full Results", batch.to_csv(index=False).encode("utf-8"), "churn_results.csv", "text/csv")

with tab3:
    st.header("How This Tool Works")
    st.write("This tool trains two models automatically on 7000 plus telecom customers. Logistic Regression predicts churn probability. Linear Regression predicts monthly bill. Both retrain fresh on each deployment so there are no version conflicts.")
