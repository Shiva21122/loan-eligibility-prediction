import os
import streamlit as st
import pickle
import numpy as np

# ─── Cache & load your trained model ────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models", "loan_approval_model_v5.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ─── App layout ────────────────────────────────────────────────────────
st.title("Loan Approval Predictor")
st.markdown("Fill in the applicant details below and click **Check Eligibility**.")

# ─── Input widgets ────────────────────────────────────────────────────
no_of_dependents         = st.number_input("Dependents", min_value=0, max_value=10, value=0)
income_annum             = st.number_input("Annual Income (₹)", min_value=0, value=500000)
loan_amount              = st.number_input("Loan Amount (₹)", min_value=0, value=100000)
loan_term                = st.number_input("Loan Term (months)", min_value=0, value=360)
cibil_score              = st.slider("CIBIL Score", 300, 900, 650)
residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=0)
commercial_assets_value  = st.number_input("Commercial Assets Value (₹)", min_value=0, value=0)
bank_asset_value         = st.number_input("Bank Assets Value (₹)", min_value=0, value=0)
education                = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed            = st.selectbox("Self-Employed?", ["Yes", "No"])

# ─── Prediction logic ─────────────────────────────────────────────────
if st.button("Check Eligibility"):
    edu_flag  = 1 if education == "Graduate" else 0
    self_flag = 1 if self_employed == "Yes" else 0

    features = np.array([[
        no_of_dependents,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        bank_asset_value,
        edu_flag,
        self_flag
    ]])

    pred = model.predict(features)[0]
    if pred == 1:
        st.success("🎉 Congratulations! Your loan is likely to be approved.")
    else:
        st.error("❌ Sorry, it looks like your loan would be declined.")

# end of file
