"""
Loan Eligibility Prediction - CLI predictor
(converted from notebooks/loan_eligibility_prediction.ipynb, bugs fixed)

Fixes vs. the original notebook:
- `except error:` was a NameError (there is no builtin `error`) -> `except ValueError:`
- Loaded `loan_approval_model_v4.pkl`, which does not exist in the repo ->
  now loads `loan_approval_model_v5.pkl` (the one the Streamlit app uses)
- Model path is resolved relative to this file, not the current working directory
- Duplicated try/except input blocks collapsed into helper functions
"""

import os

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "loan_approval_model_v5.pkl")

EXPECTED_COLUMNS = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "bank_asset_value",
    "education_ Graduate",
    "self_employed_yes",
]


def ask_number(prompt, cast=float):
    """Keep asking until the user enters a valid number."""
    while True:
        try:
            return cast(input(prompt))
        except ValueError:
            print("Please enter a valid number.")


def ask_choice(prompt, choices):
    """Keep asking until the user enters one of `choices` (case-insensitive)."""
    while True:
        value = input(prompt).strip().lower()
        if value in choices:
            return value
        print(f"Please enter one of: {', '.join(choices)}.")


def check_loan_eligibility(model):
    print("Enter customer loan application details:\n")

    no_of_dependents = ask_number("Number of Dependents: ", int)
    education = ask_choice("Education (Graduate/Not Graduate): ",
                           ["graduate", "not graduate"])
    self_employed = ask_choice("Self Employed? (Yes/No): ", ["yes", "no"])
    income_annum = ask_number("Annual Income: ")
    loan_amount = ask_number("Loan Amount: ")
    loan_term = ask_number("Loan Term (in months): ")
    cibil_score = ask_number("CIBIL Score: ")
    residential_assets_value = ask_number("Residential Assets Value: ")
    commercial_assets_value = ask_number("Commercial Assets Value: ")
    bank_asset_value = ask_number("Bank Asset Value: ")

    input_data = {
        "no_of_dependents": no_of_dependents,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "bank_asset_value": bank_asset_value,
        "education_ Graduate": 1 if education == "graduate" else 0,
        "self_employed_yes": 1 if self_employed == "yes" else 0,
    }

    customer_df = pd.DataFrame([input_data])[EXPECTED_COLUMNS]

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    print("\nPrediction Result:")
    print("Loan Approved" if prediction == 1 else "Loan Not Approved")
    print(f"Approval Probability: {probability:.2f}")


if __name__ == "__main__":
    check_loan_eligibility(joblib.load(MODEL_PATH))
# end of file
