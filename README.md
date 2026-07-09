# Loan Eligibility Prediction

Machine learning app that predicts whether a loan application is likely to be approved, based on applicant financials and credit history. Includes a Streamlit web app and a command-line predictor built on a Logistic Regression model.

## Business Question

Can a bank pre-screen loan applications automatically — using income, credit score, assets, and loan terms — to flag likely approvals and rejections before manual review?

## Demo

Run the Streamlit app, enter applicant details (income, loan amount, CIBIL score, asset values, education, employment type), and get an instant approval prediction with probability.

## Project Structure

```
loan-eligibility-prediction/
├── app.py                          # Streamlit web app
├── predict_loan_eligibility.py     # Command-line predictor
├── models/
│   └── loan_approval_model_v5.pkl  # Trained Logistic Regression model
├── data/
│   └── loan_approval_dataset.csv   # Training dataset
├── notebooks/
│   └── loan_eligibility_prediction.ipynb
├── requirements.txt
└── README.md
```

## Model

| | |
|---|---|
| Algorithm | Logistic Regression |
| Target | Loan approved (yes/no) |
| Features | Dependents, annual income, loan amount, loan term, CIBIL score, residential/commercial/bank asset values, education, self-employment |
| Strongest predictor | CIBIL (credit) score |

## Data Dictionary

| Column | Description |
|--------|-------------|
| no_of_dependents | Number of dependents |
| education | Graduate / Not Graduate |
| self_employed | Yes / No |
| income_annum | Annual income |
| loan_amount | Requested loan amount |
| loan_term | Loan term in months |
| cibil_score | Credit score (300–900) |
| residential_assets_value | Value of real estate owned |
| commercial_assets_value | Value of income-generating property |
| bank_asset_value | Total bank holdings |
| loan_status | Target: approved or not |

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py                    # web app
python predict_loan_eligibility.py      # CLI version
```

## Tech Stack

Python, pandas, scikit-learn, joblib, Streamlit

## Disclaimer

Built on a sample dataset for demonstration and learning purposes — not for real lending decisions.
