# Loan Approval Prediction App
Link app: https://loan-prediction-app-jbqhaftkaelvhcaer5dbmb.streamlit.app/
For financial institutions, it is important to evaluate loan applications carefully in order to reduce credit risk and improve the loan approval process. Therefore, this project focuses on building a machine learning web application that predicts whether a loan application will be approved or rejected based on applicant and loan information.

## 1. Step to design interactive app

### Model building:

- Cleaned and prepared the loan approval dataset for machine learning.
- Processed both numerical and categorical variables.
- Scaled numerical features using a saved scaler.
- Trained multiple machine learning models, including Decision Tree, Random Forest, AdaBoost, and XGBoost.
- Saved the trained models as `.pkl` files using `joblib`.
- Stored the trained models in the `model` folder for use in the Streamlit app.

### Server logic function:

- Loaded the selected trained model from the `model` folder.
- Rendered customer input fields through a Streamlit form.
- Collected raw input data from the user.
- Applied the same preprocessing steps used during model training.
- Encoded categorical variables and aligned the final feature columns.
- Scaled numerical columns before sending the data to the model.
- Returned the prediction result and approval probability.

### User interface part:

- Built an interactive web app using Streamlit.
- Created a sidebar for model selection.
- Displayed model loading status and feature count.
- Designed input sections for customer information:
  - Personal information
  - Employment and income information
  - Loan details
- Added a **PREDICT** button to generate the loan approval prediction.
- Displayed the final prediction result as **Approved** or **Rejected**.
- Displayed the approval probability to show the confidence of the prediction.

## 2. Explore analysis

The dataset contains information about loan applicants, including personal details, employment background, income level, credit score, existing loans, and loan information. These variables are useful for predicting whether a loan application is likely to be approved or rejected.

Some important factors in loan approval prediction include annual income, credit score, loan amount, debt-to-income ratio, employment years, interest rate, and existing loans. Applicants with higher credit scores, stable employment, and lower debt-to-income ratios are generally more likely to receive loan approval. On the other hand, applicants with high debt levels, low income, or high loan amounts may have a higher chance of rejection.

In this project, several machine learning models were used to compare prediction performance. The models include Decision Tree, Random Forest, AdaBoost, and XGBoost. These models were trained to classify loan applications into two categories: Approved and Rejected.

## 3. Application workflow

The application follows a simple prediction workflow:

- The user selects a machine learning model from the sidebar.
- The user enters applicant information into the form.
- The app collects the raw input data.
- The input data is preprocessed using the saved preprocessing objects.
- The processed data is passed into the selected model.
- The model predicts whether the loan application is approved or rejected.
- The app displays the prediction result and approval probability.

## 4. Input variables

The application uses the following input variables:

### Personal information:

- Age
- Gender
- Marital Status
- Education Level
- Home Ownership

### Employment and income information:

- Annual Income
- Employment Type
- Employment Years
- Debt-to-Income Ratio
- Existing Loans

### Loan details:

- Loan Amount
- Loan Purpose
- Loan Term
- Interest Rate
- Credit Score

## 5. Machine learning models

The following machine learning models were used in this project:

- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- XGBoost Classifier

The trained models were saved in the `model` folder:

```text
model/
├── ada_model.pkl
├── dt_model.pkl
├── rf_model.pkl
└── xgb_model.pkl
