# Loan Approval Prediction App

A Streamlit machine learning web application that predicts whether a loan application is likely to be **Approved** or **Rejected** based on applicant and loan information.

---

## 1. Project Overview

Loan approval is an important decision-making process in the banking and financial industry. Banks and financial institutions need to evaluate applicant information carefully to reduce financial risk and improve decision efficiency.

This project builds an interactive web application that allows users to enter customer information and receive a prediction result for loan approval status. The application uses trained machine learning models to classify whether a loan application is likely to be approved or rejected.

The app was developed using **Python**, **Streamlit**, and machine learning models such as **Decision Tree**, **Random Forest**, **AdaBoost**, and **XGBoost**.

---

## 2. Project Objective

The main objective of this project is to build a machine learning-based loan approval prediction system.

The application aims to:

- Predict loan approval status based on applicant information
- Provide an interactive web interface for users
- Process numerical and categorical input features
- Use saved machine learning models for prediction
- Display the final prediction as **Approved** or **Rejected**
- Show the approval probability of the prediction

---

## 3. Dataset Description

The dataset contains information about loan applicants and loan details. Each row represents one loan application.

The target variable is the loan approval status, which contains two classes:

- **Approved**
- **Rejected**

### Input Features

The application uses the following input features:

#### Personal Information

- **Age**: Age of the applicant
- **Gender**: Gender of the applicant
- **Marital Status**: Marital status of the applicant
- **Education Level**: Highest education level of the applicant
- **Home Ownership**: Home ownership status of the applicant

#### Employment and Income Information

- **Annual Income**: Yearly income of the applicant
- **Employment Type**: Type of employment
- **Employment Years**: Number of years the applicant has been employed
- **Debt-to-Income Ratio**: Ratio between debt payments and income
- **Existing Loans**: Number of existing loans

#### Loan Details

- **Loan Amount**: Amount of money requested by the applicant
- **Loan Purpose**: Purpose of the loan
- **Loan Term**: Duration of the loan in months
- **Interest Rate**: Interest rate of the loan
- **Credit Score**: Credit score of the applicant

---

## 4. Machine Learning Models

Several machine learning models were trained and saved for use in the Streamlit application.

The models used in this project include:

- **Decision Tree Classifier**
- **Random Forest Classifier**
- **AdaBoost Classifier**
- **XGBoost Classifier**

Each trained model was saved as a `.pkl` file using `joblib`.

The saved models are stored in the `model` folder:

```text
model/
├── ada_model.pkl
├── dt_model.pkl
├── rf_model.pkl
└── xgb_model.pkl
