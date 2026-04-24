import os
import joblib
import pandas as pd
import streamlit as st


# CONFIG
st.set_page_config(
    page_title="Loan Status Prediction",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


# LOAD MODEL ARTIFACT
def load_artifact(filename: str) -> dict:
    path = os.path.join(MODEL_DIR, filename)
    obj  = joblib.load(path)

    if isinstance(obj, dict):
        return obj
    return {
        "model":         obj,
        "scaler":        None,
        "num_cols":      [],
        "cat_cols":      [],
        "final_columns": [],
        "label_map":     {0: "Rejected", 1: "Approved"},
    }


def get_model_files() -> list[str]:
    if not os.path.exists(MODEL_DIR):
        return []
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]


# SIDEBAR
def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Model Settings")

        files = get_model_files()
        if not files:
            st.error("No .pkl files found in the 'model' folder.")
            st.stop()

        selected = st.selectbox("Select model", files)

        try:
            artifact = load_artifact(selected)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

        model = artifact["model"]
        st.success(f"Loaded: **{type(model).__name__}**")

        if artifact["scaler"] is not None:
            st.success("Scaler found — numeric columns will be scaled")
        else:
            st.warning("No scaler in this file — numeric columns won't be scaled")

        if artifact["final_columns"]:
            st.info(f"Feature count: **{len(artifact['final_columns'])}**")

    return artifact


# INPUT FORM
def render_input_form() -> tuple[bool, pd.DataFrame]:
    st.subheader("Customer Information")

    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Personal**")
            age             = st.number_input("Age",           min_value=18,  max_value=100, value=35)
            gender          = st.selectbox("Gender",           ["Male", "Female", "Other", "Unknown"])
            marital_status  = st.selectbox("Marital Status",   ["Single", "Married", "Divorced", "Widowed", "Unknown"])
            education_level = st.selectbox("Education Level",  ["Bachelor", "Master", "High School", "PhD", "Diploma", "Unknown"])
            home_ownership  = st.selectbox("Home Ownership",   ["Mortgage", "Own", "Rent", "Other", "Unknown"])

        with col2:
            st.markdown("**Employment & Income**")
            annual_income        = st.number_input("Annual Income",        min_value=0.0,  value=33000.0, step=1000.0, format="%.2f")
            employment_type      = st.selectbox("Employment Type",         ["Full-time", "Part-time", "Self-employed", "Unemployed", "Unknown"])
            employment_years     = st.number_input("Employment Years",     min_value=0.0,  value=10.0,   step=0.5)
            debt_to_income_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.01, max_value=0.85, value=0.43, step=0.01, format="%.3f")
            existing_loans       = st.number_input("Existing Loans",       min_value=0,    value=1,      step=1)

        with col3:
            st.markdown("**Loan Details**")
            loan_amount      = st.number_input("Loan Amount",        min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
            loan_purpose     = st.selectbox("Loan Purpose",          ["Personal", "Home", "Car", "Business", "Education", "Medical", "Unknown"])
            loan_term_months = st.number_input("Loan Term (months)", min_value=12,  max_value=360, value=60,   step=12)
            interest_rate    = st.number_input("Interest Rate (%)",  min_value=2.5, max_value=25.0, value=13.9, step=0.1)
            credit_score     = st.number_input("Credit Score",       min_value=300, max_value=850, value=679)

        submitted = st.form_submit_button("PREDICT", use_container_width=True)

    raw = pd.DataFrame([{
        "age":                  float(age),
        "gender":               gender,
        "marital_status":       marital_status,
        "education_level":      education_level,
        "annual_income":        annual_income,
        "loan_amount":          loan_amount,
        "credit_score":         float(credit_score),
        "employment_years":     employment_years,
        "employment_type":      employment_type,
        "loan_purpose":         loan_purpose,
        "home_ownership":       home_ownership,
        "debt_to_income_ratio": debt_to_income_ratio,
        "existing_loans":       float(existing_loans),
        "loan_term_months":     float(loan_term_months),
        "interest_rate":        interest_rate,
    }])

    return submitted, raw


# PREPROCESSING
def encode_categoricals(df: pd.DataFrame, cat_cols: list, feature_cols: list) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df.index)

    for col in cat_cols:
        value  = df[col].iloc[0]
        prefix = col + "_"

        dummy_cols = [f for f in feature_cols if f.startswith(prefix)]

        for dummy_col in dummy_cols:
            category = dummy_col[len(prefix):]
            encoded[dummy_col] = int(value == category)

    return encoded


def preprocess(raw: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    model        = artifact["model"]
    scaler       = artifact["scaler"]
    num_cols     = artifact["num_cols"]
    cat_cols     = artifact["cat_cols"]
    feature_cols = artifact["final_columns"]

    df = raw.copy()

    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    if scaler is not None:
        num_scaled = pd.DataFrame(
            scaler.transform(df[num_cols]),
            columns=num_cols,
            index=df.index,
        )
    else:
        num_scaled = df[num_cols].astype(float)

    if feature_cols:
        cat_encoded = encode_categoricals(df, cat_cols, feature_cols)
    else:
        cat_encoded = pd.get_dummies(df[cat_cols], drop_first=True, dtype=int)

    result = pd.concat([num_scaled, cat_encoded], axis=1)

    reference_cols = feature_cols if feature_cols else list(getattr(model, "feature_names_in_", []))
    if reference_cols:
        result = result.reindex(columns=reference_cols, fill_value=0)

    return result


# PREDICTION
def predict(model, df: pd.DataFrame) -> tuple[int, float | None]:
    label = int(model.predict(df)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        if len(probs) == 2:
            proba = float(probs[1])
    return label, proba


# RESULT DISPLAY
def render_result(label: int, proba: float | None, label_map: dict) -> None:
    st.markdown("---")
    st.subheader("Prediction Result")

    text = label_map.get(label, str(label))
    left, right = st.columns(2)

    with left:
        if text == "Approved":
            st.success(f"### {text}")
        elif text == "Rejected":
            st.error(f"### {text}")
        else:
            st.info(f"### ℹ{text}")

    with right:
        if proba is not None:
            st.metric("Approval Probability", f"{proba * 100:.2f}%")
            st.progress(proba)


# APP ENTRY POINT
st.title("Loan Status Prediction")
st.caption("Predict whether a loan application will be approved or rejected.")

artifact          = render_sidebar()
submitted, raw_df = render_input_form()

if submitted:
    with st.expander("Raw Input", expanded=False):
        st.dataframe(raw_df, use_container_width=True)

    try:
        final_df = preprocess(raw_df, artifact)

        with st.expander("Processed Data (sent to model)", expanded=True):
            st.dataframe(final_df, use_container_width=True)
            st.caption(f"{final_df.shape[1]} features")

        label, proba = predict(artifact["model"], final_df)
        render_result(label, proba, artifact["label_map"])

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)