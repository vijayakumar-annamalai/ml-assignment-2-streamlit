import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)



# Configurations

st.set_page_config(page_title="ML Assignment 2 - Classification Models", layout="wide")

ARTIFACT_DIR = os.path.join("model", "artifacts")
DEFAULT_TEST_CSV_PATH = os.path.join("data", "test.csv")
TARGET_COL = "default payment next month"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

SCALED_MODELS = {"Logistic Regression", "kNN"}  # models that use StandardScaler



# Helpers

@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))

@st.cache_resource
def load_model(model_name: str):
    path = os.path.join(ARTIFACT_DIR, MODEL_FILES[model_name])
    return joblib.load(path)

def get_proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None

def compute_metrics(y_true, y_pred, y_score):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if y_score is not None else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics



# UI - App

st.title("ML Assignment 2 — Classification Models (Streamlit App)")
st.subheader("UCI – Credit Card Default")
# st.caption("Loads pretrained models, supports test.csv download, CSV upload, predictions, and evaluation metrics.")
st.caption(
    "Dataset: UCI Credit Card Default | "
    "Loads pretrained models, supports test.csv download, CSV upload, predictions, and evaluation metrics."
)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Download test.csv")
    if os.path.exists(DEFAULT_TEST_CSV_PATH):
        with open(DEFAULT_TEST_CSV_PATH, "rb") as f:
            st.download_button(
                label="Download sample test.csv",
                data=f,
                file_name="test.csv",
                mime="text/csv"
            )
        st.info(f"File path in repo: {DEFAULT_TEST_CSV_PATH}")
        st.write(f"**Label column expected (for metrics):** `{TARGET_COL}`")
    else:
        st.warning("data/test.csv not found in repo. Please ensure it is committed.")

    st.divider()

    st.subheader("Choose model")
    model_name = st.selectbox("Select a model", list(MODEL_FILES.keys()))
    st.write("Selected:", f"**{model_name}**")

with col_right:
    st.subheader("Upload CSV for prediction / evaluation")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV to run predictions. If it includes the label column, metrics will be computed.")
        st.stop()

    try:
        input_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.write("### Preview")
    st.dataframe(input_df.head(10), use_container_width=True)

    # Split features/labels if available
    has_label = TARGET_COL in input_df.columns
    if has_label:
        y_true = input_df[TARGET_COL].astype(int)
        X_input = input_df.drop(columns=[TARGET_COL])
    else:
        y_true = None
        X_input = input_df.copy()

    # Loading the artifacts
    try:
        model = load_model(model_name)
        scaler = load_scaler()
    except Exception as e:
        st.error(f"Failed to load saved model/scaler. Ensure artifacts exist in {ARTIFACT_DIR}. Error: {e}")
        st.stop()

    # Applying the scaling for required models
    if model_name in SCALED_MODELS:
        X_used = scaler.transform(X_input)
    else:
        X_used = X_input

    # Predict
    y_pred = model.predict(X_used)
    y_score = get_proba_or_score(model, X_used)

    # Output dataframe (predictions)
    out_df = input_df.copy()
    out_df["prediction"] = y_pred

    st.write("### Predictions")
    st.dataframe(out_df.head(20), use_container_width=True)

    # Download predictions
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv"
    )

    # Metrics if labels exist
    st.divider()
    st.write("## Evaluation")
    if has_label:
        metrics = compute_metrics(y_true, y_pred, y_score)
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        mcol1.metric("AUC", f"{metrics['AUC']:.4f}" if not np.isnan(metrics["AUC"]) else "N/A")
        mcol2.metric("Precision", f"{metrics['Precision']:.4f}")
        mcol2.metric("Recall", f"{metrics['Recall']:.4f}")
        mcol3.metric("F1", f"{metrics['F1']:.4f}")
        mcol3.metric("MCC", f"{metrics['MCC']:.4f}")

        # Confusion matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=False)

        # Classification report
        st.write("### Classification Report")
        report = classification_report(y_true, y_pred, digits=4)
        st.code(report)

    else:
        st.warning(
            f"No label column found. Add `{TARGET_COL}` to your CSV to compute metrics "
            "(Accuracy, AUC, Precision, Recall, F1, MCC)."
        )
