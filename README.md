ML Assignment 2 — Classification Models

Dataset: UCI – Credit Card Default

1. Problem Statement

The objective of this assignment is to build and evaluate multiple classification models on a real-world dataset and deploy the trained models using an interactive Streamlit web application.

The task involves:

Training multiple machine learning classification models

Evaluating their performance using standard metrics

Deploying pretrained models (without retraining) in a Streamlit app

Allowing users to upload test data and view predictions and evaluation metrics

2. Dataset Description

Dataset Name: Default of Credit Card Clients
Source: UCI Machine Learning Repository

This dataset contains information about credit card clients in Taiwan and is used to predict whether a client will default on their payment in the next month.

Key Details:

Number of instances: 30,000

Number of features: 23

Target variable: default payment next month

0 → No default

1 → Default

The dataset includes demographic information, credit history, bill statements, and payment history.

A stratified train-test split (80% train, 20% test) was used to maintain class distribution.

3. Models Used and Evaluation Metrics

All models were trained on the same dataset and same train-test split to ensure fair comparison.

Models Implemented:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble)

4. Model Performance Comparison (Test Set)

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1 Score | MCC  |
|---------------------|----------|-------|-----------|--------|----------|------|
| Logistic Regression | 0.5906   | 0.7555| 0.5695    | 0.59   | 0.58     | 0.18 |
| Decision Tree       | 0.6062   | 0.6974| 0.6097    | 0.60   | 0.60     | 0.21 |
| KNN                 | 0.6094   | 0.7476| 0.5841    | 0.60   | 0.59     | 0.21 |
| Naive Bayes         | 0.5625   | 0.7377| 0.5745    | 0.56   | 0.56     | 0.12 |
| Random Forest       | 0.6750   | 0.8375| 0.6504    | 0.67   | 0.66     | 0.34 |
| XGBoost             | 0.6531   | 0.8153| 0.6480    | 0.65   | 0.65     | 0.31 |

(Metrics: Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient)

5. Observations on Model Performance

| ML Model Name       | Observation on Model Performance |
|---------------------|----------------------------------|
| Logistic Regression | Performs reasonably well with a good AUC, indicating strong ranking ability, but shows lower accuracy due to linear decision boundaries. |
| Decision Tree       | Captures non-linear patterns but exhibits moderate performance and is prone to overfitting on training data. |
| KNN                 | Achieves slightly better accuracy than Logistic Regression but is sensitive to feature scaling and data distribution. |
| Naive Bayes         | Performs comparatively worse due to the strong independence assumption between input features. |
| Random Forest       | Achieves the best overall performance across most metrics, benefiting from ensemble learning and reduced variance. |
| XGBoost             | Shows strong performance with high AUC and balanced precision–recall tradeoff, demonstrating the effectiveness of boosting. |


6. Streamlit Application

The Streamlit application provides:

Download option for test.csv

CSV upload functionality

Model selection dropdown

Predictions using pretrained models

Automatic computation of evaluation metrics when the label column is present

Confusion matrix and classification report

Streamlit App Link:
👉 https://ml-assignment-2-app-2025ab05226.streamlit.app/

7. Repository Structure


```ml-assignment-2/
│-- app.py
│-- requirements.txt
│-- README.md
│
├── data/
│   └── test.csv
│
└── model/
    ├── train_and_save_models.py
    ├── evaluate_models.py
    ├── export_test_csv.py
    └── artifacts/
        └── *.pkl```

8. Execution Environment

All model training and evaluation were performed on BITS Virtual Lab

Streamlit app deployed using Streamlit Community Cloud

Models are loaded directly in the app without retraining