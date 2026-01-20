# ML Assignment 2 — Classification Models  
*Dataset: UCI – Credit Card Default*

---

## 1. Problem Statement

In this assignment, I worked on building and evaluating multiple **classification models** using a real-world financial dataset.  
The main idea was not only to train models, but also to **compare their performance** and make them accessible through a simple **Streamlit web application**.

The work carried out in this assignment includes:

- Training several machine learning classification algorithms
- Evaluating each model using commonly used classification metrics
- Saving the trained models and loading them directly in a Streamlit app
- Allowing users to upload test data, generate predictions, and view evaluation results

The focus of this assignment is on **model comparison, evaluation, and deployment**, rather than only model training.

---

## 2. Dataset Description

- **Dataset Name:** Default of Credit Card Clients  
- **Source:** UCI Machine Learning Repository  

The dataset contains information about credit card customers in Taiwan and is used to predict **whether a customer is likely to default on their credit card payment in the next month**.

### Dataset Overview

- **Total number of records:** 30,000  
- **Number of input features:** 23  
- **Target column:** `default payment next month`  
  - `0` → Customer does not default  
  - `1` → Customer defaults  

The features include customer demographics, credit limit details, billing amounts, and payment history across multiple months.

To maintain consistency and fairness during evaluation, the dataset was split using a **stratified train–test split**:
- 80% of the data was used for training
- 20% of the data was used for testing

This ensured that the class distribution remained similar in both sets.

---

## 3. Models Used and Evaluation Metrics

All models were trained on the **same training data** and evaluated on the **same test data** so that their performance could be compared fairly.

### Models Implemented

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest  
- XGBoost  

### Evaluation Metrics

The following metrics were used to evaluate and compare the models:

- Accuracy  
- Area Under the ROC Curve (AUC)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## 4. Model Performance Comparison (Test Set)

| ML Model Name        | Accuracy | AUC   | Precision | Recall | F1 Score | MCC  |
|----------------------|----------|-------|-----------|--------|----------|------|
| Logistic Regression  | 0.5906   | 0.7555| 0.5695    | 0.59   | 0.58     | 0.18 |
| Decision Tree        | 0.6062   | 0.6974| 0.6097    | 0.60   | 0.60     | 0.21 |
| KNN                  | 0.6094   | 0.7476| 0.5841    | 0.60   | 0.59     | 0.21 |
| Naive Bayes          | 0.5625   | 0.7377| 0.5745    | 0.56   | 0.56     | 0.12 |
| Random Forest        | 0.6750   | 0.8375| 0.6504    | 0.67   | 0.66     | 0.34 |
| XGBoost              | 0.6531   | 0.8153| 0.6480    | 0.65   | 0.65     | 0.31 |

All metrics shown above were computed on the **test dataset**.

---

## 5. Observations on Model Performance

| ML Model Name | Observation |
|--------------|-------------|
| Logistic Regression | Shows a good AUC value, indicating decent class separation, but overall accuracy is limited because the model is linear in nature. |
| Decision Tree | Handles non-linear relationships but provides only moderate performance and may overfit if not constrained properly. |
| KNN | Performs slightly better than Logistic Regression in terms of accuracy, but is sensitive to scaling and choice of neighbors. |
| Naive Bayes | Produces lower performance compared to other models due to the strong assumption of feature independence. |
| Random Forest | Gives the best overall results across most metrics, benefiting from ensemble learning and reduced variance. |
| XGBoost | Performs close to Random Forest with strong AUC and balanced precision–recall, highlighting the effectiveness of boosting methods. |

---

## 6. Streamlit Application

A Streamlit web application was developed to make the trained models easy to use and evaluate.

### Application Features

- Option to download the provided `test.csv`
- CSV file upload support
- Dropdown to select a classification model
- Predictions generated using **pretrained models**
- Automatic calculation of evaluation metrics when the label column is present
- Display of confusion matrix and classification report

### Streamlit App Link
👉 https://ml-assignment-2-app-2025ab05226.streamlit.app/

---

## 7. Repository Structure

```text
ml-assignment-2/
│── app.py
│── requirements.txt
│── README.md
│
├── data/
│   └── test.csv
│
└── model/
    ├── train_and_save_models.py
    ├── evaluate_models.py
    ├── export_test_csv.py
    └── artifacts/
        └── *.pkl
```

---

## 8. Execution Environment

- All model training and evaluation were carried out on BITS Virtual Lab

- The Streamlit application was deployed using Streamlit Community Cloud

- The application loads trained models directly and does not retrain models during execution

---