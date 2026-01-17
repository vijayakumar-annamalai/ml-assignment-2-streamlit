import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


DATA_FILE = "default of credit card clients.xls"
TARGET_COL = "default payment next month"
ARTIFACT_DIR = os.path.join("model", "artifacts")


def load_dataset():
    dataset = pd.read_excel(DATA_FILE, header=1)
    X = dataset.drop(columns=[TARGET_COL])
    y = dataset[TARGET_COL]
    return X, y


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Save the split for reuse in evaluation + Streamlit app
    joblib.dump((X_train, X_test, y_train, y_test), os.path.join(ARTIFACT_DIR, "data_split.pkl"))

    # StandardScaler for models that need scaling (LR, KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

    # 1) Logistic Regression (scaled)
    lr = LogisticRegression(max_iter=3000, solver="lbfgs")
    lr.fit(X_train_scaled, y_train)
    joblib.dump(lr, os.path.join(ARTIFACT_DIR, "logistic_regression.pkl"))

    # 2) Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    joblib.dump(dt, os.path.join(ARTIFACT_DIR, "decision_tree.pkl"))

    # 3) KNN (scaled)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    joblib.dump(knn, os.path.join(ARTIFACT_DIR, "knn.pkl"))

    # 4) Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    joblib.dump(nb, os.path.join(ARTIFACT_DIR, "naive_bayes.pkl"))

    # 5) Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(ARTIFACT_DIR, "random_forest.pkl"))

    # 6) XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, os.path.join(ARTIFACT_DIR, "xgboost.pkl"))

    print("Saved artifacts to:", ARTIFACT_DIR)
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Target distribution (train):")
    print(y_train.value_counts())


if __name__ == "__main__":
    main()
