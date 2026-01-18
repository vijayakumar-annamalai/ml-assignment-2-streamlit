import os
import joblib
import pandas as pd

ARTIFACT_DIR = os.path.join("model", "artifacts")
OUT_FILE = os.path.join("data", "test.csv")

def main():
    X_train, X_test, y_train, y_test = joblib.load(os.path.join(ARTIFACT_DIR, "data_split.pkl"))

    test_df = X_test.copy()
    test_df["default payment next month"] = y_test.values  # keeping the label for metrics

    os.makedirs("data", exist_ok=True)
    test_df.to_csv(OUT_FILE, index=False)

    print("Exported:", OUT_FILE)
    print("Shape:", test_df.shape)
    print("Columns:", list(test_df.columns)[:5], "...")

if __name__ == "__main__":
    main()
