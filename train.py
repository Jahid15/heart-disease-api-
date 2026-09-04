import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/heart.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "heart_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET = "target"


def load_data():
    df = pd.read_csv(DATA_PATH)
    raw_rows = len(df)

    # the kaggle file repeats the original cleveland records, so without this
    # the same row can land in both splits and inflate the score
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    print(f"rows: {raw_rows} in file, {len(df)} after dropping duplicates")

    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df[FEATURES], df[TARGET]


def build_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, digits=4))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_type": "LogisticRegression",
        "pipeline": ["StandardScaler", "LogisticRegression"],
        "features": FEATURES,
        "target": TARGET,
        "classes": {"0": "no heart disease", "1": "heart disease"},
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_accuracy": round(float(accuracy), 4),
        "dataset": "Kaggle Heart Disease Dataset (johnsmith88)",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"saved model to {MODEL_PATH}")
    print(f"saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
