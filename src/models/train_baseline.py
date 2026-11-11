
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ParlaCAP-language-detected.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "models" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path):
    train_texts, train_labels = [], []
    dev_texts, dev_labels = [], []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            text = record.get("text", "").strip()
            label = record.get("CAP_category")
            split = record.get("split")

            if not text or not label:
                continue

            if split == "train":
                train_texts.append(text)
                train_labels.append(label)

            elif split == "dev":
                dev_texts.append(text)
                dev_labels.append(label)

    return (
        train_texts,
        train_labels,
        dev_texts,
        dev_labels
    )


print("Loading dataset...")

X_train, y_train, X_dev, y_dev = load_dataset(DATA_PATH)

print(f"Training records: {len(X_train)}")
print(f"Validation records: {len(X_dev)}")


# Word and character feature extraction
features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=300000,
            sublinear_tf=True,
            min_df=2
        )
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=300000,
            sublinear_tf=True,
            min_df=2
        )
    )
])


print("Extracting TF-IDF features...")

X_train_features = features.fit_transform(X_train)
X_dev_features = features.transform(X_dev)

print("Training Linear SVM...")

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_features, y_train)

print("Evaluating model...")

predictions = model.predict(X_dev_features)

accuracy = accuracy_score(y_dev, predictions)
macro_f1 = f1_score(
    y_dev, predictions, average="macro"
)
weighted_f1 = f1_score(
    y_dev, predictions, average="weighted"
)

print("\n===== BASELINE RESULTS =====")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Macro-F1:    {macro_f1:.4f}")
print(f"Weighted-F1: {weighted_f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_dev,
    predictions,
    zero_division=0
))

# Save artifacts
joblib.dump(features, OUTPUT_DIR / "tfidf_features.joblib")
joblib.dump(model, OUTPUT_DIR / "linear_svm.joblib")

np.save(
    OUTPUT_DIR / "predictions.npy",
    predictions
)

np.save(
    OUTPUT_DIR / "true_labels.npy",
    np.array(y_dev)
)

print(f"\nModel artifacts saved to: {OUTPUT_DIR}")
