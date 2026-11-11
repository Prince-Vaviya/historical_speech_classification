import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "ParlaCAP-language-detected.jsonl"
)

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load saved artifacts
model = joblib.load(MODEL_DIR / "linear_svm.joblib")

y_true = np.load(MODEL_DIR / "true_labels.npy", allow_pickle=True)
y_pred = np.load(MODEL_DIR / "predictions.npy", allow_pickle=True)

# Recover validation records in original order
dev_records = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        if record.get("split") == "dev":
            if record.get("text", "").strip() and record.get("CAP_category"):
                dev_records.append(record)

assert len(dev_records) == len(y_true) == len(y_pred), (
    "Validation record alignment mismatch!"
)

# Confusion matrix
labels = model.classes_

cm = confusion_matrix(y_true, y_pred, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

# Misclassified records
errors = []

for record, actual, predicted in zip(dev_records, y_true, y_pred):
    if actual != predicted:
        errors.append({
            "record_id": record.get("id"),
            "text_id": record.get("text_id"),
            "parliament": record.get("parliament"),
            "actual_category": actual,
            "predicted_category": predicted,
            "text": record.get("text")
        })

errors_df = pd.DataFrame(errors)
errors_df.to_csv(
    OUTPUT_DIR / "misclassified_records.csv",
    index=False
)

# Most common confusion pairs
pairs = (
    errors_df
    .groupby(["actual_category", "predicted_category"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

pairs.to_csv(
    OUTPUT_DIR / "confusion_pairs.csv",
    index=False
)

print("\n===== ERROR ANALYSIS =====")
print("Total validation records:", len(y_true))
print("Correct predictions:", int((y_true == y_pred).sum()))
print("Incorrect predictions:", len(errors_df))

print("\n===== TOP 15 CONFUSION PAIRS =====")
print(pairs.head(15).to_string(index=False))

print("\n===== OUTPUT FILES =====")
print("Confusion matrix:", OUTPUT_DIR / "confusion_matrix.csv")
print("Misclassified records:", OUTPUT_DIR / "misclassified_records.csv")
print("Confusion pairs:", OUTPUT_DIR / "confusion_pairs.csv")
