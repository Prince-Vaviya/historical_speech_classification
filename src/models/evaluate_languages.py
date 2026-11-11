import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "ParlaCAP-language-detected.jsonl"
)

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions
y_true = np.load(
    MODEL_DIR / "true_labels.npy",
    allow_pickle=True
)
y_pred = np.load(
    MODEL_DIR / "predictions.npy",
    allow_pickle=True
)

# Load matching validation records
records = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        if (
            record.get("split") == "dev"
            and record.get("text", "").strip()
            and record.get("CAP_category")
        ):
            records.append(record)

assert len(records) == len(y_true) == len(y_pred), (
    "Validation record alignment mismatch!"
)

df = pd.DataFrame(records)

df["actual"] = y_true
df["predicted"] = y_pred

df["correct"] = df["actual"] == df["predicted"]

df["detected_language"] = (
    df["detected_language"].fillna("Unknown")
)

df["language_label"] = (
    df["language_label"].fillna("Unknown")
)

df["language_confidence"] = pd.to_numeric(
    df["language_confidence"],
    errors="coerce"
)

# Calculate metrics for each language
results = []

for language, group in df.groupby("detected_language"):

    results.append({
        "language": language,
        "language_code": group["language_label"].mode().iloc[0],
        "records": len(group),
        "accuracy": accuracy_score(
            group["actual"],
            group["predicted"]
        ),
        "macro_f1": f1_score(
            group["actual"],
            group["predicted"],
            average="macro",
            zero_division=0
        ),
        "avg_language_confidence": (
            group["language_confidence"].mean()
        ),
        "low_confidence_records": (
            group["language_confidence"].lt(0.70).sum()
        )
    })

language_results = pd.DataFrame(results)

language_results = language_results.sort_values(
    "records",
    ascending=False
)

# Save complete results
language_results.to_csv(
    OUTPUT_DIR / "language_evaluation_all.csv",
    index=False
)

# Filter groups with sufficient support for more cautious comparisons
reliable_results = language_results[
    language_results["records"] >= 30
].copy()

reliable_results.to_csv(
    OUTPUT_DIR / "language_evaluation_min30.csv",
    index=False
)

print("\n===== LANGUAGE-WISE RESULTS =====")
print(
    language_results.round(4).to_string(index=False)
)

print("\n===== LANGUAGES WITH AT LEAST 30 RECORDS =====")
print(
    reliable_results.round(4).to_string(index=False)
)

print("\n===== SUMMARY =====")
print("Unique detected languages:", len(language_results))
print("Languages with >=30 records:", len(reliable_results))
print("Total validation records:", len(df))

print("\nSaved reports to:", OUTPUT_DIR)
