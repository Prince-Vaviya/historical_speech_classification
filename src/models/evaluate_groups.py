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

y_true = np.load(MODEL_DIR / "true_labels.npy", allow_pickle=True)
y_pred = np.load(MODEL_DIR / "predictions.npy", allow_pickle=True)

records = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        if record.get("split") == "dev":
            if record.get("text", "").strip() and record.get("CAP_category"):
                records.append(record)

assert len(records) == len(y_true), "Record alignment mismatch"

df = pd.DataFrame(records)

df["actual"] = y_true
df["predicted"] = y_pred
df["correct"] = df["actual"] == df["predicted"]


def evaluate_group(group):
    return pd.Series({
        "records": len(group),
        "accuracy": accuracy_score(
            group["actual"], group["predicted"]
        ),
        "macro_f1": f1_score(
            group["actual"],
            group["predicted"],
            average="macro",
            zero_division=0
        )
    })


# Parliament-wise evaluation
parliament_results = (
    df.groupby("parliament")
    .apply(evaluate_group, include_groups=False)
    .reset_index()
    .sort_values("records", ascending=False)
)

parliament_results.to_csv(
    OUTPUT_DIR / "parliament_evaluation.csv",
    index=False
)

print("\n===== PARLIAMENT-WISE RESULTS =====")
print(parliament_results.to_string(index=False))

# Inspect detected-language field names
print("\n===== POSSIBLE LANGUAGE FIELDS =====")
print([
    key for key in df.columns
    if "lang" in key.lower()
])

# Save enriched validation records for further analysis
df.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False
)

print("\nSaved evaluation files to:", OUTPUT_DIR)
