from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "baseline"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD EXISTING EVALUATION RESULTS
# --------------------------------------------------

predictions = pd.read_csv(
    REPORT_DIR / "validation_predictions.csv"
)

parliament = pd.read_csv(
    REPORT_DIR / "parliament_evaluation.csv"
)

language = pd.read_csv(
    REPORT_DIR / "language_evaluation_min30.csv"
)

confusion_pairs = pd.read_csv(
    REPORT_DIR / "confusion_pairs.csv"
)

print("Loaded evaluation files.")
print("Predictions:", predictions.shape)
print("Parliament:", parliament.shape)
print("Language:", language.shape)
print("Confusion pairs:", confusion_pairs.shape)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def find_column(df, candidates):
    """Find a column using possible names."""
    lookup = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


# --------------------------------------------------
# DETECT PREDICTION COLUMNS
# --------------------------------------------------

true_col = find_column(
    predictions,
    ["true_category", "y_true", "actual", "true_label"]
)

pred_col = find_column(
    predictions,
    ["predicted_category", "y_pred", "predicted", "predicted_label"]
)

y_true = predictions[true_col]
y_pred = predictions[pred_col]

labels = sorted(
    set(y_true.dropna()) | set(y_pred.dropna())
)

print("Number of evaluated categories:", len(labels))


# --------------------------------------------------
# 1. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

plt.figure(figsize=(17, 14))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix - Parliamentary Topic Classification")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

save_plot("01_confusion_matrix.png")


# --------------------------------------------------
# 2. CLASS-WISE PERFORMANCE
# --------------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)

class_report = pd.DataFrame(report).T
class_report = class_report.loc[labels]

class_report.to_csv(
    OUTPUT_DIR / "class_wise_metrics.csv"
)

metrics = class_report[
    ["precision", "recall", "f1-score"]
]

ax = metrics.plot(
    kind="bar",
    figsize=(17, 7),
    width=0.8
)

ax.set_title("Class-wise Precision, Recall and F1-score")
ax.set_xlabel("CAP Category")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
plt.xticks(rotation=45, ha="right")

save_plot("02_class_wise_performance.png")


# --------------------------------------------------
# 3. OVERALL PERFORMANCE
# --------------------------------------------------

overall = classification_report(
    y_true,
    y_pred,
    output_dict=True,
    zero_division=0
)

overall_metrics = {
    "Accuracy": overall["accuracy"],
    "Macro F1": overall["macro avg"]["f1-score"],
    "Weighted F1": overall["weighted avg"]["f1-score"]
}

plt.figure(figsize=(8, 5))

bars = plt.bar(
    overall_metrics.keys(),
    overall_metrics.values()
)

plt.title("Overall Baseline Model Performance")
plt.ylabel("Score")
plt.ylim(0, 1)

for bar, value in zip(bars, overall_metrics.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center"
    )

save_plot("03_overall_performance.png")


# --------------------------------------------------
# 4. PARLIAMENT-WISE PERFORMANCE
# --------------------------------------------------

parliament_col = find_column(
    parliament,
    ["parliament", "parliament_id", "group"]
)

accuracy_col = find_column(
    parliament,
    ["accuracy"]
)

macro_f1_col = find_column(
    parliament,
    ["macro_f1", "macro-f1", "macro f1"]
)

parliament_sorted = parliament.sort_values(
    macro_f1_col,
    ascending=True
)

ax = parliament_sorted.set_index(
    parliament_col
)[[accuracy_col, macro_f1_col]].plot(
    kind="barh",
    figsize=(12, 11),
    width=0.8
)

ax.set_title("Parliament-wise Classification Performance")
ax.set_xlabel("Score")
ax.set_ylabel("Parliament")
ax.set_xlim(0, 1)
ax.legend(["Accuracy", "Macro F1"])

save_plot("04_parliament_wise_performance.png")


# --------------------------------------------------
# 5. LANGUAGE-WISE PERFORMANCE
# --------------------------------------------------

language_col = find_column(
    language,
    ["detected_language", "language", "language_name"]
)

lang_accuracy_col = find_column(
    language,
    ["accuracy"]
)

lang_f1_col = find_column(
    language,
    ["macro_f1", "macro-f1", "macro f1"]
)

language_sorted = language.sort_values(
    lang_f1_col,
    ascending=True
)

ax = language_sorted.set_index(
    language_col
)[[lang_accuracy_col, lang_f1_col]].plot(
    kind="barh",
    figsize=(12, 12),
    width=0.8
)

ax.set_title("Language-wise Classification Performance")
ax.set_xlabel("Score")
ax.set_ylabel("Detected Language")
ax.set_xlim(0, 1)
ax.legend(["Accuracy", "Macro F1"])

save_plot("05_language_wise_performance.png")


# --------------------------------------------------
# 6. TOP CONFUSION PAIRS
# --------------------------------------------------

true_pair_col = find_column(
    confusion_pairs,
    ["true_category", "actual_category", "true_label"]
)

pred_pair_col = find_column(
    confusion_pairs,
    ["predicted_category", "predicted_label"]
)

count_col = find_column(
    confusion_pairs,
    ["count", "frequency", "total"]
)

top_pairs = confusion_pairs.nlargest(
    15,
    count_col
).copy()

top_pairs["pair"] = (
    top_pairs[true_pair_col].astype(str)
    + " → "
    + top_pairs[pred_pair_col].astype(str)
)

ax = top_pairs.sort_values(
    count_col
).set_index("pair")[count_col].plot(
    kind="barh",
    figsize=(12, 8)
)

ax.set_title("Top 15 Most Frequent Classification Errors")
ax.set_xlabel("Number of Misclassifications")
ax.set_ylabel("Actual Category → Predicted Category")

save_plot("06_top_confusion_pairs.png")


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========== VISUALIZATION COMPLETE ==========")

for filename in sorted(OUTPUT_DIR.glob("*.png")):
    print(filename.name)

print("\nSaved to:", OUTPUT_DIR)
