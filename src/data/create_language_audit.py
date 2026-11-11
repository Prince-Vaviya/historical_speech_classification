import json
import csv
import random

from pathlib import Path
from collections import defaultdict


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ParlaCAP-language-detected.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "language-audit-sample.csv"
)

SAMPLE_SIZE = 20
RANDOM_SEED = 42


# =====================================================
# VALIDATE INPUT
# =====================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {INPUT_PATH}"
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# LOAD DATASET
# =====================================================

print("Loading language-detected dataset...")

records = []

with INPUT_PATH.open("r", encoding="utf-8") as file:

    for line in file:

        if not line.strip():
            continue

        records.append(json.loads(line))

print("Total records loaded:", len(records))


# =====================================================
# CLASSIFY CONFIDENCE
# =====================================================

groups = {
    "High": [],
    "Medium": [],
    "Low": []
}

for row in records:

    confidence = float(row["language_confidence"])

    if confidence >= 0.90:
        bucket = "High"

    elif confidence >= 0.70:
        bucket = "Medium"

    else:
        bucket = "Low"

    row["confidence_bucket"] = bucket

    groups[bucket].append(row)


# =====================================================
# STRATIFIED SAMPLING BY PARLIAMENT
# =====================================================

def sample_by_parliament(data, sample_size, rng):
    """
    Select a sample spread across parliaments.

    Each parliament contributes one record per round
    before any parliament contributes a second record.
    """

    parliament_groups = defaultdict(list)

    for row in data:
        parliament_groups[row["parliament"]].append(row)

    # Shuffle records within each parliament.
    for parliament in parliament_groups:
        rng.shuffle(parliament_groups[parliament])

    selected = []

    parliaments = sorted(parliament_groups)

    while len(selected) < sample_size:

        added_in_round = False

        for parliament in parliaments:

            if parliament_groups[parliament]:

                selected.append(
                    parliament_groups[parliament].pop()
                )

                added_in_round = True

                if len(selected) >= sample_size:
                    break

        if not added_in_round:
            break

    return selected


# =====================================================
# GENERATE AUDIT SAMPLE
# =====================================================

rng = random.Random(RANDOM_SEED)

audit_records = []

for bucket in ["High", "Medium", "Low"]:

    available = len(groups[bucket])

    if available < SAMPLE_SIZE:
        raise ValueError(
            f"Not enough records in {bucket} confidence group."
        )

    selected = sample_by_parliament(
        groups[bucket],
        SAMPLE_SIZE,
        rng
    )

    for row in selected:

        audit_records.append({
            "record_id": row["id"],
            "parliament": row["parliament"],
            "date": row.get("date"),
            "topic": row.get("CAP_category"),
            "confidence_bucket": bucket,
            "predicted_language": row["detected_language"],
            "confidence": row["language_confidence"],
            "top_3_predictions": json.dumps(
                row.get("language_top3", []),
                ensure_ascii=False
            ),
            "text": row["text"],
            "human_language": "",
            "human_assessment": "",
            "review_notes": ""
        })


# Shuffle the final audit sample.
rng.shuffle(audit_records)


# =====================================================
# SAVE CSV
# =====================================================

fieldnames = [
    "record_id",
    "parliament",
    "date",
    "topic",
    "confidence_bucket",
    "predicted_language",
    "confidence",
    "top_3_predictions",
    "text",
    "human_language",
    "human_assessment",
    "review_notes"
]

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(audit_records)


# =====================================================
# SUMMARY
# =====================================================

print("\n========== AUDIT SAMPLE CREATED ==========")

print("Total audit records:", len(audit_records))

for bucket in ["High", "Medium", "Low"]:

    count = sum(
        row["confidence_bucket"] == bucket
        for row in audit_records
    )

    print(f"{bucket} confidence: {count}")

print("\nParliament coverage:")

parliament_counts = defaultdict(int)

for row in audit_records:
    parliament_counts[row["parliament"]] += 1

for parliament, count in sorted(parliament_counts.items()):
    print(f"{parliament}: {count}")

print("\nSaved to:")
print(OUTPUT_PATH)

print("\nNext: Fill the human review columns.")
