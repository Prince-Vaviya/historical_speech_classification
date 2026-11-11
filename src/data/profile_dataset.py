import json
from pathlib import Path
from collections import Counter

DATA_PATH = Path(
    "data/raw/ParlaCap-train/ParlaCAP-train.jsonl"
)

# Counters
split_counts = Counter()
category_counts = Counter()
parliament_counts = Counter()

# Quality checks
missing_text = 0
empty_text = 0
missing_category = 0
seen_ids = set()
duplicate_ids = set()

# Text statistics
total_words = 0
min_words = float("inf")
max_words = 0
total_records = 0

with DATA_PATH.open("r", encoding="utf-8") as file:

    for line in file:

        if not line.strip():
            continue

        row = json.loads(line)
        total_records += 1

        # Distribution analysis
        split_counts[row.get("split", "UNKNOWN")] += 1
        category_counts[row.get("CAP_category", "UNKNOWN")] += 1
        parliament_counts[row.get("parliament", "UNKNOWN")] += 1

        # Missing-value checks
        text = row.get("text")
        category = row.get("CAP_category")

        if text is None:
            missing_text += 1
        elif not text.strip():
            empty_text += 1

        if not category:
            missing_category += 1

        # Duplicate ID detection
        record_id = row.get("id")

        if record_id in seen_ids:
            duplicate_ids.add(record_id)

        seen_ids.add(record_id)

        # Word statistics
        words = row.get("word_count", 0)

        total_words += words
        min_words = min(min_words, words)
        max_words = max(max_words, words)


# Results
print("\n========== DATASET SUMMARY ==========")
print("Total records:", total_records)
print("Dataset splits:", dict(split_counts))

print("\n========== TOPIC DISTRIBUTION ==========")

for label, count in category_counts.most_common():
    percentage = (count / total_records) * 100
    print(f"{label:30} {count:6} ({percentage:.2f}%)")

print("\n========== PARLIAMENT DISTRIBUTION ==========")

for parliament, count in parliament_counts.most_common():
    print(f"{parliament:10} {count:6}")

print("\n========== DATA QUALITY ==========")
print("Missing text:", missing_text)
print("Empty text:", empty_text)
print("Missing category:", missing_category)
print("Duplicate IDs:", len(duplicate_ids))

print("\n========== TEXT STATISTICS ==========")
print("Average words:", total_words / total_records)
print("Minimum words:", min_words)
print("Maximum words:", max_words)
