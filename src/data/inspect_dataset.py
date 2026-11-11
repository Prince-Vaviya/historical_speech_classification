import json
from pathlib import Path
from collections import Counter

dataset_path = Path("data/raw/ParlaCap-train/ParlaCAP-train.jsonl")

if not dataset_path.exists():
    raise FileNotFoundError(
        "No JSONL file found"
    )

print("Dataset : ", dataset_path)

total_rows = 0
total_labels = Counter()
split_counts = Counter()
field_names = set()

with dataset_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)
        total_rows += 1
        field_names.update(row.keys())

        if total_rows == 1:
            print("\nFirst data : ")
            print(json.dumps(row, indent=2, ensure_ascii=False))

print("\nTotal records: ", total_rows)
print("\nAvailable fields: ", sorted(field_names))