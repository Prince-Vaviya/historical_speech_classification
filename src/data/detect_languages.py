
import json
import time
from pathlib import Path
from collections import Counter

import onnxruntime as ort
from millid import MillID
from tqdm.auto import tqdm


# =====================================================
# 1. CONFIGURATION
# =====================================================

# Automatically identify project root from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ParlaCAP-train"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ParlaCAP-language-detected.jsonl"
)

MODEL_NAME = "tiny-v1.5"

BATCH_SIZE = 128
TOP_K = 3


# =====================================================
# 2. FIND DATASET
# =====================================================

print("=" * 55)
print("HISTORICAL SPEECH CLASSIFICATION")
print("LANGUAGE DETECTION PIPELINE")
print("=" * 55)

print("\nProject root:", PROJECT_ROOT)

if not PROJECT_ROOT.exists():
    raise FileNotFoundError(
        f"Project root not found: {PROJECT_ROOT}"
    )

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATA_DIR}"
    )

# Find JSONL file automatically.
jsonl_files = list(DATA_DIR.rglob("*.jsonl"))

if not jsonl_files:
    raise FileNotFoundError(
        f"No JSONL file found inside: {DATA_DIR}"
    )

# Prefer the expected filename if available.
expected_file = [
    file for file in jsonl_files
    if file.name == "ParlaCAP-train.jsonl"
]

DATA_PATH = expected_file[0] if expected_file else jsonl_files[0]

print("\nDataset found:", DATA_PATH)
print("Dataset size:", round(DATA_PATH.stat().st_size / (1024 ** 2), 2), "MB")


# =====================================================
# 3. PREPARE OUTPUT
# =====================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\nOutput:", OUTPUT_PATH)


# =====================================================
# 4. VERIFY GPU
# =====================================================

print("\nChecking ONNX GPU support...")

providers = ort.get_available_providers()

print("Available providers:", providers)

if "CUDAExecutionProvider" not in providers:
    raise RuntimeError(
        "CUDAExecutionProvider is unavailable.\n"
        "Enable a Colab GPU runtime and install:\n"
        '!pip install "millid[gpu]"'
    )

print("CUDA GPU provider detected!")


# =====================================================
# 5. LOAD MILLID
# =====================================================

print("\nLoading MillID model...")

model = MillID.from_pretrained(
    MODEL_NAME,
    device="cuda",
    batch_size=BATCH_SIZE
)

print("MillID loaded successfully!")


# =====================================================
# 6. RESUME SUPPORT
# =====================================================

processed_ids = set()

if OUTPUT_PATH.exists():

    print("\nChecking existing results...")

    with OUTPUT_PATH.open("r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            row = json.loads(line)

            if "id" in row:
                processed_ids.add(row["id"])

    print("Previously processed records:", len(processed_ids))


# =====================================================
# 7. STATISTICS
# =====================================================

language_counts = Counter()
confidence_counts = Counter()

total_processed = 0
start_time = time.time()


# =====================================================
# 8. BATCH PROCESSING FUNCTION
# =====================================================

def process_batch(batch, outfile):

    global total_processed

    texts = [
        row["text"]
        for row in batch
    ]

    # One batched prediction call.
    predictions = model.predict(
        texts,
        k=TOP_K
    )

    if len(predictions) != len(batch):
        raise RuntimeError(
            "Prediction count does not match batch size."
        )

    for row, result in zip(batch, predictions):

        best = result[0]

        row["detected_language"] = best.language
        row["language_label"] = best.label
        row["language_confidence"] = float(best.probability)

        row["language_top3"] = [
            {
                "language": prediction.language,
                "label": prediction.label,
                "probability": float(prediction.probability)
            }
            for prediction in result
        ]

        language_counts[best.language] += 1

        if best.probability >= 0.90:
            confidence_counts["High"] += 1

        elif best.probability >= 0.70:
            confidence_counts["Medium"] += 1

        else:
            confidence_counts["Low"] += 1

        outfile.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )

    # Save every completed batch.
    outfile.flush()

    total_processed += len(batch)


# =====================================================
# 9. PROCESS DATASET
# =====================================================

print("\nStarting language detection...\n")

with DATA_PATH.open("r", encoding="utf-8") as infile, \
     OUTPUT_PATH.open("a", encoding="utf-8") as outfile:

    batch = []

    for line in tqdm(infile, desc="Detecting languages"):

        if not line.strip():
            continue

        row = json.loads(line)

        record_id = row.get("id")

        if record_id is None:
            raise ValueError("Dataset record is missing its ID.")

        # Skip previously completed records.
        if record_id in processed_ids:
            continue

        if not row.get("text", "").strip():
            raise ValueError(
                f"Record {record_id} has empty text."
            )

        batch.append(row)

        if len(batch) >= BATCH_SIZE:

            process_batch(batch, outfile)

            # Update resume tracking.
            processed_ids.update(
                row["id"] for row in batch
            )

            batch.clear()

    # Process final incomplete batch.
    if batch:

        process_batch(batch, outfile)

        processed_ids.update(
            row["id"] for row in batch
        )

        batch.clear()


# =====================================================
# 10. FINAL REPORT
# =====================================================

elapsed = time.time() - start_time

print("\n" + "=" * 55)
print("LANGUAGE DETECTION COMPLETED")
print("=" * 55)

print("\nNew records processed:", total_processed)
print("Total records in output:", len(processed_ids))

print("Execution time:", round(elapsed / 60, 2), "minutes")

if elapsed > 0:

    print(
        "Processing speed:",
        round(total_processed / elapsed, 2),
        "records/second"
    )

print("\nLANGUAGE DISTRIBUTION")
print("-" * 35)

for language, count in language_counts.most_common():

    percentage = (
        count / total_processed * 100
        if total_processed else 0
    )

    print(
        f"{language:25} {count:6} ({percentage:.2f}%)"
    )

print("\nCONFIDENCE DISTRIBUTION")
print("-" * 35)

for confidence, count in confidence_counts.items():
    print(f"{confidence:15}: {count}")

print("\nOUTPUT FILE:")
print(OUTPUT_PATH)

print("\nPipeline finished successfully!")
