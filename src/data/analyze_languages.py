
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data/processed/ParlaCAP-language-detected.jsonl"
)

language_counts = Counter()
parliament_languages = defaultdict(Counter)
confidence_by_parliament = defaultdict(list)
low_confidence = []

total = 0

with DATA_PATH.open("r", encoding="utf-8") as file:

    for line in file:

        if not line.strip():
            continue

        row = json.loads(line)

        language = row["detected_language"]
        parliament = row["parliament"]
        confidence = row["language_confidence"]
        category = row["CAP_category"]

        language_counts[language] += 1
        parliament_languages[parliament][language] += 1
        confidence_by_parliament[parliament].append(confidence)

        if confidence < 0.70:
            low_confidence.append({
                "parliament": parliament,
                "language": language,
                "confidence": confidence,
                "category": category,
                "text": row["text"][:200]
            })

        total += 1


# ==========================================
# 1. LANGUAGE DISTRIBUTION
# ==========================================

print("\nTOTAL RECORDS:", total)

print("\nTOP 20 LANGUAGES")

for language, count in language_counts.most_common(20):

    print(
        f"{language:25} {count:6} "
        f"({count / total * 100:.2f}%)"
    )


# ==========================================
# 2. LANGUAGE DISTRIBUTION PER PARLIAMENT
# ==========================================

print("\nLANGUAGES PER PARLIAMENT")

for parliament, counts in sorted(parliament_languages.items()):

    dominant = counts.most_common(1)[0]

    print(
        f"{parliament:8} "
        f"Records: {sum(counts.values()):5} "
        f"Dominant: {dominant[0]:25} "
        f"Count: {dominant[1]:5} "
        f"Languages: {len(counts)}"
    )


# ==========================================
# 3. LOW CONFIDENCE ANALYSIS
# ==========================================

print("\nLOW CONFIDENCE RECORDS:", len(low_confidence))

print("\nLOW CONFIDENCE BY PARLIAMENT")

low_by_parliament = Counter(
    row["parliament"]
    for row in low_confidence
)

for parliament, count in low_by_parliament.most_common():

    total_parliament = sum(
        parliament_languages[parliament].values()
    )

    percentage = count / total_parliament * 100

    print(
        f"{parliament:8} {count:5} "
        f"({percentage:.2f}%)"
    )


# ==========================================
# 4. LOW CONFIDENCE BY LANGUAGE
# ==========================================

print("\nLOW CONFIDENCE BY PREDICTED LANGUAGE")

low_by_language = Counter(
    row["language"]
    for row in low_confidence
)

for language, count in low_by_language.most_common(20):

    print(f"{language:25} {count:5}")


# ==========================================
# 5. SAMPLE SUSPICIOUS RECORDS
# ==========================================

print("\nSUSPICIOUS RECORDS")

for row in low_confidence[:15]:

    print("\nParliament:", row["parliament"])
    print("Language:", row["language"])
    print("Confidence:", row["confidence"])
    print("Category:", row["category"])
    print("Text:", row["text"])


# ==========================================
# 6. SUMMARY
# ==========================================

high = sum(
    1 for values in confidence_by_parliament.values()
    for confidence in values if confidence >= 0.90
)

medium = sum(
    1 for values in confidence_by_parliament.values()
    for confidence in values
    if 0.70 <= confidence < 0.90
)

low = len(low_confidence)

print("\n========== SUMMARY ==========")
print("Total:", total)
print("High confidence:", high)
print("Medium confidence:", medium)
print("Low confidence:", low)
print("Unique detected languages:", len(language_counts))
print("Parliaments:", len(parliament_languages))
