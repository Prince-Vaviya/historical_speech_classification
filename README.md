# Historical Speech Classification

An NLP and machine learning project for automatically classifying historical parliamentary texts into policy-topic categories using the Comparative Agendas Project (CAP) taxonomy.

### Problem Definition

- **Input:** A parliamentary text record.
- **Output:** One predicted policy-topic category.
- **Task:** Single-label, multiclass text classification.
- **Classification framework:** Comparative Agendas Project (CAP).

## CAP Classification Categories

The classification system uses the following 22 policy categories:

1. Agriculture
2. Civil Rights
3. Culture
4. Defense
5. Domestic Commerce
6. Education
7. Energy
8. Environment
9. Foreign Trade
10. Government Operations
11. Health
12. Housing
13. Immigration
14. International Affairs
15. Labor
16. Law and Crime
17. Macroeconomics
18. Public Lands
19. Social Welfare
20. Technology
21. Transportation
22. Other

## Dataset

**Dataset:** [ParlaCAP-train](https://www.clarin.si/repository/xmlui/handle/11356/2093)

The project uses parliamentary text records from the ParlaCAP dataset.

| Dataset characteristic | Description |
|---|---|
| Total records | 35,579 |
| Training records | 29,779 |
| Development records | 5,800 |
| Parliament identifiers | 29 |
| Classification categories | 22 |

The dataset is divided into training and development partitions. The supplied development partition is used for model evaluation.

### Dataset Fields

| Field | Description |
|---|---|
| `text` | Parliamentary text used as model input |
| `CAP_category` | Target classification category |
| `split` | Dataset partition |
| `parliament` | Parliament identifier |
| `speaker_name` | Speaker metadata |
| `date` | Record date |
| `word_count` | Dataset-provided word count |

## Project Pipeline

The implementation follows a structured machine learning workflow.

### Dataset Inspection and Profiling

- Dataset structure and schema inspection.
- Missing-value and empty-text checks.
- Duplicate record identification.
- Category distribution analysis.
- Text-length analysis.
- Parliament-wise dataset profiling.

### Language Identification

MillID was used to generate language metadata for the parliamentary texts.

The processed dataset includes:

- Detected language.
- Standardized language label.
- Language identification confidence.
- Top language predictions.

These metadata fields are also used for language-group evaluation.

### Text Feature Extraction

A combined TF-IDF representation is used to transform parliamentary text into numerical features.

**Word-level TF-IDF**
- Unigrams and bigrams.
- Sublinear term-frequency scaling.
- Vocabulary-based feature extraction.

**Character-level TF-IDF**
- Character n-grams.
- `char_wb` analyzer.
- Character-level representations to capture spelling patterns and linguistic variation.

The word and character feature representations are combined before classification.

## Repository Structure

```text
historical_speech_classification/
│
├── raw/
│   └── ParlaCAP-train/
│
├── processed/
│   └── ParlaCAP-language-detected.jsonl
│
├── src/
│   ├── data/
│   │   ├── inspect_dataset.py
│   │   └── profile_dataset.py
│   │
│   └── models/
│       ├── train_baseline.py
│       ├── predict.py
│       ├── analyze_errors.py
│       ├── evaluate_groups.py
│       ├── evaluate_languages.py
│       └── visualize_results.py
│
├── models/
│   └── baseline/
│       ├── tfidf_features.joblib
│       ├── linear_svm.joblib
│       ├── predictions.npy
│       └── true_labels.npy
│
├── reports/
│   └── baseline/
│       ├── confusion_matrix.csv
│       ├── confusion_pairs.csv
│       ├── misclassified_records.csv
│       ├── parliament_evaluation.csv
│       ├── validation_predictions.csv
│       ├── language_evaluation_all.csv
│       └── language_evaluation_min30.csv
│
│   └── visualizations/
│       ├── 01_confusion_matrix.png
│       ├── 02_class_wise_performance.png
│       ├── 03_overall_performance.png
│       ├── 04_parliament_wise_performance.png
│       ├── 05_language_wise_performance.png
│       └── 06_top_confusion_pairs.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

| Area | Technologies |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Natural Language Processing | TF-IDF |
| Language identification | MillID |
| Machine learning | Scikit-learn |
| Classification | LinearSVC |
| Model persistence | Joblib |
| Data visualization | Matplotlib, Seaborn |
| Development environment | Google Colab |

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/historical_speech_classification.git

cd historical_speech_classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Train the baseline classifier

```bash
python src/models/train_baseline.py
```

### Run prediction

```bash
python src/models/predict.py
```

### Perform error analysis

```bash
python src/models/analyze_errors.py
```

### Evaluate by parliament

```bash
python src/models/evaluate_groups.py
```

### Evaluate by language

```bash
python src/models/evaluate_languages.py
```

### Generate evaluation visualizations

```bash
python src/models/visualize_results.py
```

## Limitations

- The implementation uses a conventional TF-IDF and linear-classification approach.
- Policy categories have an uneven distribution of records.
- Some policy categories have overlapping terminology and subject matter.
- Language identification metadata is automatically generated and has not been independently verified.
- Language-wise results depend on detected-language group composition and sample size.
- Evaluation is based on the supplied training and development partitions.

## Acknowledgements

- [ParlaCAP Dataset](https://www.clarin.si/repository/xmlui/handle/11356/2093)
- [Comparative Agendas Project](https://www.comparativeagendas.net/)
- [Scikit-learn](https://scikit-learn.org/)
- [MillID](https://huggingface.co/)

