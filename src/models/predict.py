from pathlib import Path
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"

# Load trained artifacts
vectorizer = joblib.load(
    MODEL_DIR / "tfidf_features.joblib"
)

classifier = joblib.load(
    MODEL_DIR / "linear_svm.joblib"
)


def predict_topic(text):
    """
    Predict the CAP topic category of parliamentary text.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty.")

    features = vectorizer.transform([text])

    prediction = classifier.predict(features)[0]

    return {
        "predicted_category": prediction
    }


if __name__ == "__main__":

    sample_text = """
    The government must increase investment in
    public education and improve access to schools.
    """

    result = predict_topic(sample_text)

    print("\n===== PREDICTION =====")
    print("Text:", sample_text)
    print("Predicted category:", result["predicted_category"])
