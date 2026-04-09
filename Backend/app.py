from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
# import string
import nltk
import re
import unicodedata
import json
from pathlib import Path
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent

# Load model and vectorizer
model = pickle.load(open(BASE_DIR / "model.pkl", "rb"))
vectorizer = pickle.load(open(BASE_DIR / "vectorizer.pkl", "rb"))

try:
    with open(BASE_DIR / "metrics.json", "r", encoding="utf-8") as f:
        model_accuracy = float(json.load(f).get("accuracy", 0.0))
except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
    model_accuracy = None

def preprocess(text):
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = text.replace("0", "o").replace("1", "i").replace("3", "e")
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["message"]
    processed = preprocess(data)
    vector = vectorizer.transform([processed])
    result = model.predict(vector)[0]

    return jsonify({
        "prediction": "Spam" if result == 1 else "Not Spam",
        "model_accuracy": model_accuracy
    })

if __name__ == "__main__":
    app.run(debug=True)