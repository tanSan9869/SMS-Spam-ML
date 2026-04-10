import json
import pickle
import re
import unicodedata
from pathlib import Path

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

nltk.download("stopwords")

BASE_DIR = Path(__file__).resolve().parent

HOMOGLYPH_MAP = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "і": "i",
    "ѕ": "s",
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "О": "O",
    "Ρ": "P",
    "Τ": "T",
    "Χ": "X",
}


def replace_homoglyphs(text):
    return "".join(HOMOGLYPH_MAP.get(c, c) for c in text)


def preprocess(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = replace_homoglyphs(text)
    text = text.lower()
    text = text.replace("0", "o").replace("1", "i").replace("3", "e")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df = pd.read_csv(BASE_DIR / "spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df["message"] = df["message"].apply(preprocess)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

candidates = [
    (
        "LinearSVC_char_3_5",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
        ),
        LinearSVC(C=1.2),
    ),
    (
        "LogReg_word_1_2",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words="english",
        ),
        LogisticRegression(max_iter=300, C=2.0),
    ),
    (
        "MultinomialNB_word_1_2",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words="english",
        ),
        MultinomialNB(alpha=0.2),
    ),
]

best_name = None
best_vectorizer = None
best_model = None
best_accuracy = -1.0
best_report = ""

print("\nModel comparison:\n")
for name, vectorizer, model in candidates:
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"{name} accuracy: {accuracy:.4f}")
    print(report)
    print("-" * 50)

    if accuracy > best_accuracy:
        best_name = name
        best_vectorizer = vectorizer
        best_model = model
        best_accuracy = accuracy
        best_report = report

pickle.dump(best_model, open(BASE_DIR / "model.pkl", "wb"))
pickle.dump(best_vectorizer, open(BASE_DIR / "vectorizer.pkl", "wb"))

with open(BASE_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "accuracy": round(float(best_accuracy), 4),
            "model_name": best_name,
            "test_size": len(y_test),
        },
        f,
        indent=2,
    )

print("\nBest model saved")
print(f"Winner: {best_name}")
print(f"Best accuracy: {best_accuracy:.4f}")
print("\nBest classification report:\n")
print(best_report)