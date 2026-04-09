import pandas as pd
import string
import nltk
import pickle
import re
import unicodedata

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 🔥 Homoglyph mapping
HOMOGLYPH_MAP = {
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'і': 'i', 'ѕ': 's',
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Κ': 'K', 'Μ': 'M', 'Ν': 'N',
    'О': 'O', 'Ρ': 'P', 'Τ': 'T', 'Χ': 'X'
}

def replace_homoglyphs(text):
    return ''.join(HOMOGLYPH_MAP.get(c, c) for c in text)

# 🔥 ADVANCED PREPROCESSING
def preprocess(text):
    # Normalize
    text = unicodedata.normalize('NFKD', text)

    # 🔥 Replace homoglyphs
    text = replace_homoglyphs(text)

    # Lowercase
    text = text.lower()

    # Replace common obfuscations
    text = text.replace("0", "o").replace("1", "i").replace("3", "e")

    # Remove special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)

    return text
df['message'] = df['message'].apply(preprocess)

# 🔥 N-GRAM FEATURE EXTRACTION (Fix signal dilution)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

best_model = None
best_accuracy = 0
results = {}

print("\n🔍 Model Comparison:\n")

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# Save best model
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Best Model Saved!")
print(f"🏆 Best Accuracy: {best_accuracy:.4f}")