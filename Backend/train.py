import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import preprocess_text as base_preprocess_text

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = "spam.csv"

POSITIVE_LABEL_HINTS = {
    "spam",
    "phish",
    "phishing",
    "yes",
    "y",
    "true",
    "t",
    "1",
    "positive",
    "malicious",
    "fraud",
    "attack",
}
NEGATIVE_LABEL_HINTS = {
    "ham",
    "legit",
    "legitimate",
    "no",
    "n",
    "false",
    "f",
    "0",
    "negative",
    "safe",
    "benign",
    "normal",
}


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train spam/phishing text classifier")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="CSV dataset path")
    parser.add_argument("--text_col", default=None, help="Text column name")
    parser.add_argument("--label_col", default=None, help="Label column name")
    parser.add_argument("--encoding", default="latin-1", help="CSV encoding")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    parser.add_argument("--cv_folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument(
        "--use_stopwords",
        type=str2bool,
        default=True,
        help="Enable stopword removal in preprocessing",
    )
    parser.add_argument(
        "--use_homoglyph",
        type=str2bool,
        default=True,
        help="Enable homoglyph normalization",
    )
    parser.add_argument(
        "--balance_strategy",
        choices=["balanced", "none"],
        default="balanced",
        help="Class imbalance strategy",
    )
    parser.add_argument(
        "--finetune",
        type=str2bool,
        default=False,
        help="Enable GridSearchCV tuning",
    )
    parser.add_argument(
        "--compare_spam",
        type=str2bool,
        default=True,
        help="Print Dataset A vs Dataset B comparison when dataset is not spam.csv",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_value: str) -> Path:
    dataset_path = Path(dataset_value)
    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / dataset_path
    return dataset_path


def slugify_dataset(dataset_path: Path) -> str:
    slug = dataset_path.stem.lower()
    slug = "".join(char if char.isalnum() else "_" for char in slug)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "dataset"


def infer_columns(df: pd.DataFrame, text_col: Optional[str], label_col: Optional[str]) -> Tuple[str, str]:
    columns = list(df.columns)
    if text_col and label_col:
        if text_col not in columns or label_col not in columns:
            raise ValueError(f"Provided columns not found. Available columns: {columns}")
        return text_col, label_col

    text_candidates = ["message", "text", "content", "email", "body", "v2"]
    label_candidates = ["label", "target", "class", "is_spam", "spam", "v1"]

    detected_text = text_col
    detected_label = label_col

    if not detected_text:
        for candidate in text_candidates:
            if candidate in columns:
                detected_text = candidate
                break

    if not detected_label:
        for candidate in label_candidates:
            if candidate in columns:
                detected_label = candidate
                break

    if detected_text and detected_label:
        return detected_text, detected_label

    raise ValueError(
        "Could not auto-detect columns. Pass --text_col and --label_col explicitly. "
        f"Available columns: {columns}"
    )


def normalize_labels(raw_labels: pd.Series) -> pd.Series:
    labels = raw_labels.copy()
    labels = labels.dropna()
    if labels.empty:
        raise ValueError("No valid label values found after dropping missing values.")

    normalized = labels.astype(str).str.strip().str.lower()
    unique_values = sorted(normalized.unique())
    if len(unique_values) != 2:
        raise ValueError(f"Expected binary labels; found {len(unique_values)} unique values: {unique_values}")

    def score_value(value: str) -> int:
        if value in POSITIVE_LABEL_HINTS:
            return 2
        if value in NEGATIVE_LABEL_HINTS:
            return 0
        if value.replace(".", "", 1).isdigit():
            return 2 if float(value) > 0 else 0
        return 1

    value_scores = {value: score_value(value) for value in unique_values}
    sorted_values = sorted(unique_values, key=lambda item: (value_scores[item], item))
    negative_value, positive_value = sorted_values[0], sorted_values[-1]
    mapping = {negative_value: 0, positive_value: 1}
    return normalized.map(mapping)


def preprocess_text_series(
    text_series: pd.Series,
    use_stopwords: bool,
    use_homoglyph: bool,
    stopword_set: Optional[set],
) -> pd.Series:
    return text_series.apply(
        lambda value: preprocess_text(
            value,
            use_stopwords=use_stopwords,
            use_homoglyph=use_homoglyph,
            stopword_set=stopword_set,
        )
    )


def preprocess_text(
    text: object,
    use_stopwords: bool,
    use_homoglyph: bool,
    stopword_set: Optional[set],
) -> str:
    return base_preprocess_text(
        text,
        use_stopwords=use_stopwords,
        use_homoglyph=use_homoglyph,
        stopword_set=stopword_set,
    )


def load_data(args: argparse.Namespace) -> Tuple[pd.Series, pd.Series, Dict[str, object]]:
    dataset_path = resolve_dataset_path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, encoding=args.encoding)
    if df.empty:
        raise ValueError("Dataset is empty.")

    text_col, label_col = infer_columns(df, args.text_col, args.label_col)
    selected = df[[text_col, label_col]].copy()
    selected = selected.dropna(subset=[text_col, label_col])
    selected[text_col] = selected[text_col].astype(str).str.strip()
    selected = selected[selected[text_col] != ""]
    if selected.empty:
        raise ValueError("No rows left after removing missing/empty text or labels.")

    y = normalize_labels(selected[label_col])
    aligned = selected.loc[y.index].copy()

    nltk.download("stopwords", quiet=True)
    stopword_set = set(stopwords.words("english")) if args.use_stopwords else None
    aligned[text_col] = preprocess_text_series(
        aligned[text_col],
        use_stopwords=args.use_stopwords,
        use_homoglyph=args.use_homoglyph,
        stopword_set=stopword_set,
    )
    aligned = aligned[aligned[text_col] != ""]
    y = y.loc[aligned.index]

    if y.nunique() != 2:
        raise ValueError("Binary classification requires exactly 2 label classes after preprocessing.")

    metadata = {
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.name,
        "dataset_slug": slugify_dataset(dataset_path),
        "text_col": text_col,
        "label_col": label_col,
        "rows": int(len(aligned)),
        "class_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    }
    return aligned[text_col], y.astype(int), metadata


def safe_train_test_split(
    X: pd.Series,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if len(X) < 6:
        raise ValueError("Dataset too small. Need at least 6 valid rows.")

    min_class_count = int(y.value_counts().min())
    if min_class_count < 2:
        raise ValueError("Each class must have at least 2 samples for train/test split.")

    stratify_target = y if min_class_count >= 2 else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def get_candidates(balance_strategy: str) -> List[Dict[str, object]]:
    class_weight = "balanced" if balance_strategy == "balanced" else None
    return [
        {
            "name": "LinearSVC_char_3_5",
            "vectorizer": TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
            ),
            "model": LinearSVC(C=1.2, class_weight=class_weight),
            "param_grid": {
                "vectorizer__ngram_range": [(3, 5), (3, 6)],
                "model__C": [0.8, 1.2, 2.0],
            },
        },
        {
            "name": "LogReg_word_1_2",
            "vectorizer": TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
            "model": LogisticRegression(max_iter=500, C=2.0, class_weight=class_weight),
            "param_grid": {
                "vectorizer__ngram_range": [(1, 1), (1, 2)],
                "model__C": [0.5, 1.0, 2.0, 3.0],
            },
        },
        {
            "name": "MultinomialNB_word_1_2",
            "vectorizer": TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
            "model": MultinomialNB(alpha=0.2),
            "param_grid": {
                "vectorizer__ngram_range": [(1, 1), (1, 2)],
                "model__alpha": [0.05, 0.1, 0.2, 0.5],
            },
        },
    ]


def get_effective_cv_folds(y_train: pd.Series, requested_folds: int) -> int:
    min_class_count = int(y_train.value_counts().min())
    return max(2, min(requested_folds, min_class_count))


def train_and_evaluate(
    candidate: Dict[str, object],
    X_train_text: pd.Series,
    X_test_text: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int,
    finetune: bool,
) -> Dict[str, object]:
    pipeline = Pipeline(
        [
            ("vectorizer", candidate["vectorizer"]),
            ("model", candidate["model"]),
        ]
    )

    cv_scores = cross_val_score(
        pipeline,
        X_train_text,
        y_train,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )

    best_estimator = pipeline
    best_cv = float(cv_scores.mean())
    tuning_params = {}

    if finetune:
        search = GridSearchCV(
            pipeline,
            candidate["param_grid"],
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(X_train_text, y_train)
        best_estimator = search.best_estimator_
        best_cv = float(search.best_score_)
        tuning_params = search.best_params_
    else:
        best_estimator.fit(X_train_text, y_train)

    y_pred = best_estimator.predict(X_test_text)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        "name": candidate["name"],
        "pipeline": best_estimator,
        "accuracy": float(test_accuracy),
        "cv_mean_accuracy": best_cv,
        "cv_scores": [float(score) for score in cv_scores],
        "classification_report": report,
        "best_params": tuning_params,
        "y_pred": y_pred,
    }


def evaluate_models(
    candidates: List[Dict[str, object]],
    X_train_text: pd.Series,
    X_test_text: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int,
    finetune: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    results: List[Dict[str, object]] = []
    print("\nModel comparison:\n")

    for candidate in candidates:
        result = train_and_evaluate(
            candidate,
            X_train_text,
            X_test_text,
            y_train,
            y_test,
            cv_folds=cv_folds,
            finetune=finetune,
        )
        results.append(result)
        print(f"{result['name']} test accuracy: {result['accuracy']:.4f}")
        print(f"{result['name']} cv mean accuracy: {result['cv_mean_accuracy']:.4f}")
        if result["best_params"]:
            print(f"Best params: {result['best_params']}")
        print(result["classification_report"])
        print("-" * 60)

    winner = max(results, key=lambda item: (item["accuracy"], item["cv_mean_accuracy"]))
    return winner, results


def train_models(args: argparse.Namespace) -> Dict[str, object]:
    X, y, metadata = load_data(args)
    X_train_text, X_test_text, y_train, y_test = safe_train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    cv_folds = get_effective_cv_folds(y_train, args.cv_folds)
    if cv_folds < args.cv_folds:
        print(f"Requested cv_folds={args.cv_folds} adjusted to cv_folds={cv_folds} due to class counts.")

    candidates = get_candidates(args.balance_strategy)
    winner, results = evaluate_models(
        candidates,
        X_train_text,
        X_test_text,
        y_train,
        y_test,
        cv_folds=cv_folds,
        finetune=args.finetune,
    )

    winner_pipeline = winner["pipeline"]
    winner_vectorizer = winner_pipeline.named_steps["vectorizer"]
    winner_model = winner_pipeline.named_steps["model"]

    output = {
        "metadata": metadata,
        "winner": winner,
        "all_results": results,
        "vectorizer": winner_vectorizer,
        "model": winner_model,
        "test_size": len(y_test),
        "cv_folds": cv_folds,
    }
    return output


def save_outputs(training_output: Dict[str, object], args: argparse.Namespace) -> Path:
    metadata = training_output["metadata"]
    winner = training_output["winner"]
    slug = metadata["dataset_slug"]

    model_path = BASE_DIR / f"model_{slug}.pkl"
    vectorizer_path = BASE_DIR / f"vectorizer_{slug}.pkl"
    metrics_path = BASE_DIR / f"metrics_{slug}.json"

    pickle.dump(training_output["model"], open(model_path, "wb"))
    pickle.dump(training_output["vectorizer"], open(vectorizer_path, "wb"))

    metrics_payload = {
        "dataset": metadata["dataset_name"],
        "dataset_path": metadata["dataset_path"],
        "text_col": metadata["text_col"],
        "label_col": metadata["label_col"],
        "rows": metadata["rows"],
        "class_distribution": metadata["class_distribution"],
        "model_name": winner["name"],
        "accuracy": round(float(winner["accuracy"]), 4),
        "cv_mean_accuracy": round(float(winner["cv_mean_accuracy"]), 4),
        "cv_scores": [round(float(score), 4) for score in winner["cv_scores"]],
        "best_params": winner["best_params"],
        "test_size": training_output["test_size"],
        "cv_folds": training_output["cv_folds"],
        "finetune": bool(args.finetune),
        "use_stopwords": bool(args.use_stopwords),
        "use_homoglyph": bool(args.use_homoglyph),
        "balance_strategy": args.balance_strategy,
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    if slug == "spam":
        pickle.dump(training_output["model"], open(BASE_DIR / "model.pkl", "wb"))
        pickle.dump(training_output["vectorizer"], open(BASE_DIR / "vectorizer.pkl", "wb"))
        with open(BASE_DIR / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(metrics_payload, file, indent=2)

    print("\nArtifacts saved:")
    print(f"- {model_path.name}")
    print(f"- {vectorizer_path.name}")
    print(f"- {metrics_path.name}")
    if slug == "spam":
        print("- model.pkl")
        print("- vectorizer.pkl")
        print("- metrics.json")

    return metrics_path


def print_dataset_comparison(current_metrics_path: Path, current_slug: str, compare_spam: bool) -> None:
    if not compare_spam or current_slug == "spam":
        return

    spam_metrics_path = BASE_DIR / "metrics_spam.json"
    if not spam_metrics_path.exists():
        print("\nDataset comparison skipped: metrics_spam.json not found.")
        print("Run once with default dataset (spam.csv) to generate Dataset A baseline.")
        return

    with open(current_metrics_path, "r", encoding="utf-8") as file:
        current_metrics = json.load(file)
    with open(spam_metrics_path, "r", encoding="utf-8") as file:
        spam_metrics = json.load(file)

    current_acc = float(current_metrics.get("accuracy", 0.0))
    spam_acc = float(spam_metrics.get("accuracy", 0.0))
    delta = current_acc - spam_acc

    print("\nDataset A vs Dataset B comparison")
    print(f"Dataset A (spam.csv) accuracy: {spam_acc:.4f}")
    print(f"Dataset B ({current_metrics.get('dataset')}) accuracy: {current_acc:.4f}")
    print(f"Delta (B - A): {delta:+.4f}")


def main() -> None:
    args = parse_args()
    output = train_models(args)
    winner = output["winner"]
    metadata = output["metadata"]

    print("\nBest model summary")
    print(f"Winner: {winner['name']}")
    print(f"Dataset: {metadata['dataset_name']}")
    print(f"Rows used: {metadata['rows']}")
    print(f"Best accuracy: {winner['accuracy']:.4f}")
    print(f"Cross-val mean accuracy: {winner['cv_mean_accuracy']:.4f}")
    print("\nBest classification report:\n")
    print(winner["classification_report"])

    metrics_path = save_outputs(output, args)
    print_dataset_comparison(metrics_path, metadata["dataset_slug"], args.compare_spam)


if __name__ == "__main__":
    main()