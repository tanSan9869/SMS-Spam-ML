import re
import unicodedata
from typing import Optional, Set

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


LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})


def replace_homoglyphs(text: str) -> str:
    return "".join(HOMOGLYPH_MAP.get(char, char) for char in text)


def preprocess_text(
    text: object,
    *,
    use_stopwords: bool = True,
    stopword_set: Optional[Set[str]] = None,
    use_homoglyph: bool = True,
) -> str:
    cleaned = unicodedata.normalize("NFKD", str(text))

    if use_homoglyph:
        cleaned = replace_homoglyphs(cleaned)

    cleaned = cleaned.lower().translate(LEET_MAP)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if use_stopwords and stopword_set:
        tokens = [token for token in cleaned.split() if token not in stopword_set]
        cleaned = " ".join(tokens)

    return cleaned
