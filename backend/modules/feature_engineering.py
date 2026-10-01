# CELL 1 - Import future type-annotation support, regex, punctuation tools and pandas
from __future__ import annotations
import re
import string
import pandas as pd

# ============================================================
# Column order the fusion model expects. Must match the notebook.
# ============================================================
# CELL 2 - Define the exact 16 engineered linguistic features expected by the fusion model
ENGINEERED_COLS = [
    "word_count", "char_count", "avg_word_length", "sentence_count",
    "avg_sentence_length", "flesch_reading_ease", "flesch_kincaid_grade",
    "unique_word_ratio", "exclamation_ratio", "capital_word_ratio",
    "punctuation_density", "sentiment_polarity", "sentiment_subjectivity",
    "adjective_ratio", "noun_ratio", "verb_ratio",
]

# ============================================================
# Lazy NLTK setup
# ============================================================
# CELL 3 - Track whether required NLTK resources have already been checked/downloaded
_NLTK_READY = False

# CELL 4 - Download required NLTK tokenization and POS-tagging resources only once
def _ensure_nltk():
    """Downloads the small NLTK models these features need, once."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    for resource, pkg in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass  # older/newer NLTK versions don't all ship every package name above
    _NLTK_READY = True

# CELL 5 - Prepare regex, punctuation set and POS-tag prefixes used during feature extraction
_WORD_RE = re.compile(r"[A-Za-z']+")
_PUNCT_SET = set(string.punctuation)
# Penn Treebank POS tag prefixes
_ADJ_TAGS = ("JJ",)
_NOUN_TAGS = ("NN",)
_VERB_TAGS = ("VB",)

# CELL 6 - Extract English words from review text using regular expressions
def _split_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)

# CELL 7 - Split review into sentences using NLTK with regex as a fallback
def _split_sentences(text: str) -> list[str]:
    try:
        _ensure_nltk()
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
    except Exception:
        # Fallback: split on ./!/? if NLTK isn't available.
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if s.strip()]

# CELL 8 - Assign grammatical POS tags to words for adjective, noun and verb analysis
def _pos_tags(words: list[str]) -> list[tuple[str, str]]:
    try:
        _ensure_nltk()
        from nltk import pos_tag
        return pos_tag(words)
    except Exception:
        return [(w, "") for w in words]  # ratios become 0 rather than crashing

# CELL 9 - Calculate Flesch Reading Ease and Flesch-Kincaid Grade readability scores
def _flesch_scores(text: str) -> tuple[float, float]:
    try:
        import textstat
        return (
            float(textstat.flesch_reading_ease(text)),
            float(textstat.flesch_kincaid_grade(text)),
        )
    except Exception:
        return 0.0, 0.0

# CELL 10 - Calculate sentiment polarity and subjectivity using TextBlob
def _sentiment(text: str) -> tuple[float, float]:
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        return float(blob.sentiment.polarity), float(blob.sentiment.subjectivity)
    except Exception:
        return 0.0, 0.0

# CELL 11 - Extract all 16 engineered linguistic/stylometric features from one review
def extract_features(text: str) -> dict:
    """Computes all 16 ENGINEERED_COLS for a single review string."""
    text = "" if text is None else str(text)
    words = _split_words(text)
    word_count = len(words)  # Total number of words
    char_count = len(text)  # Total number of characters
    sentences = _split_sentences(text)
    sentence_count = max(len(sentences), 1)  # Total sentences, minimum 1 to avoid division by zero
    avg_word_length = (sum(len(w) for w in words) / word_count) if word_count else 0.0  # Average characters per word
    avg_sentence_length = word_count / sentence_count  # Average words per sentence
    flesch_reading_ease, flesch_kincaid_grade = _flesch_scores(text)  # Readability measurements
    unique_word_ratio = (len({w.lower() for w in words}) / word_count) if word_count else 0.0  # Vocabulary diversity

    # Convention: exclamation marks relative to sentence count (how "shouty" is
    # each sentence on average), not relative to raw character count.
    exclamation_ratio = (text.count("!") / sentence_count) if sentence_count else 0.0

    # Convention: fraction of words that are fully upper-case and at least
    # 2 characters long (so lone capital letters like "I" don't count).
    capital_words = sum(1 for w in words if len(w) > 1 and w.isupper())
    capital_word_ratio = (capital_words / word_count) if word_count else 0.0

    # Convention: punctuation characters as a fraction of all characters.
    punctuation_density = (
        sum(1 for ch in text if ch in _PUNCT_SET) / char_count
    ) if char_count else 0.0

    # Calculate sentiment-based linguistic features
    sentiment_polarity, sentiment_subjectivity = _sentiment(text)

    # Calculate adjective, noun and verb ratios from POS tags
    tags = _pos_tags(words)
    n_tags = len(tags) or 1
    adjective_ratio = sum(1 for _, t in tags if t.startswith(_ADJ_TAGS)) / n_tags
    noun_ratio = sum(1 for _, t in tags if t.startswith(_NOUN_TAGS)) / n_tags
    verb_ratio = sum(1 for _, t in tags if t.startswith(_VERB_TAGS)) / n_tags

    # Return the 16 calculated features using the names expected by the model
    return {
        "word_count": word_count,
        "char_count": char_count,
        "avg_word_length": avg_word_length,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "unique_word_ratio": unique_word_ratio,
        "exclamation_ratio": exclamation_ratio,
        "capital_word_ratio": capital_word_ratio,
        "punctuation_density": punctuation_density,
        "sentiment_polarity": sentiment_polarity,
        "sentiment_subjectivity": sentiment_subjectivity,
        "adjective_ratio": adjective_ratio,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
    }

# CELL 12 - Extract the same 16 features from multiple reviews and return them as a DataFrame
def extract_features_batch(texts) -> pd.DataFrame:
    """Vectorized-ish convenience wrapper: list/Series of strings -> DataFrame
    with one row per text and the 16 ENGINEERED_COLS as columns."""
    rows = [extract_features(t) for t in texts]  # Extract 16 features from each review
    return pd.DataFrame(rows, columns=ENGINEERED_COLS)  # Preserve exact feature order required by model