"""Handcrafted stylometric features: prompt-independent authorial habits.

impl: FR-012-02, FR-012-03
impl: FR-013-02, FR-013-03
"""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence

import numpy as np

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")

# Fixed English function-word resources. They are linguistic constants, not
# tuned thresholds: unknown-language texts simply score near zero here.
STOPWORDS = frozenset(
    "a an the and or but if then else when while of at by for with about into "
    "through during before after above below to from up down in out on off over "
    "under again further once here there all any both each few more most other "
    "some such no nor not only own same so than too very can will just should "
    "now is are was were be been being have has had having do does did doing "
    "would could ought as".split()
)
FIRST_PERSON = frozenset("i me my mine we us our ours myself ourselves".split())
SECOND_PERSON = frozenset("you your yours yourself yourselves".split())
THIRD_PERSON = frozenset(
    "he him his himself she her hers herself it its itself "
    "they them their theirs themselves".split()
)
_CONTRACTION_RES = tuple(
    re.compile(pattern) for pattern in (r"n't\b", r"'re\b", r"'ve\b", r"'ll\b", r"'d\b", r"'m\b")
)

# Rhetorical/formatting voice markers. Fixed inventory, counted with word
# boundaries on casefolded text; structural marks are per-line ratios.
DISCOURSE_TERMS = (
    "moreover", "furthermore", "however", "therefore", "thus", "hence",
    "additionally", "firstly", "secondly", "thirdly", "finally",
    "in conclusion", "in summary", "for example", "for instance",
    "on the other hand", "overall", "ultimately", "notably", "importantly",
    "crucially", "significantly", "essentially", "indeed", "certainly",
    "specifically", "particularly", "generally", "typically", "conversely",
    "nevertheless", "nonetheless", "meanwhile", "subsequently", "consequently",
)
HEDGING_TERMS = (
    "maybe", "perhaps", "might", "could", "likely", "possibly", "presumably",
    "tends to", "seems", "appears", "arguably", "roughly", "largely",
)
SELF_REFERENCE_PHRASES = (
    "as an ai", "as a language model", "my training", "my knowledge",
    "my cutoff", "language model",
)
CLOSING_PHRASES = (
    "hope this helps", "let me know", "feel free", "happy to help",
    "glad to help",
)
_DISCOURSE_RES = tuple(re.compile(r"\b" + re.escape(term) + r"\b") for term in DISCOURSE_TERMS)
_HEDGING_RES = tuple(re.compile(r"\b" + re.escape(term) + r"\b") for term in HEDGING_TERMS)
_SELF_REF_RES = tuple(re.compile(r"\b" + re.escape(term) + r"\b") for term in SELF_REFERENCE_PHRASES)
_CLOSING_RES = tuple(re.compile(r"\b" + re.escape(term) + r"\b") for term in CLOSING_PHRASES)
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s+\S")
_EMOJI_RE = re.compile(r"[🌀-🫿☀-➿⬀-⯿️]")


def _phrase_rate(patterns: Sequence[re.Pattern], lowered: str, n_words: float) -> float:
    if not n_words:
        return 0.0
    return sum(len(pattern.findall(lowered)) for pattern in patterns) / n_words * 1000.0

_PUNCT_MARKS: tuple[tuple[str, str], ...] = (
    ("comma_ratio", ","),
    ("semicolon_ratio", ";"),
    ("colon_ratio", ":"),
    ("exclamation_ratio", "!"),
    ("question_ratio", "?"),
    ("period_ratio", "."),
    ("apostrophe_ratio", "'"),
    ("quote_ratio", '"'),
    ("hyphen_ratio", "-"),
    ("em_dash_ratio", "—"),
    ("en_dash_ratio", "–"),
    ("ellipsis_ratio", "…"),
    ("open_paren_ratio", "("),
    ("close_paren_ratio", ")"),
    ("bracket_ratio", "[]{}"),
    ("slash_ratio", "/"),
    ("backslash_ratio", "\\"),
    ("ampersand_ratio", "&"),
    ("at_ratio", "@"),
    ("hash_ratio", "#"),
    ("percent_ratio", "%"),
    ("dollar_ratio", "$"),
    ("plus_ratio", "+"),
    ("equals_ratio", "="),
    ("asterisk_ratio", "*"),
    ("underscore_ratio", "_"),
    ("pipe_ratio", "|"),
    ("tilde_ratio", "~"),
    ("caret_ratio", "^"),
    ("backtick_ratio", "`"),
)

FEATURE_NAMES: tuple[str, ...] = (
    "n_chars",
    "n_words",
    "n_sentences",
    "avg_word_len",
    "avg_sentence_len_words",
    "avg_sentence_len_chars",
    "type_token_ratio",
    "hapax_ratio",
    "uppercase_ratio",
    "capitalized_word_ratio",
    "allcaps_word_ratio",
    "digit_ratio",
    *(name for name, _ in _PUNCT_MARKS),
    "paragraph_count",
    "list_item_ratio",
    "stopword_rate",
    "first_person_rate",
    "second_person_rate",
    "third_person_rate",
    "contraction_rate",
    "long_word_ratio",
    "short_word_ratio",
    "most_frequent_word_share",
    "starts_with_capital",
    "ends_with_period",
    "ends_with_exclamation",
    "ends_with_question",
    "discourse_rate",
    "hedging_rate",
    "self_reference_rate",
    "closing_rate",
    "header_line_ratio",
    "numbered_line_ratio",
    "blockquote_ratio",
    "code_fence_line_ratio",
    "bold_rate",
    "inline_code_rate",
    "emoji_rate",
)


class StylometryError(ValueError):
    """A passage cannot be turned into style features."""


def _sentence_segments(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _featurize(text: str) -> list[float]:
    if not isinstance(text, str):
        raise StylometryError(f"expected a string passage, got {type(text).__name__}")
    if not text.strip():
        raise StylometryError("blank passage has no style to measure")
    n_chars = float(len(text))
    words = _WORD_RE.findall(text)
    lowered = [word.casefold() for word in words]
    n_words = float(len(words))
    segments = _sentence_segments(text)
    n_sentences = float(len(segments))

    seg_word_counts = []
    seg_char_lens = []
    for segment in segments:
        seg_words = _WORD_RE.findall(segment)
        seg_word_counts.append(len(seg_words))
        seg_char_lens.append(len(segment))

    values: dict[str, float] = {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sentences": n_sentences,
        "avg_word_len": (sum(len(word) for word in words) / n_words) if n_words else 0.0,
        "avg_sentence_len_words": (sum(seg_word_counts) / n_sentences) if n_sentences else 0.0,
        "avg_sentence_len_chars": (sum(seg_char_lens) / n_sentences) if n_sentences else 0.0,
    }
    if n_words:
        counts: dict[str, int] = {}
        for word in lowered:
            counts[word] = counts.get(word, 0) + 1
        values["type_token_ratio"] = len(counts) / n_words
        values["hapax_ratio"] = sum(1 for count in counts.values() if count == 1) / n_words
        values["capitalized_word_ratio"] = sum(
            1 for word in words if word[:1].isupper() and word[1:].islower()
        ) / n_words
        values["allcaps_word_ratio"] = sum(1 for word in words if len(word) > 1 and word.isupper()) / n_words
        values["stopword_rate"] = sum(1 for word in lowered if word in STOPWORDS) / n_words
        values["first_person_rate"] = sum(1 for word in lowered if word in FIRST_PERSON) / n_words
        values["second_person_rate"] = sum(1 for word in lowered if word in SECOND_PERSON) / n_words
        values["third_person_rate"] = sum(1 for word in lowered if word in THIRD_PERSON) / n_words
        lowered_text = text.casefold()
        values["contraction_rate"] = sum(len(pattern.findall(lowered_text)) for pattern in _CONTRACTION_RES) / n_words
        values["long_word_ratio"] = sum(1 for word in words if len(word) > 6) / n_words
        values["short_word_ratio"] = sum(1 for word in words if len(word) <= 3) / n_words
        values["most_frequent_word_share"] = max(counts.values()) / n_words
    else:
        for key in (
            "type_token_ratio", "hapax_ratio", "capitalized_word_ratio", "allcaps_word_ratio",
            "stopword_rate", "first_person_rate", "second_person_rate", "third_person_rate",
            "contraction_rate", "long_word_ratio", "short_word_ratio", "most_frequent_word_share",
        ):
            values[key] = 0.0

    values["uppercase_ratio"] = sum(1 for char in text if char == char.upper() and char.isalpha()) / n_chars
    values["digit_ratio"] = sum(1 for char in text if char.isdigit()) / n_chars
    for name, marks in _PUNCT_MARKS:
        values[name] = sum(1 for char in text if char in marks) / n_chars

    paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    values["paragraph_count"] = float(len(paragraphs))
    lines = text.splitlines() or [text]
    values["list_item_ratio"] = sum(1 for line in lines if _LIST_ITEM_RE.match(line)) / max(len(lines), 1)

    if n_sentences:
        values["starts_with_capital"] = sum(
            1 for segment in segments if segment[:1].isupper()
        ) / n_sentences
        values["ends_with_period"] = sum(1 for segment in segments if segment.endswith((".", "…"))) / n_sentences
        stripped = text.rstrip()
        values["ends_with_exclamation"] = 1.0 if stripped.endswith("!") else 0.0
        values["ends_with_question"] = 1.0 if stripped.endswith("?") else 0.0
    else:
        values["starts_with_capital"] = 0.0
        values["ends_with_period"] = 0.0
        values["ends_with_exclamation"] = 0.0
        values["ends_with_question"] = 0.0

    lowered_text = text.casefold()
    values["discourse_rate"] = _phrase_rate(_DISCOURSE_RES, lowered_text, n_words)
    values["hedging_rate"] = _phrase_rate(_HEDGING_RES, lowered_text, n_words)
    values["self_reference_rate"] = _phrase_rate(_SELF_REF_RES, lowered_text, n_words)
    values["closing_rate"] = _phrase_rate(_CLOSING_RES, lowered_text, n_words)
    lines = text.splitlines() or [text]
    n_lines = max(len(lines), 1)
    values["header_line_ratio"] = sum(1 for line in lines if line.lstrip().startswith("#")) / n_lines
    values["numbered_line_ratio"] = sum(1 for line in lines if _NUMBERED_LINE_RE.match(line)) / n_lines
    values["blockquote_ratio"] = sum(1 for line in lines if line.lstrip().startswith(">")) / n_lines
    values["code_fence_line_ratio"] = sum(1 for line in lines if "```" in line) / n_lines
    values["bold_rate"] = text.count("**") / max(n_chars, 1.0) * 1000.0
    values["inline_code_rate"] = text.count("`") / max(n_chars, 1.0) * 1000.0
    values["emoji_rate"] = len(_EMOJI_RE.findall(text)) / max(n_chars, 1.0) * 1000.0

    return [float(values[name]) for name in FEATURE_NAMES]


def extract_style_matrix(texts: Sequence[str]) -> np.ndarray:
    """Turn passages into a fixed-order finite dense matrix, deterministically."""
    rows = [_featurize(text) for text in texts]
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(texts), len(FEATURE_NAMES)):
        raise StylometryError("style extractor returned an unexpected shape")
    if not np.all(np.isfinite(matrix)):
        raise StylometryError("style extractor produced non-finite values")
    return matrix


class StyleTransformer:
    """sklearn-compatible wrapper so style features fit inside a Pipeline."""

    def fit(self, texts, y=None):  # noqa: ANN001, ANN202 - sklearn duck-type
        return self

    def transform(self, texts):
        return extract_style_matrix(list(texts))

    def fit_transform(self, texts, y=None):
        return self.fit(texts, y).transform(texts)


def _normalize_casefold(text: str) -> str:
    return unicodedata.normalize("NFKC", text.casefold())
