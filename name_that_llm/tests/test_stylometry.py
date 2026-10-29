# verifies: FR-012-02
# verifies: FR-012-03
# verifies: FR-013-02
# verifies: FR-013-03

import numpy as np
import pytest

from ntllm import stylometry
from ntllm.stylometry import StyleTransformer, extract_style_matrix


def test_feature_names_are_fixed_and_ordered():
    first = list(stylometry.FEATURE_NAMES)
    assert len(first) > 30
    assert len(set(first)) == len(first)
    assert first == list(stylometry.FEATURE_NAMES)


def test_extract_known_text_values():
    matrix = extract_style_matrix(["Hi!"])

    assert matrix.shape == (1, len(stylometry.FEATURE_NAMES))
    assert np.all(np.isfinite(matrix))
    row = dict(zip(stylometry.FEATURE_NAMES, matrix[0]))
    assert row["n_chars"] == 3.0
    assert row["n_words"] == 1.0
    assert row["n_sentences"] == 1.0
    assert row["avg_word_len"] == pytest.approx(2.0)
    assert row["exclamation_ratio"] == pytest.approx(1.0 / 3.0)
    assert row["uppercase_ratio"] == pytest.approx(1.0 / 3.0)
    assert row["capitalized_word_ratio"] == pytest.approx(1.0)
    assert row["ends_with_exclamation"] == pytest.approx(1.0)
    assert row["ends_with_period"] == pytest.approx(0.0)


def test_extract_is_deterministic_and_finite_on_batch():
    texts = ["First passage, with words.", "Second: shorter!", "   spaced   out   "]
    first = extract_style_matrix(texts)
    second = extract_style_matrix(texts)

    assert first.shape == (3, len(stylometry.FEATURE_NAMES))
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)


def test_extract_rejects_blank_and_non_string_input():
    with pytest.raises(ValueError, match="[Bb]lank"):
        extract_style_matrix([""])
    with pytest.raises(ValueError, match="[Bb]lank"):
        extract_style_matrix(["   "])
    with pytest.raises(ValueError, match="string"):
        extract_style_matrix([None])


def test_style_transformer_fits_only_for_shape_and_scales_in_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    transformer = StyleTransformer()
    matrix = transformer.fit_transform(["Hello, brave new world!", "Short."])
    assert matrix.shape == (2, len(stylometry.FEATURE_NAMES))
    assert np.all(np.isfinite(matrix))

    pipe = Pipeline([
        ("style", StyleTransformer()),
        ("scale", StandardScaler()),
    ])
    scaled = pipe.fit_transform(["Hello, brave new world!", "Short.", "A third passage here."])
    assert scaled.shape == (3, len(stylometry.FEATURE_NAMES))
    assert np.all(np.isfinite(scaled))


def test_rhetorical_markers_counted_on_crafted_passage():
    from ntllm import stylometry

    text = "# Title\n\nMoreover, this is a test. I hope this helps!\n\n- item one\n- item two\n\n```code\nx=1\n```"
    matrix = stylometry.extract_style_matrix([text])

    assert matrix.shape == (1, len(stylometry.FEATURE_NAMES))
    assert np.all(np.isfinite(matrix))
    row = dict(zip(stylometry.FEATURE_NAMES, matrix[0]))
    assert row["discourse_rate"] == pytest.approx(1000.0 / 16.0)
    assert row["closing_rate"] == pytest.approx(1000.0 / 16.0)
    assert row["hedging_rate"] == pytest.approx(0.0)
    assert row["self_reference_rate"] == pytest.approx(0.0)
    assert row["header_line_ratio"] == pytest.approx(1.0 / 10.0)
    assert row["code_fence_line_ratio"] == pytest.approx(2.0 / 10.0)
    assert row["bold_rate"] == pytest.approx(0.0)
    assert row["emoji_rate"] == pytest.approx(0.0)


def test_rhetorical_markers_are_deterministic_and_finite():
    from ntllm import stylometry

    texts = [
        "However, **bold** claims need _no_ `code`. As an AI, I might be wrong. 😀",
        "Plain words here.",
    ]
    first = stylometry.extract_style_matrix(texts)
    second = stylometry.extract_style_matrix(texts)

    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)
    row = dict(zip(stylometry.FEATURE_NAMES, first[0]))
    assert row["discourse_rate"] > 0.0
    assert row["hedging_rate"] > 0.0
    assert row["self_reference_rate"] > 0.0
    assert row["bold_rate"] > 0.0
    assert row["inline_code_rate"] > 0.0
    assert row["emoji_rate"] > 0.0
