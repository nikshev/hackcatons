# verifies: FR-011-02
# verifies: FR-011-03

import numpy as np
import pytest

from ntllm import embeddings
from ntllm.embeddings import EmbeddingError


class StubEncoder:
    def __init__(self, dim=8):
        self.dim = dim
        self.calls = []

    def encode(self, texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i] = np.float32([(hash(f"{text}-{j}") % 1000) / 1000.0 for j in range(self.dim)])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


def test_encode_texts_returns_normalized_array_with_stable_shape(tmp_path, monkeypatch):
    stub = StubEncoder(dim=8)
    monkeypatch.setattr(embeddings, "_load_model", lambda model_dir: stub)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    first = embeddings.encode_texts(["alpha passage", "beta passage"], model_dir)
    second = embeddings.encode_texts(["alpha passage", "beta passage"], model_dir)

    assert first.shape == (2, 8)
    assert first.dtype == np.float32
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(first, second)


def test_encode_texts_rejects_missing_model_directory(tmp_path):
    with pytest.raises(EmbeddingError, match="model directory"):
        embeddings.encode_texts(["text"], tmp_path / "no-such-model")


def test_model_identity_is_stable_and_content_addressed(tmp_path):
    weights = tmp_path / "pytorch_model.bin"
    weights.write_bytes(b"fake-weights-1")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    first = embeddings.model_dir_sha(tmp_path)
    assert len(first) == 64
    weights.write_bytes(b"fake-weights-2")
    assert embeddings.model_dir_sha(tmp_path) != first
