"""Frozen sentence-embedding encoding from a vendored local model.

impl: FR-011-02
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


class EmbeddingError(ValueError):
    """Embedding model loading or encoding failed."""


def _load_model(model_dir: Path):
    """Load the vendored sentence-transformers model (imported lazily)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(f"sentence-transformers is not installed: {exc}") from exc
    try:
        return SentenceTransformer(str(model_dir))
    except Exception as exc:
        raise EmbeddingError(f"cannot load embedding model from {model_dir}: {exc}") from exc


def encode_texts(
    texts: Sequence[str],
    model_dir: str | Path,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with the frozen vendored model; never touches the network.

    Returns L2-normalized float32 vectors, one row per input text, in order.
    """
    source = Path(model_dir)
    if not source.is_dir():
        raise EmbeddingError(f"embedding model directory does not exist: {source}")
    if any(not isinstance(text, str) or not text for text in texts):
        raise EmbeddingError("cannot encode blank or non-string texts")
    model = _load_model(source)
    try:
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingError(f"embedding encoding failed: {exc}") from exc
    result = np.asarray(vectors, dtype=np.float32)
    if result.shape != (len(texts), result.shape[1]) or result.shape[1] == 0:
        raise EmbeddingError("encoder returned an unexpected array shape")
    return result


def model_dir_sha(model_dir: str | Path) -> str:
    """Content hash of the vendored weights for run provenance."""
    source = Path(model_dir)
    digest = hashlib.sha256()
    for name in ("pytorch_model.bin", "model.safetensors", "config.json"):
        path = source / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    if digest.hexdigest() == hashlib.sha256().hexdigest():
        raise EmbeddingError(f"no recognized model files in {source}")
    return digest.hexdigest()
