from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


CLASSIFICATION_ENCODER_ID = "sentence-transformers/all-mpnet-base-v2"
CLASSIFICATION_EMBEDDING_DIMENSION = 768
CLASSIFICATION_MAX_SEQUENCE_LENGTH = 384
CLASSIFICATION_EMBEDDING_BATCH_SIZE = 32


def select_embedding_device() -> str:
    """Select the best available PyTorch device for embedding inference."""
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_classification_encoder(
    *,
    device: str | None = None,
) -> SentenceTransformer:
    """Load the frozen encoder selected for job-role classification."""
    model = SentenceTransformer(
        CLASSIFICATION_ENCODER_ID,
        device=device or select_embedding_device(),
    )
    model.max_seq_length = CLASSIFICATION_MAX_SEQUENCE_LENGTH

    return model


def encode_classification_descriptions(
    model: SentenceTransformer,
    descriptions: Sequence[str],
    *,
    batch_size: int = CLASSIFICATION_EMBEDDING_BATCH_SIZE,
    show_progress_bar: bool = False,
) -> NDArray[np.float32]:
    """Encode whole descriptions using the frozen classification policy."""
    embeddings = model.encode(
        list(descriptions),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(embeddings, dtype=np.float32)