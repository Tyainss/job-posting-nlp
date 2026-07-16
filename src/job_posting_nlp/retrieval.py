from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


RETRIEVAL_ENCODER_ID = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
RETRIEVAL_EMBEDDING_DIMENSION = 384
RETRIEVAL_MODEL_SEQUENCE_LIMIT = 512
RETRIEVAL_MAX_SEQUENCE_LENGTH = 256
RETRIEVAL_CHUNK_OVERLAP = 32
RETRIEVAL_EMBEDDING_BATCH_SIZE = 64


@dataclass(frozen=True)
class RetrievalEncoding:
    embeddings: NDArray[np.float32]
    cache_status: str
    encoding_seconds: float
    cache_path: Path


def load_retrieval_encoder(
    *,
    device: str | None = None,
) -> SentenceTransformer:
    """Load the retrieval encoder with the fixed chunk input limit."""
    from sentence_transformers import SentenceTransformer

    from job_posting_nlp.embeddings import select_embedding_device

    model = SentenceTransformer(
        RETRIEVAL_ENCODER_ID,
        device=device or select_embedding_device(),
    )
    model.max_seq_length = RETRIEVAL_MAX_SEQUENCE_LENGTH

    return model


def build_retrieval_corpus(
    development_rows: pd.DataFrame,
    description_counts: pd.Series,
) -> pd.DataFrame:
    """Keep one representative posting per development description."""
    corpus = development_rows.drop_duplicates(
        subset="cleaned_description",
        keep="first",
    ).copy()

    corpus["duplicate_count"] = (
        corpus["cleaned_description"]
        .map(description_counts)
        .astype(int)
    )

    corpus.insert(
        0,
        "document_id",
        corpus["cleaned_description"].map(
            lambda text: (
                "doc_"
                + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            )
        ),
    )

    return corpus[
        [
            "document_id",
            "job_id",
            "title",
            "company_name",
            "description",
            "cleaned_description",
            "duplicate_count",
        ]
    ].reset_index(drop=True)


def count_model_tokens(
    tokenizer: Any,
    texts: Sequence[str],
) -> NDArray[np.int64]:
    """Count tokens using the retrieval model tokenizer without truncation."""
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    token_counts = []

    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        token_counts.append(
            len(encoded["input_ids"]) + special_token_count
        )

    return np.asarray(token_counts, dtype=np.int64)


def build_retrieval_chunks(
    corpus: pd.DataFrame,
    tokenizer: Any,
    *,
    max_sequence_length: int = RETRIEVAL_MAX_SEQUENCE_LENGTH,
    overlap_tokens: int = RETRIEVAL_CHUNK_OVERLAP,
) -> pd.DataFrame:
    """Split each document into fixed tokenizer-aware chunks."""
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_token_limit = max_sequence_length - special_token_count
    chunk_step = content_token_limit - overlap_tokens
    chunk_rows = []

    for document in corpus.itertuples(index=False):
        encoded = tokenizer(
            document.cleaned_description,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = encoded["offset_mapping"]

        for chunk_index, token_start in enumerate(
            range(0, len(offsets), chunk_step)
        ):
            token_end = min(
                token_start + content_token_limit,
                len(offsets),
            )
            character_start = offsets[token_start][0]
            character_end = offsets[token_end - 1][1]
            chunk_text = document.cleaned_description[
                character_start:character_end
            ].strip()

            chunk_rows.append(
                {
                    "document_id": document.document_id,
                    "chunk_id": (
                        f"{document.document_id}_chunk_{chunk_index:02d}"
                    ),
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "token_count": (
                        token_end - token_start + special_token_count
                    ),
                }
            )

            if token_end == len(offsets):
                break

    return pd.DataFrame(chunk_rows)


def retrieval_cache_fingerprint(
    texts: Sequence[str],
    *,
    model_id: str,
    max_sequence_length: int,
    overlap_tokens: int,
    normalize_embeddings: bool,
) -> str:
    """Fingerprint ordered texts and the complete retrieval policy."""
    digest = hashlib.sha256()
    configuration = {
        "model_id": model_id,
        "max_sequence_length": max_sequence_length,
        "overlap_tokens": overlap_tokens,
        "normalize_embeddings": normalize_embeddings,
    }
    digest.update(
        json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    for text in texts:
        encoded_text = text.encode("utf-8")
        digest.update(len(encoded_text).to_bytes(8, "big"))
        digest.update(encoded_text)

    return digest.hexdigest()[:16]


def encode_retrieval_chunks(
    model: SentenceTransformer,
    chunk_texts: Sequence[str],
    cache_path: Path,
    *,
    batch_size: int = RETRIEVAL_EMBEDDING_BATCH_SIZE,
    show_progress_bar: bool = False,
) -> RetrievalEncoding:
    """Encode normalized chunks or load the matching local cache."""
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            embeddings = np.asarray(
                cached["embeddings"],
                dtype=np.float32,
            )
            encoding_seconds = float(cached["encoding_seconds"])
        cache_status = "Loaded"
    else:
        started = time.perf_counter()
        embeddings = model.encode_document(
            list(chunk_texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        encoding_seconds = time.perf_counter() - started
        embeddings = np.asarray(embeddings, dtype=np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            embeddings=embeddings,
            encoding_seconds=np.asarray(encoding_seconds),
        )
        cache_status = "Encoded and cached"

    return RetrievalEncoding(
        embeddings=embeddings,
        cache_status=cache_status,
        encoding_seconds=encoding_seconds,
        cache_path=cache_path,
    )


def rank_retrieval_results(
    chunk_scores: NDArray[np.floating[Any]],
    chunks: pd.DataFrame,
    corpus: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Keep the highest-scoring chunk for each document and rank documents."""
    ranked_chunks = chunks.assign(
        similarity_score=np.asarray(chunk_scores)
    ).sort_values(
        ["similarity_score", "document_id", "chunk_index"],
        ascending=[False, True, True],
        kind="mergesort",
    )

    top_documents = (
        ranked_chunks.drop_duplicates(
            subset="document_id",
            keep="first",
        )
        .head(top_k)
        .merge(
            corpus[
                [
                    "document_id",
                    "job_id",
                    "title",
                    "company_name",
                    "duplicate_count",
                ]
            ],
            on="document_id",
            how="left",
            validate="one_to_one",
        )
        .rename(columns={"chunk_text": "excerpt"})
    )

    top_documents.insert(
        0,
        "rank",
        np.arange(1, len(top_documents) + 1),
    )

    return top_documents[
        [
            "rank",
            "similarity_score",
            "document_id",
            "job_id",
            "title",
            "company_name",
            "excerpt",
            "duplicate_count",
        ]
    ].reset_index(drop=True)


def semantic_search(
    query: str,
    model: SentenceTransformer,
    chunks: pd.DataFrame,
    chunk_embeddings: NDArray[np.float32],
    corpus: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Run normalized brute-force semantic search over retrieval chunks."""
    query_embedding = model.encode_query(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    chunk_scores = chunk_embeddings @ query_embedding

    return rank_retrieval_results(
        chunk_scores,
        chunks,
        corpus,
        top_k=top_k,
    )
