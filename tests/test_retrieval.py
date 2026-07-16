import numpy as np
import pandas as pd

from job_posting_nlp.retrieval import (
    build_retrieval_chunks,
    build_retrieval_corpus,
    rank_retrieval_results,
    retrieval_cache_fingerprint,
)


class CharacterTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
        **_: object,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        input_ids = list(range(len(text)))
        result: dict[str, list[int] | list[tuple[int, int]]] = {
            "input_ids": input_ids
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1)
                for index in range(len(text))
            ]
        return result


def test_retrieval_corpus_deduplicates_descriptions_and_adds_counts():
    cleaned_rows = pd.DataFrame(
        [
            {
                "job_id": 1,
                "title": "Data Engineer",
                "company_name": "A",
                "description": "Raw alpha",
                "cleaned_description": "alpha",
            },
            {
                "job_id": 2,
                "title": "Senior Data Engineer",
                "company_name": "B",
                "description": "Raw alpha",
                "cleaned_description": "alpha",
            },
            {
                "job_id": 3,
                "title": "Data Analyst",
                "company_name": "C",
                "description": "Raw beta",
                "cleaned_description": "beta",
            },
        ]
    )

    description_counts = pd.Series(
        {
            "alpha": 2,
            "beta": 1,
        }
    )

    corpus = build_retrieval_corpus(
        cleaned_rows,
        description_counts,
    )

    assert corpus["job_id"].tolist() == [1, 3]
    assert corpus["cleaned_description"].tolist() == [
        "alpha",
        "beta",
    ]
    assert corpus["duplicate_count"].tolist() == [2, 1]
    assert corpus["document_id"].is_unique
    assert corpus["document_id"].str.startswith("doc_").all()


def test_retrieval_chunks_respect_boundaries_overlap_and_document_ids():
    corpus = pd.DataFrame(
        {
            "document_id": ["doc_a"],
            "cleaned_description": ["abcdefghij"],
        }
    )

    chunks = build_retrieval_chunks(
        corpus,
        CharacterTokenizer(),
        max_sequence_length=6,
        overlap_tokens=1,
    )

    assert chunks["document_id"].tolist() == [
        "doc_a",
        "doc_a",
        "doc_a",
    ]
    assert chunks["chunk_text"].tolist() == [
        "abcd",
        "defg",
        "ghij",
    ]
    assert chunks["token_count"].tolist() == [6, 6, 6]


def test_ranking_keeps_best_chunk_per_document_and_orders_top_k():
    chunks = pd.DataFrame(
        {
            "document_id": ["doc_a", "doc_a", "doc_b", "doc_c"],
            "chunk_id": ["a0", "a1", "b0", "c0"],
            "chunk_index": [0, 1, 0, 0],
            "chunk_text": ["a first", "a best", "b", "c"],
            "token_count": [3, 3, 3, 3],
        }
    )
    corpus = pd.DataFrame(
        {
            "document_id": ["doc_a", "doc_b", "doc_c"],
            "job_id": [1, 2, 3],
            "title": ["A", "B", "C"],
            "company_name": ["Company A", "Company B", "Company C"],
            "duplicate_count": [1, 1, 1],
        }
    )
    scores = np.array([0.3, 0.9, 0.8, 0.2], dtype=np.float32)

    results = rank_retrieval_results(
        scores,
        chunks,
        corpus,
        top_k=2,
    )

    assert results["document_id"].tolist() == ["doc_a", "doc_b"]
    assert results["excerpt"].tolist() == ["a best", "b"]
    assert results["rank"].tolist() == [1, 2]


def test_cache_fingerprint_changes_with_text_or_configuration():
    arguments = {
        "model_id": "model-a",
        "max_sequence_length": 256,
        "overlap_tokens": 32,
        "normalize_embeddings": True,
    }
    original = retrieval_cache_fingerprint(["alpha", "beta"], **arguments)

    assert original != retrieval_cache_fingerprint(
        ["beta", "alpha"],
        **arguments,
    )
    assert original != retrieval_cache_fingerprint(
        ["alpha", "changed"],
        **arguments,
    )
    assert original != retrieval_cache_fingerprint(
        ["alpha", "beta"],
        **{**arguments, "model_id": "model-b"},
    )
    assert original != retrieval_cache_fingerprint(
        ["alpha", "beta"],
        **{**arguments, "overlap_tokens": 0},
    )
