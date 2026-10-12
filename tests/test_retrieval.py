"""Unit tests for TF-IDF cosine similarity few-shot exemplar retriever."""

from __future__ import annotations

from pathlib import Path

import pytest

from mavenmagnet.dataset_builder import DatasetBuilder, GroupedPostRecord
from mavenmagnet.retrieval import ExemplarRetriever, RetrievedExemplar


@pytest.fixture
def sample_records() -> list[GroupedPostRecord]:
    """Sample records fixture with distinct semantic categories."""
    return [
        GroupedPostRecord(
            post_id="post_001",
            original_text="Extremely hydrating moisturizer, provides deep intensity hydration.",
            clean_text="Extremely hydrating moisturizer, provides deep intensity hydration.",
            pairs=[("Moisturization", "Hydration"), ("Moisturization", "Intensity")],
            original_row_indices=[0, 1],
        ),
        GroupedPostRecord(
            post_id="post_002",
            original_text="A bit pricey but totally worth the money.",
            clean_text="A bit pricey but totally worth the money.",
            pairs=[("Pricing", "Value for money")],
            original_row_indices=[2],
        ),
        GroupedPostRecord(
            post_id="post_003",
            original_text="Deeply cleanses dirt and oil from pores.",
            clean_text="Deeply cleanses dirt and oil from pores.",
            pairs=[("Cleansing", "Dirt Removal"), ("Cleansing", "Oil Removal")],
            original_row_indices=[3, 4],
        ),
        GroupedPostRecord(
            post_id="post_004",
            original_text="Leaves skin glowing and bright every morning.",
            clean_text="Leaves skin glowing and bright every morning.",
            pairs=[("Tone", "Glow")],
            original_row_indices=[5],
        ),
    ]


def test_similarity_ranking(sample_records: list[GroupedPostRecord]) -> None:
    """Verify that retrieval ranks semantically relevant exemplars highest."""
    retriever = ExemplarRetriever()
    retriever.fit(sample_records)

    query = "Looking for hydration and moisturizing formula"
    results = retriever.retrieve(query, k=2)

    assert len(results) == 2
    top_result = results[0]
    assert top_result.post_id == "post_001"
    assert ("Moisturization", "Hydration") in top_result.pairs
    assert top_result.similarity_score > 0.0

    # Verify descending score order
    assert results[0].similarity_score >= results[1].similarity_score


def test_duplicate_handling(sample_records: list[GroupedPostRecord]) -> None:
    """Verify that results do not contain duplicate post IDs."""
    retriever = ExemplarRetriever()
    retriever.fit(sample_records)

    results = retriever.retrieve("moisturizer glowing dirt", k=4)
    retrieved_ids = [r.post_id for r in results]

    assert len(retrieved_ids) == len(set(retrieved_ids))


def test_empty_inputs_and_empty_index(sample_records: list[GroupedPostRecord]) -> None:
    """Verify graceful handling of empty text queries and empty indexes."""
    # 1. Empty index
    empty_retriever = ExemplarRetriever()
    assert empty_retriever.retrieve("Any query text") == []

    # 2. Empty query string on fitted retriever
    retriever = ExemplarRetriever()
    retriever.fit(sample_records)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_evaluation_leakage_prevented(sample_records: list[GroupedPostRecord]) -> None:
    """CRITICAL TEST: Verify validation posts and labels cannot leak into retrieval results."""
    # Suppose post_001 is the validation query
    val_post = sample_records[0]
    train_pool = sample_records[1:]  # posts 002, 003, 004 only

    # 1. Index fitted strictly on train_pool cannot contain post_001
    retriever = ExemplarRetriever()
    retriever.fit(train_pool)

    results = retriever.retrieve(val_post.original_text, k=3)
    retrieved_ids = {r.post_id for r in results}

    assert val_post.post_id not in retrieved_ids
    # None of the validation labels (Moisturization -> Hydration, Intensity) can appear from train_pool
    for r in results:
        assert ("Moisturization", "Hydration") not in r.pairs

    # 2. Explicit exclude_post_ids and exclude_texts parameter works
    full_retriever = ExemplarRetriever()
    full_retriever.fit(sample_records)

    results_excluded = full_retriever.retrieve(
        val_post.original_text,
        k=3,
        exclude_post_ids={val_post.post_id},
        exclude_texts={val_post.clean_text},
    )
    assert val_post.post_id not in {r.post_id for r in results_excluded}


def test_configurable_k(sample_records: list[GroupedPostRecord]) -> None:
    """Verify configurable top-k returns exact requested count up to corpus size."""
    retriever = ExemplarRetriever()
    retriever.fit(sample_records)

    assert len(retriever.retrieve("skin", k=1)) == 1
    assert len(retriever.retrieve("skin", k=3)) == 3
    assert len(retriever.retrieve("skin", k=10)) == len(sample_records)


def test_retriever_save_and_load(tmp_path: Path, sample_records: list[GroupedPostRecord]) -> None:
    """Verify artifact serialization and loading preserves index and scores."""
    retriever = ExemplarRetriever()
    retriever.fit(sample_records)

    save_path = tmp_path / "retriever.joblib"
    retriever.save(save_path)
    assert save_path.exists()

    loaded = ExemplarRetriever.load(save_path)
    assert loaded.is_fitted
    assert len(loaded.records) == len(sample_records)

    q = "cleanses dirt and oil"
    orig_res = retriever.retrieve(q, k=2)
    load_res = loaded.retrieve(q, k=2)

    assert [r.post_id for r in orig_res] == [r.post_id for r in load_res]
    assert [r.similarity_score for r in orig_res] == [r.similarity_score for r in load_res]
