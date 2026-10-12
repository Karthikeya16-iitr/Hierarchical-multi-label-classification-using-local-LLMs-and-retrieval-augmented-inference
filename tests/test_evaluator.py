"""Unit tests for evaluation metrics, invalid-pair tracking, and split foundations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set, Tuple

import pytest

from mavenmagnet.dataset_builder import DatasetBuilder, GroupedPostRecord
from mavenmagnet.evaluator import Evaluator, PerPairMetric


def test_perfect_predictions() -> None:
    """Verify metrics when model predictions match ground truth perfectly."""
    evaluator = Evaluator()
    y_true = [
        {("Finish", "Absorption"), ("Pricing", "Value for money")},
        {("Tone", "Glow")},
    ]
    y_pred = [
        {("Finish", "Absorption"), ("Pricing", "Value for money")},
        {("Tone", "Glow")},
    ]

    metrics = evaluator.compute_metrics(y_true, y_pred)
    assert metrics.exact_set_accuracy == 1.0
    assert metrics.exact_match_ratio == 1.0
    assert metrics.micro_precision == 1.0
    assert metrics.micro_recall == 1.0
    assert metrics.micro_f1 == 1.0
    assert metrics.hamming_loss == 0.0
    assert metrics.invalid_pair_rate == 0.0


def test_empty_predictions() -> None:
    """Verify zero_division handling when model outputs empty predictions."""
    evaluator = Evaluator()
    y_true = [
        {("Finish", "Absorption")},
        {("Tone", "Glow")},
    ]
    y_pred: List[Set[Tuple[str, str]]] = [
        set(),
        set(),
    ]

    metrics = evaluator.compute_metrics(y_true, y_pred)
    assert metrics.exact_set_accuracy == 0.0
    assert metrics.micro_precision == 0.0
    assert metrics.micro_recall == 0.0
    assert metrics.micro_f1 == 0.0
    assert metrics.macro_precision == 0.0
    assert metrics.macro_recall == 0.0
    assert metrics.macro_f1 == 0.0
    assert metrics.invalid_pair_rate == 0.0


def test_partially_correct_predictions() -> None:
    """Verify calculations for partially correct predictions with zero_division handling."""
    evaluator = Evaluator()
    y_true = [
        {("Moisturization", "Hydration"), ("Pricing", "Value for money")},
        {("Feel", "Fresh")},
        {("Tone", "Glow")},
    ]
    y_pred = [
        {("Moisturization", "Hydration")},  # Precision=1.0, Recall=0.5
        {("Feel", "Fresh"), ("Acne", "Acne Control")},  # Precision=0.5, Recall=1.0
        {("Tone", "Glow")},  # Exact match
    ]

    metrics = evaluator.compute_metrics(y_true, y_pred)
    # 1 of 3 is exact match
    assert abs(metrics.exact_set_accuracy - 1 / 3) < 1e-3
    assert 0.0 < metrics.micro_precision < 1.0
    assert 0.0 < metrics.micro_recall < 1.0
    assert 0.0 < metrics.micro_f1 < 1.0
    assert metrics.invalid_pair_rate == 0.0


def test_invalid_pair_rate_with_hallucinated_pairs() -> None:
    """Verify tracking of non-canonical, invented, or mismatched pairs."""
    evaluator = Evaluator()
    raw_predictions = [
        [("Acne", "Acne Control"), ("Invented Category", "Fake Subcategory")],
        [("Acne", "Discounts")],  # Invalid cross-combination
        [("Tone", "Glow")],
    ]
    invalid_rate, total_inv, total_preds = evaluator.calculate_invalid_pair_rate(raw_predictions)

    assert total_preds == 4
    assert total_inv == 2  # ("Invented Category", "Fake Subcategory") and ("Acne", "Discounts")
    assert invalid_rate == 0.5


def test_rare_labels_reliability_flagging() -> None:
    """Verify evaluator flags labels with insufficient or zero support."""
    evaluator = Evaluator(min_eval_support_threshold=3)
    y_true = [
        {("Finish", "Dewy")},  # Rare pair, support=1
        {("Tone", "Glow")},    # Common pair, support=2 in this mini sample
        {("Tone", "Glow")},
        {("Tone", "Glow")},
    ]
    y_pred = [
        {("Finish", "Dewy")},
        {("Tone", "Glow")},
        {("Tone", "Glow")},
        {("Tone", "Glow")},
    ]

    metrics = evaluator.compute_metrics(y_true, y_pred)
    per_pair = metrics.per_pair_metrics

    # Finish -> Dewy has support=1 < 3 -> should be flagged as unreliable
    dewy_metric = per_pair["Finish -> Dewy"]
    assert dewy_metric.support == 1
    assert not dewy_metric.is_reliable_eval
    assert "insufficient_eval_support" in dewy_metric.status

    # Tone -> Glow has support=3 >= 3 -> should be flagged as reliable
    glow_metric = per_pair["Tone -> Glow"]
    assert glow_metric.support == 3
    assert glow_metric.is_reliable_eval
    assert glow_metric.status == "sufficient_eval_support"

    # A pair with 0 occurrences in evaluation
    clean_metric = per_pair["Finish -> Clean"]
    assert clean_metric.support == 0
    assert not clean_metric.is_reliable_eval
    assert "zero_eval_support" in clean_metric.status


def test_iterative_train_test_split_properties() -> None:
    """Verify multi-label stratification preserves proportions and prevents leakage."""
    builder = DatasetBuilder()
    records = builder.load_grouped_dataset("data/interim/grouped_training_dataset.json")

    train_recs, test_recs = builder.iterative_train_test_split(records, test_size=0.2, random_seed=42)

    # 1. Total count matches
    assert len(train_recs) + len(test_recs) == len(records)
    assert len(train_recs) == 213
    assert len(test_recs) == 53

    # 2. Complete disjointness: no post text or ID appears in both partitions
    train_texts = {r.clean_text for r in train_recs}
    test_texts = {r.clean_text for r in test_recs}
    assert train_texts.isdisjoint(test_texts)

    train_ids = {r.post_id for r in train_recs}
    test_ids = {r.post_id for r in test_recs}
    assert train_ids.isdisjoint(test_ids)

    # 3. All 43 canonical labels represented in train partition
    train_support = builder.get_partition_label_support(train_recs)
    assert all(v["support"] > 0 for v in train_support.values())

    # 4. Reproducibility
    train_recs_2, test_recs_2 = builder.iterative_train_test_split(records, test_size=0.2, random_seed=42)
    assert [r.post_id for r in train_recs] == [r.post_id for r in train_recs_2]
    assert [r.post_id for r in test_recs] == [r.post_id for r in test_recs_2]
