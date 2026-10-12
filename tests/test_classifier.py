"""Unit tests for baseline multi-label TF-IDF Logistic Regression classifier."""

from __future__ import annotations

from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pytest

from mavenmagnet.classifier import TfidfLogisticBaseline
from mavenmagnet.taxonomy import get_all_valid_pairs, is_valid_association


@pytest.fixture
def sample_training_data() -> Tuple[List[str], List[Set[Tuple[str, str]]]]:
    """Fixture returning sample reviews and multi-label associations."""
    texts = [
        "A drop or two are enough to moisture entire face, great value for money.",
        "Deeply cleanses dirt and removes excess oil.",
        "Skin glows brightly and feels soft and smooth.",
        "Lightweight texture with instant absorption.",
        "Cleanses well, leaves a fresh cooling sensation, worth every penny.",
    ]
    targets = [
        {("Moisturization", "Intensity"), ("Pricing", "Value for money")},
        {("Cleansing", "Dirt Removal"), ("Cleansing", "Oil Removal")},
        {("Tone", "Glow"), ("Texture", "Softness")},
        {("Finish", "Light Weight"), ("Finish", "Absorption")},
        {("Cleansing", "Dirt Removal"), ("Feel", "Cooling Sensation"), ("Pricing", "Value for money")},
    ]
    return texts, targets


def test_baseline_model_training(sample_training_data: Tuple[List[str], List[Set[Tuple[str, str]]]]) -> None:
    """Verify baseline model fits correctly on training texts and multi-label targets."""
    texts, targets = sample_training_data
    model = TfidfLogisticBaseline(random_seed=42)
    assert not model.is_fitted

    model.fit(texts, targets)
    assert model.is_fitted
    assert model.vectorizer is not None
    assert len(model.classifiers) == len(model.valid_pairs)
    assert model.metadata["n_train_samples"] == len(texts)


def test_multiple_predicted_labels() -> None:
    """Verify model can predict multiple label pairs for a single post."""
    # Training set with posts having multiple aspects
    texts = [
        "Hydrating moisturizer worth every penny and great price value.",
        "Moisturizing formula with excellent value for money.",
        "Another post about cleansing dirt.",
        "Cleanses dirt and cleans well.",
    ]
    targets = [
        {("Moisturization", "Hydration"), ("Pricing", "Value for money")},
        {("Moisturization", "Hydration"), ("Pricing", "Value for money")},
        {("Cleansing", "Dirt Removal")},
        {("Cleansing", "Dirt Removal")},
    ]

    model = TfidfLogisticBaseline(decision_threshold=0.25, random_seed=42)
    model.fit(texts, targets)

    test_query = ["Hydrating formula that is great value for money and moisturizing."]
    predictions = model.predict(test_query)

    assert len(predictions) == 1
    pred_pairs = predictions[0]
    # Should predict both Moisturization -> Hydration and Pricing -> Value for money
    assert len(pred_pairs) >= 2
    assert ("Moisturization", "Hydration") in pred_pairs
    assert ("Pricing", "Value for money") in pred_pairs


def test_rare_and_constant_labels_handling() -> None:
    """Verify baseline handles constant classes (all 0s) and rare classes (1 positive) without crashing."""
    texts = ["Post one", "Post two", "Post three", "Post four"]
    # Label pair 0 has 1 positive; all other 42 pairs have 0 positives
    targets = [
        {("Tone", "Glow")},
        set(),
        set(),
        set(),
    ]

    model = TfidfLogisticBaseline(random_seed=42)
    # Must fit without throwing ValueError
    model.fit(texts, targets)
    assert model.is_fitted

    # 42 pairs have constant 0 class
    assert model.metadata["constant_classes_count"] == len(model.valid_pairs) - 1

    probs = model.predict_proba(["New unseen post"])
    assert probs.shape == (1, len(model.valid_pairs))
    # Unrepresented classes should output probability 0.0
    for idx, pair in enumerate(model.valid_pairs):
        if pair != ("Tone", "Glow"):
            assert probs[0, idx] == 0.0


def test_saving_and_loading_model(tmp_path: Path, sample_training_data: Tuple[List[str], List[Set[Tuple[str, str]]]]) -> None:
    """Verify model serialization and deserialization produces identical predictions."""
    texts, targets = sample_training_data
    model = TfidfLogisticBaseline(decision_threshold=0.3, random_seed=42)
    model.fit(texts, targets)

    model_file = tmp_path / "baseline.joblib"
    model.save(model_file)
    assert model_file.exists()

    loaded_model = TfidfLogisticBaseline.load(model_file)
    assert loaded_model.is_fitted
    assert loaded_model.decision_threshold == model.decision_threshold

    test_queries = ["A drop to moisture the face", "Clean bright look"]
    original_preds = model.predict(test_queries)
    loaded_preds = loaded_model.predict(test_queries)

    assert original_preds == loaded_preds
    np.testing.assert_allclose(
        model.predict_proba(test_queries),
        loaded_model.predict_proba(test_queries),
    )


def test_predictions_contain_only_valid_taxonomy_pairs(sample_training_data: Tuple[List[str], List[Set[Tuple[str, str]]]]) -> None:
    """Verify that predictions are strictly members of the canonical taxonomy."""
    texts, targets = sample_training_data
    model = TfidfLogisticBaseline(random_seed=42)
    model.fit(texts, targets)

    queries = [
        "Unseen review praising fast absorption and softness.",
        "Completely arbitrary text about something else.",
        "Acne pimples and glowing skin.",
    ]
    predictions = model.predict(queries)

    valid_pairs_set = set(model.valid_pairs)
    for pred_set in predictions:
        for pair in pred_set:
            assert pair in valid_pairs_set
            assert is_valid_association(pair[0], pair[1])
