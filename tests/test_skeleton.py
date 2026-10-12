"""Unit tests verifying project skeleton, imports, taxonomy, and configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

import mavenmagnet
from mavenmagnet.config import load_config
from mavenmagnet.evaluator import Evaluator
from mavenmagnet.taxonomy import (
    CANONICAL_TAXONOMY,
    get_all_level1_classes,
    get_all_valid_pairs,
    is_valid_association,
)
from mavenmagnet.validators import parse_and_validate_predictions


def test_package_import_and_version() -> None:
    """Verify package import and metadata."""
    assert mavenmagnet.__version__ == "0.1.0"


def test_config_loader() -> None:
    """Verify config loads correctly with pathlib paths."""
    cfg = load_config()
    assert cfg.project_name == "mavenmagnet-nlp-pipeline"
    assert isinstance(cfg.paths.input_dir, Path)
    assert isinstance(cfg.paths.training_file, Path)
    assert isinstance(cfg.paths.process_file, Path)
    assert cfg.baseline.ngram_range == (1, 2)
    assert cfg.llm.provider == "ollama"


def test_canonical_taxonomy_integrity() -> None:
    """Verify canonical taxonomy properties: exactly 17 Level 1 and 43 valid pairs."""
    l1_classes = get_all_level1_classes()
    assert len(l1_classes) == 17
    valid_pairs = get_all_valid_pairs()
    assert len(valid_pairs) == 43

    # Check known dual association (Elasticity under Anti-Aging and Texture)
    assert is_valid_association("Anti-Aging", "Elasticity")
    assert is_valid_association("Texture", "Elasticity")

    # Reject non-existent/invented association
    assert not is_valid_association("Anti-Aging", "Invented Subcategory")
    assert not is_valid_association("Invented Level 1", "Dullness")


def test_validator_rejects_hallucinations() -> None:
    """Verify prediction parser filters out invalid labels."""
    raw_json = """
    {
        "predictions": [
            {"level_1": "Anti-Aging", "level_2": "Fine Lines"},
            {"level_1": "Fake Category", "level_2": "Fake Subcategory"}
        ]
    }
    """
    validated = parse_and_validate_predictions(raw_json)
    assert ("Anti-Aging", "Fine Lines") in validated
    assert len(validated) == 1


def test_evaluator_exact_match() -> None:
    """Verify evaluator exact match calculation."""
    evaluator = Evaluator()
    y_true = [{("Finish", "Absorption")}, {("Pricing", "Value for money")}]
    y_pred = [{("Finish", "Absorption")}, {("Feel", "Fresh")}]
    metrics = evaluator.compute_metrics(y_true, y_pred)
    assert metrics.exact_match_ratio == 0.5
