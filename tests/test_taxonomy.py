"""Unit and integration tests for taxonomy extraction, validation, and serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mavenmagnet.taxonomy import (
    Taxonomy,
    build_hierarchy,
    detect_multilabel_level2,
    extract_level1_labels,
    extract_level2_labels,
    extract_valid_pairs,
    is_valid_association,
)


@pytest.fixture
def sample_training_df() -> pd.DataFrame:
    """Fixture returning sample DataFrame with valid and shared pairs."""
    return pd.DataFrame(
        {
            "Post": ["P1", "P1", "P2", "P3", "P4"],
            "Level 1": ["Anti-Aging", "Texture", "Anti-Aging", "Acne", "Pricing"],
            "Level 2": ["Elasticity", "Elasticity", "Fine Lines", "Acne Control", "Value for money"],
        }
    )


def test_every_observed_training_pair_included() -> None:
    """Verify that every single observed pair from Training.xlsx is included in the taxonomy."""
    training_file = Path("data/input/Training.xlsx")
    df = pd.read_excel(training_file, engine="openpyxl")
    taxonomy = Taxonomy.from_dataframe(df)

    observed_pairs = set()
    for _, row in df.iterrows():
        l1 = row["Level 1"]
        l2 = row["Level 2"]
        if pd.notna(l1) and pd.notna(l2):
            s1 = str(l1).strip()
            s2 = str(l2).strip()
            if s1 and s2:
                observed_pairs.add((s1, s2))

    tax_pairs_set = set(taxonomy.valid_pairs)
    assert observed_pairs.issubset(tax_pairs_set)
    assert len(observed_pairs) == 43
    assert len(taxonomy.valid_pairs) == 43


def test_no_unobserved_pair_introduced() -> None:
    """Verify that no unobserved pair, invented category, or hallucination is introduced."""
    training_file = Path("data/input/Training.xlsx")
    df = pd.read_excel(training_file, engine="openpyxl")
    taxonomy = Taxonomy.from_dataframe(df)

    observed_pairs = {
        (str(row["Level 1"]).strip(), str(row["Level 2"]).strip())
        for _, row in df.iterrows()
        if pd.notna(row["Level 1"]) and pd.notna(row["Level 2"])
    }

    tax_pairs_set = set(taxonomy.valid_pairs)
    # The set of taxonomy pairs must exactly equal the set of observed pairs
    assert tax_pairs_set == observed_pairs


def test_shared_level2_label_handling() -> None:
    """Verify that Level 2 labels shared across multiple Level 1 categories are correctly handled."""
    training_file = Path("data/input/Training.xlsx")
    df = pd.read_excel(training_file, engine="openpyxl")
    taxonomy = Taxonomy.from_dataframe(df)

    # Elasticity is known to be shared between Anti-Aging and Texture
    assert "Elasticity" in taxonomy.shared_level2_labels
    assert taxonomy.shared_level2_labels["Elasticity"] == ["Anti-Aging", "Texture"]

    # Both associations must be valid
    assert taxonomy.is_valid_pair("Anti-Aging", "Elasticity")
    assert taxonomy.is_valid_pair("Texture", "Elasticity")

    # Association with an unlinked Level 1 category must be invalid
    assert not taxonomy.is_valid_pair("Acne", "Elasticity")
    assert not taxonomy.is_valid_pair("Pricing", "Elasticity")


def test_invalid_pairs_rejected() -> None:
    """Verify that invalid associations, invented labels, and mismatched pairs are rejected."""
    training_file = Path("data/input/Training.xlsx")
    taxonomy = Taxonomy.from_training_file(training_file)

    # Invented Level 1
    assert not taxonomy.is_valid_pair("NonExistentLevel1", "Dullness")
    # Invented Level 2
    assert not taxonomy.is_valid_pair("Acne", "InventedLevel2")
    # Valid labels but invalid cross-combination
    assert not taxonomy.is_valid_pair("Acne", "Value for money")
    assert not taxonomy.is_valid_pair("Pricing", "Minimize Pores")
    # Empty strings
    assert not taxonomy.is_valid_pair("", "")


def test_taxonomy_serialization_and_loading(tmp_path: Path) -> None:
    """Verify taxonomy serialization to JSON and round-trip loading."""
    training_file = Path("data/input/Training.xlsx")
    taxonomy = Taxonomy.from_training_file(training_file)

    export_path = tmp_path / "taxonomy.json"
    taxonomy.export_json(export_path)

    assert export_path.exists()

    loaded = Taxonomy.load_json(export_path)
    assert loaded.level1_labels == taxonomy.level1_labels
    assert loaded.level2_labels == taxonomy.level2_labels
    assert loaded.valid_pairs == taxonomy.valid_pairs
    assert loaded.hierarchy == taxonomy.hierarchy
    assert loaded.shared_level2_labels == taxonomy.shared_level2_labels


def test_taxonomy_deterministic_generation() -> None:
    """Verify that taxonomy generation is strictly deterministic across multiple runs."""
    training_file = Path("data/input/Training.xlsx")
    tax1 = Taxonomy.from_training_file(training_file)
    tax2 = Taxonomy.from_training_file(training_file)

    assert tax1.to_dict() == tax2.to_dict()
    assert json.dumps(tax1.to_dict(), sort_keys=True) == json.dumps(tax2.to_dict(), sort_keys=True)


def test_artifacts_taxonomy_file_integrity() -> None:
    """Verify that the production artifact artifacts/taxonomy.json matches canonical counts."""
    artifact_path = Path("artifacts/taxonomy.json")
    assert artifact_path.exists()

    taxonomy = Taxonomy.load_json(artifact_path)
    assert len(taxonomy.level1_labels) == 17
    assert len(taxonomy.level2_labels) == 42
    assert len(taxonomy.valid_pairs) == 43
    assert len(taxonomy.shared_level2_labels) == 1
    assert "Elasticity" in taxonomy.shared_level2_labels
