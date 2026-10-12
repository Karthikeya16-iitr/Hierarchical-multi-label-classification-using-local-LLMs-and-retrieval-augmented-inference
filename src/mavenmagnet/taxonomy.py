"""Taxonomy extraction, hierarchy building, validation, and serialization.

Builds canonical Level 1 and Level 2 product review taxonomy exclusively
from labeled Training.xlsx data without inventing or hallucinating labels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger("mavenmagnet.taxonomy")


def extract_level1_labels(df: pd.DataFrame) -> List[str]:
    """Extract all unique Level 1 labels preserving exact strings, sorted deterministically.

    Args:
        df: DataFrame containing 'Level 1' column.

    Returns:
        Sorted list of unique Level 1 strings.
    """
    if "Level 1" not in df.columns:
        raise ValueError("DataFrame is missing required 'Level 1' column.")
    labels = df["Level 1"].dropna().unique().tolist()
    return sorted(str(lbl) for lbl in labels if str(lbl).strip())


def extract_level2_labels(df: pd.DataFrame) -> List[str]:
    """Extract all unique Level 2 labels preserving exact strings, sorted deterministically.

    Args:
        df: DataFrame containing 'Level 2' column.

    Returns:
        Sorted list of unique Level 2 strings.
    """
    if "Level 2" not in df.columns:
        raise ValueError("DataFrame is missing required 'Level 2' column.")
    labels = df["Level 2"].dropna().unique().tolist()
    return sorted(str(lbl) for lbl in labels if str(lbl).strip())


def extract_valid_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Extract all distinct valid (Level 1, Level 2) association pairs from training data.

    Args:
        df: DataFrame containing 'Level 1' and 'Level 2' columns.

    Returns:
        Sorted list of distinct (Level 1, Level 2) tuples.
    """
    if "Level 1" not in df.columns or "Level 2" not in df.columns:
        raise ValueError("DataFrame must contain both 'Level 1' and 'Level 2' columns.")

    valid_pairs: Set[Tuple[str, str]] = set()
    for _, row in df.iterrows():
        l1 = row["Level 1"]
        l2 = row["Level 2"]
        if pd.notna(l1) and pd.notna(l2):
            s1 = str(l1).strip()
            s2 = str(l2).strip()
            if s1 and s2:
                valid_pairs.add((str(l1), str(l2)))

    return sorted(valid_pairs)


def build_hierarchy(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build deterministic hierarchy mapping each Level 1 label to its permitted Level 2 labels.

    Args:
        df: DataFrame containing 'Level 1' and 'Level 2' columns.

    Returns:
        Dictionary where each key is a Level 1 label and the value is a sorted list of Level 2 labels.
    """
    pairs = extract_valid_pairs(df)
    hierarchy: Dict[str, Set[str]] = {}
    for l1, l2 in pairs:
        hierarchy.setdefault(l1, set()).add(l2)

    return {l1: sorted(l2_set) for l1, l2_set in sorted(hierarchy.items())}


def detect_multilabel_level2(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect any Level 2 label associated with multiple Level 1 categories.

    Args:
        df: DataFrame containing 'Level 1' and 'Level 2' columns.

    Returns:
        Dictionary mapping shared Level 2 labels to sorted list of associated Level 1 categories.
    """
    pairs = extract_valid_pairs(df)
    l2_to_l1: Dict[str, Set[str]] = {}
    for l1, l2 in pairs:
        l2_to_l1.setdefault(l2, set()).add(l1)

    shared = {l2: sorted(l1_set) for l2, l1_set in sorted(l2_to_l1.items()) if len(l1_set) > 1}
    return shared


@dataclass(frozen=True)
class Taxonomy:
    """Immutable, deterministic representation of the domain taxonomy."""

    hierarchy: Dict[str, List[str]]
    level1_labels: List[str]
    level2_labels: List[str]
    valid_pairs: List[Tuple[str, str]]
    shared_level2_labels: Dict[str, List[str]]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Taxonomy:
        """Construct Taxonomy deterministically from labeled DataFrame."""
        hierarchy = build_hierarchy(df)
        l1_labels = extract_level1_labels(df)
        l2_labels = extract_level2_labels(df)
        valid_pairs = extract_valid_pairs(df)
        shared_l2 = detect_multilabel_level2(df)

        return cls(
            hierarchy=hierarchy,
            level1_labels=l1_labels,
            level2_labels=l2_labels,
            valid_pairs=valid_pairs,
            shared_level2_labels=shared_l2,
        )

    @classmethod
    def from_training_file(
        cls,
        path: Path | str,
        sheet_name: Union[str, int] = 0,
    ) -> Taxonomy:
        """Construct Taxonomy directly from Training.xlsx workbook."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Training workbook not found at: {file_path}")
        df = pd.read_excel(file_path, engine="openpyxl", sheet_name=sheet_name)
        return cls.from_dataframe(df)

    def is_valid_pair(self, level1: str, level2: str) -> bool:
        """Check whether a (Level 1, Level 2) association exists in the taxonomy."""
        return (level1, level2) in set(self.valid_pairs)

    def get_level2_for_level1(self, level1: str) -> List[str]:
        """Return permitted Level 2 subcategories for a given Level 1 category."""
        return self.hierarchy.get(level1, [])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize taxonomy to deterministic nested dictionary."""
        return {
            "metadata": {
                "total_level1_labels": len(self.level1_labels),
                "total_level2_labels": len(self.level2_labels),
                "total_valid_pairs": len(self.valid_pairs),
                "shared_level2_count": len(self.shared_level2_labels),
            },
            "hierarchy": self.hierarchy,
            "level1_labels": self.level1_labels,
            "level2_labels": self.level2_labels,
            "valid_pairs": [[p[0], p[1]] for p in self.valid_pairs],
            "shared_level2_labels": self.shared_level2_labels,
        }

    def export_json(self, output_path: Path | str) -> Path:
        """Export taxonomy deterministically to formatted JSON file.

        Args:
            output_path: Destination path for JSON file.

        Returns:
            Pathlib Path to saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        logger.info("Exported taxonomy to %s", path)
        return path

    @classmethod
    def load_json(cls, input_path: Path | str) -> Taxonomy:
        """Load taxonomy from JSON file."""
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Taxonomy JSON file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid_pairs = [(item[0], item[1]) for item in data["valid_pairs"]]
        return cls(
            hierarchy={k: sorted(v) for k, v in sorted(data["hierarchy"].items())},
            level1_labels=sorted(data["level1_labels"]),
            level2_labels=sorted(data["level2_labels"]),
            valid_pairs=sorted(valid_pairs),
            shared_level2_labels={
                k: sorted(v) for k, v in sorted(data.get("shared_level2_labels", {}).items())
            },
        )


# Backward-compatible canonical dictionary and helper functions
# Populated from canonical training data
CANONICAL_TAXONOMY: Dict[str, Set[str]] = {
    "Acne": {"Acne Control", "Minimize Pores"},
    "Anti-Aging": {"Dullness", "Elasticity", "Fine Lines"},
    "Cleansing": {"Dirt Removal", "Exfoliation", "Oil Removal"},
    "Efficacy": {"Prolonged Use", "Quick Results", "Visible Difference"},
    "Feel": {"Cooling Sensation", "Fresh", "Soothening"},
    "Finish": {"Absorption", "Clean", "Dewy", "Light Weight"},
    "Health": {"Texture"},
    "Ingredients": {"Vitamin C"},
    "Lasting": {"Long Stay Period"},
    "Moisturization": {"Hydration", "Intensity", "Nourishment"},
    "Practicality": {"Availability"},
    "Pricing": {"Affordability", "Discounts", "Value for money"},
    "Relief": {"Damaged Skin", "Redness"},
    "Safety": {"Mildness", "Side Effects"},
    "Sensory": {"Format", "Smell", "Touch"},
    "Texture": {"Dryness Control", "Elasticity", "Oil Balance", "Smoothness", "Softness"},
    "Tone": {"Depigmentation", "Glow", "Skin Lightening"},
}


def get_all_level1_classes() -> List[str]:
    """Return sorted list of all valid Level 1 category names."""
    return sorted(CANONICAL_TAXONOMY.keys())


def get_all_valid_pairs() -> List[Tuple[str, str]]:
    """Return sorted list of all 43 valid (Level 1, Level 2) association tuples."""
    pairs: List[Tuple[str, str]] = []
    for l1, l2_set in sorted(CANONICAL_TAXONOMY.items()):
        for l2 in sorted(l2_set):
            pairs.append((l1, l2))
    return pairs


def get_valid_pairs_set() -> FrozenSet[Tuple[str, str]]:
    """Return an immutable frozen set of all valid (Level 1, Level 2) pairs."""
    return frozenset(get_all_valid_pairs())


def is_valid_association(level1: str, level2: str) -> bool:
    """Check whether a (Level 1, Level 2) association exists in the canonical taxonomy."""
    return (level1.strip(), level2.strip()) in get_valid_pairs_set()


def get_subcategories_for_level1(level1: str) -> List[str]:
    """Return valid Level 2 subcategories for a given Level 1 category."""
    return sorted(CANONICAL_TAXONOMY.get(level1.strip(), set()))
