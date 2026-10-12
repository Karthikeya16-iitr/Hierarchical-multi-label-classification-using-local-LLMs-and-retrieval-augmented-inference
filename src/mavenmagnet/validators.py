"""Validation and parsing logic for model predictions."""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from mavenmagnet.taxonomy import is_valid_association

logger = logging.getLogger("mavenmagnet.validators")


class PredictionItem(BaseModel):
    """Single predicted Level 1 and Level 2 pair with grounded evidence."""

    level_1: str = Field(..., alias="level_1")
    level_2: str = Field(..., alias="level_2")
    evidence: Optional[str] = Field(default="", alias="evidence")


class ModelOutputSchema(BaseModel):
    """Root JSON schema expected from LLM."""

    predictions: List[PredictionItem] = Field(default_factory=list)


def parse_and_validate_predictions(raw_output: str) -> Set[Tuple[str, str]]:
    """Parse raw LLM response text into validated taxonomy pairs.

    Filters out any invented, hallucinated, or malformed label associations.

    Args:
        raw_output: Raw response string from LLM.

    Returns:
        Set of valid (Level 1, Level 2) tuples.
    """
    valid_pairs: Set[Tuple[str, str]] = set()

    # Attempt to extract JSON from markdown code fences if present
    cleaned = raw_output.strip()
    if "```json" in cleaned:
        match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
    elif "```" in cleaned:
        match = re.search(r"```\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)

    try:
        data = json.loads(cleaned)
        parsed = ModelOutputSchema.model_validate(data)
        for item in parsed.predictions:
            l1 = item.level_1.strip()
            l2 = item.level_2.strip()
            if is_valid_association(l1, l2):
                valid_pairs.add((l1, l2))
            else:
                logger.warning("Rejected non-canonical prediction: (%s, %s)", l1, l2)
    except Exception as e:
        logger.warning("Failed to parse LLM JSON output (%s). Raw text: %s", e, raw_output[:200])

    return valid_pairs


def parse_predictions_with_evidence(raw_output: str) -> List[Dict[str, str]]:
    """Parse raw LLM response retaining valid pairs alongside their cited evidence text.

    Args:
        raw_output: Raw response string from LLM.

    Returns:
        List of dictionaries with keys: 'level_1', 'level_2', 'evidence'.
    """
    validated: List[Dict[str, str]] = []

    cleaned = raw_output.strip()
    if "```json" in cleaned:
        match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)
    elif "```" in cleaned:
        match = re.search(r"```\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)

    try:
        data = json.loads(cleaned)
        parsed = ModelOutputSchema.model_validate(data)
        seen = set()
        for item in parsed.predictions:
            l1 = item.level_1.strip()
            l2 = item.level_2.strip()
            pair = (l1, l2)
            if is_valid_association(l1, l2) and pair not in seen:
                seen.add(pair)
                validated.append({
                    "level_1": l1,
                    "level_2": l2,
                    "evidence": item.evidence.strip() if item.evidence else "",
                })
    except Exception as e:
        logger.warning("Failed to parse predictions with evidence (%s)", e)

    return validated

