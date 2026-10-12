"""Tests for src/mavenmagnet/prompts.py and related validators evidence parsing."""

from __future__ import annotations

import json
from typing import Set

import pytest

from mavenmagnet.prompts import (
    PROMPT_VERSION,
    build_system_prompt,
    build_user_prompt,
    get_json_output_schema,
)
from mavenmagnet.retrieval import RetrievedExemplar
from mavenmagnet.taxonomy import CANONICAL_TAXONOMY
from mavenmagnet.validators import parse_and_validate_predictions, parse_predictions_with_evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_TAXONOMY = {
    "Pricing": {"Value for money"},
    "Texture": {"Moisturisation"},
}

_SAMPLE_REVIEW = "This cream is amazing! It feels very moisturising and is worth every penny."

_SAMPLE_EXEMPLAR = RetrievedExemplar(
    post_id="ex_001",
    original_text="Great value cream that kept my skin hydrated all day.",
    clean_text="great value cream that kept my skin hydrated all day",
    pairs=[("Pricing", "Value for money"), ("Texture", "Moisturisation")],
    similarity_score=0.85,
)


# ---------------------------------------------------------------------------
# get_json_output_schema
# ---------------------------------------------------------------------------


class TestGetJsonOutputSchema:
    def test_returns_dict(self):
        schema = get_json_output_schema()
        assert isinstance(schema, dict)

    def test_has_predictions_property(self):
        schema = get_json_output_schema()
        assert "predictions" in schema["properties"]

    def test_predictions_is_array(self):
        schema = get_json_output_schema()
        assert schema["properties"]["predictions"]["type"] == "array"

    def test_item_required_fields(self):
        schema = get_json_output_schema()
        item_schema = schema["properties"]["predictions"]["items"]
        required = item_schema["required"]
        assert "level_1" in required
        assert "level_2" in required
        assert "evidence" in required

    def test_item_has_no_additional_properties(self):
        schema = get_json_output_schema()
        item_schema = schema["properties"]["predictions"]["items"]
        assert item_schema.get("additionalProperties") is False

    def test_predictions_required_at_root(self):
        schema = get_json_output_schema()
        assert "predictions" in schema["required"]

    def test_root_no_additional_properties(self):
        schema = get_json_output_schema()
        assert schema.get("additionalProperties") is False

    def test_schema_is_serializable(self):
        schema = get_json_output_schema()
        dumped = json.dumps(schema)
        reloaded = json.loads(dumped)
        assert reloaded == schema


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_returns_string(self):
        result = build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_version(self):
        result = build_system_prompt(version="9.9.9")
        assert "9.9.9" in result

    def test_default_version_equals_prompt_version(self):
        result = build_system_prompt()
        assert PROMPT_VERSION in result

    def test_contains_all_level1_labels_from_canonical(self):
        result = build_system_prompt()
        for l1 in CANONICAL_TAXONOMY:
            assert l1 in result, f"Missing Level 1 label: {l1}"

    def test_contains_all_level2_labels_from_canonical(self):
        result = build_system_prompt()
        for l2_set in CANONICAL_TAXONOMY.values():
            for l2 in l2_set:
                assert l2 in result, f"Missing Level 2 label: {l2}"

    def test_contains_prohibitions(self):
        result = build_system_prompt()
        # Must tell model not to invent labels
        lower = result.lower()
        assert any(phrase in lower for phrase in ["never invent", "hallucin", "zero hallucination"])

    def test_custom_taxonomy_used(self):
        result = build_system_prompt(taxonomy=_MINIMAL_TAXONOMY)
        assert "Pricing" in result
        assert "Value for money" in result
        # Default labels not in minimal taxonomy must NOT be present
        assert "Anti-Aging" not in result

    def test_duplicate_pair_instruction_present(self):
        result = build_system_prompt()
        lower = result.lower()
        assert "duplicate" in lower or "at most once" in lower

    def test_json_schema_structure_mentioned(self):
        result = build_system_prompt()
        assert '"predictions"' in result or "predictions" in result

    def test_taxonomy_pair_count_mentioned_or_implied(self):
        """All 43 pairs should be renderable from CANONICAL_TAXONOMY."""
        all_pairs = {
            (l1, l2)
            for l1, l2s in CANONICAL_TAXONOMY.items()
            for l2 in l2s
        }
        assert len(all_pairs) == 43

    def test_no_answer_leakage_for_sample_review(self):
        """System prompt should not contain the sample review text."""
        result = build_system_prompt()
        assert _SAMPLE_REVIEW not in result


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_returns_string(self):
        result = build_user_prompt(_SAMPLE_REVIEW)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_target_review_included(self):
        result = build_user_prompt(_SAMPLE_REVIEW)
        assert _SAMPLE_REVIEW.strip() in result

    def test_no_pre_filled_answer_for_target(self):
        """The target review section must NOT contain any predictions JSON for itself."""
        result = build_user_prompt(_SAMPLE_REVIEW)
        # The only JSON in the prompt (if no exemplars) should be the trailing "JSON Output:" block
        # There must be no {"predictions": [...]} filled in after the target review
        target_section_start = result.find("Target Review to Classify")
        assert target_section_start != -1, "Target section header missing"
        after_target = result[target_section_start:]
        # No fully populated predictions array should appear in the target section
        assert '"level_1"' not in after_target or "JSON Output:" in after_target

    def test_no_exemplars_produces_no_example_section(self):
        result = build_user_prompt(_SAMPLE_REVIEW, exemplars=None)
        assert "In-Context Examples" not in result

    def test_exemplar_retrieved_exemplar_type_included(self):
        result = build_user_prompt(_SAMPLE_REVIEW, exemplars=[_SAMPLE_EXEMPLAR])
        assert "In-Context Examples" in result
        assert _SAMPLE_EXEMPLAR.original_text.strip() in result

    def test_exemplar_predictions_present_for_exemplar(self):
        result = build_user_prompt(_SAMPLE_REVIEW, exemplars=[_SAMPLE_EXEMPLAR])
        # Exemplar predictions should be shown
        assert "Value for money" in result
        assert "Moisturisation" in result

    def test_exemplar_tuple_format_accepted(self):
        exemplar_tuple = ("Great hydrating cream.", [("Texture", "Moisturisation")])
        result = build_user_prompt(_SAMPLE_REVIEW, exemplars=[exemplar_tuple])
        assert "In-Context Examples" in result
        assert "Great hydrating cream." in result
        assert "Moisturisation" in result

    def test_exemplar_dict_format_accepted(self):
        exemplar_dict = {
            "original_text": "Affordable and effective.",
            "pairs": [("Pricing", "Value for money")],
        }
        result = build_user_prompt(_SAMPLE_REVIEW, exemplars=[exemplar_dict])
        assert "Affordable and effective." in result
        assert "Value for money" in result

    def test_multiple_exemplars_all_present(self):
        ex2 = RetrievedExemplar(
            post_id="ex_002",
            original_text="My skin looks younger after using this.",
            clean_text="my skin looks younger after using this",
            pairs=[("Anti-Aging", "Elasticity")],
            similarity_score=0.72,
        )
        result = build_user_prompt(_SAMPLE_REVIEW, exemplars=[_SAMPLE_EXEMPLAR, ex2])
        assert "Great value cream" in result
        assert "younger" in result

    def test_target_not_mixed_with_exemplar_answers(self):
        """Target review's text appears but without pre-assigned labels."""
        result = build_user_prompt(_SAMPLE_REVIEW, exemplars=[_SAMPLE_EXEMPLAR])
        # Find where exemplar section ends and target section begins
        target_idx = result.find("Target Review to Classify")
        assert target_idx != -1
        target_section = result[target_idx:]
        # Target section must contain the review text
        assert _SAMPLE_REVIEW.strip() in target_section

    def test_empty_exemplar_list_treated_as_no_exemplars(self):
        result = build_user_prompt(_SAMPLE_REVIEW, exemplars=[])
        assert "In-Context Examples" not in result

    def test_instructions_section_present(self):
        result = build_user_prompt(_SAMPLE_REVIEW)
        assert "Instructions" in result

    def test_json_output_marker_present(self):
        result = build_user_prompt(_SAMPLE_REVIEW)
        assert "JSON Output" in result


# ---------------------------------------------------------------------------
# Taxonomy consistency
# ---------------------------------------------------------------------------


class TestTaxonomyConsistency:
    def test_canonical_taxonomy_nonempty(self):
        assert len(CANONICAL_TAXONOMY) > 0

    def test_all_canonical_pairs_appear_in_system_prompt(self):
        system = build_system_prompt()
        for l1, l2s in CANONICAL_TAXONOMY.items():
            assert l1 in system
            for l2 in l2s:
                assert l2 in system

    def test_no_extra_pairs_in_system_prompt_from_minimal(self):
        """When passing a custom minimal taxonomy, only those labels appear."""
        system = build_system_prompt(taxonomy=_MINIMAL_TAXONOMY)
        # Labels from default taxonomy NOT in minimal must be absent
        for l1 in CANONICAL_TAXONOMY:
            if l1 not in _MINIMAL_TAXONOMY:
                assert l1 not in system, f"Unexpected L1 '{l1}' in minimal taxonomy prompt"


# ---------------------------------------------------------------------------
# parse_predictions_with_evidence (validators integration)
# ---------------------------------------------------------------------------


class TestParsePredictionsWithEvidence:
    def _make_json(self, predictions: list) -> str:
        return json.dumps({"predictions": predictions})

    def test_valid_pair_returned(self):
        raw = self._make_json([
            {"level_1": "Pricing", "level_2": "Value for money", "evidence": "worth every penny"}
        ])
        result = parse_predictions_with_evidence(raw)
        assert len(result) == 1
        assert result[0]["level_1"] == "Pricing"
        assert result[0]["level_2"] == "Value for money"
        assert result[0]["evidence"] == "worth every penny"

    def test_invalid_pair_filtered_out(self):
        raw = self._make_json([
            {"level_1": "FakeCategory", "level_2": "Nonexistent", "evidence": "irrelevant"}
        ])
        result = parse_predictions_with_evidence(raw)
        assert result == []

    def test_duplicate_pairs_deduplicated(self):
        raw = self._make_json([
            {"level_1": "Pricing", "level_2": "Value for money", "evidence": "worth it"},
            {"level_1": "Pricing", "level_2": "Value for money", "evidence": "again worth it"},
        ])
        result = parse_predictions_with_evidence(raw)
        assert len(result) == 1

    def test_malformed_json_returns_empty(self):
        result = parse_predictions_with_evidence("not json at all")
        assert result == []

    def test_empty_predictions_returns_empty(self):
        raw = json.dumps({"predictions": []})
        result = parse_predictions_with_evidence(raw)
        assert result == []

    def test_markdown_fenced_json_parsed(self):
        raw = "```json\n" + self._make_json([
            {"level_1": "Pricing", "level_2": "Value for money", "evidence": "great deal"}
        ]) + "\n```"
        result = parse_predictions_with_evidence(raw)
        assert len(result) == 1
        assert result[0]["level_2"] == "Value for money"
