"""Prompt templates and builders for hierarchical multi-label review classification.

Constructs versioned prompts enforcing the exact canonical taxonomy, in-context
retrieved exemplars, multi-label instructions, grounded evidence, and strict JSON schemas.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from mavenmagnet.retrieval import RetrievedExemplar
from mavenmagnet.taxonomy import CANONICAL_TAXONOMY

PROMPT_VERSION: str = "1.0.0"


def get_json_output_schema() -> Dict[str, Any]:
    """Return the strict JSON Schema for model prediction validation.

    Returns:
        JSON Schema dictionary.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SkincareReviewPredictions",
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "description": "List of distinct valid (Level 1, Level 2) mappings supported by the review.",
                "items": {
                    "type": "object",
                    "properties": {
                        "level_1": {
                            "type": "string",
                            "description": "Exact canonical Level 1 category name.",
                        },
                        "level_2": {
                            "type": "string",
                            "description": "Exact canonical Level 2 subcategory name.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Short quote or phrase from the review justifying this mapping.",
                        },
                    },
                    "required": ["level_1", "level_2", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }


def build_system_prompt(
    taxonomy: Optional[Dict[str, Set[str]]] = None,
    version: str = PROMPT_VERSION,
) -> str:
    """Build system instructions specifying strict taxonomy rules and JSON output schema.

    Args:
        taxonomy: Optional mapping of Level 1 to Level 2 sets. Defaults to CANONICAL_TAXONOMY.
        version: Prompt template version string.

    Returns:
        Formatted system instruction string.
    """
    tax = taxonomy or CANONICAL_TAXONOMY

    taxonomy_lines: List[str] = []
    for l1 in sorted(tax.keys()):
        l2_sorted = sorted(tax[l1])
        taxonomy_lines.append(f"- **{l1}**: {', '.join(l2_sorted)}")

    taxonomy_text = "\n".join(taxonomy_lines)

    return f"""You are an expert NLP classifier specialized in skincare and beauty consumer reviews.
Prompt Version: {version}

### OBJECTIVE
Your task is to assign all applicable Level 1 and Level 2 classification pairs to a customer review post.

### STRICT TAXONOMY RULES
1. **Canonical Constraint**: You MUST ONLY select (Level 1, Level 2) pairs from the ALLOWED TAXONOMY below.
2. **Zero Hallucination**: NEVER invent, extrapolate, or alter any category or subcategory names.
3. **Multi-Label Coverage**: A single review post often discusses multiple distinct aspects (e.g. price, skin finish, moisturization). You MUST identify and predict ALL distinct valid mappings supported by the text (typically 1 to 4 pairs).
4. **Textual Evidence**: Every predicted pair MUST be directly grounded in the review text. In the "evidence" field, cite the specific word or phrase from the review that justifies the mapping.
5. **No Duplicate Mappings**: Each unique (Level 1, Level 2) pair must appear at most once in your predictions list.
6. **Empty List**: Return `{{"predictions": []}}` ONLY if the review contains no content discussing any aspect of the permitted taxonomy.
7. **JSON Output Only**: Respond strictly with a valid JSON object matching the schema below. Do not wrap with conversational filler or markdown explanations outside the JSON object.

### ALLOWED TAXONOMY (17 Level 1 Categories, 43 Valid Pairs):
{taxonomy_text}

### OUTPUT JSON SCHEMA
{{
  "predictions": [
    {{
      "level_1": "<Exact Level 1 from taxonomy>",
      "level_2": "<Exact Level 2 under that Level 1>",
      "evidence": "<Direct phrase or quote from the review justifying this mapping>"
    }}
  ]
}}
"""


def _format_exemplar_predictions(pairs: List[Tuple[str, str]], text: str) -> List[Dict[str, str]]:
    """Format pairs into schema-compliant prediction dictionaries."""
    preds: List[Dict[str, str]] = []
    text_lower = text.lower()
    for l1, l2 in sorted(pairs):
        # Generate representative evidence snippet for exemplar
        evidence = f"Grounds {l1} -> {l2} in review context"
        preds.append({"level_1": l1, "level_2": l2, "evidence": evidence})
    return preds


def build_user_prompt(
    post_text: str,
    exemplars: Optional[List[Any]] = None,
) -> str:
    """Construct user prompt containing optional few-shot demonstrations and the target review.

    CRITICAL LEAKAGE PRECAUTION:
    The target review to classify NEVER contains predictions or answers. Answers are only
    included for legitimately retrieved training exemplars.

    Args:
        post_text: Review text requiring classification.
        exemplars: Optional list of RetrievedExemplar instances or (text, pairs) tuples.

    Returns:
        Formatted user prompt string.
    """
    sections: List[str] = []

    # 1. In-context few-shot exemplars (if provided)
    if exemplars:
        sections.append("### In-Context Examples of Correct Classifications:")
        for idx, ex in enumerate(exemplars, 1):
            if isinstance(ex, RetrievedExemplar):
                ex_text = ex.original_text.strip()
                ex_pairs = ex.pairs
            elif isinstance(ex, tuple) and len(ex) >= 2:
                ex_text = str(ex[0]).strip()
                ex_pairs = ex[1]
            elif isinstance(ex, dict):
                ex_text = str(ex.get("original_text", ex.get("text", ""))).strip()
                ex_pairs = ex.get("pairs", [])
            else:
                continue

            preds = _format_exemplar_predictions(ex_pairs, ex_text)
            sample_json = json.dumps({"predictions": preds}, indent=2)

            sections.append(f"Example {idx}:")
            sections.append(f'Review: "{ex_text}"')
            sections.append(f"Output:\n{sample_json}\n")

    # 2. Target review requiring classification (unlabelled)
    clean_target = post_text.strip()
    sections.append("### Target Review to Classify:")
    sections.append(f'"{clean_target}"')
    sections.append("\n### Instructions:")
    sections.append(
        "- Analyze all statements in the target review above.\n"
        "- Identify all applicable (Level 1, Level 2) pairs from the allowed taxonomy.\n"
        "- Provide supporting evidence for each prediction.\n"
        "- Return the final result strictly as a valid JSON object."
    )
    sections.append("\nJSON Output:")

    return "\n".join(sections)
