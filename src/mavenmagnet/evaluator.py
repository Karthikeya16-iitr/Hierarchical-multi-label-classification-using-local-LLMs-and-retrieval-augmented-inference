"""Comprehensive multi-label evaluation metrics for hierarchical review classification.

Calculates exact-set accuracy, micro/macro precision/recall/F1, invalid-pair rates,
and per-pair metrics with explicit support flags and zero-division handling.
Never fabricates evaluation metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

from mavenmagnet.taxonomy import get_all_valid_pairs, is_valid_association

logger = logging.getLogger("mavenmagnet.evaluator")


@dataclass(frozen=True)
class PerPairMetric:
    """Individual performance and support metrics for a single taxonomy pair."""

    pair: str
    support: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    is_reliable_eval: bool
    status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert per-pair metric to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Consolidated multi-label classification performance metrics."""

    exact_set_accuracy: float
    exact_match_ratio: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    hamming_loss: float
    invalid_pair_rate: float
    total_samples: int
    total_ground_truth_pairs: int
    total_predicted_pairs: int
    total_invalid_pairs: int
    per_pair_metrics: Dict[str, PerPairMetric] = field(default_factory=dict)

    # Backward compatibility properties
    @property
    def precision_micro(self) -> float:
        return self.micro_precision

    @property
    def recall_micro(self) -> float:
        return self.micro_recall

    @property
    def precision_macro(self) -> float:
        return self.macro_precision

    @property
    def recall_macro(self) -> float:
        return self.macro_recall

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "summary": {
                "exact_set_accuracy": self.exact_set_accuracy,
                "exact_match_ratio": self.exact_match_ratio,
                "micro_precision": self.micro_precision,
                "micro_recall": self.micro_recall,
                "micro_f1": self.micro_f1,
                "macro_precision": self.macro_precision,
                "macro_recall": self.macro_recall,
                "macro_f1": self.macro_f1,
                "hamming_loss": self.hamming_loss,
                "invalid_pair_rate": self.invalid_pair_rate,
                "total_samples": self.total_samples,
                "total_ground_truth_pairs": self.total_ground_truth_pairs,
                "total_predicted_pairs": self.total_predicted_pairs,
                "total_invalid_pairs": self.total_invalid_pairs,
            },
            "per_pair": {k: v.to_dict() for k, v in self.per_pair_metrics.items()},
        }

    def save_json(self, output_path: Path | str) -> Path:
        """Save metrics to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved evaluation metrics to %s", out)
        return out


class Evaluator:
    """Calculates ground-truth multi-label evaluation metrics."""

    def __init__(
        self,
        valid_pairs: Optional[List[Tuple[str, str]]] = None,
        min_eval_support_threshold: int = 3,
    ) -> None:
        """Initialize evaluator with target taxonomy pairs.

        Args:
            valid_pairs: List of canonical (Level 1, Level 2) tuples.
            min_eval_support_threshold: Minimum positive examples in evaluation partition
                                        required to deem a pair reliably evaluable.
        """
        self.valid_pairs = valid_pairs or get_all_valid_pairs()
        self.pair_to_idx = {pair: idx for idx, pair in enumerate(self.valid_pairs)}
        self.min_eval_support_threshold = min_eval_support_threshold

    def pairs_to_binary_matrix(
        self, list_of_pairs: List[Set[Tuple[str, str]]]
    ) -> np.ndarray:
        """Convert list of predicted/ground-truth pair sets into binary indicator matrix."""
        matrix = np.zeros((len(list_of_pairs), len(self.valid_pairs)), dtype=int)
        for i, pairs in enumerate(list_of_pairs):
            for pair in pairs:
                if pair in self.pair_to_idx:
                    matrix[i, self.pair_to_idx[pair]] = 1
        return matrix

    def calculate_invalid_pair_rate(
        self, raw_predicted_pairs: List[List[Tuple[str, str]] | Set[Tuple[str, str]]]
    ) -> Tuple[float, int, int]:
        """Compute invalid-pair rate: proportion of predicted pairs not in canonical taxonomy.

        Args:
            raw_predicted_pairs: List of raw predicted pairs per post.

        Returns:
            Tuple of (invalid_rate, total_invalid, total_predicted).
        """
        total_pred = 0
        invalid_count = 0

        for pred_set in raw_predicted_pairs:
            for pair in pred_set:
                total_pred += 1
                if len(pair) == 2:
                    if not is_valid_association(str(pair[0]), str(pair[1])):
                        invalid_count += 1
                else:
                    invalid_count += 1

        rate = (invalid_count / total_pred) if total_pred > 0 else 0.0
        return float(rate), invalid_count, total_pred

    def compute_metrics(
        self,
        y_true_pairs: List[Set[Tuple[str, str]]],
        y_pred_pairs: List[Set[Tuple[str, str]]],
        raw_predicted_pairs: Optional[List[List[Tuple[str, str]] | Set[Tuple[str, str]]]] = None,
    ) -> EvaluationMetrics:
        """Compute exact-set accuracy, micro/macro metrics, invalid rate, and per-pair metrics.

        Strictly computes verified metrics from predictions without fabrication.
        Handles zero-division safely by setting undefined metrics to 0.0.

        Args:
            y_true_pairs: Ground truth sets of valid (Level 1, Level 2) pairs.
            y_pred_pairs: Model predicted sets of valid (Level 1, Level 2) pairs.
            raw_predicted_pairs: Optional raw unvalidated predictions to check invalid rate.

        Returns:
            EvaluationMetrics instance.
        """
        if len(y_true_pairs) != len(y_pred_pairs):
            raise ValueError(
                f"Mismatched counts between true ({len(y_true_pairs)}) and predicted ({len(y_pred_pairs)}) samples."
            )

        n_samples = len(y_true_pairs)
        if n_samples == 0:
            return EvaluationMetrics(
                exact_set_accuracy=0.0,
                exact_match_ratio=0.0,
                micro_precision=0.0,
                micro_recall=0.0,
                micro_f1=0.0,
                macro_precision=0.0,
                macro_recall=0.0,
                macro_f1=0.0,
                hamming_loss=0.0,
                invalid_pair_rate=0.0,
                total_samples=0,
                total_ground_truth_pairs=0,
                total_predicted_pairs=0,
                total_invalid_pairs=0,
                per_pair_metrics={},
            )

        # Exact-set accuracy (subset accuracy)
        exact_matches = sum(1 for yt, yp in zip(y_true_pairs, y_pred_pairs) if yt == yp)
        exact_set_acc = exact_matches / n_samples

        # Convert to binary indicator matrices
        y_true = self.pairs_to_binary_matrix(y_true_pairs)
        y_pred = self.pairs_to_binary_matrix(y_pred_pairs)

        # Global micro and macro metrics with zero_division handling
        h_loss = float(hamming_loss(y_true, y_pred))
        p_micro = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        r_micro = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
        f1_mic = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

        p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Invalid pair rate
        if raw_predicted_pairs is not None:
            invalid_rate, total_inv, total_raw_preds = self.calculate_invalid_pair_rate(
                raw_predicted_pairs
            )
        else:
            invalid_rate, total_inv, total_raw_preds = self.calculate_invalid_pair_rate(
                y_pred_pairs
            )

        # Per-pair breakdown
        per_pair = self.compute_per_pair_metrics(y_true, y_pred)

        total_gt_pairs = sum(len(s) for s in y_true_pairs)
        total_pred_pairs = sum(len(s) for s in y_pred_pairs)

        return EvaluationMetrics(
            exact_set_accuracy=round(exact_set_acc, 4),
            exact_match_ratio=round(exact_set_acc, 4),
            micro_precision=round(p_micro, 4),
            micro_recall=round(r_micro, 4),
            micro_f1=round(f1_mic, 4),
            macro_precision=round(p_macro, 4),
            macro_recall=round(r_macro, 4),
            macro_f1=round(f1_mac, 4),
            hamming_loss=round(h_loss, 4),
            invalid_pair_rate=round(invalid_rate, 4),
            total_samples=n_samples,
            total_ground_truth_pairs=total_gt_pairs,
            total_predicted_pairs=total_pred_pairs,
            total_invalid_pairs=total_inv,
            per_pair_metrics=per_pair,
        )

    def compute_per_pair_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, PerPairMetric]:
        """Compute per-pair precision, recall, F1, and support with reliability flags."""
        metrics_dict: Dict[str, PerPairMetric] = {}

        for idx, pair in enumerate(self.valid_pairs):
            pair_key = f"{pair[0]} -> {pair[1]}"
            yt_col = y_true[:, idx]
            yp_col = y_pred[:, idx]

            tp = int(((yt_col == 1) & (yp_col == 1)).sum())
            fp = int(((yt_col == 0) & (yp_col == 1)).sum())
            fn = int(((yt_col == 1) & (yp_col == 0)).sum())
            support = int(yt_col.sum())
            pred_count = int(yp_col.sum())

            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            is_reliable = support >= self.min_eval_support_threshold
            if support == 0:
                status = "zero_eval_support (unrepresented in partition)"
            elif not is_reliable:
                status = f"insufficient_eval_support (support={support} < {self.min_eval_support_threshold})"
            else:
                status = "sufficient_eval_support"

            metrics_dict[pair_key] = PerPairMetric(
                pair=pair_key,
                support=support,
                predicted_count=pred_count,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                is_reliable_eval=is_reliable,
                status=status,
            )

        return metrics_dict
