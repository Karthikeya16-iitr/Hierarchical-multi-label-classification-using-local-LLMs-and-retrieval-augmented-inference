"""Dataset building, post grouping, deduplication, and leak-free multi-label stratification.

DATA LEAKAGE PREVENTION RATIONALE:
=================================
In multi-label review datasets (such as Training.xlsx), a single consumer post
frequently discusses multiple aspects of a product (e.g. skin finish and pricing),
spanning multiple rows in the original raw workbook.

If an un-grouped split were performed on individual mapping rows:
1. Row A ('This cream leaves my skin dewy and costs very little', 'Finish', 'Dewy')
   could land in the training set.
2. Row B ('This cream leaves my skin dewy and costs very little', 'Pricing', 'Affordability')
   could land in the evaluation/test set.
3. The evaluation partition would contain the EXACT verbatim text that the classifier
   or LLM already observed during training.
4. For TF-IDF and n-gram linear models, n-gram overlap would be 100%, artificially
   inflating precision, recall, and exact-match metrics.
5. For few-shot LLMs, exemplar retrieval would pull the identical test review into
   the prompt context.
6. This violates the core machine learning requirement of evaluating on independent,
   unseen data and conceals real-world generalization failure.

Therefore, the pipeline strictly groups mappings by unique post text before any
partitioning. Every post and all its associated labels are assigned atomically to
either training or evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from mavenmagnet.preprocessing import clean_text
from mavenmagnet.taxonomy import get_all_valid_pairs, is_valid_association

logger = logging.getLogger("mavenmagnet.dataset_builder")

DATA_LEAKAGE_EXPLANATION: str = (
    "A single post with multiple labels spans multiple rows in Training.xlsx. "
    "Splitting individual rows would leak the exact post text into both training "
    "and evaluation sets, artificially inflating metrics. Grouping by unique post "
    "ensures all labels for a post stay together and test posts are genuinely unseen."
)


@dataclass
class GroupedPostRecord:
    """Consolidated post record containing one unique post and all associated pairs."""

    post_id: str
    original_text: str
    clean_text: str
    pairs: List[Tuple[str, str]]
    original_row_indices: List[int]

    @property
    def text(self) -> str:
        """Alias for backward compatibility."""
        return self.original_text

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record to dictionary."""
        return {
            "post_id": self.post_id,
            "original_text": self.original_text,
            "clean_text": self.clean_text,
            "pairs": [[p[0], p[1]] for p in self.pairs],
            "original_row_indices": self.original_row_indices,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroupedPostRecord:
        """Deserialize record from dictionary."""
        pairs = [(item[0], item[1]) for item in data["pairs"]]
        return cls(
            post_id=data["post_id"],
            original_text=data["original_text"],
            clean_text=data.get("clean_text", clean_text(data["original_text"])),
            pairs=sorted(pairs),
            original_row_indices=data["original_row_indices"],
        )


# Backward-compatibility alias
PostRecord = GroupedPostRecord


@dataclass
class GroupedDatasetStats:
    """Statistical summary of the grouped multi-label dataset."""

    total_unique_posts: int
    total_mappings_raw: int
    total_mappings_deduplicated: int
    mapping_distribution: Dict[int, int]
    pair_support: Dict[str, int]
    rare_pairs: Dict[str, int]
    min_mappings_per_post: int
    max_mappings_per_post: int
    mean_mappings_per_post: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stats to dictionary."""
        return asdict(self)


class DatasetBuilder:
    """Builds multi-label datasets aggregated and partitioned strictly by post."""

    def __init__(self, valid_pairs: Optional[List[Tuple[str, str]]] = None) -> None:
        """Initialize dataset builder with canonical pairs."""
        self.valid_pairs = valid_pairs or get_all_valid_pairs()
        self.pair_to_idx = {pair: idx for idx, pair in enumerate(self.valid_pairs)}

    def build_grouped_records(self, df: pd.DataFrame) -> List[GroupedPostRecord]:
        """Aggregate row-level mappings into one record per unique post.

        Groups by normalized review text so separate rows of the same review are
        never treated as independent examples, preserves original text, deduplicates
        label pairs per post, tracks reverse mappings to row indices, and preserves
        all valid and rare labels.

        Args:
            df: DataFrame containing ['Post', 'Level 1', 'Level 2'].

        Returns:
            List of GroupedPostRecord instances.
        """
        grouped_dict: Dict[str, Dict[str, Any]] = {}

        for row_idx, row in df.iterrows():
            raw_post = str(row["Post"]) if pd.notna(row["Post"]) else ""
            cleaned = clean_text(raw_post)
            if not cleaned:
                continue

            l1 = str(row["Level 1"]).strip() if pd.notna(row["Level 1"]) else ""
            l2 = str(row["Level 2"]).strip() if pd.notna(row["Level 2"]) else ""

            if cleaned not in grouped_dict:
                grouped_dict[cleaned] = {
                    "original_text": raw_post,
                    "clean_text": cleaned,
                    "pairs": set(),
                    "row_indices": [],
                }

            grouped_dict[cleaned]["row_indices"].append(int(row_idx))

            if l1 and l2:
                pair = (l1, l2)
                if is_valid_association(l1, l2):
                    grouped_dict[cleaned]["pairs"].add(pair)
                else:
                    logger.warning(
                        "Encountered non-canonical pair (%s, %s) at row %d", l1, l2, row_idx
                    )

        records: List[GroupedPostRecord] = []
        for post_num, (cleaned_key, data) in enumerate(grouped_dict.items()):
            rec = GroupedPostRecord(
                post_id=f"post_{post_num:03d}",
                original_text=data["original_text"],
                clean_text=data["clean_text"],
                pairs=sorted(data["pairs"]),
                original_row_indices=data["row_indices"],
            )
            records.append(rec)

        logger.info(
            "Built %d unique post records from %d training rows",
            len(records),
            len(df),
        )
        return records

    def build_records_from_df(self, df: pd.DataFrame) -> List[GroupedPostRecord]:
        """Alias for build_grouped_records for backward compatibility."""
        return self.build_grouped_records(df)

    def compute_statistics(
        self, records: List[GroupedPostRecord], raw_total_rows: int = 0
    ) -> GroupedDatasetStats:
        """Compute statistics including mapping distribution and label support.

        Args:
            records: List of grouped post records.
            raw_total_rows: Total raw rows in the input workbook before grouping.

        Returns:
            GroupedDatasetStats instance.
        """
        if not records:
            return GroupedDatasetStats(
                total_unique_posts=0,
                total_mappings_raw=raw_total_rows,
                total_mappings_deduplicated=0,
                mapping_distribution={},
                pair_support={},
                rare_pairs={},
                min_mappings_per_post=0,
                max_mappings_per_post=0,
                mean_mappings_per_post=0.0,
            )

        mapping_counts = [len(r.pairs) for r in records]
        mapping_dist: Dict[int, int] = {}
        for count in sorted(set(mapping_counts)):
            mapping_dist[count] = mapping_counts.count(count)

        pair_counts: Dict[str, int] = {f"{p[0]} -> {p[1]}": 0 for p in self.valid_pairs}
        for r in records:
            for pair in r.pairs:
                pair_key = f"{pair[0]} -> {pair[1]}"
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        rare_pairs = {k: v for k, v in pair_counts.items() if v <= 2}
        total_dedup = sum(len(r.pairs) for r in records)
        total_raw = (
            raw_total_rows if raw_total_rows > 0 else sum(len(r.original_row_indices) for r in records)
        )

        return GroupedDatasetStats(
            total_unique_posts=len(records),
            total_mappings_raw=total_raw,
            total_mappings_deduplicated=total_dedup,
            mapping_distribution=mapping_dist,
            pair_support=dict(sorted(pair_counts.items(), key=lambda x: (x[1], x[0]))),
            rare_pairs=dict(sorted(rare_pairs.items(), key=lambda x: (x[1], x[0]))),
            min_mappings_per_post=min(mapping_counts),
            max_mappings_per_post=max(mapping_counts),
            mean_mappings_per_post=round(float(np.mean(mapping_counts)), 3),
        )

    def save_grouped_dataset(
        self,
        records: List[GroupedPostRecord],
        output_path: Path | str = "data/interim/grouped_training_dataset.json",
        stats_path: Optional[Path | str] = "data/interim/grouped_dataset_stats.json",
        raw_total_rows: int = 0,
    ) -> Path:
        """Save grouped records and statistics to data/interim/."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        payload = [r.to_dict() for r in records]
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved %d grouped post records to %s", len(records), out_file)

        if stats_path:
            stats_file = Path(stats_path)
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            stats = self.compute_statistics(records, raw_total_rows=raw_total_rows)
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats.to_dict(), f, indent=2)
            logger.info("Saved dataset statistics to %s", stats_file)

        return out_file

    def load_grouped_dataset(self, input_path: Path | str) -> List[GroupedPostRecord]:
        """Load grouped post records from JSON."""
        in_file = Path(input_path)
        if not in_file.exists():
            raise FileNotFoundError(f"Grouped dataset not found at: {in_file}")

        with open(in_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = [GroupedPostRecord.from_dict(item) for item in data]
        logger.info("Loaded %d grouped post records from %s", len(records), in_file)
        return records

    def to_binary_indicator_matrix(self, records: List[GroupedPostRecord]) -> np.ndarray:
        """Encode records into a binary multi-label target matrix of shape (N, 43)."""
        matrix = np.zeros((len(records), len(self.valid_pairs)), dtype=int)
        for i, rec in enumerate(records):
            for pair in rec.pairs:
                if pair in self.pair_to_idx:
                    matrix[i, self.pair_to_idx[pair]] = 1
        return matrix

    def iterative_train_test_split(
        self,
        records: List[GroupedPostRecord],
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> Tuple[List[GroupedPostRecord], List[GroupedPostRecord]]:
        """Perform iterative multi-label stratified train/test split on unique posts.

        Uses Sechidis et al. (2011) iterative greedy multi-label stratification.
        Guarantees:
        - Atomicity: all labels for a given post remain in the same partition.
        - Zero leakage: train and test post text sets are 100% disjoint.
        - Rare label handling: single-instance labels are prioritized for train
          so classifiers can observe all valid taxonomy classes.
        - Determinism: strictly reproducible with fixed random seed.

        Args:
            records: List of grouped post records.
            test_size: Proportion of records allocated to the test set.
            random_seed: Fixed seed for tie-breaking and reproducibility.

        Returns:
            Tuple of (train_records, test_records).
        """
        if not records:
            return [], []

        n_samples = len(records)
        n_test = int(round(n_samples * test_size))
        n_train = n_samples - n_test

        Y = self.to_binary_indicator_matrix(records)
        n_labels = Y.shape[1]

        rng = np.random.RandomState(random_seed)

        # Compute total label counts across dataset
        label_counts = Y.sum(axis=0)

        # Desired label quotas in test partition
        desired_test = np.round(label_counts * test_size).astype(int)
        desired_train = label_counts - desired_test

        assigned = np.zeros(n_samples, dtype=bool)
        test_indices: List[int] = []
        train_indices: List[int] = []

        curr_train = np.zeros(n_labels, dtype=int)
        curr_test = np.zeros(n_labels, dtype=int)

        # Order labels by rarity (least frequent first)
        label_order = np.argsort(label_counts)

        for l_idx in label_order:
            unassigned_with_label = np.where((Y[:, l_idx] == 1) & (~assigned))[0]
            rng.shuffle(unassigned_with_label)

            for idx in unassigned_with_label:
                if assigned[idx]:
                    continue

                need_test = (desired_test[l_idx] - curr_test[l_idx]) / max(desired_test[l_idx], 1)
                need_train = (desired_train[l_idx] - curr_train[l_idx]) / max(desired_train[l_idx], 1)

                if len(test_indices) >= n_test:
                    assign_test = False
                elif len(train_indices) >= n_train:
                    assign_test = True
                elif need_test > need_train:
                    assign_test = True
                else:
                    assign_test = False

                if assign_test:
                    test_indices.append(int(idx))
                    curr_test += Y[idx]
                else:
                    train_indices.append(int(idx))
                    curr_train += Y[idx]
                assigned[idx] = True

        # Assign remaining unassigned posts
        remaining = np.where(~assigned)[0]
        rng.shuffle(remaining)
        for idx in remaining:
            if len(test_indices) < n_test:
                test_indices.append(int(idx))
            else:
                train_indices.append(int(idx))

        train_records = [records[i] for i in sorted(train_indices)]
        test_records = [records[i] for i in sorted(test_indices)]

        # Verify zero leakage
        train_texts = {r.clean_text for r in train_records}
        test_texts = {r.clean_text for r in test_records}
        assert train_texts.isdisjoint(test_texts), "Critical: Train and test sets contain overlapping posts!"

        logger.info(
            "Iterative stratified split complete: %d train posts, %d test posts (seed=%d)",
            len(train_records),
            len(test_records),
            random_seed,
        )
        return train_records, test_records

    def get_partition_label_support(
        self, records: List[GroupedPostRecord], min_support_threshold: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """Report positive label support for each canonical pair in a partition.

        Flags labels with insufficient support (< threshold) or zero support.

        Args:
            records: Partition post records.
            min_support_threshold: Minimum positive posts needed for reliable evaluation.

        Returns:
            Dictionary mapping pair string to support count and reliability status.
        """
        counts: Dict[str, int] = {f"{p[0]} -> {p[1]}": 0 for p in self.valid_pairs}
        for r in records:
            for pair in r.pairs:
                key = f"{pair[0]} -> {pair[1]}"
                if key in counts:
                    counts[key] += 1

        support_report: Dict[str, Dict[str, Any]] = {}
        for key, count in counts.items():
            if count == 0:
                status = "unrepresented_in_partition"
                reliable = False
            elif count < min_support_threshold:
                status = f"insufficient_support (count={count} < {min_support_threshold})"
                reliable = False
            else:
                status = "sufficient_support"
                reliable = True

            support_report[key] = {
                "support": count,
                "is_reliable": reliable,
                "status": status,
            }

        return support_report

    def save_split_metadata(
        self,
        train_records: List[GroupedPostRecord],
        test_records: List[GroupedPostRecord],
        output_path: Path | str = "artifacts/split_metadata.json",
        reports_path: Optional[Path | str] = "reports/split_metadata.json",
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Save comprehensive split metadata, label supports, and leakage rationale."""
        train_support = self.get_partition_label_support(train_records)
        test_support = self.get_partition_label_support(test_records)

        unrepresented_test = [k for k, v in test_support.items() if v["support"] == 0]
        insufficient_test = [
            k for k, v in test_support.items() if 0 < v["support"] < 3
        ]

        metadata: Dict[str, Any] = {
            "split_info": {
                "random_seed": random_seed,
                "test_size": test_size,
                "total_unique_posts": len(train_records) + len(test_records),
                "train_post_count": len(train_records),
                "test_post_count": len(test_records),
                "disjoint_verified": bool(
                    {r.clean_text for r in train_records}.isdisjoint({r.clean_text for r in test_records})
                ),
                "method": "iterative_multilabel_stratification",
            },
            "data_leakage_prevention_rationale": DATA_LEAKAGE_EXPLANATION,
            "evaluation_support_analysis": {
                "total_canonical_pairs": len(self.valid_pairs),
                "train_pairs_represented": sum(1 for v in train_support.values() if v["support"] > 0),
                "test_pairs_represented": sum(1 for v in test_support.values() if v["support"] > 0),
                "unrepresented_in_test_count": len(unrepresented_test),
                "unrepresented_in_test_pairs": unrepresented_test,
                "insufficient_support_in_test_count": len(insufficient_test),
                "insufficient_support_in_test_pairs": insufficient_test,
            },
            "train_partition_support": train_support,
            "test_partition_support": test_support,
            "train_post_ids": [r.post_id for r in train_records],
            "test_post_ids": [r.post_id for r in test_records],
        }

        # Save to artifacts
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved split metadata to %s", out)

        # Save to reports if requested
        if reports_path:
            rep = Path(reports_path)
            rep.parent.mkdir(parents=True, exist_ok=True)
            with open(rep, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved split metadata copy to %s", rep)

        return metadata

    def create_grouped_splits(
        self,
        records: List[GroupedPostRecord],
        n_splits: int = 5,
        random_seed: int = 42,
    ) -> Generator[Tuple[List[GroupedPostRecord], List[GroupedPostRecord]], None, None]:
        """Generate strictly leak-free train/validation splits partitioned by post."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        indices = np.arange(len(records))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            train_records = [records[i] for i in train_idx]
            val_records = [records[i] for i in val_idx]

            # Assert zero leakage between train and val posts
            train_texts = {r.clean_text for r in train_records}
            val_texts = {r.clean_text for r in val_records}
            assert train_texts.isdisjoint(val_texts), f"Data leakage detected in fold {fold_idx}!"

            yield train_records, val_records
