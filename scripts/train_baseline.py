#!/usr/bin/env python3
"""Train and evaluate the multi-label TF-IDF Logistic Regression baseline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mavenmagnet.classifier import TfidfLogisticBaseline
from mavenmagnet.config import load_config
from mavenmagnet.data_loader import DataLoader
from mavenmagnet.dataset_builder import DatasetBuilder
from mavenmagnet.evaluator import Evaluator

logger = logging.getLogger("mavenmagnet.train_baseline")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF Logistic Regression multi-label baseline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to pipeline configuration YAML file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (default from config: 0.35).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for split and classifiers (default from config: 42).",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="artifacts/baseline_model.joblib",
        help="Path to save trained baseline model artifact.",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="reports/baseline_evaluation_metrics.json",
        help="Path to save evaluation metrics report JSON.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.random_seed if args.random_seed is not None else cfg.random_seed
    threshold = (
        args.threshold
        if args.threshold is not None
        else cfg.baseline.decision_threshold
    )

    # 1. Load or build grouped training records
    builder = DatasetBuilder()
    interim_dataset_path = cfg.paths.interim_dir / "grouped_training_dataset.json"

    if interim_dataset_path.exists():
        records = builder.load_grouped_dataset(interim_dataset_path)
    else:
        loader = DataLoader(config=cfg)
        df_train = loader.load_training_data()
        records = builder.build_grouped_records(df_train)
        builder.save_grouped_dataset(records, output_path=interim_dataset_path)

    logger.info("Loaded %d unique grouped post records", len(records))

    # 2. Partition using iterative multi-label stratification
    train_records, val_records = builder.iterative_train_test_split(
        records, test_size=0.2, random_seed=seed
    )
    builder.save_split_metadata(
        train_records=train_records,
        test_records=val_records,
        output_path=cfg.paths.artifacts_dir / "split_metadata.json",
        reports_path=cfg.paths.reports_dir / "split_metadata.json",
        test_size=0.2,
        random_seed=seed,
    )

    logger.info(
        "Partitioned into %d training posts and %d validation posts (seed=%d)",
        len(train_records),
        len(val_records),
        seed,
    )

    # 3. Fit baseline model strictly on development training partition
    train_texts = [r.clean_text for r in train_records]
    train_targets = [r.pairs for r in train_records]

    baseline = TfidfLogisticBaseline(
        ngram_range=cfg.baseline.ngram_range,
        max_features=cfg.baseline.max_features,
        c_param=cfg.baseline.c_param,
        class_weight=cfg.baseline.class_weight,
        decision_threshold=threshold,
        random_seed=seed,
        valid_pairs=builder.valid_pairs,
        fallback_top1=True,
    )
    baseline.fit(train_texts, train_targets)

    # Save fitted model artifact
    saved_model_path = baseline.save(args.output_model)
    logger.info("Saved baseline model to %s", saved_model_path)

    # 4. Evaluate strictly on validation partition
    val_texts = [r.clean_text for r in val_records]
    val_targets = [set(r.pairs) for r in val_records]

    val_predictions = baseline.predict(val_texts)

    evaluator = Evaluator(valid_pairs=builder.valid_pairs, min_eval_support_threshold=3)
    metrics = evaluator.compute_metrics(val_targets, val_predictions)

    # Save metrics JSON
    metrics_path = Path(args.output_metrics)
    metrics.save_json(metrics_path)
    logger.info("Saved validation metrics to %s", metrics_path)

    # 5. Print summary to console
    print("\n" + "=" * 60)
    print("TF-IDF LOGISTIC REGRESSION BASELINE EVALUATION REPORT")
    print("=" * 60)
    print(f"Validation Samples:       {metrics.total_samples}")
    print(f"Decision Threshold:       {threshold}")
    print(f"Exact-Set Accuracy:       {metrics.exact_set_accuracy * 100:.2f}%")
    print(f"Micro Precision:          {metrics.micro_precision:.4f}")
    print(f"Micro Recall:             {metrics.micro_recall:.4f}")
    print(f"Micro F1:                 {metrics.micro_f1:.4f}")
    print(f"Macro Precision:          {metrics.macro_precision:.4f}")
    print(f"Macro Recall:             {metrics.macro_recall:.4f}")
    print(f"Macro F1:                 {metrics.macro_f1:.4f}")
    print(f"Hamming Loss:             {metrics.hamming_loss:.4f}")
    print(f"Invalid-Pair Rate:        {metrics.invalid_pair_rate * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
