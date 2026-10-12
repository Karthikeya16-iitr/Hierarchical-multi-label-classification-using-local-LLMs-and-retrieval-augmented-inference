#!/usr/bin/env python3
"""Inspect few-shot TF-IDF exemplar retrieval results on selected skincare review queries."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mavenmagnet.dataset_builder import DatasetBuilder
from mavenmagnet.retrieval import ExemplarRetriever

logger = logging.getLogger("mavenmagnet.inspect_retrieval")

DEFAULT_QUERIES = [
    "My acne breakouts are totally under control and pores are noticeably minimized.",
    "A bit pricey product but it is worth every penny, a little goes a long way.",
    "The morning glow is so luminous and my skin tone looks bright and radiant.",
    "Absorbs instantly without feeling greasy, leaving the skin soft, clean, and light weight.",
    "Anti-aging cream that helped with fine lines, wrinkles, and restored skin elasticity.",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect TF-IDF cosine similarity few-shot exemplar retrieval."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/interim/grouped_training_dataset.json",
        help="Path to grouped dataset JSON.",
    )
    parser.add_argument(
        "--split-meta",
        type=str,
        default="artifacts/split_metadata.json",
        help="Path to split metadata to ensure index is built strictly from training partition.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        help="Number of exemplars to retrieve per query (default: 3).",
    )
    parser.add_argument(
        "--save-retriever",
        type=str,
        default="artifacts/retriever.joblib",
        help="Path to save fitted retriever artifact.",
    )
    args = parser.parse_args()

    builder = DatasetBuilder()
    records = builder.load_grouped_dataset(args.dataset)

    # CRITICAL LEAKAGE REQUIREMENT:
    # Build retrieval index strictly from the training partition
    train_records, val_records = builder.iterative_train_test_split(
        records, test_size=0.2, random_seed=42
    )

    logger.info(
        "Building retrieval index strictly over %d training partition posts...",
        len(train_records),
    )
    retriever = ExemplarRetriever()
    retriever.fit(train_records)

    if args.save_retriever:
        retriever.save(args.save_retriever)

    print("\n" + "=" * 80)
    print("TF-IDF COSINE SIMILARITY FEW-SHOT RETRIEVAL INSPECTION")
    print(f"Indexed Training Posts: {len(train_records)} | Configured k: {args.top_k}")
    print("=" * 80 + "\n")

    for i, query in enumerate(DEFAULT_QUERIES, 1):
        print(f"[{i}] QUERY:")
        print(f"    \"{query}\"")
        print("    TOP RETRIEVED EXEMPLARS FROM TRAINING PARTITION:")

        exemplars = retriever.retrieve(query, k=args.top_k)
        for rank, ex in enumerate(exemplars, 1):
            pair_strings = [f"({l1} -> {l2})" for l1, l2 in ex.pairs]
            print(f"      {rank}. [Score: {ex.similarity_score:.4f} | ID: {ex.post_id}]")
            print(f"         Text:  \"{ex.original_text.strip()}\"")
            print(f"         Pairs: {', '.join(pair_strings)}")
        print("-" * 80)


if __name__ == "__main__":
    main()
