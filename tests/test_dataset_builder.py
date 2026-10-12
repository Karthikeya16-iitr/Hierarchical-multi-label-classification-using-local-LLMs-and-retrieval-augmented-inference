"""Unit and integration tests for dataset_builder module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mavenmagnet.dataset_builder import DatasetBuilder, GroupedPostRecord


@pytest.fixture
def multi_row_sample_df() -> pd.DataFrame:
    """Fixture with multiple posts, duplicates, and multiple mappings."""
    return pd.DataFrame(
        {
            "Post": [
                "  Loved this moisturizer, glowing skin!  ",
                "  Loved this moisturizer, glowing skin!  ",
                "  Loved this moisturizer, glowing skin!  ",  # Duplicate mapping of Tone -> Glow
                "Second post about soft feel.",
                "Second post about soft feel. ",  # Slight whitespace variation of second post
            ],
            "Level 1": [
                "Moisturization",
                "Tone",
                "Tone",
                "Feel",
                "Texture",
            ],
            "Level 2": [
                "Hydration",
                "Glow",
                "Glow",
                "Fresh",
                "Softness",
            ],
        }
    )


def test_grouping_aggregates_rows_per_unique_post(multi_row_sample_df: pd.DataFrame) -> None:
    """Verify that multiple rows of the same post are aggregated into one record."""
    builder = DatasetBuilder()
    records = builder.build_grouped_records(multi_row_sample_df)

    # 5 rows should collapse into exactly 2 unique post records
    assert len(records) == 2

    first_rec = records[0]
    # Verify original text preserved
    assert first_rec.original_text == "  Loved this moisturizer, glowing skin!  "
    assert first_rec.clean_text == "Loved this moisturizer, glowing skin!"

    second_rec = records[1]
    assert len(second_rec.pairs) == 2
    assert ("Feel", "Fresh") in second_rec.pairs
    assert ("Texture", "Softness") in second_rec.pairs


def test_deduplicates_repeated_label_pairs(multi_row_sample_df: pd.DataFrame) -> None:
    """Verify that duplicate label pairs within a post are deduplicated."""
    builder = DatasetBuilder()
    records = builder.build_grouped_records(multi_row_sample_df)

    first_rec = records[0]
    # Rows 0, 1, and 2: Tone -> Glow appeared twice, so distinct pairs must be 2
    assert len(first_rec.pairs) == 2
    assert ("Moisturization", "Hydration") in first_rec.pairs
    assert ("Tone", "Glow") in first_rec.pairs


def test_reverse_mapping_tracks_all_original_rows(multi_row_sample_df: pd.DataFrame) -> None:
    """Verify reverse mapping preserves all row indices from original DataFrame."""
    builder = DatasetBuilder()
    records = builder.build_grouped_records(multi_row_sample_df)

    first_rec = records[0]
    # Rows 0, 1, 2 correspond to first post
    assert first_rec.original_row_indices == [0, 1, 2]

    second_rec = records[1]
    # Rows 3, 4 correspond to second post
    assert second_rec.original_row_indices == [3, 4]


def test_preserves_rare_labels() -> None:
    """Verify that single-occurrence rare labels are strictly preserved."""
    df_rare = pd.DataFrame(
        {
            "Post": ["Rare post with single dewy finish", "Common post with glow"],
            "Level 1": ["Finish", "Tone"],
            "Level 2": ["Dewy", "Glow"],
        }
    )
    builder = DatasetBuilder()
    records = builder.build_grouped_records(df_rare)
    stats = builder.compute_statistics(records)

    assert ("Finish", "Dewy") in records[0].pairs
    assert "Finish -> Dewy" in stats.rare_pairs
    assert stats.pair_support["Finish -> Dewy"] == 1


def test_save_and_load_grouped_dataset(tmp_path: Path, multi_row_sample_df: pd.DataFrame) -> None:
    """Verify serialization and round-trip loading of grouped records."""
    builder = DatasetBuilder()
    records = builder.build_grouped_records(multi_row_sample_df)

    data_file = tmp_path / "grouped.json"
    stats_file = tmp_path / "stats.json"

    builder.save_grouped_dataset(records, output_path=data_file, stats_path=stats_file)
    assert data_file.exists()
    assert stats_file.exists()

    loaded = builder.load_grouped_dataset(data_file)
    assert len(loaded) == len(records)
    assert loaded[0].post_id == records[0].post_id
    assert loaded[0].pairs == records[0].pairs
    assert loaded[0].original_row_indices == records[0].original_row_indices


def test_binary_indicator_matrix_encoding(multi_row_sample_df: pd.DataFrame) -> None:
    """Verify conversion of grouped records to binary multi-label indicator matrix."""
    builder = DatasetBuilder()
    records = builder.build_grouped_records(multi_row_sample_df)
    matrix = builder.to_binary_indicator_matrix(records)

    assert matrix.shape == (2, len(builder.valid_pairs))
    assert matrix.sum(axis=1).tolist() == [2, 2]


def test_leak_free_grouped_splits(multi_row_sample_df: pd.DataFrame) -> None:
    """Verify that grouped splits have zero overlap in review texts."""
    builder = DatasetBuilder()
    records = builder.build_grouped_records(multi_row_sample_df)
    splits = list(builder.create_grouped_splits(records, n_splits=2, random_seed=42))

    assert len(splits) == 2
    for train_recs, val_recs in splits:
        train_texts = {r.clean_text for r in train_recs}
        val_texts = {r.clean_text for r in val_recs}
        assert train_texts.isdisjoint(val_texts)


def test_real_training_workbook_grouped_dataset() -> None:
    """Verify grouping properties on actual production Training.xlsx."""
    df_real = pd.read_excel("data/input/Training.xlsx", engine="openpyxl")
    builder = DatasetBuilder()
    records = builder.build_grouped_records(df_real)
    stats = builder.compute_statistics(records, raw_total_rows=len(df_real))

    assert len(records) == 266
    assert stats.total_unique_posts == 266
    assert stats.total_mappings_raw == 512
    assert stats.total_mappings_deduplicated == 510
    assert stats.mapping_distribution == {1: 73, 2: 149, 3: 37, 4: 7}
    assert stats.min_mappings_per_post == 1
    assert stats.max_mappings_per_post == 4

    # Verify that data/interim/ files exist and are valid
    interim_data = Path("data/interim/grouped_training_dataset.json")
    interim_stats = Path("data/interim/grouped_dataset_stats.json")
    assert interim_data.exists()
    assert interim_stats.exists()
