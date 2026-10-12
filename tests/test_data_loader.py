"""Unit and integration tests for data_loader module using temporary workbooks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mavenmagnet.data_loader import DataLoader, DataQualityReport


def test_load_training_valid_input(tmp_path: Path) -> None:
    """Verify loading a valid training workbook preserves text and generates quality report."""
    excel_path = tmp_path / "valid_training.xlsx"
    reports_dir = tmp_path / "reports"

    df_sample = pd.DataFrame(
        {
            "Post": [
                "  Original post text with leading space  ",
                "Second review post text",
                "  Original post text with leading space  ",
            ],
            "Level 1": ["Pricing", "Finish", "Feel"],
            "Level 2": ["Value for money", "Light Weight", "Fresh"],
        }
    )
    df_sample.to_excel(excel_path, index=False, engine="openpyxl")

    loader = DataLoader(
        training_path=excel_path,
        process_path=excel_path,
        reports_dir=reports_dir,
    )
    df_loaded = loader.load_training_data()

    assert len(df_loaded) == 3
    # Requirement: preserve original post text
    assert df_loaded.loc[0, "Post"] == "  Original post text with leading space  "
    assert df_loaded.loc[1, "Post"] == "Second review post text"

    report_path = reports_dir / "data_quality_report.json"
    assert report_path.exists()

    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    assert report_data["training"]["total_rows"] == 3
    assert report_data["training"]["unique_posts_exact"] == 2
    assert report_data["training"]["unique_posts_normalized"] == 2
    assert report_data["training"]["missing_post_rows"] == 0
    assert report_data["training"]["missing_label_rows"] == 0
    assert report_data["training"]["duplicate_mapping_rows"] == 0


def test_load_process_valid_input(tmp_path: Path) -> None:
    """Verify loading a valid process workbook."""
    excel_path = tmp_path / "valid_process.xlsx"
    reports_dir = tmp_path / "reports"

    df_sample = pd.DataFrame(
        {
            "Post": ["Review for prediction 1", "Review for prediction 2"],
            "Level 1": [None, None],
            "Level 2": [None, None],
        }
    )
    df_sample.to_excel(excel_path, index=False, engine="openpyxl")

    loader = DataLoader(
        training_path=excel_path,
        process_path=excel_path,
        reports_dir=reports_dir,
    )
    df_loaded = loader.load_process_data()

    assert len(df_loaded) == 2
    assert df_loaded.loc[0, "Post"] == "Review for prediction 1"
    assert loader.quality_report.process is not None
    assert loader.quality_report.process.unique_posts_exact == 2


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised when workbook does not exist."""
    non_existent = tmp_path / "does_not_exist.xlsx"
    loader = DataLoader(training_path=non_existent, process_path=non_existent)

    with pytest.raises(FileNotFoundError, match="Training workbook not found"):
        loader.load_training_data()

    with pytest.raises(FileNotFoundError, match="Process workbook not found"):
        loader.load_process_data()


def test_missing_required_column_raises_valueerror(tmp_path: Path) -> None:
    """Verify ValueError is raised when required columns are missing."""
    invalid_train = tmp_path / "missing_col_train.xlsx"
    # Missing 'Level 2' column
    df = pd.DataFrame({"Post": ["A review"], "Level 1": ["Pricing"]})
    df.to_excel(invalid_train, index=False, engine="openpyxl")

    loader = DataLoader(training_path=invalid_train, process_path=invalid_train)
    with pytest.raises(ValueError, match="missing required columns"):
        loader.load_training_data()

    invalid_proc = tmp_path / "missing_col_proc.xlsx"
    # Missing 'Post' column
    df_proc = pd.DataFrame({"Text": ["A review"]})
    df_proc.to_excel(invalid_proc, index=False, engine="openpyxl")

    loader_proc = DataLoader(training_path=invalid_train, process_path=invalid_proc)
    with pytest.raises(ValueError, match="missing required columns"):
        loader_proc.load_process_data()


def test_missing_post_text_detected(tmp_path: Path) -> None:
    """Verify missing/blank post text is detected without silently discarding rows."""
    excel_path = tmp_path / "missing_post.xlsx"
    df = pd.DataFrame(
        {
            "Post": ["Valid post", None, "   ", "nan"],
            "Level 1": ["Pricing", "Feel", "Finish", "Tone"],
            "Level 2": ["Discounts", "Fresh", "Dewy", "Glow"],
        }
    )
    df.to_excel(excel_path, index=False, engine="openpyxl")

    loader = DataLoader(training_path=excel_path, process_path=excel_path, reports_dir=tmp_path)
    df_loaded = loader.load_training_data()

    # Records must not be silently discarded
    assert len(df_loaded) == 4
    summary = loader.quality_report.training
    assert summary is not None
    assert summary.missing_post_rows == 3
    assert summary.unique_posts_exact == 1


def test_missing_training_label_detected(tmp_path: Path) -> None:
    """Verify missing training labels are recorded in data quality report."""
    excel_path = tmp_path / "missing_labels.xlsx"
    df = pd.DataFrame(
        {
            "Post": ["Post 1", "Post 2", "Post 3"],
            "Level 1": ["Pricing", None, "Feel"],
            "Level 2": ["Discounts", "Dewy", ""],
        }
    )
    df.to_excel(excel_path, index=False, engine="openpyxl")

    loader = DataLoader(training_path=excel_path, process_path=excel_path, reports_dir=tmp_path)
    df_loaded = loader.load_training_data()

    assert len(df_loaded) == 3
    summary = loader.quality_report.training
    assert summary is not None
    assert summary.missing_label_rows == 2


def test_duplicate_mapping_row_detected(tmp_path: Path) -> None:
    """Verify identical duplicate mapping rows are detected."""
    excel_path = tmp_path / "duplicates.xlsx"
    df = pd.DataFrame(
        {
            "Post": ["Post 1", "Post 1", "Post 1"],
            "Level 1": ["Pricing", "Pricing", "Finish"],
            "Level 2": ["Discounts", "Discounts", "Clean"],
        }
    )
    df.to_excel(excel_path, index=False, engine="openpyxl")

    loader = DataLoader(training_path=excel_path, process_path=excel_path, reports_dir=tmp_path)
    df_loaded = loader.load_training_data()

    assert len(df_loaded) == 3
    summary = loader.quality_report.training
    assert summary is not None
    assert summary.duplicate_mapping_rows == 1  # 1 duplicate occurrence of (Post 1, Pricing, Discounts)


def test_unusable_workbook_raises_valueerror(tmp_path: Path) -> None:
    """Verify unusable workbooks raise informative ValueError."""
    corrupted_file = tmp_path / "corrupted.xlsx"
    corrupted_file.write_text("This is not a zip/excel file")

    loader = DataLoader(training_path=corrupted_file, process_path=corrupted_file)
    with pytest.raises(ValueError, match="Unusable or unreadable"):
        loader.load_training_data()


def test_configurable_sheet_name(tmp_path: Path) -> None:
    """Verify sheet name configurability."""
    excel_path = tmp_path / "multi_sheet.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame({"Post": ["Sheet 1 post"], "Level 1": ["A"], "Level 2": ["B"]}).to_excel(
            writer, sheet_name="FirstSheet", index=False
        )
        pd.DataFrame({"Post": ["Target post"], "Level 1": ["Tone"], "Level 2": ["Glow"]}).to_excel(
            writer, sheet_name="TargetSheet", index=False
        )

    loader = DataLoader(
        training_path=excel_path,
        process_path=excel_path,
        training_sheet="TargetSheet",
        reports_dir=tmp_path,
    )
    df_loaded = loader.load_training_data()
    assert len(df_loaded) == 1
    assert df_loaded.loc[0, "Post"] == "Target post"


def test_real_workbooks_ingestion_and_quality_report() -> None:
    """Verify ingestion of the actual production workbooks."""
    loader = DataLoader()
    df_train = loader.load_training_data()
    df_proc = loader.load_process_data()

    assert len(df_train) == 512
    assert len(df_proc) == 190

    report_path = Path("reports/data_quality_report.json")
    assert report_path.exists()

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["training"]["total_rows"] == 512
    assert data["training"]["unique_posts_normalized"] == 266
    assert data["training"]["missing_post_rows"] == 0
    assert data["training"]["missing_label_rows"] == 0
    assert data["process"]["total_rows"] == 190
