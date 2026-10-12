"""Data ingestion module for MavenMagnet reviews with data-quality reporting."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from mavenmagnet.config import AppConfig, load_config

logger = logging.getLogger("mavenmagnet.data_loader")


@dataclass
class DataQualityIssue:
    """Individual anomaly detected during dataset ingestion."""

    row_index: int
    issue_type: str
    description: str
    post_snippet: str = ""


@dataclass
class DatasetSummary:
    """Summary metrics and anomaly breakdown for an ingested dataset."""

    file_path: str
    sheet_name: Union[str, int]
    total_rows: int
    unique_posts_exact: int
    unique_posts_normalized: int
    missing_post_rows: int
    missing_label_rows: int
    duplicate_mapping_rows: int
    issues: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report for input workbooks."""

    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    training: Optional[DatasetSummary] = None
    process: Optional[DatasetSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert quality report to serializable dictionary."""
        return {
            "generated_at": self.generated_at,
            "training": asdict(self.training) if self.training else None,
            "process": asdict(self.process) if self.process else None,
        }

    def save(self, output_path: Path | str) -> Path:
        """Write machine-readable report as formatted JSON.

        Args:
            output_path: Destination path for JSON report.

        Returns:
            Path to saved JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved data quality report to %s", path)
        return path


class DataLoader:
    """Loads, validates, and reports on Training.xlsx and Process.xlsx workbooks."""

    REQUIRED_TRAINING_COLUMNS = ["Post", "Level 1", "Level 2"]
    REQUIRED_PROCESS_COLUMNS = ["Post"]

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        training_path: Optional[Path | str] = None,
        process_path: Optional[Path | str] = None,
        reports_dir: Optional[Path | str] = None,
        training_sheet: Union[str, int] = 0,
        process_sheet: Union[str, int] = 0,
    ) -> None:
        """Initialize data loader with optional configuration and file overrides.

        Args:
            config: Pipeline configuration instance. If omitted, loaded from default.
            training_path: Path to Training.xlsx. If omitted, resolved via config.
            process_path: Path to Process.xlsx. If omitted, resolved via config.
            reports_dir: Directory where data quality reports will be saved.
            training_sheet: Sheet name or index for Training workbook (defaults to first sheet).
            process_sheet: Sheet name or index for Process workbook (defaults to first sheet).
        """
        self.config = config or load_config()

        self.training_path = self._resolve_input_path(
            explicit_path=training_path,
            config_path=self.config.paths.training_file,
            fallback_name="Training.xlsx",
        )
        self.process_path = self._resolve_input_path(
            explicit_path=process_path,
            config_path=self.config.paths.process_file,
            fallback_name="Process.xlsx",
        )

        self.reports_dir = Path(reports_dir) if reports_dir else self.config.paths.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.training_sheet = training_sheet
        self.process_sheet = process_sheet

        self.quality_report = DataQualityReport()

    def _resolve_input_path(
        self,
        explicit_path: Optional[Path | str],
        config_path: Path,
        fallback_name: str,
    ) -> Path:
        """Resolve workbook path prioritizing explicit paths, then config, then data/ fallback."""
        if explicit_path is not None:
            return Path(explicit_path)

        if config_path.exists():
            return config_path

        fallback_path = self.config.paths.base_dir / "data" / fallback_name
        if fallback_path.exists():
            return fallback_path

        return config_path

    def load_training_data(
        self, sheet_name: Optional[Union[str, int]] = None
    ) -> pd.DataFrame:
        """Load, validate, and audit Training.xlsx workbook.

        Args:
            sheet_name: Optional sheet override; defaults to configured training_sheet.

        Returns:
            DataFrame containing loaded training records with original text preserved.

        Raises:
            FileNotFoundError: If the training file cannot be found.
            ValueError: If the file is unusable or missing required columns.
        """
        path = self.training_path
        if not path.exists():
            raise FileNotFoundError(f"Training workbook not found at: '{path.resolve()}'")

        sheet = self.training_sheet if sheet_name is None else sheet_name

        try:
            df = pd.read_excel(path, engine="openpyxl", sheet_name=sheet)
        except Exception as e:
            raise ValueError(
                f"Unusable or unreadable training workbook at '{path}' (sheet={sheet}): {e}"
            ) from e

        if df.empty or len(df.columns) == 0:
            raise ValueError(
                f"Training workbook at '{path}' (sheet={sheet}) contains no data rows or columns."
            )

        missing_cols = [c for c in self.REQUIRED_TRAINING_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Training workbook '{path}' is missing required columns: {missing_cols}. "
                f"Expected {self.REQUIRED_TRAINING_COLUMNS}, found {list(df.columns)}."
            )

        # Audit quality without altering original post text
        summary = self._audit_training_dataframe(df, file_path=str(path), sheet_name=sheet)
        self.quality_report.training = summary

        # Persist updated quality report
        report_file = self.reports_dir / "data_quality_report.json"
        self.quality_report.save(report_file)

        logger.info(
            "Training workbook loaded: %d rows, %d unique posts (%d normalized), "
            "%d missing posts, %d missing labels, %d duplicate mapping rows",
            summary.total_rows,
            summary.unique_posts_exact,
            summary.unique_posts_normalized,
            summary.missing_post_rows,
            summary.missing_label_rows,
            summary.duplicate_mapping_rows,
        )

        return df

    def load_process_data(
        self, sheet_name: Optional[Union[str, int]] = None
    ) -> pd.DataFrame:
        """Load, validate, and audit Process.xlsx workbook.

        Args:
            sheet_name: Optional sheet override; defaults to configured process_sheet.

        Returns:
            DataFrame containing reviews requiring classification with original text preserved.

        Raises:
            FileNotFoundError: If the process file cannot be found.
            ValueError: If the file is unusable or missing required columns.
        """
        path = self.process_path
        if not path.exists():
            raise FileNotFoundError(f"Process workbook not found at: '{path.resolve()}'")

        sheet = self.process_sheet if sheet_name is None else sheet_name

        try:
            df = pd.read_excel(path, engine="openpyxl", sheet_name=sheet)
        except Exception as e:
            raise ValueError(
                f"Unusable or unreadable process workbook at '{path}' (sheet={sheet}): {e}"
            ) from e

        if df.empty or len(df.columns) == 0:
            raise ValueError(
                f"Process workbook at '{path}' (sheet={sheet}) contains no data rows or columns."
            )

        missing_cols = [c for c in self.REQUIRED_PROCESS_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Process workbook '{path}' is missing required columns: {missing_cols}. "
                f"Expected {self.REQUIRED_PROCESS_COLUMNS}, found {list(df.columns)}."
            )

        # Audit quality without altering original post text
        summary = self._audit_process_dataframe(df, file_path=str(path), sheet_name=sheet)
        self.quality_report.process = summary

        # Persist updated quality report
        report_file = self.reports_dir / "data_quality_report.json"
        self.quality_report.save(report_file)

        logger.info(
            "Process workbook loaded: %d rows, %d unique posts (%d normalized), "
            "%d missing posts, %d duplicate post rows",
            summary.total_rows,
            summary.unique_posts_exact,
            summary.unique_posts_normalized,
            summary.missing_post_rows,
            summary.duplicate_mapping_rows,
        )

        return df

    def _audit_training_dataframe(
        self, df: pd.DataFrame, file_path: str, sheet_name: Union[str, int]
    ) -> DatasetSummary:
        """Audit training dataframe for missing fields and duplicate mappings."""
        total_rows = len(df)
        issues: List[Dict[str, Any]] = []

        # Detect missing post text
        is_missing_post = (
            df["Post"].isna()
            | (df["Post"].astype(str).str.strip() == "")
            | (df["Post"].astype(str).str.lower().isin(["nan", "null", "none"]))
        )
        missing_post_rows = int(is_missing_post.sum())
        for idx in df[is_missing_post].index:
            issues.append(
                asdict(
                    DataQualityIssue(
                        row_index=int(idx),
                        issue_type="missing_post",
                        description="Post text is null, empty, or whitespace-only",
                    )
                )
            )

        # Detect missing labels
        is_missing_l1 = (
            df["Level 1"].isna()
            | (df["Level 1"].astype(str).str.strip() == "")
            | (df["Level 1"].astype(str).str.lower().isin(["nan", "null", "none"]))
        )
        is_missing_l2 = (
            df["Level 2"].isna()
            | (df["Level 2"].astype(str).str.strip() == "")
            | (df["Level 2"].astype(str).str.lower().isin(["nan", "null", "none"]))
        )
        is_missing_label = is_missing_l1 | is_missing_l2
        missing_label_rows = int(is_missing_label.sum())
        for idx in df[is_missing_label].index:
            post_snippet = str(df.loc[idx, "Post"])[:50]
            issues.append(
                asdict(
                    DataQualityIssue(
                        row_index=int(idx),
                        issue_type="missing_label",
                        description=f"Missing label: Level 1='{df.loc[idx, 'Level 1']}', Level 2='{df.loc[idx, 'Level 2']}'",
                        post_snippet=post_snippet,
                    )
                )
            )

        # Detect duplicate mapping rows (same Post, Level 1, Level 2)
        duplicate_mask = df.duplicated(subset=["Post", "Level 1", "Level 2"], keep="first")
        duplicate_rows = int(duplicate_mask.sum())
        for idx in df[duplicate_mask].index:
            post_snippet = str(df.loc[idx, "Post"])[:50]
            issues.append(
                asdict(
                    DataQualityIssue(
                        row_index=int(idx),
                        issue_type="duplicate_mapping",
                        description="Duplicate (Post, Level 1, Level 2) row",
                        post_snippet=post_snippet,
                    )
                )
            )

        valid_posts = df.loc[~is_missing_post, "Post"]
        unique_posts_exact = int(valid_posts.nunique())
        unique_posts_norm = int(valid_posts.astype(str).str.strip().nunique())

        return DatasetSummary(
            file_path=file_path,
            sheet_name=sheet_name,
            total_rows=total_rows,
            unique_posts_exact=unique_posts_exact,
            unique_posts_normalized=unique_posts_norm,
            missing_post_rows=missing_post_rows,
            missing_label_rows=missing_label_rows,
            duplicate_mapping_rows=duplicate_rows,
            issues=issues,
        )

    def _audit_process_dataframe(
        self, df: pd.DataFrame, file_path: str, sheet_name: Union[str, int]
    ) -> DatasetSummary:
        """Audit process dataframe for missing posts and duplicate queries."""
        total_rows = len(df)
        issues: List[Dict[str, Any]] = []

        is_missing_post = (
            df["Post"].isna()
            | (df["Post"].astype(str).str.strip() == "")
            | (df["Post"].astype(str).str.lower().isin(["nan", "null", "none"]))
        )
        missing_post_rows = int(is_missing_post.sum())
        for idx in df[is_missing_post].index:
            issues.append(
                asdict(
                    DataQualityIssue(
                        row_index=int(idx),
                        issue_type="missing_post",
                        description="Post text is null, empty, or whitespace-only",
                    )
                )
            )

        # Duplicate post check in Process.xlsx
        duplicate_mask = df.duplicated(subset=["Post"], keep="first")
        duplicate_rows = int(duplicate_mask.sum())
        for idx in df[duplicate_mask].index:
            post_snippet = str(df.loc[idx, "Post"])[:50]
            issues.append(
                asdict(
                    DataQualityIssue(
                        row_index=int(idx),
                        issue_type="duplicate_post",
                        description="Duplicate query post in Process.xlsx",
                        post_snippet=post_snippet,
                    )
                )
            )

        valid_posts = df.loc[~is_missing_post, "Post"]
        unique_posts_exact = int(valid_posts.nunique())
        unique_posts_norm = int(valid_posts.astype(str).str.strip().nunique())

        return DatasetSummary(
            file_path=file_path,
            sheet_name=sheet_name,
            total_rows=total_rows,
            unique_posts_exact=unique_posts_exact,
            unique_posts_normalized=unique_posts_norm,
            missing_post_rows=missing_post_rows,
            missing_label_rows=0,
            duplicate_mapping_rows=duplicate_rows,
            issues=issues,
        )
