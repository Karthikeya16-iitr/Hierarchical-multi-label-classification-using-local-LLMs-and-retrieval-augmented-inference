"""Top-level pipeline orchestrator for MavenMagnet review classification."""

from __future__ import annotations

import logging
from typing import Optional

from mavenmagnet.config import AppConfig, load_config
from mavenmagnet.data_loader import DataLoader
from mavenmagnet.dataset_builder import DatasetBuilder
from mavenmagnet.evaluator import Evaluator
from mavenmagnet.exporter import ResultExporter

logger = logging.getLogger("mavenmagnet.pipeline")


class PipelineRunner:
    """Orchestrates end-to-end baseline evaluation and local LLM inference."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize pipeline with configuration."""
        self.config = config or load_config()
        self.data_loader = DataLoader(
            training_path=self.config.paths.training_file,
            process_path=self.config.paths.process_file,
        )
        self.dataset_builder = DatasetBuilder()
        self.evaluator = Evaluator()
        self.exporter = ResultExporter(output_dir=self.config.paths.output_dir)

    def run_baseline_evaluation(self) -> None:
        """Execute leak-free grouped cross-validation for the TF-IDF baseline."""
        logger.info("Starting baseline cross-validation pipeline...")
        # Pipeline orchestration step to be implemented in Component 2
        df_train = self.data_loader.load_training_data()
        records = self.dataset_builder.build_records_from_df(df_train)
        logger.info("Loaded and aggregated %d post records for baseline evaluation.", len(records))

    def run_process_inference(self) -> None:
        """Execute classification on unlabelled Process.xlsx reviews and export output."""
        logger.info("Starting Process.xlsx inference pipeline...")
        # Pipeline orchestration step to be implemented in Component 5
        df_process = self.data_loader.load_process_data()
        logger.info("Loaded %d unlabelled posts from Process.xlsx.", len(df_process))
