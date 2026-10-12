"""Typed configuration management for MavenMagnet pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class PathConfig:
    """Filesystem path configurations using pathlib."""

    base_dir: Path
    input_dir: Path
    training_file: Path
    process_file: Path
    interim_dir: Path
    output_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class BaselineConfig:
    """TF-IDF and Logistic Regression baseline configuration."""

    model_type: str = "tfidf_logistic_regression"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 5000
    c_param: float = 1.0
    class_weight: str = "balanced"
    decision_threshold: float = 0.35


@dataclass(frozen=True)
class LLMConfig:
    """Local Ollama LLM configuration."""

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"
    fallback_model: str = "qwen3.5:0.8b"
    temperature: float = 0.0
    top_p: float = 0.9
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation protocol configuration."""

    n_splits: int = 5
    stratify_by_post: bool = True
    metrics: list[str] = field(
        default_factory=lambda: [
            "exact_match_ratio",
            "micro_f1",
            "macro_f1",
            "hamming_loss",
            "precision",
            "recall",
        ]
    )


@dataclass(frozen=True)
class AppConfig:
    """Top-level pipeline configuration."""

    project_name: str
    random_seed: int
    paths: PathConfig
    baseline: BaselineConfig
    llm: LLMConfig
    evaluation: EvaluationConfig


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load configuration from YAML file and environment defaults.

    Args:
        config_path: Optional path to config YAML file. Defaults to config/config.yaml.

    Returns:
        Structured AppConfig instance.
    """
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        resolved_config_path = base_dir / "config" / "config.yaml"
    else:
        resolved_config_path = Path(config_path).resolve()
        base_dir = resolved_config_path.parent.parent

    if resolved_config_path.exists():
        with open(resolved_config_path, "r", encoding="utf-8") as f:
            raw_cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    else:
        raw_cfg = {}

    proj = raw_cfg.get("project", {})
    paths_raw = raw_cfg.get("paths", {})
    bl_raw = raw_cfg.get("baseline", {})
    llm_raw = raw_cfg.get("llm", {})
    eval_raw = raw_cfg.get("evaluation", {})

    paths = PathConfig(
        base_dir=base_dir,
        input_dir=base_dir / paths_raw.get("input_dir", "data/input"),
        training_file=base_dir / paths_raw.get("training_file", "data/input/Training.xlsx"),
        process_file=base_dir / paths_raw.get("process_file", "data/input/Process.xlsx"),
        interim_dir=base_dir / paths_raw.get("interim_dir", "data/interim"),
        output_dir=base_dir / paths_raw.get("output_dir", "data/output"),
        artifacts_dir=base_dir / paths_raw.get("artifacts_dir", "artifacts"),
        reports_dir=base_dir / paths_raw.get("reports_dir", "reports"),
        logs_dir=base_dir / paths_raw.get("logs_dir", "logs"),
    )

    baseline = BaselineConfig(
        model_type=bl_raw.get("model_type", "tfidf_logistic_regression"),
        ngram_range=tuple(bl_raw.get("ngram_range", [1, 2])),  # type: ignore
        max_features=bl_raw.get("max_features", 5000),
        c_param=bl_raw.get("c_param", 1.0),
        class_weight=bl_raw.get("class_weight", "balanced"),
        decision_threshold=bl_raw.get("decision_threshold", 0.35),
    )

    llm = LLMConfig(
        provider=llm_raw.get("provider", "ollama"),
        base_url=llm_raw.get("base_url", "http://localhost:11434"),
        model=llm_raw.get("model", "qwen3.5:9b"),
        fallback_model=llm_raw.get("fallback_model", "qwen3.5:0.8b"),
        temperature=llm_raw.get("temperature", 0.0),
        top_p=llm_raw.get("top_p", 0.9),
        timeout_seconds=llm_raw.get("timeout_seconds", 120),
        max_retries=llm_raw.get("max_retries", 3),
        retry_backoff_seconds=llm_raw.get("retry_backoff_seconds", 2.0),
    )

    evaluation = EvaluationConfig(
        n_splits=eval_raw.get("n_splits", 5),
        stratify_by_post=eval_raw.get("stratify_by_post", True),
        metrics=eval_raw.get(
            "metrics",
            [
                "exact_match_ratio",
                "micro_f1",
                "macro_f1",
                "hamming_loss",
                "precision",
                "recall",
            ],
        ),
    )

    return AppConfig(
        project_name=proj.get("name", "mavenmagnet-nlp-pipeline"),
        random_seed=proj.get("random_seed", 42),
        paths=paths,
        baseline=baseline,
        llm=llm,
        evaluation=evaluation,
    )
