"""MavenMagnet NLP Classification Pipeline.

A production-quality local pipeline for multi-label hierarchical
classification of product reviews using local LLMs and linear baselines.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__version__ = "0.1.0"

_LOGGER_NAME = "mavenmagnet"


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        level: Minimum logging level (e.g. logging.INFO or 'INFO').
        log_file: Optional path to write log output to disk using pathlib.
        log_format: Formatter specification string.

    Returns:
        The configured root logger for mavenmagnet.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logging is invoked repeatedly
    if not logger.handlers:
        formatter = logging.Formatter(log_format)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = setup_logging()
