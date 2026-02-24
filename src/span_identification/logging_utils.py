"""Logging setup for span identification pipeline."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_span_id_logger(
    log_dir: str | Path,
    script_name: str = "span_id",
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup logger with file and console handlers. log_dir: data/logs/<domain>/span_id/"""
    logger = logging.getLogger("span_id")
    # Remove existing file handlers when switching log dir (e.g. new domain)
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{timestamp}_{script_name}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler (add only if none)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info("Log file: %s", log_file.resolve())
    return logger


def get_logger() -> logging.Logger:
    """Get the span_id logger (setup must have been done first)."""
    return logging.getLogger("span_id")
