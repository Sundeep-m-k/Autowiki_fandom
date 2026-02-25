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
    """Setup logger with file and console handlers. log_dir: data/logs/<domain>/span_id/

    Safe to call multiple times (e.g. when switching domains in a sweep): all
    existing handlers are closed and removed before new ones are added, so
    there is never more than one file handler or one console handler active.
    """
    logger = logging.getLogger("span_id")
    for h in list(logger.handlers):
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

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file.resolve())
    return logger


def get_logger() -> logging.Logger:
    """Get the span_id logger (setup must have been done first)."""
    return logging.getLogger("span_id")
