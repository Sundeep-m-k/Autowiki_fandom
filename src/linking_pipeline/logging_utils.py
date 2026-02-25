"""Logging setup for the linking pipeline. Same pattern as Tasks 1 & 2."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    log_dir: str | Path,
    script_name: str = "linking",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger("linking")
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
    log_file  = log_path / f"{timestamp}_{script_name}.log"

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
    return logging.getLogger("linking")
