"""
Logging utilities for YOLOWorld-mini"""

import logging
import sys
from pathlib import Path


def get_logger(name: str = "yoloworld", log_level=logging.INFO) -> logging.Logger:
    """
    Returns a logger with a standard timestamped format.
    Ensures we don't create duplicate handlers on every call.
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Stream handler (stdout)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # Optional: file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        fh = logging.FileHandler(log_dir / "training.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Avoid propagating to root logger
    logger.propagate = False

    return logger