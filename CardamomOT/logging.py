"""
Logging configuration and utilities for CARDAMOM.

Provides standardized logging across the entire package with support
for both console and file output, configurable verbosity levels,
and consistent formatting.

Example:
    >>> from CardamomOT.logging import get_logger
    >>> logger = get_logger("my_module")
    >>> logger.info("Processing started...")
    >>> logger.warning("Unusual condition detected")
    >>> logger.error("Failed to process", exc_info=True)
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for terminal output
_COLORS = {
    'DEBUG': '\033[36m',      # Cyan
    'INFO': '\033[32m',       # Green
    'WARNING': '\033[33m',    # Yellow
    'ERROR': '\033[31m',      # Red
    'CRITICAL': '\033[35m',   # Magenta
    'RESET': '\033[0m'        # Reset
}


class _ColoredFormatter(logging.Formatter):
    """Custom formatter with optional ANSI colors for terminal output."""

    def __init__(self, fmt: str, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            levelname = record.levelname
            if levelname in _COLORS:
                record.levelname = (
                    f"{_COLORS[levelname]}{levelname}{_COLORS['RESET']}"
                )
        return super().format(record)


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    verbose: bool = False,
) -> None:
    """
    Configure the root logger for CARDAMOM.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
               Default is logging.INFO.
        log_file: Optional path to write logs to file. If provided,
                  logs will be written to both console and file.
        verbose: If True, sets level to DEBUG. Overrides `level` parameter.

    Example:
        >>> from CardamomOT.logging import configure_logging
        >>> import logging
        >>> configure_logging(level=logging.DEBUG, log_file=Path("output.log"))
    """
    if verbose:
        level = logging.DEBUG

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    console_handler.setFormatter(_ColoredFormatter(console_fmt, use_color=True))
    root_logger.addHandler(console_handler)

    # File handler if requested
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_fmt = (
            "%(asctime)s | %(name)s | %(levelname)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(logging.Formatter(file_fmt))
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.

    Example:
        >>> from CardamomOT.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


# Default configuration on import
configure_logging(level=logging.INFO)
