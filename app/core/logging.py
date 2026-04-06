"""
Structured logging setup for the Edunary AI Center.
"""

import logging
import sys
import time
from contextlib import contextmanager


def setup_logging(debug: bool = False) -> None:
    """Configure application-wide logging."""
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)


_timer_logger = logging.getLogger("timer")


@contextmanager
def timer(label: str):
    """Log execution time of a block. Usage: `with timer("label"): ...`"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    _timer_logger.info(f"⏱  {label}: {elapsed:.2f}s")
