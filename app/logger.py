import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
BACKUP_COUNT = 5

INFO_LOG = LOGS_DIR / "info.log"
WARNING_LOG = LOGS_DIR / "warning.log"
ERROR_LOG = LOGS_DIR / "error.log"


class _LevelFilter(logging.Filter):
    """Allow only records at exactly the given level."""
    def __init__(self, level: int):
        super().__init__()
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self._level


def _rotating(path: Path, level: int, exact: bool = False) -> RotatingFileHandler:
    handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    if exact:
        handler.addFilter(_LevelFilter(level))
    return handler


def setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(logging.DEBUG)

    # Console — INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    root.addHandler(console)

    # info.log — INFO only
    root.addHandler(_rotating(INFO_LOG, logging.INFO, exact=True))

    # warning.log — WARNING only
    root.addHandler(_rotating(WARNING_LOG, logging.WARNING, exact=True))

    # error.log — ERROR and above (ERROR + CRITICAL)
    root.addHandler(_rotating(ERROR_LOG, logging.ERROR))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
