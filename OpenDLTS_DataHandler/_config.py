import logging
import sys
from ._typing import Path
import cvxpy




LOGGER_CVXPY = logging.getLogger("__cvxpy__")
LOGGER_ODDH = logging.getLogger("__oddh__")

LOGGER_CVXPY.setLevel(logging.INFO)
LOGGER_ODDH.setLevel(logging.INFO)
# clear handler
LOGGER_CVXPY.handlers = []
LOGGER_ODDH.handlers = []
LOGGER_CVXPY_STREAM_HANDLER = logging.StreamHandler(sys.stderr)
LOGGER_ODDH_STREAM_HANDLER = logging.StreamHandler(sys.stderr)
LOGGER_CVXPY_FORMATTER = logging.Formatter('(CVXPY) %(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER_ODDH_FORMATTER = logging.Formatter('(ODDH) %(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER_CVXPY_STREAM_HANDLER.setFormatter(LOGGER_CVXPY_FORMATTER)
LOGGER_ODDH_STREAM_HANDLER.setFormatter(LOGGER_ODDH_FORMATTER)

LOGGER_CVXPY.addHandler(LOGGER_CVXPY_STREAM_HANDLER)
LOGGER_ODDH.addHandler(LOGGER_ODDH_STREAM_HANDLER)

LOGGER_CVXPY_FILE_HANDLER = None
LOGGER_ODDH_FILE_HANDLER = None

def INIT_LOG_FILE(filepath: str | Path | None = None, clear_exist_log: bool = False) -> None:
    """Initialize log file for ODDH and CVXPY loggers.
    Args:
        filepath (str | Path | None, optional): Log file path. Defaults to None.
    """
    global LOGGER_CVXPY_FILE_HANDLER, LOGGER_ODDH_FILE_HANDLER
    if filepath is not None:
        filepath = Path(filepath).resolve()
        if clear_exist_log:
            if filepath.exists():
                filepath.unlink()
        if LOGGER_CVXPY_FILE_HANDLER is not None:
            LOGGER_CVXPY.removeHandler(LOGGER_CVXPY_FILE_HANDLER)
        LOGGER_CVXPY_FILE_HANDLER = logging.FileHandler(filepath)
        if LOGGER_ODDH_FILE_HANDLER is not None:
            LOGGER_ODDH.removeHandler(LOGGER_ODDH_FILE_HANDLER)
        LOGGER_ODDH_FILE_HANDLER = logging.FileHandler(filepath)
        LOGGER_CVXPY_FILE_HANDLER.setFormatter(LOGGER_CVXPY_FORMATTER)
        LOGGER_ODDH_FILE_HANDLER.setFormatter(LOGGER_ODDH_FORMATTER)
        LOGGER_CVXPY.addHandler(LOGGER_CVXPY_FILE_HANDLER)
        LOGGER_ODDH.addHandler(LOGGER_ODDH_FILE_HANDLER)
        LOGGER_ODDH.info(f"ODDH log file initialized at: {filepath}")


__all__ = [
    "LOGGER_CVXPY",
    "LOGGER_ODDH",
    "INIT_LOG_FILE"
]