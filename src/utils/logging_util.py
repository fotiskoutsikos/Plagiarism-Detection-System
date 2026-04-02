import os
import sys
import logging
from pathlib import Path


class TeeStream:
    """Duplicate output to stdout/stderr and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for s in self.streams:
            s.write(message)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_logging(script_path=None, log_dir="logs", level=logging.INFO):
    """Initialize logging and also duplicate stdout/stderr to logs/{script_name}.txt."""
    if script_path is None:
        script_name = Path(sys.argv[0]).stem if sys.argv else "script"
    else:
        script_name = Path(script_path).stem

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{script_name}.txt")

    # Open file with append to preserve history
    log_fh = open(log_file, "a", encoding="utf-8")

    # Duplicate stdout/stderr
    sys.stdout = TeeStream(sys.stdout, log_fh)
    sys.stderr = TeeStream(sys.stderr, log_fh)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid duplicate handlers in case setup_logging called multiple times
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        console_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"Logging initialized for {script_name}, file: {log_file}")
    return logger
