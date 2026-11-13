"""Structured, correlated, tiered diagnostic logging and diagnostic bundle export for MicroTopos."""

from __future__ import annotations

import collections
from contextlib import contextmanager
import json
import logging
import platform
import sys
import time
from pathlib import Path
import uuid
import zipfile

# Canonical human log format: timestamp, level, logger, function:line, message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"

# Ring buffer for retaining recent logs in-memory for diagnostic bundles
_RING_BUFFER: collections.deque[str] = collections.deque(maxlen=2000)

_CURRENT_RUN_ID: str = uuid.uuid4().hex[:8]


def get_run_id() -> str:
    """Return the current operation/session run ID."""
    global _CURRENT_RUN_ID
    return _CURRENT_RUN_ID


def new_run_id() -> str:
    """Generate and return a new run ID for a distinct operation."""
    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = uuid.uuid4().hex[:8]
    return _CURRENT_RUN_ID


class RingBufferHandler(logging.Handler):
    """Stores formatted log records in a fixed-size ring buffer for diagnostic export."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            _RING_BUFFER.append(msg)
        except Exception:
            self.handleError(record)


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects for CI and automated analysis."""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "func": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", get_run_id()),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_logging(
    verbose: bool = False,
    json_format: bool = False,
    stream=None,
    log_file: Path | str | None = None,
) -> None:
    """Configure structured logging for the application.

    - verbose=False: INFO level for general events, WARNING/ERROR for issues.
    - verbose=True: DEBUG level for fine-grained algorithm internals and telemetry.
    - json_format=True: emit JSON lines for automated parsing in CI/diagnostics.
    """
    root_logger = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    formatter: logging.Formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(LOG_FORMAT)

    # Configure OpenCV internal C++ logger level and noisy third-party INFO logs
    try:
        import cv2.utils.logging as cv_log
        cv_log.setLogLevel(cv_log.LOG_LEVEL_DEBUG if verbose else cv_log.LOG_LEVEL_ERROR)
    except (ImportError, AttributeError):
        pass
    logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.DEBUG if verbose else logging.WARNING)

    # Console / stream handler
    stream_out = stream if stream is not None else sys.stderr
    stream_handler = logging.StreamHandler(stream_out)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Always attach the in-memory ring buffer handler formatted in standard text
    ring_handler = RingBufferHandler()
    ring_handler.setLevel(logging.DEBUG)
    ring_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(ring_handler)

    # Optional file handler
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


@contextmanager
def timed_operation(name: str, logger: logging.Logger | None = None):
    """Context manager that logs the start and elapsed execution time of an operation."""
    log = logger or logging.getLogger("microtopos.timing")
    run_id = get_run_id()
    log.debug("[run=%s] Starting operation '%s'", run_id, name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.debug("[run=%s] Operation '%s' completed in %.4fs", run_id, name, elapsed)


def log_environment(logger: logging.Logger | None = None) -> dict[str, str]:
    """Log and return system environment details (Python, NumPy, OpenCV, OS)."""
    log = logger or logging.getLogger("microtopos.env")

    versions: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }

    try:
        import numpy as np
        versions["numpy"] = np.__version__
    except ImportError:
        versions["numpy"] = "not installed"

    try:
        import cv2
        versions["opencv"] = cv2.__version__
    except ImportError:
        versions["opencv"] = "not installed"

    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        versions["scipy"] = "not installed"

    try:
        from PySide6 import __version__ as pyside_ver
        versions["pyside6"] = pyside_ver
    except ImportError:
        versions["pyside6"] = "not installed"

    try:
        from . import __version__ as mt_ver
        versions["microtopos"] = mt_ver
    except Exception:
        versions["microtopos"] = "unknown"

    log.info(
        "Environment: microtopos=%s python=%s numpy=%s opencv=%s scipy=%s pyside6=%s",
        versions.get("microtopos"),
        sys.version.split()[0],
        versions.get("numpy"),
        versions.get("opencv"),
        versions.get("scipy"),
        versions.get("pyside6"),
    )
    return versions


def get_recent_logs() -> list[str]:
    """Return all logs currently retained in the ring buffer."""
    return list(_RING_BUFFER)


def create_diagnostic_bundle(out_path: Path | str | None = None) -> Path:
    """Export a self-contained diagnostic zip bundle for reproducing and diagnosing issues.

    Contains:
    - versions.json: Runtime, OS, and library versions
    - commandline.txt: Command line arguments
    - logs.txt: Recent in-memory log buffer
    - calibration.json: Objective calibration metadata if accessible
    - config.json: Active runtime configuration details
    """
    logger = logging.getLogger("microtopos.diagnostics")
    logger.info("Creating diagnostic bundle...")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if out_path is None:
        target_zip = Path(f"microtopos_diagnostics_{timestamp}.zip")
    else:
        target_zip = Path(out_path)

    target_zip.parent.mkdir(parents=True, exist_ok=True)

    versions = log_environment(logger)
    recent_logs = get_recent_logs()

    # Locate calibration file if it exists
    calibration_data = {}
    config_dir = Path.home() / ".config" / "microtopos"
    cal_file = config_dir / "calibration.json"
    if cal_file.exists():
        try:
            calibration_data = json.loads(cal_file.read_text(encoding="utf-8"))
        except Exception as exc:
            calibration_data = {"error": f"Failed to read calibration: {exc}"}

    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("versions.json", json.dumps(versions, indent=2))
        zf.writestr("commandline.txt", " ".join(sys.argv) + "\n")
        zf.writestr("logs.txt", "\n".join(recent_logs) + "\n")
        zf.writestr("calibration.json", json.dumps(calibration_data, indent=2))
        zf.writestr(
            "config.json",
            json.dumps(
                {
                    "run_id": get_run_id(),
                    "timestamp": timestamp,
                    "platform": sys.platform,
                    "argv": sys.argv,
                },
                indent=2,
            ),
        )

    logger.info("Diagnostic bundle created at: %s (%d bytes)", target_zip, target_zip.stat().st_size)
    return target_zip
