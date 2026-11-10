"""Command-line entry point: `uv run microtopos [--simulator]`."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .logging import configure_logging, create_diagnostic_bundle, log_environment
from .validation import parse_crop, parse_fusion_mode

logger = logging.getLogger("microtopos.app")

# harmless Qt warnings that otherwise flood the console (window-manager size hints, XCB/Wayland notes)
_QT_NOISE = ("propagateSizeHints", "QXcbConnection", "Wayland does not support", "requestActivate")


def _quiet_qt(verbose: bool):
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler

        def handler(kind, _ctx, msg):
            if not verbose and (kind == QtMsgType.QtDebugMsg or any(n in msg for n in _QT_NOISE)):
                return
            sys.stderr.write(msg + "\n")

        qInstallMessageHandler(handler)
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser(description="MicroTopos: live stitching, shape from focus and surface metrology")
    ap.add_argument("--simulator", action="store_true", help="ignore cameras, use the synthetic specimen")
    ap.add_argument("--replay", metavar="DIR", help="play back a recording made with microtopos-record")
    ap.add_argument(
        "--selftest",
        nargs=argparse.REMAINDER,
        metavar="ARGS",
        help="run the ground-truth self-test (e.g. --selftest --quick --gui) and exit",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="print verbose debug logging to stderr")
    ap.add_argument("--json-logs", action="store_true", help="format log lines as JSON for automated parsing")
    ap.add_argument("--crop", type=parse_crop, default=None, help="active processing crop region as x,y,w,h (integers)")
    ap.add_argument(
        "--fusion",
        type=parse_fusion_mode,
        default="sharpest",
        help="mosaic fusion mode for overlapping/revisited tiles ('sharpest', 'weighted', 'average', 'superres')",
    )
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="create a self-contained diagnostic zip bundle with logs, config and environment details, then exit",
    )
    ap.add_argument("--version", action="version", version=f"MicroTopos {__version__}")
    args = ap.parse_args()

    # Configure structured diagnostic logging
    configure_logging(verbose=args.verbose, json_format=args.json_logs)
    log_environment(logger)

    if args.diagnostics:
        bundle_path = create_diagnostic_bundle()
        print(f"Diagnostic bundle exported to: {bundle_path.resolve()}")
        sys.exit(0)

    if args.selftest is not None:
        try:
            from .selftest import main as selftest_main
        except ImportError as exc:
            logger.exception("Failed to import selftest module: %s", exc)
            sys.exit(1)
        selftest_main(args.selftest)
        sys.exit(0)

    try:
        import pyqtgraph as pg
        from PySide6.QtWidgets import QApplication

        pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
        _quiet_qt(args.verbose)
        app = QApplication(sys.argv)
        app.setApplicationName("MicroTopos")
        app.setOrganizationName("MicroTopos")
        from .gui.main_window import MainWindow

        w = MainWindow(
            simulator_only=args.simulator or bool(args.replay),
            replay=args.replay,
            verbose=args.verbose,
            fusion=args.fusion,
            crop=args.crop,
        )
        w.show()
        sys.exit(app.exec())
    except Exception as exc:
        logger.exception("Fatal error at application boundary: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
