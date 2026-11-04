"""Command-line entry point: `uv run microtopos [--simulator]`."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main():
    ap = argparse.ArgumentParser(description="MicroTopos: live stitching, shape from focus and surface metrology")
    ap.add_argument("--simulator", action="store_true", help="ignore cameras, use the synthetic specimen")
    ap.add_argument("--replay", metavar="DIR", help="play back a recording made with microtopos-record")
    ap.add_argument("--selftest", nargs=argparse.REMAINDER, metavar="ARGS",
                    help="run the ground-truth self-test (e.g. --selftest --quick --gui) and exit")
    ap.add_argument("--version", action="version", version=f"MicroTopos {__version__}")
    args = ap.parse_args()
    if args.selftest is not None:
        from .selftest import main as selftest_main

        selftest_main(args.selftest)
    import pyqtgraph as pg
    from PySide6.QtWidgets import QApplication

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    app = QApplication(sys.argv)
    app.setApplicationName("MicroTopos")
    app.setOrganizationName("MicroTopos")
    from .gui.main_window import MainWindow

    w = MainWindow(simulator_only=args.simulator or bool(args.replay), replay=args.replay)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
