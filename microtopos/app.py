"""Command-line entry point: `uv run microtopos [--simulator]`."""

from __future__ import annotations

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="MicroTopos: live stitching, shape from focus and surface metrology")
    ap.add_argument("--simulator", action="store_true", help="ignore cameras, use the synthetic specimen")
    ap.add_argument("--replay", metavar="DIR", help="play back a recording made with microtopos-record")
    args = ap.parse_args()
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
