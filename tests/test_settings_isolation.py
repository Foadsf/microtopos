"""The window's saved settings never come from, or go to, the user's own store while the tests run.

On Windows the native QSettings store is the registry and QSettings.setPath does not redirect it: every GUI test
used to read the user's saved objective and write its own back (HKCU\\Software\\MicroTopos\\MicroTopos), so results
depended on what the last run, or the user, had left there.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication


@pytest.fixture
def app(tmp_path, monkeypatch):
    QStandardPaths.setTestModeEnabled(True)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return QApplication.instance() or QApplication([])


def _pump(app, n=20):
    for _ in range(n):
        app.processEvents()


def test_a_window_saves_its_settings_in_the_folder_of_the_test_not_in_the_users_store(app):
    from microtopos.gui.main_window import MainWindow, app_settings

    folder = Path(os.environ["MICROTOPOS_SETTINGS_DIR"])  # set per test by conftest.py
    assert app_settings().allKeys() == []  # empty before the window exists: nothing from an earlier test or the user
    w = MainWindow(simulator_only=True)
    w.show()
    _pump(app)
    w.timer.stop()
    assert w.settings.format() == QSettings.Format.IniFormat
    assert Path(w.settings.fileName()).parent == folder  # the registry reports "\HKEY_CURRENT_USER\..." here
    w.cmb_obj.setCurrentText("40x / 0.85")
    _pump(app)
    w.close()
    _pump(app)
    saved = QSettings(str(folder / "MicroTopos.ini"), QSettings.Format.IniFormat)
    assert saved.value("objective") == "40x / 0.85"  # what the window saved went to that folder


def test_the_next_window_of_another_test_starts_from_empty_settings(app):
    from microtopos.gui.main_window import MainWindow, app_settings

    assert app_settings().allKeys() == []  # the test above saved an objective; this test must not see it
    w = MainWindow(simulator_only=True)
    w.show()
    _pump(app)
    w.timer.stop()
    assert w.cmb_obj.currentText() != "40x / 0.85"
    w.close()
    _pump(app)


def test_without_the_override_the_settings_are_the_platforms_native_store(monkeypatch):
    from microtopos.gui.main_window import app_settings

    monkeypatch.delenv("MICROTOPOS_SETTINGS_DIR")
    s = app_settings()  # constructed only: nothing is read or written
    assert s.format() == QSettings.Format.NativeFormat
    assert (s.organizationName(), s.applicationName()) == ("MicroTopos", "MicroTopos")
