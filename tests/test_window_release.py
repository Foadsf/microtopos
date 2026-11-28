"""A closed window that ran a job can be freed: its finished jobs' signal connections must not pin it.

Each job's slots are lambdas that capture the window; the connections are held on the C++ side, which Python's
garbage collector cannot see, so before the fix every window that had run a job (a reconstruction, a stack) stayed
alive with its engine and results: the test suite grew by 100-380 MB per GUI test, to 10.6 GB on a 15 GB laptop,
where the kernel's OOM killer then took other applications down (2026-11-28).
"""
import gc
import os
import time
import weakref

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from PySide6.QtCore import QCoreApplication, QEvent, QStandardPaths
from PySide6.QtWidgets import QApplication

from microtopos.gui.main_window import Job, MainWindow


def _trivial(progress=None, cancelled=None):
    return 42


def test_a_closed_window_that_ran_a_job_is_freed(tmp_path):
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance() or QApplication([])
    pg.setConfigOptions(imageAxisOrder="row-major")
    w = MainWindow(simulator_only=True)
    w.show()
    got = []
    job = Job(_trivial, label="trivial")
    assert w._start_job(job, got.append)
    t0 = time.time()
    while (job.isRunning() or not got or w.jobs) and time.time() - t0 < 10:
        app.processEvents()
        time.sleep(0.01)
    assert got == [42] and not w.jobs
    for _ in range(5):
        app.processEvents()  # the deferred disconnect runs after the queued signals were delivered
    ref = weakref.ref(w)
    w.close()
    w.deleteLater()
    del w, job
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()
    assert ref() is None, [type(r).__name__ for r in gc.get_referrers(ref())][:8]
