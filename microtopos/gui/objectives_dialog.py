"""Edit the objective list: add, edit, remove and reorder objectives, or start from a preset.

Values are checked as you type (objectives.problems); OK stays disabled until the list is valid. Applying goes
through objectives.commit_edit, which carries calibrations across renames and keeps those of removed objectives.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from .. import objectives as objlib
from ..optics import Objective

COLUMNS = ("Name", "Magnification", "NA", "Immersion n", "Cover glass", "Maker", "Series", "Stored calibration")
ORIG = Qt.UserRole  # on the name cell: the name the row had when the dialog opened (None for a new row)


class ObjectivesDialog(QDialog):
    committed = Signal(object, object, object)  # (ObjectiveSet, notes, renames)

    def __init__(self, parent, objset: objlib.ObjectiveSet, store, data_dir):
        super().__init__(parent)
        self.setWindowTitle("Objectives")
        self.setWindowModality(Qt.WindowModal)
        self.resize(900, 420)
        self.old, self.store, self.data_dir = objset, store, data_dir
        self.source = objset.source
        self.preset_applied: str | None = None
        lay = QVBoxLayout(self)
        intro = QLabel("One row per objective on your nosepiece, in nosepiece order (keys 1–9 select the first "
                       "nine). Enter the values engraved on each objective. Calibrations belong to the name: "
                       "renaming an objective takes its calibration along; removing one keeps its calibration "
                       "for when you add it again.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        prov = QLabel(f"Current list: {objset.source or 'unknown origin'}, revision {objset.revision}"
                      + (f", changed {objset.date}" if objset.date else ""))
        prov.setObjectName("muted")
        lay.addWidget(prov)
        pre = QHBoxLayout()
        pre.addWidget(QLabel("Preset"))
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(objlib.PRESETS))
        pre.addWidget(self.cmb_preset, 1)
        self.btn_preset = QPushButton("Apply preset")
        self.btn_preset.setToolTip("Replace the rows with the preset's objectives (nothing is saved until OK)")
        self.btn_preset.clicked.connect(lambda: self.apply_preset(self.cmb_preset.currentText()))
        pre.addWidget(self.btn_preset)
        lay.addLayout(pre)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        hdr = self.table.horizontalHeader()
        for c in range(len(COLUMNS)):
            hdr.setSectionResizeMode(c, QHeaderView.Stretch if c == 0 else QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(True)
        self.table.itemChanged.connect(lambda _it: self._check())
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        for text, fn in (("Add", self.add_row), ("Remove", self.remove_selected), ("Move up", lambda: self._move(-1)),
                         ("Move down", lambda: self._move(1))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.lbl_problems = QLabel()
        self.lbl_problems.setWordWrap(True)
        self.lbl_problems.setObjectName("muted")
        lay.addWidget(self.lbl_problems)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self._fill(objset.objectives, keep_origin=True)

    # ------------------------------------------------------------ rows
    def _fill(self, objs, keep_origin: bool):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for o in objs:
            self._append(o, o.name if keep_origin else (o.name if o.name in self.old else None))
        self.table.blockSignals(False)
        self._check()

    def _append(self, o: Objective | None, origin: str | None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = ([o.name, f"{o.magnification:g}", f"{o.na:g}", f"{o.immersion_n:g}", o.cover, o.maker, o.series]
                if o else ["", "", "", "1", "", "", ""])
        for c, v in enumerate(vals):
            it = QTableWidgetItem(v)
            if c == 0:
                it.setData(ORIG, origin)
            self.table.setItem(r, c, it)
        info = QTableWidgetItem(self._cal_text(origin))
        info.setFlags(info.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, len(COLUMNS) - 1, info)

    def _cal_text(self, name: str | None) -> str:
        c = self.store.cals.get(name) if (self.store is not None and name) else None
        if c is None or not c.has_data:
            return "—"
        bits = (["lateral"] if c.lateral_um_per_px_full else []) + (["Z estimate"] if c.z_um_per_frame else []) \
            + (["dark/flat"] if c.darks or c.flats else [])
        return ", ".join(bits)

    def add_row(self):
        self.table.blockSignals(True)
        self._append(None, None)
        self.table.blockSignals(False)
        self.table.setCurrentCell(self.table.rowCount() - 1, 0)
        self._check()

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()} or {self.table.currentRow()}, reverse=True)
        for r in rows:
            if r >= 0:
                self.table.removeRow(r)
        self._check()

    def _move(self, d: int):
        r = self.table.currentRow()
        t = r + d
        if r < 0 or not (0 <= t < self.table.rowCount()):
            return
        self.table.blockSignals(True)
        for c in range(self.table.columnCount()):
            a, b = self.table.takeItem(r, c), self.table.takeItem(t, c)
            self.table.setItem(r, c, b)
            self.table.setItem(t, c, a)
        self.table.blockSignals(False)
        self.table.setCurrentCell(t, self.table.currentColumn())
        self._check()

    def apply_preset(self, name: str):
        """Replace the rows with a preset. Rows whose name already exists keep that objective's calibration."""
        self._fill(objlib.PRESETS[name], keep_origin=False)
        self.preset_applied = name
        self.cmb_preset.setCurrentText(name)

    # ------------------------------------------------------------ reading and checking
    def rows(self) -> tuple[list[Objective], dict[str, str], list[str]]:
        """(objectives, renames old->new, parse problems)."""
        objs, renames, errs = [], {}, []
        for r in range(self.table.rowCount()):
            cell = [(self.table.item(r, c).text() if self.table.item(r, c) else "") for c in range(7)]
            nums = []
            for label, txt in (("magnification", cell[1]), ("NA", cell[2]), ("immersion n", cell[3])):
                try:
                    nums.append(float(txt.replace(",", ".").replace("×", "").replace("x", "").strip()))
                except ValueError:
                    errs.append(f"Row {r + 1}: {label} {txt!r} is not a number.")
                    nums.append(float("nan"))
            o = Objective(cell[0], nums[0], nums[1], nums[2], cell[4].strip(), cell[5].strip(), cell[6].strip())
            objs.append(o)
            orig = self.table.item(r, 0).data(ORIG) if self.table.item(r, 0) else None
            if orig and orig != o.name:
                renames[orig] = o.name
        return objs, renames, errs

    def _check(self) -> list[str]:
        objs, _renames, errs = self.rows()
        probs = errs + [p for p in objlib.problems(objs) if "not a finite number" not in p or not errs]
        ok = not probs
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
        if ok:
            gone = [n for n in self.old.names if n not in {o.name for o in objs} and self.store is not None
                    and self.store.has_data(n)]
            self.lbl_problems.setText(
                (f"{len(objs)} objectives. " if len(objs) != 1 else "1 objective. ")
                + (f"Calibrations of {', '.join(gone)} will be kept, not deleted." if gone else ""))
        else:
            self.lbl_problems.setText("Fix before saving: " + " ".join(probs))
        return probs

    def accept(self):
        objs, renames, errs = self.rows()
        if errs or self._check():
            return
        src = self.preset_applied if self.preset_applied else None
        if src and [o.to_dict() for o in objs] != [o.to_dict() for o in objlib.PRESETS[src]]:
            src = f"{src} + edited"
        try:
            new, notes = objlib.commit_edit(self.data_dir, self.store, self.old, objs, renames, src)
        except (objlib.ObjectiveError, OSError) as e:
            self.lbl_problems.setText(f"Not saved: {e}")
            return
        self.committed.emit(new, notes, renames)
        super().accept()
