"""Recordings: one writer for the GUI and `microtopos-record`, one validated manifest for replay.

Layout of a recording folder:
  000000.jpg ...    the camera's exact MJPEG bytes ("mjpeg-raw"), or
  000000.png ...    lossless PNG of decoded frames ("decoded-png"), when the backend does not deliver MJPEG bytes
  frames.json       manifest: format, version, camera, effective settings, mode, and one entry per frame
                    (file, seq, t, bytes, crc32); "complete" is false until the recording is closed cleanly

8-bit MJPEG cannot be turned back into higher bit depth; "decoded-png" only avoids a second compression.
"""

from __future__ import annotations

import json
import os
import threading
import zlib
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

FORMAT = "microtopos-recording"
VERSION = 2
MANIFEST = "frames.json"


def is_jpeg(data: bytes) -> bool:
    """Start-of-image marker and an end-of-image marker near the end (some cameras pad after EOI)."""
    return len(data) > 4 and data[:2] == b"\xff\xd8" and b"\xff\xd9" in data[-64:]


def encoded_buffer(raw: np.ndarray) -> np.ndarray | None:
    """The flat byte buffer when `raw` is an encoded frame, else None (a decoded image).

    OpenCV hands an undecoded frame over as 1-D, or as a single row: V4L2 with CONVERT_RGB off gives MJPEG
    bytes shaped (1, N) (measured on a MikrOkular Full HD, 2026-11-14). A one-pixel-tall decoded image does
    not occur, so any single row or column of uint8 is treated as bytes, as camera.py already does."""
    if raw.dtype == np.uint8 and (raw.ndim == 1 or (raw.ndim == 2 and 1 in raw.shape)):
        return raw.reshape(-1)
    return None


class RecordingError(RuntimeError):
    pass


class RecordingWriter:
    """Writes frames and keeps the manifest current, so a disconnect or crash leaves a usable recording.

    add() runs on the camera thread and close() on the GUI thread, so both hold one lock: without it a Stop
    that lands during a periodic manifest flush can be overwritten by that flush, leaving a finished
    recording marked "not closed cleanly" (the likely cause of exactly that on a real recording, 2026-11-14:
    its last manifest write was the flush at frame 1250, 6 ms after the last frame, and Stop wrote nothing later)."""

    FLUSH_EVERY = 50

    def __init__(self, folder: Path | str, camera: str = "", readback: dict | None = None,
                 controls: dict | None = None):
        self.folder = Path(folder)
        if self.folder.exists() and any(self.folder.iterdir()):
            raise FileExistsError(f"{self.folder} is not empty: a recording never mixes with older files")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.meta = {"format": FORMAT, "version": VERSION, "camera": camera, "readback": readback or {},
                     "controls": controls or {}, "mode": None, "complete": False}
        self.entries: list[dict] = []
        self.bytes = 0
        self.error: str | None = None
        self.closed = False
        self._lock = threading.Lock()

    @property
    def n(self) -> int:
        return len(self.entries)

    def add(self, raw: np.ndarray, seq: int, t: float) -> None:
        """raw: the encoded buffer (uint8, 1-D or a single row) or a decoded image. Raises RecordingError on failure."""
        with self._lock:
            self._add(raw, seq, t)

    def _fail(self, msg: str, cause: BaseException | None = None):
        self.error = msg
        if cause is not None:
            raise RecordingError(msg) from cause
        raise RecordingError(msg)

    def _add(self, raw: np.ndarray, seq: int, t: float) -> None:
        if self.closed:
            raise RecordingError("recording already closed")
        enc = encoded_buffer(raw)
        if enc is not None:
            data = enc.tobytes()
            if not is_jpeg(data):
                self._fail("the camera buffer is not a JPEG image; raw recording is not possible")
            try:
                decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR) if self.n % 100 == 0 else True
            except cv2.error as e:
                self._fail(f"the camera's JPEG data does not decode: {e}")
            if decoded is None:
                self._fail("the camera's JPEG data does not decode")
            mode, ext = "mjpeg-raw", "jpg"
        else:
            try:
                ok, buf = cv2.imencode(".png", raw)
            except cv2.error as e:
                self._fail(f"PNG encoding failed: {e}")
            if not ok:
                self._fail("PNG encoding failed")
            data, mode, ext = buf.tobytes(), "decoded-png", "png"
        if self.meta["mode"] is None:
            self.meta["mode"] = mode
        elif self.meta["mode"] != mode:
            self._fail(f"the frame format changed during the recording ({self.meta['mode']} -> {mode})")
        name = f"{self.n:06d}.{ext}"
        try:
            with open(self.folder / name, "wb") as f:
                written = f.write(data)
        except OSError as e:  # disk full, permissions, unplugged drive: a visible failure, not a silent gap
            self._fail(f"could not write {name}: {e}", e)
        if written != len(data):
            self._fail(f"short write for {name}")
        self.entries.append({"file": name, "seq": int(seq), "t": round(float(t), 4), "bytes": len(data),
                             "crc32": zlib.crc32(data)})
        self.bytes += len(data)
        if self.n % self.FLUSH_EVERY == 0:
            self._write_manifest()

    def _write_manifest(self) -> None:
        meta = {**self.meta, "frames": self.entries, "timestamps": [e["t"] for e in self.entries],
                "bytes": self.bytes, "error": self.error}
        tmp = self.folder / f".{MANIFEST}.tmp"
        tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        os.replace(tmp, self.folder / MANIFEST)

    def close(self) -> int:
        """Finalise the manifest (also after an error). Returns the number of frames saved."""
        with self._lock:
            if not self.closed:
                self.meta["complete"] = self.error is None
                self._write_manifest()
                self.closed = True
            return self.n


@dataclass
class Manifest:
    folder: Path
    files: list[Path]
    stamps: list[float]
    seqs: list[int]
    mode: str
    meta: dict
    warnings: list[str] = field(default_factory=list)


def load_manifest(folder: Path | str) -> Manifest:
    """Validated list of frames. Version-2 manifests name every file; older recordings (timestamps only) are
    matched to the sorted JPEG files, and a count mismatch is reported, not hidden by zip()."""
    folder = Path(folder)
    mpath = folder / MANIFEST
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath} not found: not a MicroTopos recording")
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    warnings: list[str] = []
    if "frames" in meta:
        entries = meta["frames"]
        files, stamps, seqs = [], [], []
        for e in entries:
            p = folder / e["file"]
            if not p.exists():
                warnings.append(f"missing frame file {e['file']}")
                continue
            files.append(p)
            stamps.append(float(e["t"]))
            seqs.append(int(e.get("seq", len(seqs))))
        mode = meta.get("mode") or "mjpeg-raw"
        if not meta.get("complete", False):
            warnings.append("the recording was not closed cleanly (camera disconnect or crash); frames up to the "
                            "last manifest update are used")
        listed = {e["file"] for e in entries}
        extra = [p.name for p in folder.iterdir() if p.suffix in (".jpg", ".png") and p.name not in listed]
        if extra:
            warnings.append(f"{len(extra)} image file(s) in the folder are not in the manifest and are ignored")
    else:  # version 1: timestamps only
        files = sorted(folder.glob("*.jpg"))
        stamps = [float(t) for t in meta.get("timestamps", [])]
        if len(stamps) != len(files):
            warnings.append(f"{len(files)} JPEG files but {len(stamps)} timestamps: using the first "
                            f"{min(len(files), len(stamps))}")
            n = min(len(files), len(stamps))
            files, stamps = files[:n], stamps[:n]
        seqs = list(range(len(files)))
        mode = "mjpeg-raw"
    if not files:
        raise ValueError(f"{folder}: the recording contains no usable frames")
    back = sum(1 for a, b in pairwise(stamps) if b < a)
    if back:
        warnings.append(f"timestamps go backwards {back} time(s); playback keeps file order")
    return Manifest(folder, files, stamps, seqs, mode, meta, warnings)


def decode(path: Path) -> np.ndarray | None:
    """Decoded BGR frame (or 16-bit PNG if recorded in 16-bit), or None when the file is damaged."""
    try:
        buf = np.fromfile(path, np.uint8)
    except OSError:
        return None
    if not buf.size:
        return None
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)  # the one decode of this file; the rest is conversion
    if raw is None:
        return None
    if raw.dtype == np.uint16:
        return raw
    if raw.dtype != np.uint8:  # an exotic depth: let OpenCV reduce it to 8-bit BGR
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if raw.ndim == 2:
        if raw.shape[0] == 1 and is_jpeg(raw.tobytes()):  # MJPEG bytes stored as a 1xN PNG strip: decode the JPEG inside
            return cv2.imdecode(raw.reshape(-1), cv2.IMREAD_COLOR)
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    return raw
