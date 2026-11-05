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
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

FORMAT = "microtopos-recording"
VERSION = 2
MANIFEST = "frames.json"


def is_jpeg(data: bytes) -> bool:
    """Start-of-image marker and an end-of-image marker near the end (some cameras pad after EOI)."""
    return len(data) > 4 and data[:2] == b"\xff\xd8" and b"\xff\xd9" in data[-64:]


class RecordingError(RuntimeError):
    pass


class RecordingWriter:
    """Writes frames and keeps the manifest current, so a disconnect or crash leaves a usable recording."""

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

    @property
    def n(self) -> int:
        return len(self.entries)

    def add(self, raw: np.ndarray, seq: int, t: float) -> None:
        """raw: the encoded buffer (1-D uint8) or a decoded image (2-D / 3-D). Raises RecordingError on failure."""
        if self.closed:
            raise RecordingError("recording already closed")
        if raw.ndim == 1:
            data = raw.tobytes()
            if not is_jpeg(data):
                raise RecordingError("the camera buffer is not a JPEG image; raw recording is not possible")
            if self.n % 100 == 0 and cv2.imdecode(raw, cv2.IMREAD_COLOR) is None:
                raise RecordingError("the camera's JPEG data does not decode")
            mode, ext = "mjpeg-raw", "jpg"
        else:
            ok, buf = cv2.imencode(".png", raw)
            if not ok:
                raise RecordingError("PNG encoding failed")
            data, mode, ext = buf.tobytes(), "decoded-png", "png"
        if self.meta["mode"] is None:
            self.meta["mode"] = mode
        elif self.meta["mode"] != mode:
            raise RecordingError(f"the frame format changed during the recording ({self.meta['mode']} -> {mode})")
        name = f"{self.n:06d}.{ext}"
        try:
            with open(self.folder / name, "wb") as f:
                written = f.write(data)
        except OSError as e:  # disk full, permissions, unplugged drive: a visible failure, not a silent gap
            self.error = f"could not write {name}: {e}"
            raise RecordingError(self.error) from e
        if written != len(data):
            self.error = f"short write for {name}"
            raise RecordingError(self.error)
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
    back = sum(1 for a, b in zip(stamps, stamps[1:]) if b < a)
    if back:
        warnings.append(f"timestamps go backwards {back} time(s); playback keeps file order")
    return Manifest(folder, files, stamps, seqs, mode, meta, warnings)


def decode(path: Path) -> np.ndarray | None:
    """Decoded BGR frame, or None when the file is damaged."""
    try:
        buf = np.fromfile(path, np.uint8)
    except OSError:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    return img
