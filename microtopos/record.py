"""Record the camera's raw MJPEG stream (lossless: the exact bytes the camera sent) for offline replay.

    uv run microtopos-record OUTDIR [--seconds 180] [--exposure 300]

Writes OUTDIR/000000.jpg ... and OUTDIR/frames.json (timestamps, camera settings). Replay with
`uv run microtopos --replay OUTDIR`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .camera import CameraSource, list_cameras
from .validation import parse_crop, parse_nonnegative_int, parse_positive_float


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--seconds", type=parse_positive_float, default=180, help="duration in seconds (positive float)")
    ap.add_argument("--exposure", type=parse_nonnegative_int, default=300, help="device units (UVC: 100 us); 0 = automatic")
    ap.add_argument("--camera", default="okular", help="substring of the camera name")
    ap.add_argument("--crop", type=parse_crop, default=None, help="crop region as x,y,w,h (integers)")
    a = ap.parse_args()
    cams = [c for c in list_cameras() if a.camera.lower() in c.name.lower()] or list_cameras()
    if not cams:
        sys.exit("no camera found")
    src = CameraSource(cams[0])
    controls = src.set_manual_exposure(None if a.exposure == 0 else a.exposure)
    cap = src.open()
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # deliver the MJPEG bytes undecoded
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stamps = []
    t0 = time.perf_counter()
    last_print = -10
    while (t := time.perf_counter() - t0) < a.seconds:
        ok, buf = cap.read()
        if not ok or buf is None:
            continue
        (out / f"{len(stamps):06d}.jpg").write_bytes(buf.tobytes())
        stamps.append(round(t, 4))
        if t - last_print >= 10:
            print(f"{t:5.0f} s  {len(stamps)} frames", flush=True)
            last_print = t
    cap.release()
    meta = {"camera": cams[0].name, "readback": src.readback, "controls": controls, "timestamps": stamps}
    (out / "frames.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {len(stamps)} frames to {out}")


if __name__ == "__main__":
    main()
