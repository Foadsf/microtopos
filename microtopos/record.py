"""Record the camera stream for offline replay: the camera's exact MJPEG bytes when the backend delivers them,
otherwise lossless PNG of the decoded frames (the manifest says which).

    uv run microtopos-record OUTDIR [--seconds 180] [--exposure 300]

Writes OUTDIR/000000.jpg (or .png) ... and OUTDIR/frames.json (manifest: per-frame file, sequence, time, size,
checksum, and the effective camera settings). OUTDIR must be new or empty. Replay with
`uv run microtopos --replay OUTDIR`. Exits non-zero when no frame was saved or a write failed.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from .camera import CameraSource, list_cameras
from .recording import RecordingError, RecordingWriter


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--exposure", type=int, default=300, help="device units (UVC: 100 us); 0 = automatic")
    ap.add_argument("--camera", default="okular", help="substring of the camera name")
    a = ap.parse_args(argv)
    cams = [c for c in list_cameras() if a.camera.lower() in c.name.lower()] or list_cameras()
    if not cams:
        sys.exit("no camera found")
    src = CameraSource(cams[0])
    controls = src.set_manual_exposure(None if a.exposure == 0 else a.exposure)
    cap = src.open()
    raw_ok = cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # ask for the MJPEG bytes undecoded
    try:
        writer = RecordingWriter(a.outdir, cams[0].name, {**src.readback, "raw_mode_accepted": bool(raw_ok)},
                                 controls)
    except FileExistsError as e:
        sys.exit(str(e))
    t0 = time.perf_counter()
    last_print, seq, failed = -10.0, 0, None
    try:
        while (t := time.perf_counter() - t0) < a.seconds:
            ok, buf = cap.read()
            if not ok or buf is None:
                continue
            seq += 1
            writer.add(buf, seq, t)
            if t - last_print >= 10:
                print(f"{t:5.0f} s  {writer.n} frames ({writer.meta['mode']})", flush=True)
                last_print = t
    except RecordingError as e:
        failed = str(e)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        n = writer.close()
    mb = writer.bytes / 1e6
    print(f"saved {n} frames ({mb:.1f} MB, {writer.meta['mode']}) to {writer.folder}")
    if writer.meta["mode"] == "decoded-png":
        print("note: the backend did not deliver MJPEG bytes; frames were saved as lossless PNG of the decoded image")
    if failed:
        sys.exit(f"recording failed: {failed}")
    if n == 0:
        sys.exit("no frames were saved")


if __name__ == "__main__":
    main()
