"""Deterministic offline reconstruction of a recording: every frame, in order, independent of playback speed.

    uv run microtopos-batch RECORDING OUTDIR [--objective "10x / 0.25"] [--z-um-per-frame 0.5 --direction up]
                            [--objectives objectives.json]

--objective takes any name in your objective list (the one edited in the GUI's step 2, objectives.json in the
data folder) or in a file given with --objectives; `--list-objectives` prints them.

Runs the same live engine as the GUI, synchronously, over every frame of the manifest (no frame is skipped),
reconstructs each detected focus sweep and writes OUTDIR/sweep-NN/ (report bundle and height exports) plus
OUTDIR/batch.json (inputs, manifest checksum, parameters, one entry per sweep). The same recording and
parameters give the same results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__, export, units, zstack
from .calibration import CalibrationStore
from .live import LiveEngine
from .objectives import ObjectiveError, ObjectiveSet, default_data_dir, load_set, resolve
from .recording import decode, load_manifest


def run_batch(recording: str | Path, outdir: str | Path, objective: str = "10x / 0.25",
              z_um_per_frame: float | None = None, direction: int | None = None,
              lateral_um_per_px_full: float | None = None, settings: zstack.ZSettings | None = None,
              log=print, objset: ObjectiveSet | None = None) -> dict:
    obj = resolve(objective, objset=objset)  # ObjectiveError (a clear message) for a name not in the list
    man = load_manifest(recording)
    store = CalibrationStore(None)  # nothing is read from or written to the user's calibration
    if lateral_um_per_px_full:
        store.get(obj).lateral_um_per_px_full = lateral_um_per_px_full
    eng = LiveEngine(store, obj)
    eng.sync_callbacks = True
    caps = []
    eng.on_sweep = caps.append
    corrupt = 0
    for f, ts, seq in zip(man.files, man.stamps, man.seqs):
        img = decode(f)
        if img is None:
            corrupt += 1
            continue
        eng.process_frame(img, ts, seq)
    eng.flush()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    st = settings or zstack.ZSettings(workers=1)  # one worker: identical results on every run
    sweeps = []
    for k, cap in enumerate(caps, 1):
        cal = store.get(obj)
        um = cal.um_per_px(cap.shape[1], obj)
        unit = units.UM if z_um_per_frame else units.FRAMES
        entry = {"index": k, "frames": len(cap.frames), "completeness": cap.completeness,
                 "seq_first": cap.seqs[0] if cap.seqs else None, "seq_last": cap.seqs[-1] if cap.seqs else None}
        try:
            r = zstack.reconstruct(cap.frames, None, obj, um, unit, z_um_per_frame, st,
                                   sample_pos=np.asarray(cap.seqs, float), z_sign=direction)
        except (ValueError, MemoryError) as e:
            entry["error"] = str(e)
            sweeps.append(entry)
            log(f"sweep {k}: not reconstructed: {e}")
            continue
        r.update(z_status="CALIBRATED" if z_um_per_frame else "NOMINAL",
                 lateral_status="CALIBRATED" if lateral_um_per_px_full else "NOMINAL")
        d = out / f"sweep-{k:02d}"
        export.write_report_bundle(d, r, {"microtopos": __version__})
        export.export_height(d / "height.tif", "tif", r["sl_surface"], r["um_per_px"], r["z_unit"])
        entry.update({k2: r.get(k2) for k2 in ("frames_used", "valid_fraction", "z_unit", "z_orientation", "Sq",
                                               "Sa", "Sz")})
        entry["folder"] = d.name
        sweeps.append(entry)
        log(f"sweep {k}: {r['frames_used']} frames, {r['valid_fraction']:.0%} valid, Sq {r.get('Sq')}")
    summary = {"microtopos": __version__, "recording": str(Path(recording).resolve()),
               "manifest_sha256": hashlib.sha256((Path(recording) / "frames.json").read_bytes()).hexdigest(),
               "frames": len(man.files), "corrupt_frames": corrupt, "manifest_warnings": man.warnings,
               "objective": objective, "objective_spec": obj.to_dict(), "z_um_per_frame": z_um_per_frame, "direction": direction,
               "lateral_um_per_px_full": lateral_um_per_px_full, "proc_width": st.proc_width, "measure": st.measure,
               "engine_errors": eng.errors, "sweeps": sweeps}
    (out / "batch.json").write_text(json.dumps(export._jsonable(summary), indent=1, allow_nan=False),
                                    encoding="utf-8")
    return summary


def main(argv=None):
    from . import safe_console

    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", nargs="?")
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--objective", help="objective name from your objective list (default: 10x / 0.25 if listed, "
                                        "else the first objective)")
    ap.add_argument("--objectives", metavar="FILE", help="objective list to use instead of the one in the data folder")
    ap.add_argument("--list-objectives", action="store_true", help="print the objective list and exit")
    ap.add_argument("--z-um-per-frame", type=float, help="knob travel per source frame for these sweeps (µm)")
    ap.add_argument("--direction", choices=("up", "down"), help="focal plane moved up (toward the objective) or down")
    ap.add_argument("--um-per-px", type=float, help="calibrated lateral scale at 1920 px width (µm/px)")
    a = ap.parse_args(argv)
    if a.objectives:
        src = Path(a.objectives)
        try:
            objset = ObjectiveSet.from_json(json.loads(src.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:  # ObjectiveError and JSONDecodeError are ValueErrors
            sys.exit(f"--objectives {src}: {e}")
    else:
        src = default_data_dir() / "objectives.json"
        objset = load_set(src.parent)
        if objset.load_error:
            sys.exit(objset.load_error)
    if a.list_objectives:
        print(f"Objective list: {src if src.exists() else 'built-in default'} ({objset.source})")
        for o in objset.objectives:
            print(f"  {o.name!r}: {o.magnification:g}x, NA {o.na:g}, n {o.immersion_n:g}"
                  + (f", cover {o.cover}" if o.cover else ""))
        sys.exit(0)
    if not (a.recording and a.outdir):
        ap.error("RECORDING and OUTDIR are required")
    name = a.objective or ("10x / 0.25" if "10x / 0.25" in objset else objset.names[0])
    if name not in objset:
        sys.exit(f"--objective {name!r} is not in the objective list; choose one of: "
                 + ", ".join(repr(n) for n in objset.names))
    if a.z_um_per_frame is not None and a.z_um_per_frame <= 0:
        sys.exit("--z-um-per-frame must be positive")
    try:
        s = run_batch(a.recording, a.outdir, name, a.z_um_per_frame,
                      {"up": 1, "down": -1}.get(a.direction), a.um_per_px, objset=objset)
    except ObjectiveError as e:
        sys.exit(str(e))
    ok = sum("error" not in x for x in s["sweeps"])
    print(f"{len(s['sweeps'])} sweep(s) detected, {ok} reconstructed; summary in {Path(a.outdir) / 'batch.json'}")
    sys.exit(0 if s["sweeps"] else 2)


if __name__ == "__main__":
    main()
