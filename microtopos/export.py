"""Exports: Gwyddion Simple Field (.gsf), binary STL, 32-bit float TIFF, report bundle, session files."""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import shutil
import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile


class ExportError(OSError):
    pass


def _atomic_bytes(path: Path | str, data: bytes) -> Path:
    """Write to a temporary file next to the target and rename it into place (no half-written files)."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def _png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ExportError("PNG encoding failed")
    return buf.tobytes()


def write_mask_png(path: Path | str, valid: np.ndarray) -> Path:
    """Validity mask (255 = measured, 0 = no reliable depth) beside a height export."""
    return _atomic_bytes(path, _png_bytes(valid.astype(np.uint8) * 255))


def write_sidecar(path: Path | str, info: dict) -> Path:
    """Provenance JSON beside an export (units, scale, how invalid pixels were written)."""
    return _atomic_bytes(path, json.dumps(_jsonable(info), indent=2, allow_nan=False).encode("utf-8"))

def write_gwyddion_gsf(
    filepath: Path | str,
    height_map: np.ndarray,
    pixel_um: float,
    z_calibrated_in_um: bool = True,
    title: str = "MicroTopos_Height",
) -> float:
    """
    Writes a Gwyddion Simple Field 1.0 (.gsf) file in SI meters.
    Replaces NaNs with the median valid height (documented in header comment).
    Returns the NaN fill value used (in output units).
    """
    h, w = height_map.shape
    z_scale = 1e-6 if z_calibrated_in_um else 1.0
    z_unit = "m" if z_calibrated_in_um else "frames"

    valid = np.isfinite(height_map)
    fill_val = float(np.median(height_map[valid])) * z_scale if np.any(valid) else 0.0
    data_si = np.where(valid, height_map * z_scale, fill_val).astype("<f4")

    xreal_m = float(w * pixel_um * 1e-6)
    yreal_m = float(h * pixel_um * 1e-6)

    header_lines = [
        "Gwyddion Simple Field 1.0",
        f"XRes = {w}",
        f"YRes = {h}",
        f"XReal = {xreal_m:.9e}",
        f"YReal = {yreal_m:.9e}",
        "XOffset = 0.0",
        "YOffset = 0.0",
        "XYUnits = m",
        f"ZUnits = {z_unit}",
        f"Title = {title}",
        f"NaNFillValue = {fill_val:.9e}",
        "",
    ]
    header_bytes = "\n".join(header_lines).encode("ascii")
    # GSF spec: pad with 1 to 4 NUL bytes so total header length is divisible by 4
    pad_len = 4 - (len(header_bytes) % 4)
    header_padded = header_bytes + (b"\x00" * pad_len)

    _atomic_bytes(filepath, header_padded + data_si.tobytes(order="C"))
    return fill_val


def read_gwyddion_gsf(filepath: Path | str) -> tuple[dict[str, str], np.ndarray]:
    """Reads back a Gwyddion .gsf file to verify header metadata and float32 payload."""
    raw = Path(filepath).read_bytes()
    nul_pos = raw.find(b"\x00")
    if nul_pos < 0:
        raise ValueError("Invalid GSF file: missing NUL padding.")
    data_offset = nul_pos + (4 - (nul_pos % 4))
    header_text = raw[:nul_pos].decode("ascii", errors="replace")

    meta: dict[str, str] = {}
    for line in header_text.splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()

    xres = int(meta["XRes"])
    yres = int(meta["YRes"])
    arr = np.frombuffer(raw, dtype="<f4", count=xres * yres, offset=data_offset).reshape((yres, xres))
    return meta, arr.copy()


def write_binary_stl(
    filepath: Path | str,
    height_map: np.ndarray,
    pixel_um: float,
    max_grid_dim: int = 160,
) -> tuple[int, tuple[float, float]]:
    """
    Writes an IEEE 754 binary STL with true cross-product facet normals
    while preserving the exact aspect ratio of `height_map` and rescaling `dx, dy` accordingly!
    Returns (num_triangles, (span_x_um, span_y_um)).
    """
    h, w = height_map.shape
    valid = np.isfinite(height_map)
    fill = float(np.median(height_map[valid])) if np.any(valid) else 0.0
    z_clean = np.where(valid, height_map, fill).astype(np.float32)

    scale = min(1.0, float(max_grid_dim) / float(max(h, w)))
    gh = max(4, int(round(h * scale)))
    gw = max(4, int(round(w * scale)))
    z_mesh = cv2.resize(z_clean, (gw, gh), interpolation=cv2.INTER_AREA) if (gh, gw) != (h, w) else z_clean

    # Rescale dx and dy so total physical span (w - 1)*pixel_um is preserved!
    dx = float((w - 1) * pixel_um) / float(gw - 1)
    dy = float((h - 1) * pixel_um) / float(gh - 1)

    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    X = xx * dx
    Y = yy * dy
    Z = z_mesh

    # Build two triangles per grid quad vectorized
    v00 = np.stack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]], axis=-1)
    v10 = np.stack([X[:-1, 1:], Y[:-1, 1:], Z[:-1, 1:]], axis=-1)
    v11 = np.stack([X[1:, 1:], Y[1:, 1:], Z[1:, 1:]], axis=-1)
    v01 = np.stack([X[1:, :-1], Y[1:, :-1], Z[1:, :-1]], axis=-1)

    # Triangle 1: (v00, v10, v11), Triangle 2: (v00, v11, v01)
    t1 = np.stack([v00, v10, v11], axis=-2).reshape(-1, 3, 3)
    t2 = np.stack([v00, v11, v01], axis=-2).reshape(-1, 3, 3)
    tris = np.concatenate([t1, t2], axis=0)

    # True facet unit normals via cross product (v1 - v0) x (v2 - v0)
    e1 = tris[:, 1, :] - tris[:, 0, :]
    e2 = tris[:, 2, :] - tris[:, 0, :]
    normals = np.cross(e1, e2)
    n_len = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
    normals = (normals / n_len).astype("<f4")

    n_tri = len(tris)
    stl_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    records = np.empty(n_tri, dtype=stl_dtype)
    records["normal"] = normals
    records["v0"] = tris[:, 0, :].astype("<f4")
    records["v1"] = tris[:, 1, :].astype("<f4")
    records["v2"] = tris[:, 2, :].astype("<f4")
    records["attr"] = 0

    header = b"MicroTopos binary STL visualisation surface (units: um; invalid pixels filled)".ljust(80, b"\x00")
    _atomic_bytes(filepath, header + struct.pack("<I", n_tri) + records.tobytes())

    return n_tri, (float((w - 1) * pixel_um), float((h - 1) * pixel_um))


def read_binary_stl_bounds(filepath: Path | str) -> tuple[int, tuple[float, float], np.ndarray]:
    """Reads back a binary STL file and returns (num_triangles, (span_x, span_y), normals)."""
    raw = Path(filepath).read_bytes()
    n_tri = struct.unpack("<I", raw[80:84])[0]
    stl_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    recs = np.frombuffer(raw, dtype=stl_dtype, count=n_tri, offset=84)
    xs = np.concatenate([recs["v0"][:, 0], recs["v1"][:, 0], recs["v2"][:, 0]])
    ys = np.concatenate([recs["v0"][:, 1], recs["v1"][:, 1], recs["v2"][:, 1]])
    span_x = float(np.max(xs) - np.min(xs))
    span_y = float(np.max(ys) - np.min(ys))
    return n_tri, (span_x, span_y), recs["normal"].copy()


def write_float32_tiff(filepath: Path | str, height_map: np.ndarray, pixel_um: float):
    """Writes standard single-channel 32-bit float TIFF (not .ome.tiff) with cm resolution tags."""
    px_per_cm = 1e4 / max(pixel_um, 1e-4)
    buf = io.BytesIO()
    tifffile.imwrite(buf, height_map.astype(np.float32), resolution=(px_per_cm, px_per_cm),
                     resolutionunit="CENTIMETER")  # NaN (no reliable depth) is kept as NaN
    _atomic_bytes(filepath, buf.getvalue())


def _jsonable(v):
    """Strict-JSON value: numpy scalars become Python numbers, non-finite numbers become null, arrays are dropped."""
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, np.ndarray):
        return None
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items() if not isinstance(x, np.ndarray)}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def write_report_bundle(folder: Path | str, result: dict, versions: dict) -> list[Path]:
    """report.json (all scalar results + provenance), summary.csv, edf.png, confidence.png, valid_mask.png.

    Every file is produced in a staging folder and checked first; only a complete bundle is moved into place,
    so a failed encoder or a full disk never leaves a partial bundle that looks complete.
    """
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: _jsonable(v) for k, v in result.items() if not isinstance(v, np.ndarray)}
    meta["software"] = {"python": platform.python_version(), **versions}
    stage = Path(tempfile.mkdtemp(prefix=".microtopos-report-", dir=out))
    try:
        files = {"report.json": json.dumps(meta, indent=2, allow_nan=False).encode("utf-8")}
        csv_buf = io.StringIO()
        wr = csv.writer(csv_buf, lineterminator="\n")
        wr.writerow(["parameter", "value"])
        for k, v in meta.items():
            if not isinstance(v, (dict, list)):
                wr.writerow([k, "" if v is None else v])
        files["summary.csv"] = csv_buf.getvalue().encode("utf-8")
        files["edf.png"] = _png_bytes(np.clip(result["edf"], 0, 255).astype(np.uint8))
        if isinstance(result.get("confidence"), np.ndarray):
            files["confidence.png"] = _png_bytes(np.clip(result["confidence"] * 255, 0, 255).astype(np.uint8))
        if isinstance(result.get("height"), np.ndarray):
            files["valid_mask.png"] = _png_bytes(np.isfinite(result["height"]).astype(np.uint8) * 255)
        for name, data in files.items():
            (stage / name).write_bytes(data)
            if (stage / name).stat().st_size != len(data):
                raise ExportError(f"short write for {name}")
        json.loads((stage / "report.json").read_text(encoding="utf-8"))  # parses strictly
        done = []
        for name in files:
            os.replace(stage / name, out / name)
            done.append(out / name)
        return done
    finally:
        shutil.rmtree(stage, ignore_errors=True)


SESSION_SCHEMA = 2
SESSION_ARRAYS = ("edf", "edf_proc", "height", "height_raw", "sl_surface", "confidence", "support", "mosaic_image",
                  "height_mosaic")
_LIMIT_PX = 400_000_000  # refuse absurd arrays from a damaged or foreign file before touching the UI


def save_session(path: Path | str, result: dict | None, settings: dict, extra: dict | None = None) -> None:
    """Schema-2 session: the result's original arrays (incl. the processing EDF and filtered surface), all scalars
    and warnings, settings, and optionally a mosaic. `result` may be None for a mosaic-only session."""
    result = dict(result or {})
    result.update(extra or {})
    arrays = {k: result[k] for k in SESSION_ARRAYS if isinstance(result.get(k), np.ndarray)}
    meta = {k: _jsonable(v) for k, v in result.items() if not isinstance(v, np.ndarray)}
    blob = json.dumps({"schema": SESSION_SCHEMA, "result": meta, "settings": _jsonable(settings),
                       "has_result": "height" in arrays}, allow_nan=False)
    buf = io.BytesIO()
    np.savez_compressed(buf, __meta__=np.array(blob), **arrays)
    _atomic_bytes(path, buf.getvalue())


def load_session(path: Path | str) -> tuple[dict, dict]:
    """Load and validate a session. Raises ValueError for a file that is inconsistent; returns (result, settings).

    result["session_schema"] says which schema the file had; schema-1 files lack the processing EDF and the
    filtered surface, and result["migrated"] lists what the caller has to recompute."""
    with np.load(path, allow_pickle=False) as z:
        if "__meta__" not in z.files:
            raise ValueError("not a MicroTopos session")
        meta = json.loads(str(z["__meta__"]))
        result = dict(meta.get("result", {}))
        for k in SESSION_ARRAYS:
            if k in z.files:
                a = z[k]
                if a.size > _LIMIT_PX:
                    raise ValueError(f"array {k} is too large ({a.shape})")
                result[k] = a
    schema = int(meta.get("schema", 1))
    if schema > SESSION_SCHEMA:
        raise ValueError(f"session schema {schema} is newer than this program supports ({SESSION_SCHEMA})")
    result["session_schema"] = schema
    result["migrated"] = [k for k in ("edf_proc", "sl_surface") if "height" in result and k not in result]
    if "height" in result:
        h = result["height"]
        if h.ndim != 2:
            raise ValueError("height map is not 2-D")
        for k in ("height_raw", "confidence", "edf_proc", "sl_surface", "support"):
            if k in result and result[k].shape != h.shape:
                raise ValueError(f"{k} has shape {result[k].shape}, height has {h.shape}")
        um = result.get("um_per_px")
        if not isinstance(um, (int, float)) or not np.isfinite(um) or um <= 0:
            raise ValueError("missing or invalid pixel size")
        if "z_unit" not in result:
            raise ValueError("missing height unit")
        from . import units
        result["z_unit"] = units.canon(result["z_unit"])
    return result, meta.get("settings", {})


def export_height(path: Path | str, fmt: str, height: np.ndarray, um_per_px: float, z_unit: str,
                  provenance: dict | None = None, title: str = "MicroTopos height") -> list[Path]:
    """Height export plus its validity mask (<name>.mask.png) and provenance (<name>.json).

    gsf: SI metres (z in frames if uncalibrated); invalid pixels are filled with the median for Gwyddion, so the
         mask is the record of which pixels were measured.
    tif: 32-bit float, invalid pixels are NaN.
    stl: visualisation mesh, invalid pixels filled with the median, z in µm (frames x pixel size if uncalibrated);
         not a measurement archive.
    """
    from . import units

    p = Path(path)
    is_um = units.is_um(z_unit)
    info = {"file": p.name, "format": fmt, "pixel_um": um_per_px, "z_unit": units.label(z_unit),
            "shape": list(height.shape), "valid_fraction": float(np.mean(np.isfinite(height))),
            "mask": p.with_suffix(".mask.png").name, **(provenance or {})}
    if fmt == "gsf":
        fill = write_gwyddion_gsf(p, height, um_per_px, is_um, title)
        info["invalid_pixels"] = f"filled with the median ({fill:.6g} {'m' if is_um else 'frames'}); see mask"
    elif fmt == "tif":
        write_float32_tiff(p, height, um_per_px)
        info["invalid_pixels"] = "NaN"
    elif fmt == "stl":
        z = height if is_um else height * um_per_px
        n, _ = write_binary_stl(p, z, um_per_px, max_grid_dim=400)
        info.update(triangles=n, invalid_pixels="filled with the median (display surface)",
                    note="visualisation mesh, not a measurement archive"
                    + ("" if is_um else "; z = focus frames x pixel size, not µm"))
    else:
        raise ValueError(f"unknown export format {fmt!r}")
    return [p, write_mask_png(p.with_suffix(".mask.png"), np.isfinite(height)), write_sidecar(p.with_suffix(".json"),
                                                                                              info)]
