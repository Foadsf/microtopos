"""Exports: Gwyddion Simple Field (.gsf), binary STL, 32-bit float TIFF, report bundle, session files."""

from __future__ import annotations

import csv
import json
import platform
import struct
from pathlib import Path

import cv2
import numpy as np
import tifffile

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

    with open(filepath, "wb") as f:
        f.write(header_padded)
        f.write(data_si.tobytes(order="C"))
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

    header = b"MicroTopos v5 Binary STL Surface Mesh (units: um)".ljust(80, b"\x00")
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n_tri))
        f.write(records.tobytes())

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
    tifffile.imwrite(
        str(filepath),
        height_map.astype(np.float32),
        resolution=(px_per_cm, px_per_cm),
        resolutionunit="CENTIMETER",
    )


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items() if not isinstance(x, np.ndarray)}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def write_report_bundle(folder: Path | str, result: dict, versions: dict) -> list[Path]:
    """report.json (all scalar results + provenance), summary.csv, edf.png, confidence.png."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: _jsonable(v) for k, v in result.items() if not isinstance(v, np.ndarray)}
    meta["software"] = {"python": platform.python_version(), **versions}
    files = [out / "report.json", out / "summary.csv", out / "edf.png"]
    files[0].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with open(files[1], "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["parameter", "value"])
        for k, v in meta.items():
            if not isinstance(v, (dict, list)):
                wr.writerow([k, v])
    cv2.imwrite(str(files[2]), np.clip(result["edf"], 0, 255).astype(np.uint8))
    if "confidence" in result:
        p = out / "confidence.png"
        cv2.imwrite(str(p), np.clip(result["confidence"] * 255, 0, 255).astype(np.uint8))
        files.append(p)
    return files


SESSION_ARRAYS = ("edf", "height", "confidence", "height_raw")


def save_session(path: Path | str, result: dict, settings: dict) -> None:
    arrays = {k: result[k] for k in SESSION_ARRAYS if isinstance(result.get(k), np.ndarray)}
    meta = {k: _jsonable(v) for k, v in result.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, __meta__=np.array(json.dumps({"result": meta, "settings": settings})), **arrays)


def load_session(path: Path | str) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        result = dict(meta["result"])
        for k in SESSION_ARRAYS:
            if k in z.files:
                result[k] = z[k]
    return result, meta.get("settings", {})
