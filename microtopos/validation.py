"""Input validation and sanitization utilities for CLI arguments and configuration parameters."""

from __future__ import annotations

import argparse
import math

import numpy as np


def parse_crop(value: str) -> tuple[int, int, int, int]:
    """Parse and strictly validate a crop specification in 'x,y,w,h' format.

    Raises argparse.ArgumentTypeError with an informative message if validation fails.
    """
    if not value or not isinstance(value, str):
        raise argparse.ArgumentTypeError("Crop must be a non-empty string in format 'x,y,w,h'")

    parts = [v.strip() for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Crop must consist of exactly 4 comma-separated values 'x,y,w,h' (got {len(parts)}: {value!r})"
        )

    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Crop values must be integers (got {value!r})"
        ) from exc

    if x < 0 or y < 0:
        raise argparse.ArgumentTypeError(
            f"Crop origin (x={x}, y={y}) must be non-negative"
        )
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(
            f"Crop dimensions (w={w}, h={h}) must be strictly positive"
        )

    return x, y, w, h


def parse_positive_float(value: str) -> float:
    """Parse a strictly positive float (> 0)."""
    try:
        v = float(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(f"Value must be a valid float (got {value!r})") from exc

    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"Value must be strictly positive and finite (got {value})")
    return v


def parse_nonnegative_int(value: str) -> int:
    """Parse a non-negative integer (>= 0)."""
    try:
        v = int(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(f"Value must be a valid integer (got {value!r})") from exc

    if v < 0:
        raise argparse.ArgumentTypeError(f"Value must be non-negative (got {value})")
    return v


def parse_fusion_mode(value: str) -> str:
    """Validate mosaic fusion mode."""
    v = value.strip().lower()
    valid = ("sharpest", "weighted", "average", "superres")
    if v not in valid:
        raise argparse.ArgumentTypeError(
            f"Invalid fusion mode {value!r}. Must be one of: {', '.join(valid)}"
        )
    return v


def validate_warp_matrix(W: np.ndarray, max_translation: float = 1000.0) -> bool:
    """Validate that a 2x3 affine warp matrix is finite and within physically reasonable bounds."""
    if not isinstance(W, np.ndarray) or W.shape != (2, 3):
        return False
    if not np.isfinite(W).all():
        return False

    # Check determinant of linear part (scale & rotation)
    lin = W[:, :2]
    det = float(np.linalg.det(lin))
    if det <= 0.2 or det >= 5.0:  # scale changes should be within reasonable optical range
        return False

    # In microscopy without 180-deg physical camera rotation, diagonal elements must remain positive
    if lin[0, 0] <= 0.0 or lin[1, 1] <= 0.0:
        return False

    # Check translation magnitudes
    tx, ty = float(W[0, 2]), float(W[1, 2])
    return math.hypot(tx, ty) <= max_translation
