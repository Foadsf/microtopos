"""JPEG-aware restoration of the camera's MJPEG frames: header parsing, exact luma, QCS deblocking, guided chroma.

The MikrOkular sends baseline MJPEG (4:2:2; its tables are closest to IJG quality 43), and the app keeps the
camera's exact bytes. Those bytes say exactly how each 8x8 luma block was quantised, so the true image is known
to lie in a convex set: every block's DCT coefficients lie inside their quantisation interval. `deblock` looks
for a smoother image inside that set (projection onto convex sets, POCS), which removes block edges without
inventing detail the bitstream rules out. `upsample_chroma` replaces the decoder's blind chroma upsampling by a
guided upsampling that makes colour edges follow luma edges.

What this module does not do: it does not decode the entropy-coded data itself. The quantised coefficients are
recovered from the decoder's 8-bit Y plane (DCT of y - 128, divided by the step, rounded); see `block_coefficients`
for why that recovery is exact for the step sizes the camera uses and where it is not (saturated blocks, very
fine quantisation). Chroma is not deblocked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# zig-zag position k -> natural (row-major) index within the 8x8 block (ITU-T T.81 figure A.6)
ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
])

# ITU-T T.81 Annex K example tables (the IJG base tables), natural row-major order
IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]).reshape(8, 8)
IJG_CHROMA = np.full((8, 8), 99)
IJG_CHROMA[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]])

# A table counts as IJG-like when the best-fitting IJG quality reproduces it to within this mean |log ratio|
# (0.25 is about 28 %). Tables libjpeg wrote give 0 up to integer rounding; the MikrOkular's tables (one frame,
# 2026-11-28) are not exactly Annex K scaled but fit quality 43 to 0.022 (luma) and 0.003 (chroma).
IJG_LIKE_MAX_LOG_DEV = 0.25

_SOF_NAMES = {
    0xC0: "baseline", 0xC1: "extended sequential", 0xC2: "progressive", 0xC3: "lossless",
    0xC5: "differential sequential", 0xC6: "differential progressive", 0xC7: "differential lossless",
    0xC9: "arithmetic extended sequential", 0xCA: "arithmetic progressive", 0xCB: "arithmetic lossless",
    0xCD: "arithmetic differential sequential", 0xCE: "arithmetic differential progressive",
    0xCF: "arithmetic differential lossless",
}


def ijg_table(quality: int, base: np.ndarray = IJG_LUMA, force_baseline: bool = True) -> np.ndarray:
    """The quantisation table libjpeg's jpeg_set_quality builds for `quality` (1..100), natural order."""
    q = min(max(int(quality), 1), 100)
    scale = 5000 // q if q < 50 else 200 - 2 * q
    t = (base.astype(np.int64) * scale + 50) // 100
    return np.clip(t, 1, 255 if force_baseline else 32767).astype(np.int64)


@dataclass(frozen=True)
class Component:
    id: int
    h: int  # horizontal sampling factor
    v: int  # vertical sampling factor
    tq: int  # quantisation table id


@dataclass
class JpegInfo:
    width: int
    height: int
    precision: int
    components: list[Component]
    qtables: dict[int, np.ndarray] = field(default_factory=dict)  # id -> (8, 8) int, natural row-major order
    restart_interval: int = 0
    frame_type: str = "baseline"

    @property
    def subsampling(self) -> list[tuple[int, int]]:
        """Per component (horizontal, vertical) subsampling factor relative to full resolution, e.g. (2, 1)."""
        hmax = max(c.h for c in self.components)
        vmax = max(c.v for c in self.components)
        return [(hmax // c.h, vmax // c.v) for c in self.components]

    def qtable(self, component: int = 0) -> np.ndarray:
        return self.qtables[self.components[component].tq]

    def estimated_quality(self, component: int = 0) -> int | None:
        """IJG quality (1..100) whose scaled Annex K table best fits this component's table; None if none fits.

        The inverse of libjpeg's scaling is searched over all 100 qualities (component 0 is compared with the
        luma base table, others with the chroma one). For a table libjpeg wrote, the answer is the quality it was
        written with. For other encoders the answer is the nearest IJG equivalent, or None when even the best
        fit is off by more than IJG_LIKE_MAX_LOG_DEV on average (a table of a different shape).
        """
        q, dev = ijg_fit(self.qtable(component), IJG_LUMA if component == 0 else IJG_CHROMA)
        return q if dev <= IJG_LIKE_MAX_LOG_DEV else None


def ijg_fit(table: np.ndarray, base: np.ndarray = IJG_LUMA) -> tuple[int, float]:
    """(quality, mean |log(table / ijg_table(quality))|) for the best-fitting IJG quality."""
    t = np.log(np.maximum(np.asarray(table, np.float64), 1.0))
    best = (0, np.inf)
    for q in range(1, 101):
        dev = float(np.mean(np.abs(t - np.log(ijg_table(q, base, force_baseline=False)))))
        if dev < best[1]:
            best = (q, dev)
    return best


def parse_jpeg(data: bytes) -> JpegInfo:
    """Frame header, quantisation tables (natural order) and restart interval of a sequential Huffman JPEG.

    Raises ValueError for anything that is not a JPEG, and for progressive, lossless, hierarchical or
    arithmetic-coded JPEGs (whose coefficients this module's block model does not describe).
    """
    data = bytes(data)
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("not a JPEG: no SOI marker")
    qtables: dict[int, np.ndarray] = {}
    restart = 0
    frame: tuple | None = None
    i, n = 2, len(data)
    while i < n:
        if data[i] != 0xFF:
            # entropy-coded data after SOS: find the next marker that is not stuffing (FF00) or a restart (RSTn)
            j = data.find(b"\xff", i)
            if j < 0:
                break
            i = j
            continue
        while i < n and data[i] == 0xFF:  # fill bytes
            i += 1
        if i >= n:
            break
        m = data[i]
        i += 1
        if m == 0x00 or 0xD0 <= m <= 0xD7 or m == 0x01:  # stuffed byte, RSTn, TEM: no length field
            continue
        if m == 0xD9:  # EOI
            break
        if i + 2 > n:
            raise ValueError("truncated JPEG marker segment")
        length = int.from_bytes(data[i:i + 2], "big")
        seg = data[i + 2:i + length]
        if length < 2 or len(seg) != length - 2:
            raise ValueError(f"truncated JPEG segment 0xFF{m:02X}")
        if m in _SOF_NAMES:
            if m not in (0xC0, 0xC1):
                raise ValueError(f"{_SOF_NAMES[m]} JPEG (SOF{m - 0xC0}) is not supported: only sequential Huffman "
                                 "(baseline or extended) frames")
            precision, height, width, nc = seg[0], int.from_bytes(seg[1:3], "big"), int.from_bytes(seg[3:5], "big"), seg[5]
            if precision != 8:
                raise ValueError(f"{precision}-bit JPEG is not supported: only 8-bit samples")
            comps = [Component(seg[6 + 3 * k], seg[7 + 3 * k] >> 4, seg[7 + 3 * k] & 15, seg[8 + 3 * k]) for k in range(nc)]
            frame = (width, height, precision, comps, _SOF_NAMES[m])
        elif m == 0xDB:  # DQT: one or more tables, 8- or 16-bit
            k = 0
            while k < len(seg):
                pq, tq = seg[k] >> 4, seg[k] & 15
                k += 1
                if pq == 0:
                    zz = np.frombuffer(seg[k:k + 64], np.uint8).astype(np.int64)
                    k += 64
                else:
                    zz = np.frombuffer(seg[k:k + 128], ">u2").astype(np.int64)
                    k += 128
                if zz.size != 64:
                    raise ValueError("truncated DQT segment")
                nat = np.empty(64, np.int64)
                nat[ZIGZAG] = zz
                qtables[tq] = nat.reshape(8, 8)
        elif m == 0xDD:  # DRI
            restart = int.from_bytes(seg[0:2], "big")
        i += length
    if frame is None:
        raise ValueError("JPEG has no frame header (SOF)")
    width, height, precision, comps, kind = frame
    missing = sorted({c.tq for c in comps} - set(qtables))
    if missing:
        raise ValueError(f"JPEG lacks quantisation table(s) {missing}")
    return JpegInfo(width, height, precision, comps, qtables, restart, kind)


# --------------------------------------------------------------------------- decoding

_IMREAD_FLAGS = cv2.IMREAD_IGNORE_ORIENTATION  # an EXIF rotation would move the 8x8 block grid


def decode_luma(data: bytes) -> np.ndarray:
    """The decoder's own Y plane (uint8), not a luma recomputed from its BGR output.

    OpenCV asks libjpeg for JCS_GRAYSCALE output when reading a YCbCr JPEG as grayscale; libjpeg then emits
    component 0 straight from its inverse DCT, with no colour conversion, chroma upsampling or clipping of
    reconstructed RGB. Recomputing Y from the 8-bit BGR decode instead re-rounds and loses whatever the RGB
    clip removed.
    """
    y = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE | _IMREAD_FLAGS)
    if y is None:
        raise ValueError("OpenCV could not decode the JPEG")
    return y


def decode_bgr(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | _IMREAD_FLAGS)
    if bgr is None:
        raise ValueError("OpenCV could not decode the JPEG")
    return bgr


# --------------------------------------------------------------------------- block DCT

def _dct_matrix() -> np.ndarray:
    """Orthonormal 8-point DCT-II. Its 2-D separable product is exactly JPEG's FDCT normalisation
    F(u,v) = 1/4 C(u) C(v) sum f(x,y) cos(...) cos(...), with C(0) = 1/sqrt(2)."""
    k = np.arange(8)
    m = np.cos((2 * k[None, :] + 1) * k[:, None] * np.pi / 16) * np.sqrt(2 / 8)
    m[0] /= np.sqrt(2)
    return m


DCT8 = _dct_matrix()


def _dct_image(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Blockwise 8x8 (I)DCT kept in image layout: pixel (8i+u, 8j+v) holds coefficient (u, v) of block (i, j).

    Two large matrix products (along x within each block row, then along y) instead of one tiny product per
    block, which is what makes a full HD frame cost milliseconds.
    """
    h, w = x.shape
    d = DCT8.astype(x.dtype) if x.dtype == np.float32 else DCT8
    a, b = (d, d.T) if inverse else (d.T, d)
    t = (x.reshape(h, w // 8, 8) @ a).reshape(h, w)
    return (b @ t.reshape(h // 8, 8, w)).reshape(h, w)


def _to_blocks(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    return img.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)


def block_dct(img: np.ndarray) -> np.ndarray:
    """(H/8, W/8, 8, 8) JPEG-normalised DCT coefficients of every 8x8 block of img (dimensions multiples of 8)."""
    return _to_blocks(_dct_image(img))


def _pad8(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    return np.pad(img, ((0, -h % 8), (0, -w % 8)), mode="edge")


def block_coefficients(y: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    """Integer quantised coefficients c of every 8x8 block, recovered from the decoded Y: round(DCT(y - 128) / q).

    Why this recovers the encoder's coefficients: the decoder output is y = IDCT(c q) + 128 + e, where e is the
    IDCT's rounding to integers (measured for OpenCV's libjpeg: |e| <= 1 grey level from the exact IDCT, and
    nonzero in 0.3-1.8 % of pixels beyond the ideal rounding). The DCT is orthonormal,
    so DCT(e) has the same energy as e: each coefficient is off by about 0.3-0.6 RMS, and rounding recovers c
    whenever that error is below q/2. With the camera's steps (q >= 13) that margin is more than 10 standard
    deviations. Measured against a bitstream (Huffman) decoder on libjpeg encodes: exact in every unclipped block
    at qualities 30-75; at 90 and 95 (steps down to 2 and 1), 0.005-0.05 % and 0.4-1 % of the coefficients are
    off. Recovery also fails where the decoder clipped the block at 0 or 255 (see `saturated_blocks`), and for
    blocks cut by the right/bottom edge of an image whose size is not a multiple of 8 (the encoder's padding is
    not in the decoded image; edge replication stands in for it).
    """
    yp = _pad8(np.asarray(y, np.float64))
    return np.rint(block_dct(yp - 128.0) / np.asarray(qtable, np.float64)).astype(np.int32)


def _tile(table: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.tile(np.asarray(table, np.float32), (shape[0] // 8, shape[1] // 8))


# --------------------------------------------------------------------------- deblocking

def _shift_threshold_average(x: np.ndarray, qtable: np.ndarray, grid: int, threshold: float) -> np.ndarray:
    """Average of x and of x re-coded on the 8x8 block grids shifted by multiples of `grid` pixels.

    For every shift (dy, dx) other than (0, 0), the image is cut into 8x8 blocks that straddle the JPEG block
    boundaries, coefficients smaller than threshold * q are set to zero (hard thresholding at a fraction of the
    quantisation step), and the blocks are transformed back. A block edge of the decoded image is an
    off-grid discontinuity for those shifted blocks, i.e. many small high-frequency coefficients, which the
    thresholding removes; real texture that survived quantisation is larger than the threshold and stays. This is
    the "re-application of JPEG" estimate of Nosratinia (2001) with hard thresholding instead of requantisation.
    x has dimensions that are multiples of 8; qtable is the 8x8 quantisation table in natural order.
    """
    h, w = x.shape
    xp = np.pad(x, 8, mode="reflect")  # every shifted grid then covers the whole image, edges included
    acc = np.zeros(xp.shape, np.float32)
    acc[8:8 + h, 8:8 + w] = x
    t = _tile(qtable, (h + 8, w + 8)) * np.float32(threshold)
    shifts = [(dy, dx) for dy in range(0, 8, grid) for dx in range(0, 8, grid) if (dy, dx) != (0, 0)]
    for dy, dx in shifts:
        win = (slice(8 - dy, 8 - dy + h + 8), slice(8 - dx, 8 - dx + w + 8))  # blocks start dy, dx px early
        c = _dct_image(xp[win])
        np.multiply(c, np.abs(c) >= t, out=c)
        acc[win] += _dct_image(c, inverse=True)  # back onto the pixels it came from
    return acc[8:8 + h, 8:8 + w] / (len(shifts) + 1)


def deblock(y: np.ndarray, qtable: np.ndarray, iterations: int = 1, grid: int = 2, threshold: float = 0.3,
            coefficients: np.ndarray | None = None) -> np.ndarray:
    """Quantisation-constraint-set (QCS) restoration of a decoded JPEG luma plane, float32.

    Projection onto convex sets: `iterations` times, a smoothing estimate is formed by
    `_shift_threshold_average` (block grids shifted by multiples of `grid` px, coefficients below
    `threshold` x the quantisation step zeroed, all versions averaged), then every 8x8 block's DCT coefficients
    are clipped back into their quantisation interval [q (c - 1/2), q (c + 1/2)]. The projection is the last
    operation, so the result is an image from which the encoder would have produced exactly these quantised
    coefficients. c comes from `block_coefficients` unless the caller passes the true coefficients (block
    layout, as `block_coefficients` returns them). The result is not clipped to 0..255.

    One iteration is the default: measured on simulator renders and on downscaled MikrOkular frames coded at
    IJG quality 50, it gave +0.5 to +1.4 dB PSNR and cut the block-edge excess to 0.07-0.3 of its decoded value; repeating the
    smoothing keeps lowering blockiness but drifts towards the flattest image in the constraint set and gives
    PSNR back.
    """
    y = np.asarray(y)
    h, w = y.shape
    if grid not in (1, 2, 4):
        raise ValueError("grid must be 1, 2 or 4 pixels")
    x = _pad8(y.astype(np.float32)) - 128.0
    q = _tile(qtable, x.shape)
    c = block_coefficients(y, qtable) if coefficients is None else np.asarray(coefficients)
    c = c.astype(np.float32).transpose(0, 2, 1, 3).reshape(x.shape)  # block layout -> image layout
    lo, hi = (c - 0.5) * q, (c + 0.5) * q
    decoded = x
    for _ in range(int(iterations)):
        x = _shift_threshold_average(x, qtable, grid, threshold)
        x = _dct_image(np.clip(_dct_image(x), lo, hi), inverse=True)
    sat = saturated_blocks(y)
    if sat.any():  # c is not trustworthy where the decoder clipped: keep the decoder's pixels there
        mask = np.kron(sat, np.ones((8, 8), bool))
        x = np.where(mask, decoded, x)
    return (x[:h, :w] + 128.0).astype(np.float32)


def saturated_blocks(y: np.ndarray) -> np.ndarray:
    """(H/8, W/8) bool: blocks of the decoded Y (edge-padded to whole blocks) that contain a 0 or a 255."""
    yp = _pad8(np.asarray(y))
    return _to_blocks((yp <= 0) | (yp >= 255)).any(axis=(2, 3))


def blockiness(y: np.ndarray) -> float:
    """Mean |horizontal and vertical difference| across 8-px block boundaries minus the mean within blocks."""
    y = np.asarray(y, np.float64)
    dx = np.abs(np.diff(y, axis=1))
    dy = np.abs(np.diff(y, axis=0))
    bx = (np.arange(dx.shape[1]) % 8) == 7  # difference between column 8k+7 and 8k+8
    by = (np.arange(dy.shape[0]) % 8) == 7
    across = np.concatenate([dx[:, bx].ravel(), dy[by, :].ravel()]).mean()
    within = np.concatenate([dx[:, ~bx].ravel(), dy[~by, :].ravel()]).mean()
    return float(across - within)


# --------------------------------------------------------------------------- chroma

def bgr_to_ycbcr(bgr: np.ndarray) -> np.ndarray:
    """JFIF (full-range BT.601) Y, Cb, Cr in float32, channel order Y, Cb, Cr."""
    b, g, r = (bgr[..., k].astype(np.float32) for k in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.dstack([y, cb, cr])


def ycbcr_to_bgr(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """JFIF inverse, float32 BGR (not clipped)."""
    cb = cb - 128.0
    cr = cr - 128.0
    return np.dstack([y + 1.772 * cb, y - 0.344136 * cb - 0.714136 * cr, y + 1.402 * cr]).astype(np.float32)


def _box(img: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def _downsample(img: np.ndarray, fx: int, fy: int) -> np.ndarray:
    """Average over fy x fx cells (the libjpeg encoder's chroma downsampling), edge-replicated to whole cells."""
    h, w = img.shape
    p = np.pad(img, ((0, -h % fy), (0, -w % fx)), mode="edge")
    return p.reshape(p.shape[0] // fy, fy, p.shape[1] // fx, fx).mean(axis=(1, 3), dtype=np.float32)


def _undo_fancy_upsampling(low: np.ndarray, fx: int, fy: int, rounds: int = 8) -> np.ndarray:
    """Undo the smoothing that libjpeg's "fancy" (triangle) upsampling leaves after cell averaging.

    For a factor of 2, libjpeg interpolates out[2i] = (3 c[i] + c[i-1]) / 4 and out[2i+1] = (3 c[i] + c[i+1]) / 4,
    so averaging each pair back gives (c[i-1] + 6 c[i] + c[i+1]) / 8. A few Jacobi rounds solve that diagonally
    dominant system for c along every axis subsampled by 2 (OpenCV's libjpeg does this; measured: the decoder's
    chroma matches triangle interpolation of the encoder's samples to 1.1 grey levels RMS at quality 95, against
    2.9 for sample replication). For other factors libjpeg replicates samples (jdsample.c), which needs no
    correction; that case was not measured.
    """
    for axis, f in ((1, fx), (0, fy)):
        if f != 2 or low.shape[axis] < 3:
            continue
        m = low
        c = low
        for _ in range(rounds):
            prev = np.concatenate([np.take(c, [0], axis), np.take(c, range(c.shape[axis] - 1), axis)], axis)
            nxt = np.concatenate([np.take(c, range(1, c.shape[axis]), axis), np.take(c, [-1], axis)], axis)
            c = (8.0 * m - prev - nxt) / 6.0
        low = c.astype(np.float32)
    return low


def _upsample(img: np.ndarray, fx: int, fy: int, w: int, h: int) -> np.ndarray:
    """Linear interpolation from cell centres back to the full grid (half-pixel-centre convention)."""
    up = cv2.resize(img, (img.shape[1] * fx, img.shape[0] * fy), interpolation=cv2.INTER_LINEAR)
    return up[:h, :w]


def upsample_chroma(bgr: np.ndarray, info: JpegInfo, guide_y: np.ndarray, radius: int = 1,
                    eps: float = 4.0) -> np.ndarray:
    """Full-resolution BGR (float32, clipped to 0..255) whose chroma follows the luma guide's edges.

    `bgr` is the decoder's output (uint8 or float) for the JPEG `info` describes. For each
    chroma component that the JPEG subsampled, the decoder's chroma and the guide are averaged back onto the
    encoder's chroma grid, a guided filter (He et al., square box windows of 2 `radius` + 1 chroma samples, regulariser `eps`
    in grey levels squared) fits chroma as a local linear function a Y + b there, and a, b are interpolated to
    full resolution and applied to the full-resolution guide. Colour edges therefore land where the luma edges
    are instead of being smeared over the decoder's interpolation footprint. Components that were not
    subsampled keep the decoder's values. The returned image's luma is `guide_y` (up to the 0..255 clip).
    """
    ycc = bgr_to_ycbcr(np.asarray(bgr))
    if len(info.components) != 3:
        raise ValueError("chroma upsampling needs a three-component (YCbCr) JPEG")
    guide = np.asarray(guide_y, np.float32)
    h, w = guide.shape
    if ycc.shape[:2] != (h, w):
        raise ValueError("image and guide sizes differ")
    out = []
    for k in (1, 2):
        fx, fy = info.subsampling[k]
        p = ycc[..., k]
        if fx == 1 and fy == 1:
            out.append(p)
            continue
        gl = _downsample(guide, fx, fy)
        pl = _undo_fancy_upsampling(_downsample(p, fx, fy), fx, fy)
        mi, mp = _box(gl, radius), _box(pl, radius)
        cov = _box(gl * pl, radius) - mi * mp
        var = _box(gl * gl, radius) - mi * mi
        a = cov / (var + eps)
        b = mp - a * mi
        a, b = _box(a, radius), _box(b, radius)
        out.append(_upsample(a, fx, fy, w, h) * guide + _upsample(b, fx, fy, w, h))
    return np.clip(ycbcr_to_bgr(guide, out[0], out[1]), 0.0, 255.0)


# --------------------------------------------------------------------------- convenience

def restore_frame(data: bytes, deblock_iters: int = 1, colour: bool = True) -> dict:
    """Restore one camera JPEG from its exact bytes.

    Returns {"y": float32 luma (deblocked unless deblock_iters == 0), "bgr": float32 BGR or None,
    "info": JpegInfo, "applied": list of what was done, "seconds": wall time}.
    """
    t0 = time.perf_counter()
    info = parse_jpeg(data)
    y8 = decode_luma(data)
    applied = ["exact decoder luma"]
    if deblock_iters > 0:
        y = deblock(y8, info.qtable(0), iterations=deblock_iters)
        applied.append(f"QCS deblock ({deblock_iters} iterations)")
    else:
        y = y8.astype(np.float32)
    bgr = None
    if colour and len(info.components) == 3:
        bgr = upsample_chroma(decode_bgr(data), info, np.clip(y, 0.0, 255.0))
        if any(s != (1, 1) for s in info.subsampling[1:]):
            applied.append("guided chroma upsampling " + "/".join(f"{fx}x{fy}" for fx, fy in info.subsampling[1:]))
        else:
            applied.append("chroma not subsampled: decoder chroma kept")
    return {"y": y, "bgr": bgr, "info": info, "applied": applied, "seconds": time.perf_counter() - t0}
