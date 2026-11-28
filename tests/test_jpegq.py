"""JPEG-aware restoration (microtopos.jpegq) against ground truth built independently of the module.

Ground truth here is (1) JPEGs that OpenCV's libjpeg encodes from known uncompressed images, (2) the Annex K
tables and libjpeg's quality scaling written out in this file, and (3) a small baseline Huffman decoder in this
file that reads the exact quantised coefficients out of the bitstream, so the module's coefficient recovery and
quantisation constraints are checked against the encoder's own numbers rather than against themselves.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from microtopos import jpegq
from microtopos.objectives import resolve
from microtopos.simulator import SpecimenSimulator

# ITU-T T.81 Annex K tables K.1 and K.2, natural row-major order, typed here rather than imported
ANNEX_K_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
])
ANNEX_K_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
] + [[99] * 8] * 4)


def _libjpeg_scaled(base: np.ndarray, quality: int) -> np.ndarray:
    """jcparam.c: jpeg_quality_scaling, then jpeg_add_quant_table with force_baseline."""
    s = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((base * s + 50) // 100, 1, 255)


def _zigzag_natural_index() -> list[int]:
    """Zig-zag scan built by walking the anti-diagonals (independent of the module's lookup table)."""
    order = []
    for d in range(15):
        cells = [(r, d - r) for r in range(8) if 0 <= d - r < 8]
        order += [r * 8 + c for r, c in (cells if d % 2 else cells[::-1])]
    return order


def _reference_coefficients(data: bytes) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Baseline sequential Huffman decoding of the quantised coefficients: per component (by, bx, 8, 8) ints
    in natural order, and the per-component quantisation tables (8, 8). Interleaved single-scan files only."""
    zz = _zigzag_natural_index()
    pos, qt, huff, frame, ri = 2, {}, {}, None, 0
    while True:
        while data[pos] == 0xFF:
            pos += 1
        m = data[pos]
        length = int.from_bytes(data[pos + 1:pos + 3], "big")
        seg = data[pos + 3:pos + 1 + length]
        pos += 1 + length
        if m == 0xDB:
            k = 0
            while k < len(seg):
                t = np.zeros(64, int)
                t[zz] = list(seg[k + 1:k + 65])
                qt[seg[k] & 15] = t.reshape(8, 8)
                k += 65
        elif m == 0xC4:
            k = 0
            while k < len(seg):
                cls, tid, counts = seg[k] >> 4, seg[k] & 15, seg[k + 1:k + 17]
                syms = list(seg[k + 17:k + 17 + sum(counts)])
                table, code, i = {}, 0, 0
                for bits, n in enumerate(counts, start=1):
                    for _ in range(n):
                        table[(bits, code)] = syms[i]
                        code, i = code + 1, i + 1
                    code <<= 1
                huff[(cls, tid)] = table
                k += 17 + sum(counts)
        elif m == 0xC0:
            h, w = int.from_bytes(seg[1:3], "big"), int.from_bytes(seg[3:5], "big")
            frame = (w, h, [(seg[6 + 3 * k], seg[7 + 3 * k] >> 4, seg[7 + 3 * k] & 15, seg[8 + 3 * k]) for k in range(seg[5])])
        elif m == 0xDD:
            ri = int.from_bytes(seg[0:2], "big")
        elif m == 0xDA:
            scan = [(seg[1 + 2 * k], seg[2 + 2 * k] >> 4, seg[2 + 2 * k] & 15) for k in range(seg[0])]
            break
    entropy = data[pos:data.rindex(b"\xff\xd9")]
    segments, cur, i = [], bytearray(), 0
    while i < len(entropy):
        b = entropy[i]
        if b == 0xFF and entropy[i + 1] == 0x00:
            cur.append(0xFF)
            i += 2
        elif b == 0xFF and 0xD0 <= entropy[i + 1] <= 0xD7:
            segments.append(bytes(cur))
            cur = bytearray()
            i += 2
        else:
            cur.append(b)
            i += 1
    segments.append(bytes(cur))
    w, h, comps = frame
    hmax, vmax = max(c[1] for c in comps), max(c[2] for c in comps)
    mx, my = -(-w // (8 * hmax)), -(-h // (8 * vmax))
    coef = [np.zeros((my * c[2], mx * c[1], 8, 8), int) for c in comps]
    tables = [{"dc": huff[(0, td)], "ac": huff[(1, ta)]} for _, td, ta in scan]
    assert [s[0] for s in scan] == [c[0] for c in comps]
    bits, bp, seg_i = "", 0, -1

    def read(n: int) -> int:
        nonlocal bp
        v = int(bits[bp:bp + n], 2) if n else 0
        bp += n
        return v

    def decode(table: dict) -> int:
        code = 0
        for length in range(1, 17):
            code = (code << 1) | read(1)
            if (length, code) in table:
                return table[(length, code)]
        raise AssertionError("bad Huffman code")

    def extend(v: int, s: int) -> int:
        return v - (1 << s) + 1 if s and v < (1 << (s - 1)) else v

    pred = [0] * len(comps)
    for n in range(mx * my):
        if n % (ri or mx * my) == 0:
            seg_i += 1
            bits, bp, pred = "".join(f"{b:08b}" for b in segments[seg_i]), 0, [0] * len(comps)
        yy, xx = divmod(n, mx)
        for ci, (_, hs, vs, _) in enumerate(comps):
            for by in range(vs):
                for bx in range(hs):
                    blk = np.zeros(64, int)
                    s = decode(tables[ci]["dc"])
                    pred[ci] += extend(read(s), s)
                    blk[0] = pred[ci]
                    k = 1
                    while k < 64:
                        rs = decode(tables[ci]["ac"])
                        r, s = rs >> 4, rs & 15
                        if s == 0:
                            if r != 15:
                                break
                            k += 16
                            continue
                        k += r
                        blk[zz[k]] = extend(read(s), s)
                        k += 1
                    coef[ci][yy * vs + by, xx * hs + bx] = blk.reshape(8, 8)
    return coef, [qt[c[3]] for c in comps]


def _dct8() -> np.ndarray:
    """T.81 A.3.3 basis: C(u)/2 cos((2x+1) u pi / 16)."""
    u, x = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    return np.where(u == 0, 1 / np.sqrt(2), 1.0) / 2 * np.cos((2 * x + 1) * u * np.pi / 16)


def _blocks_dct(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    b = np.asarray(img, np.float64).reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    return _dct8() @ b @ _dct8().T


def _ideal_luma(coef: np.ndarray, q: np.ndarray, h: int, w: int) -> np.ndarray:
    """Exact (float) IDCT of the dequantised luma, rounded and clipped as a decoder must."""
    pix = _dct8().T @ (coef * q) @ _dct8()
    img = pix.transpose(0, 2, 1, 3).reshape(coef.shape[0] * 8, coef.shape[1] * 8)[:h, :w]
    return np.clip(np.rint(img + 128.0), 0, 255)


def _encode(img: np.ndarray, quality: int, sampling: int | None = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422,
            extra: tuple = ()) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality, *extra]
    if sampling is not None:
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, sampling]
    ok, buf = cv2.imencode(".jpg", img, params)
    assert ok
    return buf.tobytes()


def _specimen(seed: int, w: int = 320, h: int = 240) -> np.ndarray:
    """A noiseless, in-focus simulator render: the uncompressed image the camera would have coded."""
    sim = SpecimenSimulator(seed, resolve("10x / 0.25"), w, h)
    _, hz = sim.truth_patch()
    sim.focus_z_um = float(np.median(hz))
    return sim.render(tremor=False, noise=False, breathing=False)[0]


def _colour_scene(seed: int, w: int = 320, h: int = 240) -> np.ndarray:
    """Overlapping flat-coloured discs and rectangles: sharp colour edges everywhere."""
    r = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = r.integers(40, 200, 3)
    for _ in range(40):
        col = tuple(int(v) for v in r.integers(0, 256, 3))
        x, y = int(r.integers(0, w)), int(r.integers(0, h))
        if r.random() < 0.5:
            cv2.circle(img, (x, y), int(r.integers(5, 40)), col, -1, lineType=cv2.LINE_8)
        else:
            cv2.rectangle(img, (x, y), (x + int(r.integers(3, 60)), y + int(r.integers(3, 60))), col, -1)
    return img


def _jfif_ycbcr(bgr: np.ndarray) -> np.ndarray:
    """T.871 full-range conversion, float64, channel order Y, Cb, Cr (the encoder's input, before rounding)."""
    b, g, r = (bgr[..., k].astype(np.float64) for k in range(3))
    return np.dstack([0.299 * r + 0.587 * g + 0.114 * b,
                      128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
                      128 + 0.5 * r - 0.418688 * g - 0.081312 * b])


def _psnr(a: np.ndarray, ref: np.ndarray) -> float:
    return float(10 * np.log10(255.0**2 / np.mean((np.asarray(a, np.float64) - ref) ** 2)))


# ------------------------------------------------------------------------------------------ parse_jpeg

@pytest.mark.parametrize("quality", [30, 50, 75, 90])
def test_parse_returns_the_encoders_tables_in_natural_order_and_its_quality(quality):
    info = jpegq.parse_jpeg(_encode(_specimen(3), quality))
    assert (info.width, info.height) == (320, 240)
    assert [(c.h, c.v) for c in info.components] == [(2, 1), (1, 1), (1, 1)]  # 4:2:2 as the MikrOkular sends
    assert info.subsampling == [(1, 1), (2, 1), (2, 1)]
    np.testing.assert_array_equal(info.qtable(0), _libjpeg_scaled(ANNEX_K_LUMA, quality))
    np.testing.assert_array_equal(info.qtable(1), _libjpeg_scaled(ANNEX_K_CHROMA, quality))
    np.testing.assert_array_equal(info.qtable(2), _libjpeg_scaled(ANNEX_K_CHROMA, quality))
    assert info.estimated_quality() == quality
    assert info.estimated_quality(1) == quality


def test_parse_reads_420_sampling():
    info = jpegq.parse_jpeg(_encode(_specimen(3), 75, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420))
    assert [(c.h, c.v) for c in info.components] == [(2, 2), (1, 1), (1, 1)]
    assert info.subsampling == [(1, 1), (2, 2), (2, 2)]


def test_a_table_of_another_shape_has_no_ijg_quality():
    """A flat table is not a scaled Annex K table at any quality."""
    info = jpegq.parse_jpeg(_encode(_specimen(3), 75))
    info.qtables[info.components[0].tq] = np.full((8, 8), 20)
    assert info.estimated_quality() is None


def _with_awkward_markers(data: bytes) -> bytes:
    """Rewrite a libjpeg file: an APP15 segment full of 0xFF bytes, fill bytes before every header marker, and
    every 8-bit DQT table stored with 16-bit precision instead (T.81 B.2.4.1, Pq = 1)."""
    out, pos = bytearray(data[:2]), 2
    out += b"\xff\xef" + (2 + 6).to_bytes(2, "big") + b"\xff\xff\x00\xff\xd9\xff"
    while True:
        m = data[pos + 1]
        length = int.from_bytes(data[pos + 2:pos + 4], "big")
        seg = data[pos + 4:pos + 2 + length]
        if m == 0xDB:
            new, k = bytearray(), 0
            while k < len(seg):
                new.append(0x10 | (seg[k] & 15))
                for v in seg[k + 1:k + 65]:
                    new += int(v).to_bytes(2, "big")
                k += 65
            seg = bytes(new)
        out += b"\xff\xff\xff" + bytes([m]) + (len(seg) + 2).to_bytes(2, "big") + seg
        pos += 2 + length
        if m == 0xDA:
            return bytes(out) + data[pos:]


def test_parse_survives_fill_bytes_app_segments_restart_intervals_and_16_bit_tables():
    plain = _encode(_specimen(11), 60, extra=(cv2.IMWRITE_JPEG_RST_INTERVAL, 7))
    data = _with_awkward_markers(plain)
    # the rewritten file is still a valid JPEG: libjpeg decodes it to the same pixels
    np.testing.assert_array_equal(jpegq.decode_luma(data), jpegq.decode_luma(plain))
    info = jpegq.parse_jpeg(data)
    assert info.restart_interval == 7
    np.testing.assert_array_equal(info.qtable(0), _libjpeg_scaled(ANNEX_K_LUMA, 60))
    np.testing.assert_array_equal(info.qtable(1), _libjpeg_scaled(ANNEX_K_CHROMA, 60))
    assert info.estimated_quality() == 60


def test_progressive_jpeg_is_refused():
    data = _encode(_specimen(3), 75, extra=(cv2.IMWRITE_JPEG_PROGRESSIVE, 1))
    assert b"\xff\xc2" in data  # OpenCV really wrote a progressive frame
    with pytest.raises(ValueError, match="progressive"):
        jpegq.parse_jpeg(data)
    with pytest.raises(ValueError, match="progressive"):
        jpegq.restore_frame(data)


# ------------------------------------------------------------------------------------------ exact luma

def test_decode_luma_is_the_decoders_y_plane_not_a_luma_recomputed_from_bgr():
    """Against the exact IDCT of the bitstream's coefficients the decoder's Y is off by at most 1 grey level
    (libjpeg's integer IDCT). A luma recomputed from the 8-bit BGR decode loses whatever the RGB clip removed."""
    data = _encode(_colour_scene(2), 75)
    coef, qts = _reference_coefficients(data)
    ideal = _ideal_luma(coef[0], qts[0], 240, 320)
    y = jpegq.decode_luma(data).astype(np.float64)
    assert np.abs(y - ideal).max() <= 1
    assert np.mean(y != ideal) < 0.02
    via_bgr = cv2.cvtColor(jpegq.decode_bgr(data), cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float64)
    assert np.abs(via_bgr - ideal).max() >= 5  # the route decode_luma avoids is measurably worse here


@pytest.mark.parametrize("quality", [30, 50, 75])
@pytest.mark.parametrize("scene", ["specimen", "colour"])
def test_block_coefficients_recovers_the_bitstreams_quantised_coefficients(quality, scene):
    img = _specimen(29) if scene == "specimen" else _colour_scene(1)
    data = _encode(img, quality)
    coef, _ = _reference_coefficients(data)
    y = jpegq.decode_luma(data)
    got = jpegq.block_coefficients(y, jpegq.parse_jpeg(data).qtable(0))
    ok = ~jpegq.saturated_blocks(y)  # where the decoder clipped, the recovery is documented not to hold
    assert ok.mean() > 0.99
    np.testing.assert_array_equal(got[ok], coef[0][ok])


def test_saturated_blocks_are_left_as_decoded():
    """The colour scene at quality 50 has a few blocks the decoder clipped; in one of them (block 13, 8) the
    recovered DC is off by one step. deblock must not apply a constraint it cannot trust there."""
    data = _encode(_colour_scene(1), 50)
    y = jpegq.decode_luma(data)
    sat = jpegq.saturated_blocks(y)
    coef, _ = _reference_coefficients(data)
    assert sat[13, 8] and not np.array_equal(jpegq.block_coefficients(y, jpegq.parse_jpeg(data).qtable(0))[13, 8], coef[0][13, 8])
    out = jpegq.deblock(y, jpegq.parse_jpeg(data).qtable(0))
    mask = np.kron(sat, np.ones((8, 8), bool))
    np.testing.assert_array_equal(out[mask], y[mask].astype(np.float32))
    assert np.abs(out[~mask] - y[~mask]).max() > 2


# ------------------------------------------------------------------------------------------ deblocking

@pytest.mark.parametrize("seed", [3, 11, 29])
def test_deblock_brings_quality_50_luma_closer_to_the_original(seed):
    """Measured gains at quality 50 on these 320x240 renders: 0.98-1.01 dB; the bound is 0.8 dB."""
    img = _specimen(seed)
    truth = _jfif_ycbcr(img)[..., 0]
    data = _encode(img, 50)
    y = jpegq.decode_luma(data)
    out = jpegq.deblock(y, jpegq.parse_jpeg(data).qtable(0))
    assert out.dtype == np.float32 and out.shape == y.shape
    assert _psnr(out, truth) - _psnr(y, truth) >= 0.8


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_deblock_removes_most_of_the_block_edge_excess(seed):
    """Measured blockiness ratio at quality 50: 0.07-0.12; the bound is 0.25. The original has none."""
    img = _specimen(seed)
    data = _encode(img, 50)
    y = jpegq.decode_luma(data)
    assert jpegq.blockiness(_jfif_ycbcr(img)[..., 0]) < 0.1
    assert jpegq.blockiness(y) > 1.0
    assert jpegq.blockiness(jpegq.deblock(y, jpegq.parse_jpeg(data).qtable(0))) <= 0.25 * jpegq.blockiness(y)


@pytest.mark.parametrize("quality", [30, 50, 75])
def test_deblocked_image_still_codes_to_the_same_quantised_coefficients(quality):
    """Every block's DCT of the result lies in [q (c - 1/2), q (c + 1/2)] for the bitstream's own c."""
    data = _encode(_specimen(11), quality)
    coef, qts = _reference_coefficients(data)
    out = jpegq.deblock(jpegq.decode_luma(data), jpegq.parse_jpeg(data).qtable(0))
    ratio = _blocks_dct(out.astype(np.float64) - 128.0) / qts[0]
    assert np.abs(ratio - coef[0]).max() <= 0.5 + 1e-3
    assert np.abs(out - jpegq.decode_luma(data)).max() > 2  # and it did change the image


def test_blockiness_measures_block_edges():
    flat = np.full((64, 64), 100.0)
    steps = flat + np.kron(np.arange(64).reshape(8, 8) % 3, np.ones((8, 8)))  # one level per block, 0..2
    assert jpegq.blockiness(flat) == 0.0
    # differences across block edges are |0..2| steps; within blocks zero
    edges = np.abs(np.diff(steps, axis=1))[:, 7::8].ravel().tolist() + np.abs(np.diff(steps, axis=0))[7::8, :].ravel().tolist()
    assert jpegq.blockiness(steps) == pytest.approx(np.mean(edges), rel=1e-12)
    assert np.mean(edges) > 0.5


# ------------------------------------------------------------------------------------------ chroma

def _edge_band(truth_ycc: np.ndarray) -> np.ndarray:
    g = sum(np.hypot(cv2.Sobel(truth_ycc[..., k], cv2.CV_64F, 1, 0), cv2.Sobel(truth_ycc[..., k], cv2.CV_64F, 0, 1)) for k in (1, 2))
    return cv2.dilate((g > 20).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


@pytest.mark.parametrize("sampling,bound", [(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422, 0.8),
                                            (cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420, 0.75)])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_guided_chroma_is_closer_to_the_original_colour_edges_than_the_decoders(seed, sampling, bound):
    """RMS Cb/Cr error within 2 px of colour edges, quality 75. Measured ratios (guided / decoder):
    4:2:2 0.66-0.73 (bound 0.8), 4:2:0 0.62-0.69 (bound 0.75)."""
    img = _colour_scene(seed)
    truth = _jfif_ycbcr(img)
    band = _edge_band(truth)
    data = _encode(img, 75, sampling)
    info = jpegq.parse_jpeg(data)
    bgr = jpegq.decode_bgr(data)
    out = jpegq.upsample_chroma(bgr, info, jpegq.decode_luma(data))

    def err(b: np.ndarray) -> float:
        return float(np.sqrt(np.mean((_jfif_ycbcr(b)[..., 1:][band] - truth[..., 1:][band]) ** 2)))

    assert out.dtype == np.float32 and out.shape == img.shape
    assert err(out) <= bound * err(bgr)


# ------------------------------------------------------------------------------------------ restore_frame

def test_restore_frame_returns_the_deblocked_luma_and_guided_colour():
    data = _encode(_specimen(3), 50)
    r = jpegq.restore_frame(data)
    info = jpegq.parse_jpeg(data)
    y = jpegq.deblock(jpegq.decode_luma(data), info.qtable(0))
    np.testing.assert_array_equal(r["y"], y)
    expected = jpegq.upsample_chroma(jpegq.decode_bgr(data), info, np.clip(y, 0, 255))
    np.testing.assert_array_equal(r["bgr"], expected)
    assert r["info"].estimated_quality() == 50
    assert r["applied"] == ["exact decoder luma", "QCS deblock (1 iterations)", "guided chroma upsampling 2x1/2x1"]


def test_restore_frame_of_a_grayscale_jpeg_has_no_colour():
    img = cv2.cvtColor(_specimen(3), cv2.COLOR_BGR2GRAY)
    data = _encode(img, 50, sampling=None)
    r = jpegq.restore_frame(data)
    assert r["bgr"] is None
    np.testing.assert_array_equal(r["y"], jpegq.deblock(jpegq.decode_luma(data), jpegq.parse_jpeg(data).qtable(0)))


def test_chroma_that_was_not_subsampled_is_kept_and_luma_comes_from_the_guide():
    data = _encode(_colour_scene(1), 75, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)
    info = jpegq.parse_jpeg(data)
    assert info.subsampling == [(1, 1), (1, 1), (1, 1)]
    bgr = jpegq.decode_bgr(data)
    guide = np.full(bgr.shape[:2], 120.0, np.float32)
    out = _jfif_ycbcr(jpegq.upsample_chroma(bgr, info, guide))
    inner = np.all(np.abs(out - [120, 128, 128]) < 60, axis=2)  # pixels whose BGR the 0..255 clip did not touch
    assert inner.mean() > 0.5
    np.testing.assert_allclose(out[..., 0][inner], 120.0, atol=1e-3)
    np.testing.assert_allclose(out[..., 1:][inner], _jfif_ycbcr(bgr)[..., 1:][inner], atol=1e-3)
    assert jpegq.restore_frame(data)["applied"][-1] == "chroma not subsampled: decoder chroma kept"
