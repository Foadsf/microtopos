"""Optional OpenCL compute backend for the focus measures and the guided filter, with the CPU result as reference.

    uv run microtopos-bench [--size 960x540] [--reps 20] [--json FILE]

The CPU path is the default everywhere. OpenCL (OpenCV's transparent API, cv2.UMat) is used only when asked for
(ZSettings.backend = "opencl"), only for measures that have an OpenCL implementation of the SAME measure
(DoG band-pass, Tenengrad, SML; same kernels, window and border mode as focus.focus_map) and for the guided
filter, and falls back to the CPU on any OpenCL error; the result records which backend actually ran.
OpenCL is NOT claimed to be faster: on a given computer it can be slower (kernel launch and transfer costs);
the benchmark below measures it for every offered measure, with the input already on the device and
including upload and download, synchronised, after warm-up, and checks that each output agrees with the CPU
reference (and that a deliberately corrupted output does not).
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import cv2
import numpy as np

from .focus import focus_map

PARITY_REL_L2 = 1e-5


def opencl_info() -> dict:
    """What OpenCV reports about OpenCL here (device name, availability). No device is assumed from this."""
    info = {"have_opencl": bool(cv2.ocl.haveOpenCL()), "use_opencl": bool(cv2.ocl.useOpenCL())}
    if info["have_opencl"]:
        try:
            d = cv2.ocl.Device.getDefault()
            info.update(device=d.name(), vendor=d.vendorName(), driver=d.driverVersion(),
                        global_mem_mb=int(d.globalMemSize() / 2**20))
        except cv2.error as e:
            info["device_error"] = str(e)
    try:
        info["cuda_devices"] = int(cv2.cuda.getCudaEnabledDeviceCount())
    except (cv2.error, AttributeError):
        info["cuda_devices"] = 0
    return info


def _dog_umat(u: cv2.UMat, window: int, psf_sigma_px: float) -> cv2.UMat:
    s1 = max(0.8, 0.8 * psf_sigma_px)
    bp = cv2.subtract(cv2.GaussianBlur(u, (0, 0), s1), cv2.GaussianBlur(u, (0, 0), 2.5 * s1))
    return cv2.boxFilter(cv2.multiply(bp, bp), cv2.CV_32F, (window, window), borderType=cv2.BORDER_REFLECT)


def _box_umat(u: cv2.UMat, window: int) -> cv2.UMat:
    """Same box average as focus._box (mean over window x window, BORDER_REFLECT)."""
    return cv2.boxFilter(u, cv2.CV_32F, (window, window), borderType=cv2.BORDER_REFLECT)


def _tenengrad_umat(u: cv2.UMat, window: int) -> cv2.UMat:
    """Tenengrad, as focus.focus_map: squared 3x3 Sobel gradient magnitude, box-averaged over `window`."""
    gx = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
    return _box_umat(cv2.add(cv2.multiply(gx, gx), cv2.multiply(gy, gy)), window)


def _sml_umat(u: cv2.UMat, window: int) -> cv2.UMat:
    """Sum-modified Laplacian, as focus.focus_map: |[-1 2 -1]| along x plus along y, box-averaged over `window`."""
    k = np.array([[-1, 2, -1]], np.float32)
    lx = cv2.absdiff(cv2.filter2D(u, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT), 0.0)
    ly = cv2.absdiff(cv2.filter2D(u, cv2.CV_32F, k.T, borderType=cv2.BORDER_REFLECT), 0.0)
    return _box_umat(cv2.add(lx, ly), window)


# Measures with an OpenCL implementation that matches focus.focus_map (same kernels, window and border mode,
# checked to rel. L2 <= 1e-5 by the tests). Any other measure runs on the CPU and is reported as "cpu".
OPENCL_MEASURES = ("DoG band-pass", "Tenengrad", "SML")


def focus_map_backend(img: np.ndarray, measure: str, window: int, psf_sigma_px: float,
                      backend: str = "cpu") -> tuple[np.ndarray, str]:
    """(focus map, backend that actually ran). The CPU implementation (focus.focus_map) is the reference.
    OpenCL is used only when asked for, when a device exists and the measure has an OpenCL implementation of the
    SAME measure (OPENCL_MEASURES); otherwise, and on any OpenCL error, the CPU runs and "cpu" is returned."""
    used = "cpu"
    if backend == "opencl" and cv2.ocl.haveOpenCL() and measure in OPENCL_MEASURES:
        try:
            cv2.ocl.setUseOpenCL(True)
            u = cv2.UMat(img.astype(np.float32, copy=False))
            if measure == "DoG band-pass":
                out = _dog_umat(u, window, psf_sigma_px)
            elif measure == "Tenengrad":
                out = _tenengrad_umat(u, window)
            else:
                out = _sml_umat(u, window)
            return out.get(), "opencl"
        except cv2.error:
            used = "cpu"
    return focus_map(img, measure, window, psf_sigma_px), used

def _guided_filter_cpu(w: np.ndarray, p: np.ndarray, guide: np.ndarray, radius: int,
                       eps: float) -> np.ndarray:
    """Guided filter (He et al.) with per-pixel data weights w; NaN in p counts as weight 0."""
    w = np.where(np.isfinite(p), w, 0.0).astype(np.float32)
    p0 = np.where(np.isfinite(p), p, 0.0).astype(np.float32)
    I = guide.astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)

    def box(x):
        return cv2.boxFilter(x, cv2.CV_32F, k, normalize=False, borderType=cv2.BORDER_REFLECT)

    sw = box(w) + 1e-6
    mI = box(w * I) / sw
    mp = box(w * p0) / sw
    vI = box(w * I * I) / sw - mI * mI
    cIp = box(w * I * p0) / sw - mI * mp
    A = cIp / (vI + eps)
    B = mp - A * mI
    n = box(np.ones_like(I))
    support = box((w > 0).astype(np.float32)) / n
    A = box(A) / n
    B = box(B) / n
    out = A * I + B
    return np.where(support > 0.15, out, np.nan).astype(np.float32)


def _guided_filter_umat(w: np.ndarray, p: np.ndarray, guide: np.ndarray, radius: int,
                        eps: float) -> np.ndarray:
    """Same maths as _guided_filter_cpu on OpenCV UMat (OpenCL) buffers; returns a numpy array."""
    shape = guide.shape
    # NaN in p counts as weight 0 (same contract as _guided_filter_cpu): substitute before any box
    # filter so NaN cannot leak through UMat reductions.
    finite = np.isfinite(p)
    p = np.where(finite, p, 0.0).astype(np.float32)
    w = np.where(finite, w, 0.0).astype(np.float32)
    uw = cv2.UMat(w)
    uP = cv2.UMat(p)
    uI = cv2.UMat(guide)
    one = cv2.UMat(np.ones(shape, np.float32))
    eps_u = cv2.UMat(np.full(shape, eps, np.float32))
    onee6_u = cv2.UMat(np.full(shape, 1e-6, np.float32))
    k = (2 * radius + 1, 2 * radius + 1)

    def box(x):
        return cv2.boxFilter(x, cv2.CV_32F, k, normalize=False, borderType=cv2.BORDER_REFLECT)

    sw = cv2.add(box(uw), onee6_u)
    mI = cv2.divide(box(cv2.multiply(uw, uI)), sw)
    mp = cv2.divide(box(cv2.multiply(uw, uP)), sw)
    vI = cv2.subtract(cv2.divide(box(cv2.multiply(uw, cv2.multiply(uI, uI))), sw),
                      cv2.multiply(mI, mI))
    cIp = cv2.subtract(cv2.divide(box(cv2.multiply(uw, cv2.multiply(uI, uP))), sw),
                       cv2.multiply(mI, mp))
    A = cv2.divide(cIp, cv2.add(vI, eps_u))
    B = cv2.subtract(mp, cv2.multiply(A, mI))
    n = box(one)
    support = cv2.divide(box(cv2.UMat((w > 0).astype(np.float32))), n)
    A = cv2.divide(box(A), n)
    B = cv2.divide(box(B), n)
    out = cv2.add(cv2.multiply(A, uI), B)
    sp = support.get() if isinstance(support, cv2.UMat) else np.asarray(support)
    o = out.get() if isinstance(out, cv2.UMat) else np.asarray(out)
    return np.where(sp > 0.15, o, np.nan).astype(np.float32)


def guided_filter_backend(p: np.ndarray, w: np.ndarray, guide: np.ndarray, radius: int,
                          eps: float = 1e-2, backend: str = "cpu") -> tuple[np.ndarray, str]:
    """Edge-preserving weighted average of p guided by `guide`; returns (result, backend that actually ran).

    NaN in p counts as weight 0. OpenCL is used only when requested and available; any failure falls back
    to the CPU reference. p, w and guide must have the same shape.
    """
    p = np.asarray(p, np.float32)
    w = np.asarray(w, np.float32)
    guide = np.asarray(guide, np.float32)
    if p.shape != w.shape or p.shape != guide.shape:
        raise ValueError("guided_filter_backend: p, w and guide must have the same shape")
    used = "cpu"
    if backend == "opencl" and cv2.ocl.haveOpenCL():
        try:
            return _guided_filter_umat(w, p, guide, radius, eps), "opencl"
        except cv2.error:
            used = "cpu"
    return _guided_filter_cpu(w, p, guide, radius, eps), used


def parity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Relative L2 difference and maximum absolute difference over finite pixels (NaN tolerated on either
    side; all-NaN input yields NaN, NaN)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan"), float("nan")
    d = a[m] - b[m]
    return float(np.linalg.norm(d) / max(np.linalg.norm(a[m]), 1e-30)), float(np.abs(d).max())


def _median_ms(fn, reps: int, sync: bool = False) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            cv2.ocl.finish()  # synchronise: queued work is not finished work
        ts.append(time.perf_counter() - t0)
    return round(1e3 * float(np.median(ts)), 3)


def _parity_entry(ref: np.ndarray, out: np.ndarray) -> dict:
    rel, mx = parity(ref, out)
    bad_rel, _ = parity(ref, out * 1.01)  # negative control: a 1 % corruption must fail the parity test
    return {"rel_l2": rel, "max_abs": mx, "ok": bool(rel <= PARITY_REL_L2),
            "negative_control_detected": bool(bad_rel > PARITY_REL_L2)}


def benchmark(w: int = 960, h: int = 540, reps: int = 20, window: int = 9, psf: float = 1.0, threads=(1, 4, 0)) -> dict:
    """CPU vs OpenCL for every offered measure and the guided filter: timing and parity with the CPU reference.

    res["measures"][name] = {cpu_ms (default threads), opencl_resident_ms, opencl_with_transfer_ms, parity,
    opencl_faster_than_cpu}; res["cpu_threads_ms"] is the DoG CPU time per OpenCV thread count."""
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 1.2)
    res = {"size": [w, h], "reps": reps, "window": window, "opencl": opencl_info(), "cpu_threads_ms": {},
           "measures": {}, "platform": platform.platform(), "opencv": cv2.__version__}
    keep = cv2.getNumThreads()
    for t in threads:
        cv2.setNumThreads(t)
        focus_map(img, "DoG band-pass", window, psf)  # warm-up
        res["cpu_threads_ms"][str(t if t else f"default({keep})")] = _median_ms(
            lambda: focus_map(img, "DoG band-pass", window, psf), reps)
    cv2.setNumThreads(keep)
    have = res["opencl"]["have_opencl"]
    if have:
        cv2.ocl.setUseOpenCL(True)
    fns = {"DoG band-pass": lambda u: _dog_umat(u, window, psf), "Tenengrad": lambda u: _tenengrad_umat(u, window),
           "SML": lambda u: _sml_umat(u, window)}
    for name in OPENCL_MEASURES:
        ref = focus_map(img, name, window, psf)
        ent = {"cpu_ms": _median_ms(lambda n=name: focus_map(img, n, window, psf), reps)}
        if have:
            u = cv2.UMat(img)
            fns[name](u).get()  # warm-up: kernel compilation is a cold-start cost, reported apart
            t0 = time.perf_counter()
            fns[name](cv2.UMat(img)).get()
            ent["opencl_cold_ms"] = round(1e3 * (time.perf_counter() - t0), 3)
            ent["opencl_resident_ms"] = _median_ms(lambda f=fns[name], u=u: f(u), reps, sync=True)
            ent["opencl_with_transfer_ms"] = _median_ms(lambda f=fns[name]: f(cv2.UMat(img)).get(), reps)
            out = fns[name](cv2.UMat(img)).get()
            ent["parity"] = _parity_entry(ref, out)
            ent["opencl_faster_than_cpu"] = bool(ent["opencl_with_transfer_ms"] < ent["cpu_ms"])
        res["measures"][name] = ent
    # guided filter (the depth-refinement step): inputs are numpy, so OpenCL timing includes the transfers
    p = (50.0 * cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), 6)).astype(np.float32)
    p[::13, ::17] = np.nan
    wgt = rng.uniform(0.2, 1.0, (h, w)).astype(np.float32)
    ent = {"cpu_ms": _median_ms(lambda: _guided_filter_cpu(wgt, p, img, 4, 1e-2), reps)}
    if have:
        _guided_filter_umat(wgt, p, img, 4, 1e-2)
        ent["opencl_with_transfer_ms"] = _median_ms(lambda: _guided_filter_umat(wgt, p, img, 4, 1e-2), reps)
        ref = _guided_filter_cpu(wgt, p, img, 4, 1e-2)
        out = _guided_filter_umat(wgt, p, img, 4, 1e-2)
        ent["parity"] = _parity_entry(ref, out)
        ent["opencl_faster_than_cpu"] = bool(ent["opencl_with_transfer_ms"] < ent["cpu_ms"])
    res["measures"]["guided filter"] = ent
    return res


def main(argv=None):
    from . import safe_console

    safe_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json", metavar="FILE")
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    r = benchmark(w, h, a.reps)
    print(json.dumps(r, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
    bad = [n for n, m in r["measures"].items()
           if "parity" in m and not (m["parity"]["ok"] and m["parity"]["negative_control_detected"])]
    if bad:
        raise SystemExit(f"OpenCL output does not match the CPU reference for: {', '.join(bad)}")

if __name__ == "__main__":
    main()
