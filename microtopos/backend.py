"""Optional compute backend for one stage (the DoG focus measure), with the CPU result as the reference.

    uv run microtopos-bench [--size 960x540] [--reps 20] [--json FILE]

The CPU path is the default everywhere. OpenCL (OpenCV's transparent API, cv2.UMat) is used only when asked for
(ZSettings.backend = "opencl"), only for the DoG measure, and falls back to the CPU on any OpenCL error; the
result records which backend actually ran. The benchmark measures CPU, OpenCL with the input already on the
device, and OpenCL including upload and download, synchronised, after warm-up, and checks that the outputs
agree (and that a deliberately corrupted output does not).
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import cv2
import numpy as np

from .focus import focus_map

PARITY_REL_L2 = 1e-4


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


def focus_map_backend(img: np.ndarray, measure: str, window: int, psf_sigma_px: float,
                      backend: str = "cpu") -> tuple[np.ndarray, str]:
    """(focus map, backend that actually ran). Anything but OpenCL+DoG, or any OpenCL failure, runs on the CPU."""
    if backend == "opencl" and measure == "DoG band-pass" and cv2.ocl.haveOpenCL():
        try:
            cv2.ocl.setUseOpenCL(True)
            out = _dog_umat(cv2.UMat(img.astype(np.float32, copy=False)), window, psf_sigma_px).get()
            return out, "opencl"
        except cv2.error:
            pass
    return focus_map(img, measure, window, psf_sigma_px), "cpu"


def parity(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Relative L2 difference and maximum absolute difference."""
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.linalg.norm(d) / (np.linalg.norm(a) + 1e-30)), float(np.abs(d).max())


def benchmark(w: int = 960, h: int = 540, reps: int = 20, window: int = 9, psf: float = 1.0, threads=(1, 4, 0)) -> dict:
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 1.2)
    ref = focus_map(img, "DoG band-pass", window, psf)
    res = {"size": [w, h], "reps": reps, "opencl": opencl_info(), "cpu_ms": {}, "platform": platform.platform(),
           "opencv": cv2.__version__}
    keep = cv2.getNumThreads()
    for t in threads:
        cv2.setNumThreads(t)
        focus_map(img, "DoG band-pass", window, psf)  # warm-up
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            focus_map(img, "DoG band-pass", window, psf)
            ts.append(time.perf_counter() - t0)
        res["cpu_ms"][str(t if t else f"default({keep})")] = round(1e3 * float(np.median(ts)), 3)
    cv2.setNumThreads(keep)
    if res["opencl"]["have_opencl"]:
        cv2.ocl.setUseOpenCL(True)
        u = cv2.UMat(img)
        _dog_umat(u, window, psf).get()  # warm-up: kernel compilation is a cold-start cost, reported apart
        t0 = time.perf_counter()
        _dog_umat(cv2.UMat(img), window, psf).get()
        res["opencl_cold_ms"] = round(1e3 * (time.perf_counter() - t0), 3)
        resident, total = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            r = _dog_umat(u, window, psf)
            cv2.ocl.finish()  # synchronise: queued work is not finished work
            resident.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            out = _dog_umat(cv2.UMat(img), window, psf).get()
            total.append(time.perf_counter() - t0)
        r.get()
        res["opencl_resident_ms"] = round(1e3 * float(np.median(resident)), 3)
        res["opencl_with_transfer_ms"] = round(1e3 * float(np.median(total)), 3)
        rel, mx = parity(ref, out)
        bad_rel, _ = parity(ref, out * 1.01)  # negative control: a 1 % corruption must fail the parity test
        res["parity"] = {"rel_l2": rel, "max_abs": mx, "ok": rel < PARITY_REL_L2,
                         "negative_control_detected": bad_rel >= PARITY_REL_L2}
        best_cpu = min(res["cpu_ms"].values())
        res["opencl_faster_than_best_cpu"] = res["opencl_with_transfer_ms"] < best_cpu
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
    if "parity" in r and not (r["parity"]["ok"] and r["parity"]["negative_control_detected"]):
        raise SystemExit("OpenCL output does not match the CPU reference")


if __name__ == "__main__":
    main()
