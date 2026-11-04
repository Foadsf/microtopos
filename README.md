# MicroTopos

Encoderless computational microscopy for a manual microscope with a USB eyepiece camera. Move the X-Y stage and
the view is stitched into a mosaic. Turn the fine focus through the surface and a height map, an all-in-focus
image and ISO 25178-style roughness, texture and grain statistics are computed automatically. Free software
(GPL-3.0-or-later); Linux, Windows and macOS.

## Download

Portable executables are attached to each [release](https://github.com/Foadsf/microtopos/releases); no Python
installation needed. Every executable runs its own self-test on the build machine before it is published.

| Platform | File | Requirement |
|---|---|---|
| Linux x86_64 | `microtopos-<version>-linux-x86_64` (single file; `chmod +x`, run) | glibc 2.34+ (Ubuntu 22.04, Debian 12, Fedora 35 or newer) |
| Linux arm64 (e.g. Raspberry Pi 5) | `microtopos-<version>-linux-arm64` | glibc 2.39+ (Ubuntu 24.04, Debian 13, Raspberry Pi OS Trixie) |
| Windows x86_64 | `microtopos-<version>-windows-x86_64.exe` | Windows 10/11 (also runs on Windows-on-ARM via emulation) |
| macOS Apple Silicon | `microtopos-<version>-macos-arm64.zip` (unzip, open `microtopos.app`) | macOS 14+ |
| macOS Intel | `microtopos-<version>-macos-x86_64.zip` | macOS 14+ |

The binaries are not code-signed: on macOS right-click → Open the first time; on Windows choose "More info → Run
anyway". Linux needs the usual desktop libraries (OpenGL/EGL, fontconfig) and, for camera controls, `v4l-utils`.
A native Windows-on-ARM build is not possible yet (no OpenCV wheels for that platform).
Verify a download with `microtopos --selftest --quick` (add `--gui` for the interface check).

## Run from source

```bash
uv run microtopos              # GUI; picks the microscope camera if one is connected
uv run microtopos --simulator  # GUI with the synthetic specimen only
uv run microtopos-selftest     # ground-truth checks, 3 seeds (~3 min); --quick for one seed, --gui adds a GUI run
uv run --group dev pytest      # the same checks plus an offscreen GUI smoke test
uv run microtopos-record DIR   # save the camera's raw MJPEG stream (also: "Record raw stream" in the GUI)
uv run microtopos --replay DIR # analyse a recording offline, exactly as it was captured
```

## Status (v0.6, November 2026)

Early but working. Tested with a Bresser MikrOkular Full HD eyepiece camera on Linux Mint: live capture,
automatic focus-sweep detection, height maps and stitching all run on real specimens. The algorithms are
validated against simulator ground truth (below); on real specimens they are **not yet validated
quantitatively** (no calibrated reference sample measured so far), and stitching on real footage has only been
checked visually. Windows and macOS should work (OpenCV capture fallback) but have not been tested.

## Workflow

The left panel is a checklist; each step shows a status chip.

1. **Camera**: exposure is locked (manual) and in-camera sharpening is switched off, because both corrupt focus
   measurement. The device's actual settings are read back and shown.
2. **Objective** (keys 1–4): sets NA, depth of field, focus window and the roughness S-filter.
3. **Dark & flat** (optional): 24-frame averages, stored per objective, capture size and exposure.
4. **Lateral scale**: drag the blue line over a stage micrometer interval and enter its length. Until then all
   lateral lengths are labelled NOMINAL.
5. **Z scale**: the knob has no encoder, so heights are in *focus frames* until you calibrate (knob travel during a
   sweep) or apply the experimental DOF ruler.
6. **Inspect**: move X-Y to grow the mosaic; turn the fine focus slowly from clearly below to clearly above the
   surface and stop. The sweep is detected, checked for completeness and reconstructed in the background.
   Space starts/stops a manual capture. "Repeat sweep here" measures repeatability (σ_inst).

Tabs: Live (with in-focus overlay), Mosaic (2D and 2.5D height mosaic), All-in-focus (with grain boundaries,
experimental super-resolution), Height (height / raw / confidence, profile tool), 3D, Report.
Exports: Gwyddion `.gsf`, 32-bit TIFF, binary STL, report bundle (JSON, CSV, PNG), session save/open.

## Pipeline

| Stage | Method |
|---|---|
| Capture | MJPG via V4L2 (Linux, `v4l2-ctl` for controls) or OpenCV; latest-frame pull model, no backlog |
| Motion state | phase correlation + sharpness trend on a 320 px copy, every feature normalised by its own quiet-time noise, entry/exit hysteresis, scene-cut detection |
| Mosaic | keyframe-relative tracking, sharpness gating, unbounded 512 px tiles, loop-closure least squares, relocalisation after tracking loss |
| Sweep | monotonic segmentation from z labels or from tiles that come into focus twice (reversal); pause removal when z is unknown |
| Registration | translation-only ECC between consecutive frames (parallel), chained; failed pairs inherit neighbour motion |
| Depth | focus volume (DoG band-pass default), Gaussian peak fit (log-parabola, 5 samples) on the actual z values, confidence from peak prominence and curve noise, NaN where unreliable, confidence-weighted guided filter with outlier rejection |
| All-in-focus | streaming Laplacian-pyramid fusion with softmax weights at full capture resolution |
| Metrology | polynomial form removal, ISO 16610-61 Gaussian S/L filters (σ = 0.1874 λ), Sa Sq Sp Sv Sz Ssk Sku Sdr, radial PSD slope and fractal dimension (only for 2 < β < 4), 64-level GLCM, watershed grains with true max Feret; withheld when < 30 % of pixels are reliable |
| Super-resolution | drizzle (true drop-overlap weights) of still frames with ECC sub-pixel shifts, Richardson–Lucy with a discrepancy stop; experimental |

## Verified (`microtopos-selftest`, seeds 11/22/33: 83 checks passed, 0 failed, 30 informational)

Simulator ground truth (every instrument parameter randomised per seed):
- Depth at 40×/0.85 with known z: Pearson r 0.994–0.998, height slope 0.979–0.994, RMSE 0.06–0.08 µm
  (0.1 DOF), 94–97 % valid pixels. At 10×/0.25: r 0.990–0.997, slope 0.973–0.986.
- Hand sweep with unknown z (variable speed, pause): rank correlation 0.994–0.998.
- Registration on informative frames: 0.10–0.16 px RMS. Mosaic loop: 0.50–0.77 px RMS.
- Tracking loss (jump to unmapped ground and back): every tile within about 1 px of its true place.
- No false Z sweeps in 300 still or 200 panning frames; reversals found within 2 frames; objective change detected
  for an instant switch and a turret with dark frames.
- Textureless patch: 99–100 % masked. ISO Gaussian filter: 0.500 transmission at the cutoff.
- 48 frames at 1920×1080: 2.8 s end to end on a 4-core laptop (target ≤ 15 s).

Real specimen (`--real recordings/…`, texture and noise from the Bresser camera on a metallurgical sample):
- Stitching along a known path: 40/40 frames tracked, 0.21 px RMS.
- Depth over a known synthetic relief (real texture blurred by the optical model): r 0.986, slope 0.993,
  RMSE 0.91 µm (0.09 DOF), 80 % valid.
- Camera: MJPG 1920×1080 at **12.5 fps** (it advertises 25), all frames processed; 3 min still: 2,236 frames,
  no false detections, memory flat.

## Known limits (stated, not hidden)

- Super-resolution beats a bicubic single frame on 2 of 3 seeds and ties on the third; it is marked experimental and reported, not gated, by the self-test.
- The DOF ruler's µm/frame estimate was off by −34 % to +13 % in simulation; it is labelled experimental.
- Focus breathing (magnification change with focus) is not corrected: two estimators were biased by defocus more
  than the breathing itself. Effect: about 1 px at the image edges in far-defocused frames.
- The magnification guess after an objective change is a texture-scale heuristic and is not validated; the
  prompt always offers all objectives.
- The simulator scales its relief to each objective's depth of field so every objective can be demonstrated.
- Not implemented: high-magnification insets inside a low-magnification mosaic, a motorised Z drive, multiband
  blending (feathering is used).
- OpenCV 5.0's `phaseCorrelate` modifies its inputs when given a window; all calls go through a wrapper that
  avoids this (`registration.phase_shift`).
