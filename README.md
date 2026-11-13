# MicroTopos

Encoderless computational microscopy for a manual microscope with a USB eyepiece camera. Move the X-Y stage and
the view is stitched into a mosaic. Turn the fine focus through the surface and a height map, an all-in-focus
image and ISO 25178-style roughness, texture and grain statistics are computed automatically. Free software
(GPL-3.0-or-later); Linux, Windows and macOS.

**Website, manual and tutorials: <https://foadsf.github.io/microtopos/>**

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
uv run microtopos-selftest     # ground-truth checks, 3 seeds; --quick for one seed, --gui adds a GUI run,
                               # --json FILE writes every row with thresholds and versions
uv run --group dev pytest      # the same checks, pipeline tests and an offscreen GUI smoke test
uv run --group dev pytest -m slow  # full generalisation matrix (tests/test_gen_*.py: seeds x objectives x
                                   # conditions, negative controls); the default run keeps a subset
uv run microtopos-record DIR   # record the camera (also: "Record raw stream" in the GUI)
uv run microtopos --replay DIR # replay a recording at its recorded timing (frames may be skipped under load)
uv run microtopos-batch DIR OUT  # analyse EVERY frame of a recording in order, reconstruct each sweep
uv run microtopos-bench        # CPU vs OpenCL focus-measure timing with a parity check, on this computer
```

Python 3.12 or newer (CI tests 3.12 and 3.13; development runs on 3.14). On Python 3.11 the Qt bindings abort with a reference-count error, so 3.11 is not supported.

## Status (v0.8, November 2026)

v0.8.1 adds a user-editable objective list (step 2, "Edit objectives…") with presets, including the Leitz NPL
5×/0.09, 10×/0.20, 20×/0.40 and Fluotar 50×/0.85 set.

Early but working. Tested with a Bresser MikrOkular Full HD eyepiece camera on Linux Mint: live capture,
automatic focus-sweep detection, height maps and stitching all run on real specimens. The algorithms are
validated against simulator ground truth (below); on real specimens they are **not yet validated
quantitatively** (no calibrated reference sample measured so far), and stitching on real footage has only been
checked visually. The Windows and macOS builds pass their self-tests on the build servers but have not been run
on a real desktop.

v0.8 follows an external source review (November 2026). It adds:

- a self-test gate that cannot pass a failing NumPy verdict;
- signed hand-sweep heights with source-frame spacing;
- result, calibration and mosaic identity;
- unit-safe mosaics and repeatability;
- validated recordings with a deterministic batch mode;
- atomic exports with validity masks;
- versioned sessions;
- several metrology edge-case fixes.

## Workflow

The left panel is a checklist; each step shows a status chip.

1. **Camera**: exposure is locked (manual) and in-camera sharpening, automatic white balance and gain are switched
   off, because they corrupt focus measurement. The settings are re-applied by the capture thread after the stream
   starts and after a reconnect, and the device's actual values are read back and shown. "Set once" picks a manual
   exposure just below saturation. **View rotation** turns the 2D views when the camera sits rotated in the
   eyepiece tube; "Measure from X pan" sets it from a move of the X knob alone (display only).
2. **Objective** (keys 1–9, in list order): sets NA, depth of field, focus window and the roughness S-filter.
   *Edit objectives…* holds the objectives on your nosepiece (name, magnification, NA, immersion index, cover-glass
   note, maker/series) with validation, or loads a preset such as Leitz NPL 5×/10×/20×/50× Fluotar. The list is
   saved as `objectives.json` next to the calibration; calibrations stay with the objective's name (renaming
   carries them along, removing keeps them). The lateral scale still needs a stage micrometer for each objective.
3. **Dark & flat** (optional): 24-frame averages, stored per objective, capture size and exposure.
4. **Lateral scale**: drag the blue line over a stage micrometer interval and enter its length. Until then all
   lateral lengths are labelled NOMINAL.
5. **Z scale**: the knob has no encoder. A hand sweep's heights are *relative focus progression* in camera frames
   (constant knob speed assumed), with an unknown direction unless you state it. *Set Z…* binds the knob travel
   and direction to one sweep (CALIBRATED, that sweep only). A stored µm-per-frame value, or the experimental DOF
   ruler, is applied to later sweeps as ESTIMATED.
6. **Inspect**: move X-Y to grow the mosaic; turn the fine focus slowly from clearly below to clearly above the
   surface and stop. The sweep is detected, checked for completeness and reconstructed in the background
   (or only kept, with "Reconstruct sweeps automatically" off; "Reconstruct last sweep" builds it on demand).
   The view stays on the current tab; the result line under the coach describes the selected map (its own
   objective, X-Y and Z status, valid area, warnings), separately from the live capture in the header.
   Space starts/stops a manual capture. "Repeat sweep here" measures repeatability (σ_inst).

Tabs: Live (with in-focus outlines and magenta stripes on clipped pixels; "Move slower" hint on fast pans), Mosaic (image mosaic with the stitched height maps as an adjustable colour
overlay, or heights alone; live camera box), All-in-focus (with grain boundaries,
experimental super-resolution), Height (height / raw / confidence, profile tool), 3D, Report.
Exports: Gwyddion `.gsf`, 32-bit TIFF (NaN = no depth), binary STL (visualisation), each with a validity mask and
a provenance JSON; report bundle (strict JSON, CSV, PNG, mask), written completely or not at all; versioned
session files (schema 2), also for a mosaic alone.

## Pipeline

| Stage | Method |
|---|---|
| Capture | MJPG via V4L2 (Linux, `v4l2-ctl` for controls) or OpenCV; latest-frame pull model for the live view; every frame keeps its source sequence number and time |
| Recording / replay | one writer (MJPEG bytes, or lossless PNG when the backend decodes), checked writes, a manifest with per-frame file, sequence, time, size and CRC; replay validates the manifest, skips damaged frames as gaps; an ordered mode with back-pressure for exhaustive, deterministic analysis (`microtopos-batch`) |
| Motion state | phase correlation + sharpness trend on a 320 px copy, every feature normalised by its own quiet-time noise, entry/exit hysteresis, scene-cut detection that first checks for a fast (blurred) pan with a band-pass match |
| Mosaic | band-pass phase correlation refined by ECC (motion blur splits the phase-only peak), verified by normalised cross-correlation, motion-predicted acceptance of weaker matches, keyframe pose graph with consistency-checked loop closures, unbounded 512 px tiles, relocalisation after tracking loss, separate areas when a view cannot be placed |
| Height mosaic | every height map pinned to its keyframe (moves with pose-graph corrections) in the mosaic it was captured in; least-squares levelling and feathered blending only among maps of the same height quantity (unit, sign, scale basis); others are kept as separate groups; .gsf export with mask; display rendering bounded to ~1600 px; sparse pose-graph solve |
| Sweep | monotonic segmentation from z labels or from tiles that come into focus twice (reversal); pause removal when z is unknown; hand-sweep z = source frames of knob movement × the sweep's scale × its stated sign |
| Registration | translation-only ECC between consecutive frames (parallel), chained; failed pairs inherit ECC-valid neighbour motion; unobservable motion (no texture) is zero, never an invented shift; pixels not observed in every frame are excluded from depth |
| Depth | focus volume (DoG band-pass default), Gaussian peak fit (log-parabola, 5 samples) on the actual z values, confidence from peak prominence and curve noise, NaN where unreliable, confidence-weighted guided filter with outlier rejection |
| All-in-focus | streaming Laplacian-pyramid fusion with softmax weights at full capture resolution |
| Metrology | polynomial form removal (0 = mean only … 3 = cubic), ISO 16610-61 Gaussian S/L filters (σ = 0.1874 λ; refused below 4 px per cut-off), Sa Sq Sp Sv Sz Ssk Sku, triangulated Sdr (µm only), Sk Spk Svk Smr1 Smr2 (40 % secant, tie-safe crossings), Sal Str Std (lag-support-aware autocorrelation), radial PSD slope and fractal dimension (only for 2 < β < 4); height groups withheld when < 30 % of pixels are reliable, with the reason per group. GLCM texture and h-minima watershed regions (Feret, Heyn intercept, ASTM E112-style G) come from the all-in-focus image and are labelled image-based |
| Super-resolution | drizzle (true drop-overlap weights) onto a 2× (linear) grid of still frames with ECC sub-pixel shifts, Richardson–Lucy with a discrepancy stop; frames without a valid registration are dropped, and too little sub-pixel motion is refused; experimental |
| Compute | CPU reference; optional OpenCL DoG focus measure (`ZSettings.backend`), CPU fallback, parity-checked; jobs check their memory need against available RAM before starting |

## Verified (`microtopos-selftest`, seeds 11/22/33: 130 checks passed, 0 failed, 30 informational)

Measured 2026-11-05 on an i5-8365U laptop (4 cores, Linux Mint, Python 3.14, NumPy 2.5, OpenCV 5.0);
`--json FILE` records every row with thresholds and versions. The gate counts every failing verdict, NumPy booleans
included; an exception, an empty check or a non-boolean verdict is a failure; CI checks that a deliberately failing
row makes the command exit 1.

Simulator ground truth (every instrument parameter randomised per seed):
- Depth at 40×/0.85 with known z: Pearson r 0.994–0.998, height slope 0.979–0.994, RMSE 0.07–0.08 µm
  (0.1 DOF), 94–97 % valid pixels. At 10×/0.25: r 0.990–0.996, slope 0.972–0.985.
- Signed step (two terraces, +1.60 µm, rendered with a blur law independent of the reconstruction's optics model):
  known z up and down, a hand sweep with the direction stated, the same frames reversed, and every other frame
  kept all give +1.60 µm (±0.003). A reversed sweep without a stated direction is flagged as unknown orientation;
  the old index spacing, kept as a negative control, halves the step (+0.80 µm).
- Hand sweep with unknown z (variable speed, pause) and the direction stated: signed rank correlation 0.994–0.998.
- Registration on informative frames: 0.09–0.16 px RMS; identical blank frames give no motion (the old code
  returned half the frame size) and super-resolution refuses them. Mosaic loop: 0.45–0.89 px RMS.
- Tracking loss (jump to unmapped ground and back): every tile within about 1.1 px of its true place.
- Fast hand pans with motion blur (30 and 60 px/frame at 640 px): no false view changes, one mosaic area, end
  position error 0.5–1.2 px. At 90 px/frame tracking holds but drifts 8–21 px (measured for v0.7, not rerun).
- Metrology on synthetic surfaces: Voronoi grain count within 25 %, 50 µm grid intercept within 5 µm, Sdr exact on a
  tilted plane, Std of grooves along their direction, Str < 0.3 for grooves and > 0.5 isotropic, plateau surface
  Svk > 3 Spk. Edge cases: patterned missing pixels withhold Sal/Str; a flat field gives no Std and no numeric
  warning; form order 0 keeps the tilt; Gaussian transmission 0.502 at 4 px and 0.500 at 8 px, refused below 4 px;
  an exact and a 1e-6-perturbed plateau give the same Smr2 (90 %) and Svk; Vollath F4/F5 have no wrap-around;
  the DOF ruler gives no estimate on a blank crop.
- µm and focus-frame maps are never blended in the height mosaic; "um" and "µm" are the same unit.
- No false Z sweeps in 300 still or 200 panning frames; reversals found within 2 frames; objective change detected
  for an instant switch and a turret with dark frames.
- Textureless patch: 91–97 % masked. ISO Gaussian filter: 0.500 transmission at the cutoff.
- 48 frames at 1920×1080: 4.1 s end to end on this laptop (target ≤ 15 s; 2.8–4.2 s in earlier runs).

Pipeline tests (`pytest`, 59 tests): recordings (raw JPEG and decoded PNG modes, a non-JPEG buffer refused, a
non-empty folder refused, a failed write visible with the manifest marked incomplete), legacy manifests with a
count mismatch, damaged replay frames as sequence gaps with the engine still running, an engine error that does not
stop the engine, dark frames cancelled by a source or size change, a late height map not pinned into a restarted
mosaic, a source switch that does not skip the new source's first frame, exact keyframe correction, strict report
JSON, an encoder failure that leaves no partial bundle, session round trip (original arrays and warnings) and
validation, export masks and provenance, a deterministic batch run, and sign/scale rescaling.

Real specimen (`--real recordings/…`, texture and noise from the Bresser camera on a metallurgical sample):
- Stitching along a known path: 40/40 frames tracked, 0.21 px RMS.
- Depth over a known synthetic relief (real texture blurred by the optical model): r 0.986, slope 0.993,
  RMSE 0.91 µm (0.09 DOF), 80 % valid.
- Camera: MJPG 1920×1080 at **12.5 fps** (it advertises 25), all frames processed; 3 min still: 2,236 frames,
  no false detections, memory flat. These real-camera numbers are from v0.6/v0.7 runs; the recording itself is
  not in the repository.

## Known limits (stated, not hidden)

- Manual Z: a hand sweep assumes a constant knob speed while moving; its scale holds for that sweep only, and its
  sign is unknown unless stated. Only a measured Z (encoder, motorised drive) removes this; none is supported yet.
- Physical accuracy, uncertainty and ISO/ASTM equivalence are not established: no certified flatness, step-height,
  pitch-grid or roughness reference has been measured, and no clause-by-clause standards audit was done. The
  simulator shares the reconstruction's optics model; the new signed-step control uses an independent blur law.
- The optional OpenCL focus measure matched the CPU (relative L2 1e-7) and was 4 % faster than the best CPU
  setting for one 960×540 frame on an Intel UHD 620 (i5-8365U laptop). That is not a measured end-to-end gain,
  so the CPU stays the default. CUDA, Vulkan, SYCL and NPU paths are not implemented.
- The reconstruction still holds the whole processing stack in memory (now one preallocated stack per stage and an
  admission check); tiled or spilled processing for very long sweeps is not implemented.
- The GUI has not had a usability study, a keyboard-only walkthrough or native high-DPI checks; the offscreen smoke
  test drives it programmatically only. The window fits 1280×720 with both panels.

- Super-resolution beat a bicubic single frame on all three seeds in this run (nRMSE 0.29–0.46 vs 0.37–0.50) but
  tied on one seed earlier; it is experimental, reported and not gated. Its 2× is per axis (4× the pixels), and a
  real resolution gain on optical targets has not been measured.
- The DOF ruler's µm/frame estimate was off by −34 % to +13 % in simulation; it is stored as an ESTIMATED scale.
- Focus breathing (magnification change with focus) is not corrected: two estimators were biased by defocus more
  than the breathing itself. Effect: about 1 px at the image edges in far-defocused frames.
- The magnification guess after an objective change is a texture-scale heuristic and is not validated; the
  prompt always offers all objectives in the list.
- The simulator scales its relief to each objective's depth of field so every objective can be demonstrated.
- Not implemented: high-magnification insets inside a low-magnification mosaic, a motorised Z drive, multiband
  blending (feathering is used).
- OpenCV 5.0's `phaseCorrelate` modifies its inputs when given a window; all calls go through a wrapper that
  avoids this (`registration.phase_shift`).
