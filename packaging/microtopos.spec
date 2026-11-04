# PyInstaller spec for the portable MicroTopos executables (used by packaging/build.py).
# -*- mode: python -*-
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden = collect_submodules("pyqtgraph", filter=lambda m: not m.startswith(("pyqtgraph.examples", "pyqtgraph.jupyter")))
hidden += collect_submodules("OpenGL") + collect_submodules("skimage")
datas = collect_data_files("pyqtgraph", excludes=["**/examples/**"]) + collect_data_files("skimage")

a = Analysis(
    ["entry.py"],
    hiddenimports=hidden,
    datas=datas,
    excludes=["matplotlib", "tkinter", "IPython", "pytest", "pyqtgraph.examples", "pyqtgraph.jupyter"],
)

if sys.platform.startswith("linux"):
    # Never bundle the host's graphics, X11, font or system-bus libraries: they must match the user's own
    # drivers and desktop (AppImage "excludelist" practice). Bundling Ubuntu 22.04's libX11/libxcb/libGL
    # made Mesa's GLX fail to initialise on newer systems.
    SYSTEM = (
        "libGL.so", "libGLX.so", "libGLdispatch.so", "libOpenGL.so", "libEGL.so", "libGLESv2.so", "libgbm.so",
        "libdrm.so", "libX11.so", "libX11-xcb.so", "libxcb.so", "libxcb-glx.so", "libxcb-dri2.so", "libxcb-dri3.so",
        "libxcb-present.so", "libxcb-sync.so", "libxshmfence.so", "libXext.so", "libXau.so", "libXdmcp.so",
        "libfontconfig.so", "libfreetype.so", "libexpat.so", "libdbus-1.so", "libasound.so", "libwayland-",
        "libglib-2.0.so", "libgobject-2.0.so", "libgio-2.0.so", "libgmodule-2.0.so", "libgthread-2.0.so",
        "libpcre.so", "libpcre2-8.so", "libuuid.so", "libsystemd.so", "libudev.so",
    )
    a.binaries = [b for b in a.binaries if not b[0].split("/")[-1].startswith(SYSTEM)]

pyz = PYZ(a.pure)
if sys.platform == "darwin":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="microtopos", console=False)
    coll = COLLECT(exe, a.binaries, a.datas, name="microtopos")
    app = BUNDLE(coll, name="microtopos.app", bundle_identifier="org.microtopos.MicroTopos")
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="microtopos", console=True,
              hide_console="hide-early" if sys.platform == "win32" else None)
