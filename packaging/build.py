"""Build a portable MicroTopos executable for the current platform with PyInstaller.

    uv run --with pyinstaller python packaging/build.py

Linux: one self-contained file. Windows: one .exe (console hidden when started from Explorer).
macOS: a .app bundle (the portable form on macOS), zipped.
"""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from microtopos import __version__  # noqa: E402

system = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}[platform.system()]
arch = {"x86_64": "x86_64", "AMD64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine(),
                                                                                       platform.machine())
name = f"microtopos-{__version__}-{system}-{arch}"
subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "microtopos.spec"], check=True,
               cwd=ROOT / "packaging")

dist = ROOT / "packaging" / "dist"
out = ROOT / "release"
out.mkdir(exist_ok=True)
if system == "macos":
    target = out / f"{name}.zip"
    subprocess.run(["ditto", "-c", "-k", "--keepParent", str(dist / "microtopos.app"), str(target)], check=True)
    exe = dist / "microtopos.app" / "Contents" / "MacOS" / "microtopos"
elif system == "windows":
    target = out / f"{name}.exe"
    shutil.copy2(dist / "microtopos.exe", target)
    exe = target
else:
    target = out / name
    shutil.copy2(dist / "microtopos", target)
    target.chmod(0o755)
    exe = target
print(f"built {target} ({target.stat().st_size / 1e6:.0f} MB)")
print(f"EXE={exe.as_posix()}")
