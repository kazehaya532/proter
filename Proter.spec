# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Proter beta executable.

Folder distribution (onedir): dist/Proter/ contains Proter.exe, _internal/,
and (copied after the build) the YOLO weights + calibration file so the app
runs offline. Build with:

    .venv\\Scripts\\pyinstaller Proter.spec --noconfirm --clean
"""
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("ultralytics",):          # bundle YOLO configs/trackers + torch deps
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["proter_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Proter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                    # GUI app: no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Proter",
)
