# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file  BEAC Anomaly Detector
#
# Build command:
#   pyinstaller BEACDetect.spec
#
# Output: dist/BEACDetect.exe
#
# Requirements:
#   pip install pyinstaller>=6.0
#   pip install kaleido  (for PNG figure export)
#
# Notes:
# - Qt WebEngine DLLs are collected automatically by the PyQt6 hook.
# - If building CPU-only, uncomment the torch CUDA exclude lines.
# - Adjust `data` paths if running from a different working directory.

import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# ── Collect all data / binaries from key packages ────────────────────────────
dash_data,       dash_bins,       dash_hid       = collect_all("dash")
plotly_data,     plotly_bins,     plotly_hid     = collect_all("plotly")
statsm_data,     statsm_bins,     statsm_hid     = collect_all("statsmodels")
shap_data,       shap_bins,       shap_hid       = collect_all("shap")
kaleido_data,    kaleido_bins,    kaleido_hid    = collect_all("kaleido")
waitress_data,   waitress_bins,   waitress_hid   = collect_all("waitress")

# ── Analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=dash_bins + plotly_bins + statsm_bins + shap_bins + kaleido_bins + waitress_bins,
    datas=[
        # Project data
        ("data", "data"),
        ("assets", "assets"),
        ("models", "models"),
        # Package data
        *dash_data,
        *plotly_data,
        *statsm_data,
        *shap_data,
        *kaleido_data,
        *waitress_data,
    ],
    hiddenimports=[
        # PyQt6 WebEngine (must be explicit)
        "PyQt6.QtWebEngineWidgets",
        "PyQt6.QtWebEngineCore",
        "PyQt6.QtWebEngineQuick",
        # Dash internals
        "dash.dcc",
        "dash.html",
        "dash._dash_renderer",
        "flask",
        "flask.templating",
        "werkzeug.sansio.utils",
        # PyTorch
        "torch",
        "torch.nn",
        "torch.optim",
        "torch.utils.data",
        # Project packages
        "beac_lof",
        "beac_lof.pipeline",
        "beac_lof.graphiques",
        "bivat",
        "bivat.pipeline",
        "bivat.graphiques",
        "utils",
        # ML
        "sklearn.neighbors",
        "sklearn.decomposition",
        "sklearn.impute",
        # Submodule sweeps
        *dash_hid,
        *plotly_hid,
        *statsm_hid,
        *shap_hid,
        *kaleido_hid,
        *waitress_hid,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused large modules to reduce bundle size
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "setuptools",
        # Exclude CUDA if building CPU-only:
        # "torch.cuda",
        # "torch.backends.cuda",
        # "torch.backends.cudnn",
    ],
    noarchive=False,
    optimize=1,
    cipher=block_cipher,
)

# ── PYZ ──────────────────────────────────────────────────────────────────────
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── EXE ──────────────────────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # COLLECT mode  smaller exe, data in dist/ folder
    name="BEACDetect",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_dir=None,
    console=False,           # no console window in production
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.ico" if sys.platform == "win32" else None,
    version_file=None,
)

# ── COLLECT  bundle everything into dist/BEACDetect/ ────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BEACDetect",
)
