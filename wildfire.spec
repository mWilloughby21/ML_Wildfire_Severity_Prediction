# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Wildfire Burn Severity Predictor.
# Build with:  pyinstaller wildfire.spec --clean --noconfirm

from PyInstaller.utils.hooks import collect_all

datas = [('data/processed/cleaned_data.csv', 'data/processed')]
binaries = []
hiddenimports = [
    'src.main',
    'src.models.xgboost_model',
    'src.models.mlp_model',
    'src.models.ensemble_model',
    'src.visualization.plots',
]

for pkg in ('xgboost', 'sklearn', 'matplotlib', 'seaborn', 'pandas', 'numpy', 'scipy'):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WildfireSeverity',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
)
