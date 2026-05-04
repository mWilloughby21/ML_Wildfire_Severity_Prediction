"""Entry point used by the PyInstaller bundle.

Sits at the project root so the ``src`` package is importable without
``python -m src.main``. When run from a frozen bundle, switches CWD to
the unpacked resource directory so that relative paths like
``data/processed/cleaned_data.csv`` resolve to the bundled copy.
"""

import os
import sys

if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)

from src.main import main

if __name__ == '__main__':
    main()
