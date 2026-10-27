"""Arranca a app a partir de qualquer pasta: python run.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sfxtagger.__main__ import main  # noqa: E402

main()
