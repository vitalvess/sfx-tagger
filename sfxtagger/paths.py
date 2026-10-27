"""Pastas de dados da app (fora das bibliotecas de som)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def app_dir() -> Path:
    if os.environ.get("SFXTAGGER_HOME"):   # definido pelo arrancador (e nos testes)
        d = Path(os.environ["SFXTAGGER_HOME"])
        d.mkdir(parents=True, exist_ok=True)
        return d
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    d = base / "SFXTagger"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = app_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def uploads_dir() -> Path:
    """Onde ficam os sons carregados pela app (arrastados para o browser)."""
    docs = Path.home() / "Documents"
    try:
        d = (docs if docs.is_dir() else Path.home()) / "SFX Tagger"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:
        # no macOS o acesso a Documentos pode ser recusado; usa a pasta da app
        d = app_dir() / "Sons carregados"
        d.mkdir(parents=True, exist_ok=True)
        return d


def settings_file() -> Path:
    return app_dir() / "definicoes.json"
