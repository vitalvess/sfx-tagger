"""Pastas de dados da app (fora das bibliotecas de som)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "Earmark"
OLD_APP_NAME = "SFXTagger"      # nome da pasta de dados nas versões chamadas SFX Tagger


def env(name: str, default: str | None = None) -> str | None:
    """Variável EARMARK_<name>; aceita ainda a antiga SFXTAGGER_<name>."""
    return os.environ.get(f"EARMARK_{name}") or os.environ.get(f"SFXTAGGER_{name}") or default


def app_dir() -> Path:
    home = env("HOME")              # definido pelo arrancador (e nos testes)
    if home:
        d = Path(home)
        d.mkdir(parents=True, exist_ok=True)
        return d
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    d = base / APP_NAME
    old = base / OLD_APP_NAME
    if not d.exists() and old.is_dir():
        # instalação antiga: o Python e a IA instalados guardam este caminho, por isso a
        # pasta não muda de nome (mover partia o ambiente e obrigava a descarregar ~3 GB)
        d = old
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
        base = docs if docs.is_dir() else Path.home()
        d = base / "Earmark"
        if not d.exists() and (base / "SFX Tagger").is_dir():
            d = base / "SFX Tagger"   # sons já carregados com o nome antigo: os caminhos ficam válidos
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:
        # no macOS o acesso a Documentos pode ser recusado; usa a pasta da app
        d = app_dir() / "Sons carregados"
        d.mkdir(parents=True, exist_ok=True)
        return d


def settings_file() -> Path:
    return app_dir() / "definicoes.json"
