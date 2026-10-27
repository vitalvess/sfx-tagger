"""Instalação automática dos componentes de IA (PyTorch + transformers + modelo).

A app arranca primeiro só com as bibliotecas base (servidor web, áudio). Se o PyTorch
ainda não estiver instalado, este módulo instala-o em segundo plano enquanto a
interface mostra o progresso — assim quem usa a app nunca precisa de abrir o Terminal.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import threading
import time

STATE = {"state": "ok", "step": "", "log": [], "error": "", "started": None}
_lock = threading.Lock()


def ai_installed() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def _packages() -> list[list[str]]:
    """Comandos de instalação para este sistema (cada lista é um passo)."""
    mac_intel = sys.platform == "darwin" and platform.machine() == "x86_64"
    if mac_intel:
        # o PyTorch mais recente para Mac Intel é o 2.2.2, que pede numpy 1.x e transformers 4.4x
        return [["torch==2.2.2", "numpy<2"], ["transformers>=4.40,<4.47", "numpy<2"]]
    if sys.platform == "win32":
        return [["torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
                ["transformers>=4.40"]]
    return [["torch"], ["transformers>=4.40"]]


def _installer() -> list[str]:
    uv = os.environ.get("SFXTAGGER_UV") or shutil.which("uv")
    if uv and os.path.exists(uv):
        return [uv, "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]


def _log(line: str) -> None:
    line = line.rstrip()
    if line:
        STATE["log"] = (STATE["log"] + [line])[-12:]


def _run(cmd: list[str]) -> None:
    _log("$ " + " ".join(cmd[-4:]))
    # no Windows, sem isto cada passo abriria uma janela preta por cima do browser
    flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", creationflags=flags)
    for line in proc.stdout:
        _log(line)
    if proc.wait() != 0:
        raise RuntimeError("a instalação de pacotes falhou (ver detalhes acima)")


def _install() -> None:
    try:
        if not ai_installed():
            STATE["step"] = "A instalar os componentes de IA (PyTorch, cerca de 300 MB)…"
            for pkgs in _packages():
                _run(_installer() + pkgs)
            importlib.invalidate_caches()
            if not ai_installed():
                raise RuntimeError("os componentes de IA não ficaram disponíveis")
        STATE["step"] = "A descarregar os modelos de IA (cerca de 1,4 GB, só desta vez)…"
        from .ai import get_tagger
        get_tagger().ensure_loaded()
        STATE.update(state="ok", step="Pronto")
    except Exception as e:  # noqa: BLE001 — mostrado na interface
        STATE.update(state="erro", error=str(e))
        _log(f"ERRO: {e}")


def ensure_started() -> None:
    """Se faltar a IA, começa a instalá-la (uma vez) em segundo plano."""
    with _lock:
        if STATE["state"] in ("a instalar",):
            return
        if ai_installed() and STATE["state"] == "ok":
            return
        STATE.update(state="a instalar", step="A preparar…", error="", log=[],
                     started=time.time())
    threading.Thread(target=_install, daemon=True).start()


def status() -> dict:
    s = dict(STATE)
    if s["state"] == "ok" and not ai_installed():
        s["state"] = "pendente"
    if s["started"]:
        s["elapsed"] = int(time.time() - s["started"])
    return s
