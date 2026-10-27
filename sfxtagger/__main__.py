"""Arranque: python -m sfxtagger [--port 8765] [--no-browser]"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser


def _free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("Não encontrei nenhuma porta livre")


def _open_when_ready(port: int, url: str, timeout: float = 1800.0) -> None:
    """Abre o browser só quando o servidor já responde. Abrir antes (um tempo fixo) falhava
    em computadores mais lentos: o browser mostrava "não foi possível abrir a página".
    O primeiro arranque num Mac pode demorar minutos (o macOS verifica cada biblioteca nova)."""
    import time
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2):
                print(f"  Servidor pronto em {time.time() - t0:.0f} s; a abrir o browser.", flush=True)
                webbrowser.open(url)
                return
        except Exception:
            time.sleep(0.5)
    print("  O servidor não respondeu a tempo; o browser não foi aberto.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="SFX Tagger")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    # Sem janela de consola (pythonw no Windows, app de Mac): tudo vai para um ficheiro de registo
    if sys.stdout is None or sys.stderr is None:
        from . import paths
        log = open(paths.app_dir() / "registo-app.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log

    import platform
    import time
    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} · Python {platform.python_version()} "
          f"· {platform.machine()} · {platform.platform()}", flush=True)

    import uvicorn

    from . import __version__, setup_ai
    print(f"  SFX Tagger {__version__}", flush=True)
    if not setup_ai.ai_installed():
        print("  Primeira utilização: a instalar os componentes de IA em segundo plano…")
        setup_ai.ensure_started()

    port = _free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    print(f"\n  SFX Tagger a correr em {url}")
    print("  Fecha esta janela para terminar.\n")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(port, url), daemon=True).start()
    uvicorn.run("sfxtagger.server:app", host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
