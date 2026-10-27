"""Teste de ponta a ponta de uma app já lançada (Mac ou Windows), como um colega a usaria.

    python ferramentas/ci/teste_fim_a_fim.py [--port 8765] [--sem-ia]

1. espera que o servidor responda (o arranque instala o Python e as bibliotecas);
2. espera que a instalação da IA termine (PyTorch + modelos);
3. cria uma pasta com sons sintéticos, analisa-a e confirma que há sugestões UCS;
4. grava os metadados num ficheiro e fecha a app.

Só usa a biblioteca padrão: corre com qualquer Python, fora do ambiente da app.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

BASE = ""


def api(method: str, path: str, body: dict | None = None, timeout: float = 30) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def esperar(desc: str, limite: float, teste) -> object:
    t0 = time.time()
    ultimo = ""
    while time.time() - t0 < limite:
        try:
            r = teste()
            if r:
                print(f"OK  {desc} ({time.time() - t0:.0f} s)", flush=True)
                return r
        except (urllib.error.URLError, OSError, ValueError) as e:
            msg = str(e)
            if msg != ultimo:
                print(f"    ... {desc}: {msg}", flush=True)
                ultimo = msg
        time.sleep(3)
    raise SystemExit(f"FALHOU: {desc} (mais de {limite:.0f} s)")


def som(path: Path, tipo: str, segundos: float = 2.0, sr: int = 48000) -> None:
    rnd = random.Random(tipo)
    n = int(segundos * sr)
    amostras = []
    for i in range(n):
        t = i / sr
        if tipo == "tom":
            v = 0.4 * math.sin(2 * math.pi * 440 * t) * min(1, (n - i) / (0.2 * sr))
        elif tipo == "impacto":
            v = rnd.uniform(-1, 1) * math.exp(-t * 18) * 0.9
        else:  # ruído contínuo (vento, chuva)
            v = rnd.uniform(-1, 1) * 0.3
        amostras.append(int(max(-1, min(1, v)) * 32767))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{n}h", *amostras))


def main() -> None:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--arranque", type=float, default=600, help="segundos até o servidor responder")
    ap.add_argument("--ia", type=float, default=2400, help="segundos para instalar a IA")
    ap.add_argument("--sem-ia", action="store_true", help="só verifica o arranque")
    a = ap.parse_args()
    BASE = f"http://127.0.0.1:{a.port}"

    st = esperar("servidor a responder", a.arranque, lambda: api("GET", "/api/status"))
    print(f"    versão {st['version']}, standalone={st['standalone']}")
    r = urllib.request.urlopen(BASE + "/", timeout=10).read().decode("utf-8", "replace")
    assert "<html" in r.lower(), "a página principal não carregou"
    print("OK  página principal")
    if a.sem_ia:
        return

    def ia_pronta():
        s = api("GET", "/api/status")["setup"]
        if s["state"] == "erro":
            print("\n".join(s.get("log", [])))
            raise SystemExit(f"FALHOU: instalação da IA: {s['error']}")
        return s["state"] == "ok"
    esperar("instalação da IA", a.ia, ia_pronta)

    pasta = Path(tempfile.mkdtemp(prefix="sfxtagger-teste-"))
    for tipo in ("tom", "impacto", "ruido"):
        som(pasta / f"teste_{tipo}.wav", tipo)
    api("POST", "/api/open", {"path": str(pasta)})
    api("POST", "/api/analyze", {})
    esperar("análise", 900, lambda: not api("GET", "/api/job")["job"]["running"])
    recs = api("GET", "/api/project")["records"]
    assert len(recs) == 3, f"esperava 3 sons, há {len(recs)}"
    for rec in recs:
        cats = (rec.get("suggestion") or {}).get("categories") or []
        assert rec.get("error") is None, f"{rec['path']}: {rec['error']}"
        assert cats and rec["fields"]["catid"], f"{rec['path']}: sem sugestões"
        print(f"    {rec['path']}: {rec['fields']['catid']} ({rec['fields']['fxname']})")
    print("OK  sugestões UCS")

    res = api("POST", "/api/write", {"ids": [recs[0]["id"]], "rename": False, "backup": True})["result"]
    assert res["ok"] == 1 and not res["errors"], f"gravação: {res}"
    dados = (pasta / recs[0]["path"]).read_bytes()
    assert b"iXML" in dados and b"bext" in dados, "metadados não ficaram no ficheiro"
    print("OK  metadados gravados no WAV")

    api("POST", "/api/quit", {})
    print("OK  tudo a funcionar")


if __name__ == "__main__":
    sys.exit(main())
