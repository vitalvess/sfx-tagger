"""Treino credível: o que acontece se retirarmos (ou reduzirmos) as fontes menos fiáveis?

    python ferramentas/benchmark/exp_niveis.py            (corre todas as versões e compara)

Níveis das fontes (autor dos exemplos):
  prata  - curadas à mão: ESC-50, FSD50K, BBC, Nonspeech7k, VocalSound (e cópias do ESC-50)
  bronze - etiqueta por gravação ou por corte automático: TUT 2018, CochlScene, UrbanSound8K
Cada versão é uma cópia filtrada do conhecimento numa pasta à parte (a IA em uso não é tocada),
medida com o "Testar a IA" da própria app nas 6 pastas de teste.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONHECIMENTO = Path(os.environ.get("LOCALAPPDATA", "")) / "SFXTagger" / "conhecimento"
TESTES = Path.home() / "Documents" / "Earmark Treino" / "Testes"
BRONZE = {"US8K", "TUT2018", "CochlScene"}
VERSOES = {
    "atual": lambda m: True,
    "sem bronze": lambda m: m.get("author") not in BRONZE,
    "bronze reduzido (30/categoria)": None,   # tratado em filtrar()
}


def filtrar(nome: str, destino: Path) -> int:
    meta = json.loads((CONHECIMENTO / "exemplos.json").read_text(encoding="utf-8"))
    if nome == "bronze reduzido (30/categoria)":
        conta: dict = defaultdict(int)
        keep = {}
        for i, m in meta.items():
            if m.get("author") in BRONZE:
                key = (m.get("author"), m.get("catid"))
                conta[key] += 1
                if conta[key] > 30:
                    continue
            keep[i] = m
    else:
        keep = {i: m for i, m in meta.items() if VERSOES[nome](m)}
    d = destino / "conhecimento"
    d.mkdir(parents=True)
    (d / "exemplos.json").write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    for f in ("embeddings.npz", "embeddings2.npz"):
        with np.load(CONHECIMENTO / f) as z:
            np.savez_compressed(d / f, **{k: z[k] for k in z.files if k in keep})
    return len(keep)


def medir(home: str) -> None:
    """Corre num processo próprio (o conhecimento da app é um singleton por pasta)."""
    os.environ["EARMARK_HOME"] = home
    sys.path.insert(0, str(ROOT))
    from earmark.knowledge import get_knowledge
    from earmark.project import Project
    t0 = time.time()
    get_knowledge().classifier(wait=True)
    out = {"treino_s": round(time.time() - t0)}
    for d in sorted(p for p in TESTES.iterdir() if p.is_dir()):
        p = Project(d)
        p.scan()
        assert p.start_evaluation()
        while p.job["running"]:
            time.sleep(0.3)
        s = (p.evaluation or {}).get("summary") or {}
        out[d.name] = {k: s.get(k) for k in ("n", "known", "top1", "top5", "family")}
        out[d.name]["alta"] = next((l for l in s.get("levels", []) if l["level"] == "alta"), None)
    print("RESULTADO " + json.dumps(out, ensure_ascii=False), flush=True)
    os._exit(0)


def main() -> None:
    res = {}
    for nome in VERSOES:
        home = Path(tempfile.mkdtemp(prefix="niveis-"))
        n = filtrar(nome, home)
        print(f"{nome}: {n} exemplos", flush=True)
        r = subprocess.run([sys.executable, "-X", "utf8", __file__, "medir", str(home)],
                           capture_output=True, text=True, encoding="utf-8")
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULTADO ")), None)
        if line is None:
            print(r.stdout[-2000:], r.stderr[-3000:])
            raise SystemExit(f"falhou: {nome}")
        res[nome] = dict(json.loads(line[10:]), exemplos=n)
    testes = [k for k in res["atual"] if k.startswith("Teste")]
    print("\n1.ª sugestão certa (nas 5)")
    print(f"{'':42}" + "".join(f"{v[:22]:>26}" for v in res))
    for t in testes:
        print(f"{t[:42]:42}" + "".join(f"{res[v][t]['top1']:>17} ({res[v][t]['top5']:>5})" for v in res))
    for label, subset in (("média dos 6", testes),
                          ("média sem testes bronze", [t for t in testes if not any(b in t for b in ("TUT", "CochlScene", "UrbanSound"))])):
        print(f"{label:42}" + "".join(f"{np.mean([res[v][t]['top1'] for t in subset]):>17.1f}{'':9}" for v in res))
    print("confiança alta no Teste fixo:", {v: res[v].get("Teste fixo (copia das partes teste)", {}).get("alta") for v in res})
    (Path(__file__).parent / "relatorios" / "niveis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "medir":
        medir(sys.argv[2])
    else:
        main()
