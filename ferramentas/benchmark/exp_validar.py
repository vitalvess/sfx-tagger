"""Confirma, com o "Testar a IA" da própria app, o ganho do 3.º modelo no classificador e das
cópias com fundos, comparando o conhecimento antes e depois de ferramentas/completar_conhecimento.py.

    python ferramentas/benchmark/exp_validar.py <cópia de antes> [pasta do conhecimento atual]

Cada versão corre numa pasta da app à parte (cópia do conhecimento + definições), por isso a IA
em uso não é tocada. Precisa da app desligada (escreve o estado das pastas de teste).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTES = Path.home() / "Documents" / "Earmark Treino" / "Testes"


def medir(home: str) -> None:
    os.environ["EARMARK_HOME"] = home
    sys.path.insert(0, str(ROOT))
    from earmark import precisao
    from earmark.knowledge import get_knowledge
    from earmark.project import Project
    if precisao.enabled():
        precisao._load()
        precisao.STATE.update(state="ok")
    t0 = time.time()
    clf = get_knowledge().classifier(wait=True)
    out = {"treino_s": round(time.time() - t0), "3.º modelo no classificador": bool(clf and clf.use_third)}
    for d in sorted(p for p in TESTES.iterdir() if p.is_dir()):
        p = Project(d)
        p.scan()
        assert p.start_evaluation()
        while p.job["running"]:
            time.sleep(0.3)
        s = (p.evaluation or {}).get("summary") or {}
        out[d.name] = [s.get("top1"), s.get("top5"), s.get("n")]
    print("RESULTADO " + json.dumps(out, ensure_ascii=False), flush=True)
    os._exit(0)


def main(antes: Path, depois: Path) -> None:
    versoes = {
        "antes, normal": (antes, False), "depois, normal": (depois, False),
        "antes, Precisão máxima": (antes, True), "depois, Precisão máxima": (depois, True),
    }
    res = {}
    for nome, (conh, prec) in versoes.items():
        home = Path(tempfile.mkdtemp(prefix="validar-"))
        shutil.copytree(conh, home / "conhecimento")
        (home / "definicoes.json").write_text(json.dumps({"max_precision": prec}), encoding="utf-8")
        r = subprocess.run([sys.executable, "-X", "utf8", __file__, "medir", str(home)],
                           capture_output=True, text=True, encoding="utf-8")
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULTADO ")), None)
        if line is None:
            print(r.stdout[-2000:], r.stderr[-3000:])
            raise SystemExit(f"falhou: {nome}")
        res[nome] = json.loads(line[10:])
        print(nome, res[nome], flush=True)
    testes = [k for k in res["antes, normal"] if k.startswith("Teste")]
    print("\n1.ª sugestão certa (nas 5)")
    print(f"{'':38}" + "".join(f"{v:>26}" for v in versoes))
    for t in testes:
        print(f"{t[:38]:38}" + "".join(f"{res[v][t][0]:>18} ({res[v][t][1]:>5})" for v in versoes))
    print(f"{'média':38}" + "".join(f"{np.mean([res[v][t][0] for t in testes]):>18.1f}{'':8}" for v in versoes))
    (Path(__file__).parent / "relatorios" / "validar_3modelo_fundos.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "medir":
        medir(sys.argv[2])
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else
             Path(os.environ["LOCALAPPDATA"]) / "SFXTagger" / "conhecimento")
