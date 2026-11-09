"""Testar a IA com sons já classificados: compara o que ela ouve com a resposta certa.

A resposta certa vem, por esta ordem, de:
  * respostas.csv na pasta ("ficheiro,catid"; várias respostas aceites separadas por | );
  * do CatID gravado dentro do WAV, ou do nome do ficheiro se seguir o padrão UCS.
O nome do ficheiro nunca é dado à IA neste teste: tem de acertar só pelo que ouve.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import metadata, paths, ucs

ANSWERS_FILE = "respostas.csv"
HISTORY_KEEP = 50


def load_answers(root: Path) -> dict[str, list[str]]:
    f = root / ANSWERS_FILE
    out: dict[str, list[str]] = {}
    if not f.exists():
        return out
    with open(f, newline="", encoding="utf-8-sig") as h:
        for row in csv.reader(h):
            if len(row) >= 2 and row[1].strip() and row[0].strip().lower() != "ficheiro":
                ids = [c.strip() for c in row[1].split("|") if ucs.lookup(c.strip())]
                if ids:
                    out[Path(row[0].strip()).name.lower()] = ids
    return out


def truth_of(path: Path, answers: dict[str, list[str]]) -> list[str]:
    """As categorias aceites para este som, lidas do ficheiro (nunca do rascunho da IA)."""
    if path.name.lower() in answers:
        return answers[path.name.lower()]
    try:
        catid = metadata.read_metadata(path)["fields"].get("catid", "")
    except (OSError, metadata.MetadataError):
        catid = ""
    catid = catid or ucs.parse_filename(path.stem).get("catid", "")
    return [catid] if ucs.lookup(catid) else []


def score(cats: list[dict], truth: list[str]) -> dict:
    ids = [c["catid"] for c in cats]
    families = {ucs.lookup(t).category for t in truth}
    return {"top1": bool(ids) and ids[0] in truth, "top5": any(i in truth for i in ids),
            "family": bool(ids) and ucs.lookup(ids[0]).category in families}


def summarize(rows: list[dict], no_label: int, known: int, errors: int) -> dict:
    n = len(rows)
    pct = lambda k, rs: round(100 * sum(r[k] for r in rs) / len(rs), 1) if rs else None  # noqa: E731
    levels = []
    for level in ("alta", "media", "baixa"):
        rs = [r for r in rows if r["level"] == level]
        levels.append({"level": level, "n": len(rs), "right": pct("top1", rs)})
    fam = defaultdict(list)
    for r in rows:
        fam[ucs.lookup(r["truth"][0]).category].append(r)
    families = sorted(({"family": f, "n": len(rs), "right": pct("top1", rs)} for f, rs in fam.items() if len(rs) >= 3),
                      key=lambda x: (x["right"], -x["n"]))
    confusions = Counter((r["ai"], r["truth"][0]) for r in rows if not r["top1"])
    return {"n": n, "no_label": no_label, "known": known, "errors": errors,
            "top1": pct("top1", rows), "top5": pct("top5", rows), "family": pct("family", rows),
            "levels": levels, "families": families[:12],
            "confusions": [{"ai": a, "real": b, "n": c} for (a, b), c in confusions.most_common(10)]}


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Ficheiro", "Resposta certa", "1.ª sugestão da IA", "Confiança", "Acertou à 1.ª", "Nas 5 sugestões",
                "Categoria principal certa"])
    yes = {True: "sim", False: "não"}
    for r in rows:
        w.writerow([r["file"], " | ".join(r["truth"]), r["ai"], f"{r['confidence']:.2f}", yes[r["top1"]],
                    yes[r["top5"]], yes[r["family"]]])
    return buf.getvalue()


def _history_file() -> Path:
    return paths.app_dir() / "testes.json"


def history() -> list[dict]:
    try:
        return json.loads(_history_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def remember(summary: dict, folder: str, examples: int, models: int) -> None:
    """Guarda o resultado, para se ver se a IA melhora depois de treinar."""
    h = (history() + [{"at": time.strftime("%Y-%m-%d %H:%M"), "folder": folder, "n": summary["n"],
                       "top1": summary["top1"], "top5": summary["top5"], "family": summary["family"],
                       "examples": examples, "models": models}])[-HISTORY_KEEP:]
    f = _history_file()
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(h, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, f)
