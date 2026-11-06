"""Precisão da IA nos sons REAIS do curso (o teste que conta).

    python ferramentas/benchmark/avaliar_curso.py colega1.sfxk colega2.sfxk ... [--meu]
    python ferramentas/benchmark/avaliar_curso.py --pasta <biblioteca> --respostas respostas.csv

Modo 1 (ficheiros .sfxk, ou --meu para o conhecimento desta instalação): cada exemplo é um
som confirmado por uma pessoa. Mede-se:
  * zero-shot: a IA sozinha, sem ter aprendido nada;
  * com aprendizagem: kNN + classificador treinados com os sons dos OUTROS colegas
    (deixa-se um autor de fora de cada vez, como um colega novo que recebe os .sfxk) e,
    se só houver um autor, com validação cruzada em 5 partes.

Modo 2 (uma pasta + CSV "ficheiro,catid" com as respostas certas, ex.: o relatório da
FOLDER04): analisa a pasta com o código da app e compara.

Escreve também um CSV com cada som, a sugestão e a resposta certa (para ver os erros).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sfxtagger import ai, classifier  # noqa: E402
from sfxtagger.ai import Tagger  # noqa: E402


def load_sfxk(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        items = json.loads(z.read("exemplos.json"))
        X1 = np.load(io.BytesIO(z.read("embeddings.npy"))).astype(np.float32)
        X2 = (np.load(io.BytesIO(z.read("embeddings2.npy"))).astype(np.float32)
              if "embeddings2.npy" in z.namelist() else None)
    for n, it in enumerate(items):
        it["x1"] = X1[n] / (np.linalg.norm(X1[n]) + 1e-9)
        v2 = X2[n] if X2 is not None else None
        it["x2"] = v2 / np.linalg.norm(v2) if v2 is not None and np.abs(v2).sum() > 0 else None
    return items


def load_local() -> list[dict]:
    from sfxtagger.knowledge import get_knowledge
    kb = get_knowledge()
    return [dict(kb.meta[i], id=i, x1=kb.emb[i], x2=kb.emb2.get(i)) for i in kb.meta]


def knn_memory(x1, X1m, catids, k=10):
    sims = X1m @ x1
    order = np.argsort(sims)[::-1][:k]
    return [(catids[i], float(sims[i])) for i in order if sims[i] >= ai.MEMORY["min_sim"]]


def evaluate(items: list[dict], tagger: Tagger) -> None:
    index = {c.catid: n for n, c in enumerate(tagger.categories)}
    items = [it for it in items if it.get("catid") in index]
    # o mesmo som confirmado por duas pessoas conta uma vez (fica a última confirmação)
    uniq = {}
    for it in items:
        uniq[it.get("id") or id(it)] = it
    items = list(uniq.values())
    n = len(items)
    if not n:
        raise SystemExit("Não há exemplos com categorias UCS válidas.")
    y = np.array([index[it["catid"]] for it in items])
    X1 = np.stack([it["x1"] for it in items])
    has2 = all(it["x2"] is not None for it in items)
    X2 = np.stack([it["x2"] for it in items]) if has2 else None
    authors = np.array([it.get("author") or "anónimo" for it in items])
    fam = np.array([tagger.categories[i].category for i in y])
    print(f"{n} sons confirmados, {len(set(y))} categorias, {len(set(fam))} categorias principais, "
          f"autores: {dict(Counter(authors))}")
    if not has2:
        print("(alguns exemplos não têm o segundo modelo: exportados antes da v0.7 — zero-shot só com o principal)")

    # partições: deixar um autor de fora, ou 5 partes se só houver um
    if len(set(authors)) >= 2:
        folds = [(authors == a) for a in sorted(set(authors))]
        how = "deixando cada autor de fora"
    else:
        rng = np.random.default_rng(0)
        part = rng.integers(0, 5, n)
        folds = [part == k for k in range(5)]
        how = "validação cruzada em 5 partes"

    res = {"zero-shot": np.zeros((n, 5), int), "com aprendizagem": np.zeros((n, 5), int)}
    conf = np.zeros(n)
    for test in folds:
        train_idx = np.where(~test)[0]
        clf = (classifier.train(X1[train_idx], X2[train_idx] if has2 else None, y[train_idx],
                                **dict(classifier.TRAIN, use_helper=has2 and classifier.TRAIN.get("use_helper", False)))
               if len(train_idx) >= classifier.MIN_EXAMPLES else None)
        if clf is not None:   # como na app: a "porta" ajustada ao conjunto de sons
            clf = classifier.adapt(clf, classifier.features(X1, X2 if clf.use_helper else None),
                                   gate_at=classifier.BLEND["gate_at"], **classifier.ADAPT)
        catids = [items[i]["catid"] for i in train_idx]
        for i in np.where(test)[0]:
            x2 = X2[i] if has2 else None
            s0 = tagger.suggest(X1[i], [], helper_emb=x2)
            s1 = tagger.suggest(X1[i], [], helper_emb=x2, clf=clf,
                                memory=knn_memory(X1[i], X1[train_idx], catids) if len(train_idx) else None)
            for name, sg in (("zero-shot", s0), ("com aprendizagem", s1)):
                res[name][i] = [index[c["catid"]] for c in sg["categories"]]
            conf[i] = s1["confidence"]["value"]

    print(f"\nPRECISÃO ({how})")
    for name, top in res.items():
        t1 = (top[:, 0] == y).mean() * 100
        t5 = (top == y[:, None]).any(1).mean() * 100
        f1 = (np.array([tagger.categories[i].category for i in top[:, 0]]) == fam).mean() * 100
        print(f"  {name:18} 1.ª sugestão {t1:5.1f}%   nas 5 {t5:5.1f}%   categoria principal certa {f1:5.1f}%")
    top = res["com aprendizagem"]
    hit = top[:, 0] == y
    print("\nCONFIANÇA (com aprendizagem): quando a IA diz...")
    for lvl in ("alta", "media", "baixa"):
        sel = np.array([ai.confidence_level(c) for c in conf]) == lvl
        if sel.any():
            print(f"  {lvl:6} acerta {100 * hit[sel].mean():5.1f}%  ({sel.sum()} sons)")
    print("\nPIORES CATEGORIAS PRINCIPAIS (com aprendizagem)")
    for f, cnt in Counter(fam).most_common():
        sel = fam == f
        if cnt >= 3:
            print(f"  {f:22} {100 * hit[sel].mean():5.1f}%  ({cnt} sons)")
    print("\nCONFUSÕES MAIS FREQUENTES (IA -> certa)")
    conf_pairs = Counter((tagger.categories[top[i, 0]].catid, items[i]["catid"]) for i in np.where(~hit)[0])
    for (a, b), c in conf_pairs.most_common(12):
        print(f"  {a:10} -> {b:10} {c}x")

    out = Path("avaliacao_curso.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ficheiro", "autor", "certa", "IA sozinha", "IA com aprendizagem", "confiança", "acertou"])
        for i, it in enumerate(items):
            w.writerow([it.get("file", ""), authors[i], it["catid"],
                        tagger.categories[res["zero-shot"][i, 0]].catid,
                        tagger.categories[top[i, 0]].catid, f"{conf[i]:.2f}", "sim" if hit[i] else "não"])
    print(f"\nDetalhe por som: {out.resolve()}")


def evaluate_folder(folder: Path, answers: Path, tagger: Tagger) -> None:
    from sfxtagger.project import Project
    truth = {}
    with open(answers, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1].strip() and not row[0].lower().startswith("ficheiro"):
                truth[Path(row[0].strip()).name.lower()] = row[1].strip()
    proj = Project(folder)
    proj.scan()
    todo = [rid for rid, r in proj.records.items() if Path(r["path"]).name.lower() in truth]
    print(f"{len(todo)} sons com resposta; a analisar…", flush=True)
    proj.start_analysis(todo, force=True)
    import time
    while proj.job["running"]:
        time.sleep(1)
    index = {c.catid: n for n, c in enumerate(tagger.categories)}
    t1 = t5 = fam = 0
    rows = []
    for rid in todo:
        r = proj.records[rid]
        right = truth[Path(r["path"]).name.lower()]
        cats = [c["catid"] for c in (r.get("suggestion") or {}).get("categories", [])]
        ok1 = bool(cats) and cats[0] == right
        t1 += ok1
        t5 += right in cats
        fam += bool(cats) and right in index and cats[0] in index and \
            tagger.categories[index[cats[0]]].category == tagger.categories[index[right]].category
        rows.append((r["path"], right, cats[0] if cats else "", ok1))
    n = max(len(todo), 1)
    print(f"1.ª sugestão {100 * t1 / n:.1f}%   nas 5 {100 * t5 / n:.1f}%   categoria principal {100 * fam / n:.1f}%")
    for path, right, got, ok in rows:
        if not ok:
            print(f"  {path}: IA {got}, certa {right}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ficheiros", nargs="*", type=Path)
    ap.add_argument("--meu", action="store_true", help="usa o conhecimento desta instalação")
    ap.add_argument("--pasta", type=Path)
    ap.add_argument("--respostas", type=Path)
    a = ap.parse_args()
    t = Tagger()
    t.ensure_loaded()
    if a.pasta:
        evaluate_folder(a.pasta, a.respostas, t)
    else:
        items = [it for f in a.ficheiros for it in load_sfxk(f)] + (load_local() if a.meu else [])
        evaluate(items, t)
