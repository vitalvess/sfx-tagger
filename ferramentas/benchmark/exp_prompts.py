"""Experiências de afinação do texto das categorias e da pontuação (zero-shot).

    python ferramentas/benchmark/exp_prompts.py <pasta ESC-50> [modelo]

Afina em folds 1-3 e reporta também em folds 4-5 (nunca usados para escolher).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench import Data, metrics, per_class  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sfxtagger.ai import MODEL_NAME, NON_SOUND_CATEGORIES, Tagger  # noqa: E402


def norm(m):
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


def template_matrices(tagger: Tagger) -> dict[str, np.ndarray]:
    cats = tagger.categories
    groups: dict[str, list[list[str]]] = {k: [] for k in
                                          ("atual", "nome", "explic", "sinon_lista", "sinon_cada", "sfx", "familia")}
    for c in cats:
        cat, sub = c.category.lower(), c.subcategory.lower()
        name = cat if sub in ("misc", "general") else f"{sub} {cat}"
        groups["atual"].append(c.prompts())
        groups["nome"].append([f"the sound of {name}"])
        groups["explic"].append([c.explanation or f"the sound of {name}"])
        groups["sinon_lista"].append(["the sound of " + ", ".join(c.synonyms[:6])] if c.synonyms
                                     else [f"the sound of {name}"])
        groups["sinon_cada"].append([f"the sound of {s}" for s in c.synonyms[:8]] or [f"the sound of {name}"])
        groups["sfx"].append([f"{name} sound effect"])
        groups["familia"].append([f"the sound of {cat}"])
    out = {}
    for k, g in groups.items():
        flat = [p for ps in g for p in ps]
        e = tagger.embed_text(flat)
        rows, i = [], 0
        for ps in g:
            rows.append(norm(e[i:i + len(ps)].mean(0)))
            i += len(ps)
        out[k] = np.stack(rows)
        print(f"  textos '{k}': {len(flat)}", flush=True)
    return out


def main(esc: Path, model: str = MODEL_NAME):
    data = Data(esc, model)
    tagger = Tagger(model)
    tagger.ensure_loaded()
    sugg = np.array([c.category not in NON_SOUND_CATEGORIES for c in tagger.categories])
    T = template_matrices(tagger)
    X = data.X

    def score(mat):
        return X @ mat.T

    def report(name, s):
        d, t = metrics(s, data, data.dev, sugg), metrics(s, data, data.test, sugg)
        print(f"{name:42} dev top1 {d['top1']:5}  top5 {d['top5']:5} fam {d['familia']:5} | "
              f"teste top1 {t['top1']:5}  top5 {t['top5']:5} fam {t['familia']:5}", flush=True)
        return d["top1"]

    print("\n== modelos de texto isolados ==")
    for k, m in T.items():
        report(k, score(m))

    print("\n== combinação (média de embeddings com pesos) — procura em dev ==")
    keys = ["nome", "explic", "sinon_lista", "sinon_cada", "sfx", "familia"]
    best = (0, None)
    for w in itertools.product([0, 1, 2], repeat=len(keys)):
        if sum(w) == 0:
            continue
        m = norm(sum(wi * T[k] for wi, k in zip(w, keys)))
        d = metrics(score(m), data, data.dev, sugg)["top1"]
        if d > best[0]:
            best = (d, w)
    w = best[1]
    print("melhores pesos:", dict(zip(keys, w)))
    M = norm(sum(wi * T[k] for wi, k in zip(w, keys)))
    report("combinação ótima", score(M))
    report("atual (referência)", score(T["atual"]))

    print("\n== correção de 'categorias íman' (hubness) ==")
    S = score(M)
    bias = S[data.dev].mean(0)           # só áudio de dev, sem etiquetas
    for lam in (0.25, 0.5, 0.75, 1.0):
        report(f"íman λ={lam}", S - lam * bias)

    print("\n== bónus de família (log-soma-exp das subcategorias da mesma família) ==")
    fam = np.array([c.category for c in tagger.categories])
    fams = {f: np.where(fam == f)[0] for f in set(fam)}
    scale = tagger.logit_scale
    for beta in (0.1, 0.2, 0.3):
        s2 = S * scale
        bonus = np.zeros_like(s2)
        for idx in fams.values():
            lse = np.log(np.exp(s2[:, idx] - s2.max(1, keepdims=True)).sum(1)) + s2.max(1)
            bonus[:, idx] = lse[:, None]
        report(f"família β={beta}", s2 + beta * bonus)

    print("\n== piores classes (combinação ótima, teste) ==")
    for c, acc, wrong in per_class(score(M), data, data.test, sugg)[:12]:
        print(f"   {c:18} {acc:3}%  confunde com {wrong}")
    np.save(esc.parent / f"best_weights_{model.replace('/', '_')}.npy", np.array(w))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME)
