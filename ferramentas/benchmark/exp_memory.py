"""Curva de aprendizagem da memória de exemplos (sons confirmados -> kNN).

    python ferramentas/benchmark/exp_memory.py <pasta ESC-50>

Simula pessoas a confirmar k sons por classe e mede o acerto em sons novos.
Afinação: memória = folds 1-2, avaliação = fold 3.  Resultado final: memória = folds 1-3,
teste = folds 4-5 (nunca usado para afinar).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench import MAPPING, Data, metrics  # noqa: E402
from exp_prompts import norm, template_matrices  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sfxtagger.ai import MODEL_NAME, NON_SOUND_CATEGORIES, Tagger  # noqa: E402

KEYS = ["nome", "explic", "sinon_lista", "sinon_cada", "sfx", "familia"]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def with_memory(P, Xq, Xm, ym, n_cat, k=10, min_sim=0.8, tau=0.04, start=0.8, full=0.95, maxw=0.85):
    """Versão vetorizada de Tagger._apply_memory."""
    if len(Xm) == 0:
        return P
    S = Xq @ Xm.T
    kk = min(k, S.shape[1])
    top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    st = np.take_along_axis(S, top, 1)
    w = np.where(st >= min_sim, np.exp((st - 1.0) / tau), 0.0)
    votes = np.zeros((len(Xq), n_cat))
    np.add.at(votes, (np.repeat(np.arange(len(Xq)), kk), ym[top].ravel()), w.ravel())
    best = np.where(st >= min_sim, st, -1).max(1)
    weight = np.clip((best - start) / (full - start), 0, maxw)
    has = votes.sum(1) > 0
    out = P.copy()
    vn = votes[has] / votes[has].sum(1, keepdims=True)
    out[has] = (1 - weight[has, None]) * P[has] + weight[has, None] * vn
    return out


def pick(data, pool_mask, k, rng):
    idx = []
    for c in sorted(set(data.cls)):
        cand = np.where(pool_mask & (np.array(data.cls) == c))[0]
        idx += list(rng.choice(cand, size=min(k, len(cand)), replace=False))
    return np.array(idx, int)


def main(esc: Path):
    data = Data(esc)
    tagger = Tagger()
    tagger.ensure_loaded()
    sugg = np.array([c.category not in NON_SOUND_CATEGORIES for c in tagger.categories])
    T = template_matrices(tagger)
    w = np.load(esc.parent / f"best_weights_{MODEL_NAME.replace('/', '_')}.npy")
    M = norm(sum(wi * T[k] for wi, k in zip(w, KEYS)))
    logits = tagger.logit_scale * (data.X @ M.T)
    logits[:, ~sugg] = -1e9
    P = softmax(logits)
    catidx = {c.catid: i for i, c in enumerate(tagger.categories)}
    y = np.array([catidx[MAPPING[c][0]] for c in data.cls])  # etiqueta que a pessoa escolheria
    n_cat = len(tagger.categories)

    classes = sorted(set(data.cls))
    cls_arr = np.array(data.cls)

    def run(mem_pool, eval_mask, k_per_class, params, reps=5):
        """Memória com só METADE dos tipos de som (o resto é desconhecido, como na vida real).
        Devolve o acerto (top1) nos sons de tipos conhecidos e desconhecidos."""
        known_acc, unknown_acc, top5 = [], [], []
        for r in range(reps):
            rng = np.random.default_rng(r)
            known = set(rng.choice(classes, size=len(classes) // 2, replace=False))
            pool = mem_pool & np.isin(cls_arr, list(known))
            m = pick(data, pool, k_per_class, rng) if k_per_class else np.array([], int)
            Q = np.where(eval_mask)[0]
            Pq = with_memory(P[Q], data.X[Q], data.X[m], y[m], n_cat, **params) if len(m) else P[Q]
            full = np.full_like(P, -1.0)
            full[Q] = Pq
            kn = eval_mask & np.isin(cls_arr, list(known))
            un = eval_mask & ~np.isin(cls_arr, list(known))
            known_acc.append(metrics(full, data, kn)["top1"])
            unknown_acc.append(metrics(full, data, un)["top1"])
            top5.append(metrics(full, data, eval_mask)["top5"])
        return (round(float(np.mean(known_acc)), 1), round(float(np.mean(unknown_acc)), 1),
                round(float(np.mean(top5)), 1))

    tune_pool = data.fold <= 2
    tune_eval = data.fold == 3
    grid = list(itertools.product([0.65, 0.7, 0.75, 0.8, 0.85], [0.01, 0.02, 0.03, 0.05],
                                  [0.65, 0.7, 0.75, 0.8], [0.85, 0.9, 0.95], [0.5, 0.6, 0.7, 0.85]))
    print(f"a afinar {len(grid)} combinações (memória folds 1-2, avaliação fold 3)…", flush=True)
    base_known, base_unknown, _ = run(tune_pool, tune_eval, 0, {}, reps=3)
    best = None
    for ms, tau, st, fu, mw in grid:
        if fu <= st:
            continue
        params = dict(min_sim=ms, tau=tau, start=st, full=fu, maxw=mw)
        res = [run(tune_pool, tune_eval, k, params, reps=3) for k in (1, 3, 5)]
        gain = np.mean([r[0] for r in res]) - base_known
        harm = base_unknown - np.mean([r[1] for r in res])
        score = gain - 3 * max(harm, 0)   # estragar sons desconhecidos pesa 3x mais
        if best is None or score > best[0]:
            best = (score, params, gain, harm)
    print("melhores parâmetros:", best[1], f"(ganho {best[2]:.1f} pts, perda {best[3]:.1f} pts no fold 3)")
    old = dict(min_sim=0.8, tau=0.04, start=0.8, full=0.95, maxw=0.85)

    print()
    print("TESTE (folds 4-5), memória com metade dos tipos de som (vinda dos folds 1-3):")
    print(f"{'exemplos/tipo':>14} | {'atuais: conhecidos / desconhecidos':>36} | {'afinados: conhecidos / desconhecidos':>38}")
    for k in (0, 1, 3, 5, 10, 20):
        a = run(data.dev, data.test, k, old)
        b = run(data.dev, data.test, k, best[1])
        print(f"{k:>14} | {a[0]:>16} / {a[1]:<17} | {b[0]:>18} / {b[1]:<17}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
