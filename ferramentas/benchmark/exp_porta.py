"""A "porta" do classificador: quando é que ele pode mexer nas sugestões?

    python ferramentas/benchmark/exp_porta.py <pasta ESC-50>

Cenários de tipos de som conhecidos (confirmados por alguém):
  * metade dos tipos (uma turma já com bastante trabalho feito);
  * só 1 ou 3 tipos (alguém que acabou de começar a treinar) — o caso que o ESC-50 com
    metade dos tipos escondia: um classificador que só conhece uma categoria vota sempre nela.
A biblioteca de fundo para adaptar o limiar é o próprio conjunto de sons (como na app).
Afina com a memória dos folds 1-2 e avaliação no fold 3; reporta nos folds 4-5.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp_otimizacao import Setup, metrics, pick  # noqa: E402

from sfxtagger import classifier  # noqa: E402

SCENARIOS = (("metade", None), ("1 tipo", 1), ("3 tipos", 3))


def run(s: Setup, F, pool_mask, eval_mask, k, n_known, blend, adapt, reps=6):
    d = s.data
    kn, un = [], []
    for r in range(reps):
        rng = np.random.default_rng(r)
        known = set(rng.choice(s.classes, size=n_known or len(s.classes) // 2, replace=False))
        m = pick(d, pool_mask & np.isin(s.cls, list(known)), k, rng)
        Q = np.where(eval_mask)[0]
        clf = classifier.train(d.X[m], s.data2.X[m], s.y[m], **classifier.TRAIN)
        if adapt is not None:
            clf = classifier.adapt(clf, F, gate_at=blend["gate_at"], **adapt)
        Pq = classifier.blend(s.P[Q], clf, d.X[Q], s.data2.X[Q], **blend)
        full = np.full_like(s.P, -1.0)
        full[Q] = Pq
        kn.append(metrics(full, d, eval_mask & np.isin(s.cls, list(known)))["top1"])
        un.append(metrics(full, d, eval_mask & ~np.isin(s.cls, list(known)))["top1"])
    return np.mean(kn), np.mean(un)


def main(esc: Path):
    s = Setup(esc)
    d = s.data
    F = classifier.features(d.X, s.data2.X)
    tune_pool, tune_eval = d.fold <= 2, d.fold == 3
    base = {}
    for name, nk in SCENARIOS:
        base[name] = run(s, F, tune_pool, tune_eval, 1, nk, dict(weight=0.0, gate_at=1, gate_width=1), None, reps=3)

    def score(blend, adapt):
        tot = 0.0
        for name, nk in SCENARIOS:
            for k in (1, 3, 5):
                a, b = run(s, F, tune_pool, tune_eval, k, nk, blend, adapt, reps=3)
                tot += (a - base[name][0]) - 3 * max(base[name][1] - b, 0)
        return tot

    cands = [("v0.7 sem adaptar", dict(weight=0.8, gate_at=0.65, gate_width=0.06), None)]
    for w, ga, gw, q, mg in itertools.product((0.6, 0.8), (0.55, 0.65), (0.06,), (0.5, 0.75),
                                              (0.2, 0.25, 0.3, 0.35)):
        cands.append((f"w={w} porta>={ga} largura {gw} quantil {q} margem {mg}",
                      dict(weight=w, gate_at=ga, gate_width=gw), dict(quantile=q, margin=mg)))
    scored = sorted(((score(b, a), n, b, a) for n, b, a in cands), key=lambda t: -t[0])
    print("melhores (afinação, fold 3):")
    for sc, n, _, _ in scored[:5]:
        print(f"  {sc:7.1f}  {n}")
    print(f"  {next(sc for sc, n, _, _ in scored if n.startswith('v0.7')):7.1f}  v0.7 sem adaptar")
    best = scored[0]
    print(f"\nTESTE (folds 4-5): conhecidos / desconhecidos (sem classificador: ver k=0)")
    for name, nk in SCENARIOS:
        b0 = run(s, F, d.dev, d.test, 1, nk, dict(weight=0.0, gate_at=1, gate_width=1), None)
        print(f"  {name}: sem classificador {b0[0]:.1f} / {b0[1]:.1f}")
        for k in (1, 3, 5, 20):
            old = run(s, F, d.dev, d.test, k, nk, dict(weight=0.8, gate_at=0.65, gate_width=0.06), None)
            new = run(s, F, d.dev, d.test, k, nk, best[2], best[3])
            print(f"     k={k:2}: v0.7 sem adaptar {old[0]:5.1f} / {old[1]:5.1f}    adaptado {new[0]:5.1f} / {new[1]:5.1f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
