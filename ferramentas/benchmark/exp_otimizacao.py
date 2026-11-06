"""Experiências da ronda de otimização v0.7 (ESC-50 -> UCS, os dois modelos da app).

    python ferramentas/benchmark/exp_otimizacao.py <pasta ESC-50> [hier|conf|grupo|treino|tudo]

  hier    categoria principal primeiro (hierarquia UCS)
  conf    calibração da confiança (quando a IA diz "tenho a certeza", acerta quanto?)
  grupo   consenso entre takes da mesma gravação (séries)
  treino  classificador treinado vs memória kNN, com metade dos tipos de som desconhecidos

Afina-se sempre nos folds 1-3 e reporta-se nos folds 4-5 (nunca usados para escolher).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench import MAPPING, Data, metrics  # noqa: E402
from exp_memory import pick, with_memory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sfxtagger import ai, classifier  # noqa: E402
from sfxtagger.ai import HELPER_MODEL, Tagger  # noqa: E402


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Setup:
    def __init__(self, esc: Path):
        self.data = Data(esc)
        self.data2 = Data(esc, HELPER_MODEL)
        t = Tagger()
        t.ensure_loaded()
        self.tagger = t
        self.sugg = t.suggestable
        self.P = np.stack([t.audio_probs(a, b) for a, b in zip(self.data.X, self.data2.X)])
        self.n_cat = len(t.categories)
        idx = {c.catid: i for i, c in enumerate(t.categories)}
        self.y = np.array([idx[MAPPING[c][0]] for c in self.data.cls])
        self.cls = np.array(self.data.cls)
        self.classes = sorted(set(self.data.cls))
        self.source = np.array([r["filename"].split("-")[1] for r in self.data.meta])

    def report(self, name, P, mask=None):
        d = self.data
        a = metrics(P, d, d.dev if mask is None else mask)
        b = metrics(P, d, d.test)
        print(f"{name:42} dev top1 {a['top1']:5.1f} top5 {a['top5']:5.1f} | "
              f"teste top1 {b['top1']:5.1f} top5 {b['top5']:5.1f} fam {b['familia']:5.1f}", flush=True)
        return a, b


# --- hierarquia ------------------------------------------------------------------
def exp_hier(s: Setup):
    s.report("base (app v0.6)", s.P)
    best = (None, -1)
    for g in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        Q = ai.family_rescore(s.P, s.tagger.family_index, g)
        a = metrics(Q, s.data, s.data.dev)
        print(f"   gama {g:4}: dev top1 {a['top1']} top5 {a['top5']}")
        if a["top1"] > best[1]:
            best = (g, a["top1"])
    s.report(f"hierarquia gama={best[0]}", ai.family_rescore(s.P, s.tagger.family_index, best[0]))


# --- confiança --------------------------------------------------------------------
def exp_conf(s: Setup):
    P = s.P
    top = P.argmax(1)
    hit = s.data.ok[np.arange(len(P)), top]
    conf = ai.confidence_value(P)
    for name, m in (("dev", s.data.dev), ("teste", s.data.test)):
        print(f"\n{name}: confiança -> acerto real (n)")
        for lo, hi in ((0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, .8), (.8, 1.01)):
            sel = m & (conf >= lo) & (conf < hi)
            if sel.sum():
                print(f"  {lo:.2f}-{hi:.2f}: {100 * hit[sel].mean():5.1f}%  ({sel.sum()})")
        for lvl in ("alta", "media", "baixa"):
            sel = m & (np.array([ai.confidence_level(c) for c in conf]) == lvl)
            if sel.sum():
                print(f"  nível {lvl:6}: acerta {100 * hit[sel].mean():5.1f}%  "
                      f"({100 * sel.sum() / m.sum():.0f}% dos sons)")


# --- grupos ----------------------------------------------------------------------
def exp_grupo(s: Setup):
    groups = {}
    for i, src in enumerate(s.source):
        groups.setdefault((s.data.fold[i], src), []).append(i)
    multi = np.zeros(len(s.P), bool)
    for ids in groups.values():
        if len(ids) >= 2:
            multi[ids] = True
    print(f"sons em séries (>=2 takes da mesma gravação): {multi.sum()}")
    base = s.report("base, só sons em séries", s.P, s.data.dev & multi)
    best = (0.0, -1)
    for beta in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75):
        Q = s.P.copy()
        for ids in groups.values():
            if len(ids) >= 2:
                Q[ids] = ai.group_consensus(s.P[ids], beta)
        a = metrics(Q, s.data, s.data.dev & multi)
        print(f"   beta {beta}: dev top1 {a['top1']} top5 {a['top5']}")
        if a["top1"] > best[1]:
            best = (beta, a["top1"])
    Q = s.P.copy()
    for ids in groups.values():
        if len(ids) >= 2:
            Q[ids] = ai.group_consensus(s.P[ids], best[0])
    print(f"beta escolhido {best[0]}")
    d = s.data
    print("  teste (séries):", metrics(s.P, d, d.test & multi), "->", metrics(Q, d, d.test & multi))


# --- treino ----------------------------------------------------------------------
def exp_treino(s: Setup):
    d = s.data
    X1, X2 = d.X, s.data2.X

    def run(pool_mask, eval_mask, k, method, params, reps=5):
        kn_acc, un_acc, t5 = [], [], []
        for r in range(reps):
            rng = np.random.default_rng(r)
            known = set(rng.choice(s.classes, size=len(s.classes) // 2, replace=False))
            pool = pool_mask & np.isin(s.cls, list(known))
            m = pick(d, pool, k, rng) if k else np.array([], int)
            Q = np.where(eval_mask)[0]
            Pq = s.P[Q]
            if len(m):
                if method == "knn":
                    Pq = with_memory(Pq, X1[Q], X1[m], s.y[m], s.n_cat, **params)
                else:
                    bl = dict(params.get("blend", {}))
                    kp = bl.pop("knn", None)
                    clf = classifier.train(X1[m], X2[m], s.y[m], **params.get("train", {}))
                    Pq = classifier.blend(Pq, clf, X1[Q], X2[Q], **bl)
                    if kp:
                        Pq = with_memory(Pq, X1[Q], X1[m], s.y[m], s.n_cat, **kp)
            full = np.full_like(s.P, -1.0)
            full[Q] = Pq
            kn = eval_mask & np.isin(s.cls, list(known))
            un = eval_mask & ~np.isin(s.cls, list(known))
            kn_acc.append(metrics(full, d, kn)["top1"])
            un_acc.append(metrics(full, d, un)["top1"])
            t5.append(metrics(full, d, eval_mask)["top5"])
        return np.mean(kn_acc), np.mean(un_acc), np.mean(t5)

    knn = dict(min_sim=ai.MEMORY["min_sim"], tau=ai.MEMORY["tau"], start=ai.MEMORY["start"],
               full=ai.MEMORY["full"], maxw=ai.MEMORY["max_weight"])
    tune_pool, tune_eval = d.fold <= 2, d.fold == 3
    base_kn, base_un, _ = run(tune_pool, tune_eval, 0, "knn", {}, reps=3)
    print(f"sem memória (fold 3): conhecidos {base_kn:.1f} desconhecidos {base_un:.1f}")
    best = None
    grid = itertools.product([0.6, 0.8, 1.0], [0.55, 0.65, 0.75], [0.03, 0.06], [10.0, 20.0],
                             [False, True])
    for w, t, sc, temp, uh in grid:
        params = {"blend": dict(weight=w, gate_at=t, gate_width=sc),
                  "train": dict(temperature=temp, use_helper=uh)}
        res = [run(tune_pool, tune_eval, k, "clf", params, reps=3) for k in (1, 3, 5)]
        gain = np.mean([r[0] for r in res]) - base_kn
        harm = base_un - np.mean([r[1] for r in res])
        score = gain - 3 * max(harm, 0)
        if best is None or score > best[0]:
            best = (score, params, gain, harm)
    print("melhor classificador:", best[1], f"ganho {best[2]:.1f} perda {best[3]:.1f}")
    print(f"\nTESTE folds 4-5 (memória com metade dos tipos, vinda dos folds 1-3)")
    print(f"{'exemplos/tipo':>13} | {'kNN v0.6: conh / desc / top5':>30} | {'classificador: conh / desc / top5':>34} | {'os dois':>22}")
    for k in (0, 1, 3, 5, 10, 20):
        a = run(d.dev, d.test, k, "knn", knn)
        b = run(d.dev, d.test, k, "clf", best[1])
        both = {"blend": dict(best[1]["blend"], knn=knn), "train": best[1]["train"]}
        c = run(d.dev, d.test, k, "clf", both)
        print(f"{k:>13} | {a[0]:9.1f} / {a[1]:5.1f} / {a[2]:5.1f}    | {b[0]:12.1f} / {b[1]:5.1f} / {b[2]:5.1f}"
              f"    | {c[0]:5.1f} / {c[1]:5.1f} / {c[2]:5.1f}", flush=True)


if __name__ == "__main__":
    esc = Path(sys.argv[1])
    what = sys.argv[2] if len(sys.argv) > 2 else "tudo"
    s = Setup(esc)
    for name, fn in (("hier", exp_hier), ("conf", exp_conf), ("grupo", exp_grupo), ("treino", exp_treino)):
        if what in (name, "tudo"):
            print(f"\n===== {name} =====")
            fn(s)
