"""Otimizar o classificador com o que já foi treinado (um .sfxk real).

    python ferramentas/benchmark/exp_treinado.py <pasta ESC-50> <ficheiro.sfxk>

Treino: os exemplos do .sfxk. Afinação: sons dos folds 1-3 que não estão no treino nem vêm das
mesmas gravações. Teste: folds 4-5, lidos só para o vencedor. Um método só conta se não piorar
os sons de tipos desconhecidos (cenários com metade, 3 e 1 tipos conhecidos).
"""
from __future__ import annotations

import io
import itertools
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp_otimizacao import Setup, metrics  # noqa: E402

from sfxtagger import classifier  # noqa: E402


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def lda_logits(F, y):
    """Médias por classe e covariância partilhada (fechado, sem iterar)."""
    classes, yi = np.unique(y, return_inverse=True)
    means = np.stack([F[yi == c].mean(0) for c in range(len(classes))])
    R = F - means[yi]
    return classes, means, R.T @ R / max(len(F) - len(classes), 1)


class Metodo:
    def __init__(self, F, y, kind, tau, lam=1.0):
        self.kind, self.tau = kind, tau
        self.classes, means, S = lda_logits(F, y)
        self.protos = classifier._norm(means)
        self.counts = np.array([(y == c).sum() for c in self.classes])
        if kind == "lda":
            d = S.shape[0]
            A = np.linalg.solve(S + lam * (np.trace(S) / d) * np.eye(d), means.T).T
            self.A, self.b = A, 0.5 * (A * means).sum(1)
        elif kind == "cos":
            m = classifier.train(F, None, y, temperature=tau)
            self.W = m.W

    def q(self, F):
        if self.kind == "lda":
            return softmax(self.tau * (F @ self.A.T - self.b))
        return softmax(self.tau * (F @ (self.W if self.kind == "cos" else self.protos).T))


def aplicar(P, m, F, bg, w0, w1, k, gate=0.55, width=0.06, quant=0.75, margin=0.35, top=0, w_off=0.3):
    sims = F @ m.protos.T
    near = sims.argmax(1)
    at = np.maximum(gate, np.quantile(bg @ m.protos.T, quant, axis=0) + margin)[near]
    g = 1 / (1 + np.exp(-(sims[np.arange(len(F)), near] - at) / width))
    q = m.q(F)
    best = q.argmax(1)
    wc = w0 + (w1 - w0) * np.minimum(1.0, m.counts[best] / k)
    if top:   # a IA de base também acha esta categoria plausível?
        kth = np.partition(P, -top, axis=1)[:, -top]
        wc = np.where(P[np.arange(len(P)), m.classes[best]] >= kth, wc, np.minimum(wc, w_off))
    w = (wc * g)[:, None]
    Q = np.zeros_like(P)
    Q[:, m.classes] = q
    out = (1 - w) * P + w * Q
    return out / out.sum(1, keepdims=True)


def main(esc: Path, sfxk: Path):
    s = Setup(esc)
    d = s.data
    F = classifier.features(d.X, s.data2.X)
    with zipfile.ZipFile(sfxk) as z:
        Xk = np.load(io.BytesIO(z.read("embeddings.npy"))).astype(np.float32)
    Xk /= np.linalg.norm(Xk, axis=1, keepdims=True)
    sim = Xk @ d.X.T
    train = sorted(set(sim.argmax(1)[sim.max(1) > 0.995].tolist()))
    train = np.array(train)
    print("treino:", len(train), "sons", flush=True)
    srcs = set(zip(s.source[train], s.cls[train]))
    tune = d.dev.copy()
    for i in range(len(tune)):
        if (s.source[i], s.cls[i]) in srcs:
            tune[i] = False
    print("afinação:", tune.sum(), "| teste:", d.test.sum(), flush=True)

    def full(idx_train, mask, kind, tau, lam, w0, w1, k, adapt_margin=0.35, top=0, w_off=0.3):
        m = Metodo(F[idx_train], None if False else s.y[idx_train], kind, tau, lam)
        Q = np.where(mask)[0]
        out = np.full_like(s.P, -1.0)
        out[Q] = aplicar(s.P[Q], m, F[Q], F[mask], w0, w1, k, margin=adapt_margin, top=top, w_off=w_off)
        return out

    def abertos(mask, kind, tau, lam, w0, w1, k, adapt_margin=0.35, top=0, w_off=0.3, reps=4):
        res = {}
        for nk in (None, 3, 1):
            kn, un = [], []
            for r in range(reps):
                rng = np.random.default_rng(r)
                known = set(rng.choice(s.classes, size=nk or len(s.classes) // 2, replace=False))
                idx = train[np.isin(s.cls[train], list(known))]
                out = full(idx, mask, kind, tau, lam, w0, w1, k, adapt_margin, top, w_off)
                kn.append(metrics(out, d, mask & np.isin(s.cls, list(known)))["top1"])
                un.append(metrics(out, d, mask & ~np.isin(s.cls, list(known)))["top1"])
            res[nk] = (np.mean(kn), np.mean(un))
        return res

    base = metrics(s.P, d, tune)
    print("sem classificador (afinação):", base, flush=True)
    atual = ("cos", 10.0, 1.0, 0.6, 0.6, 10.0, 0.35, 0, 0.3)
    ref_all = metrics(full(train, tune, *atual), d, tune)
    ref_open = abertos(tune, *atual)
    print("atual (v0.7.5):", ref_all, flush=True)

    cands = [(kind, tau, lam, 0.6, w1, 10.0, mg, top, off)
             for (kind, tau, lam), w1, mg, top, off in itertools.product(
                 (("lda", 1.0, 3.0), ("lda", 1.0, 1.0), ("cos", 40.0, 1.0)), (0.7, 0.85, 0.95), (0.35, 0.45),
                 (0, 5, 20, 50), (0.15, 0.3, 0.5)) if top or off == 0.3]
    scored = []
    for c in cands:
        a = metrics(full(train, tune, *c), d, tune)
        scored.append((a["top1"], a["top5"], c))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    print("melhores (todos os tipos conhecidos):")
    for t1, t5, c in scored[:6]:
        print(f"  {t1:5.1f} {t5:5.1f}  {c}", flush=True)

    # critério: ganho nos conhecidos, e estragar um som de tipo desconhecido pesa 3x mais
    rows = []
    for t1, t5, c in scored:
        o = abertos(tune, *c, reps=3)
        harm = max(ref_open[nk][1] - o[nk][1] for nk in o)
        gain = (t1 - ref_all["top1"]) + (o[None][0] - ref_open[None][0])
        rows.append((gain - 3 * max(harm, 0), t1, t5, o[None][0], o[None][1], harm, c))
    rows.sort(key=lambda r: -r[0])
    for r in rows[:10]:
        print(f"  nota {r[0]:6.1f} | todos {r[1]:5.1f}/{r[2]:5.1f} | metade {r[3]:5.1f} desc {r[4]:5.1f} | piora {r[5]:4.1f} | {r[6]}", flush=True)
    winner = rows[0][6] if rows[0][0] > 0 and rows[0][5] <= 1.0 else None
    print("referência aberto: " + " ".join(f"{nk}: {ref_open[nk][0]:.1f}/{ref_open[nk][1]:.1f}" for nk in ref_open))
    print("VENCEDOR", winner, flush=True)
    if winner:
        print("teste atual  ", metrics(full(train, d.test, *atual), d, d.test))
        print("teste vencedor", metrics(full(train, d.test, *winner), d, d.test))
        o0, o1 = abertos(d.test, *atual), abertos(d.test, *winner)
        print("teste aberto atual   ", {k: tuple(round(x, 1) for x in v) for k, v in o0.items()})
        print("teste aberto vencedor", {k: tuple(round(x, 1) for x in v) for k, v in o1.items()})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
