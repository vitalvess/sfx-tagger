"""Como combinar os modelos: pesos, escalas, modo de fusão, hierarquia e categorias-íman.

    python ferramentas/benchmark/exp_fusao.py textos <pasta ESC-50>   (ambiente msclap: frases do 3.º modelo)
    python ferramentas/benchmark/exp_fusao.py frases <pasta ESC-50>   (pesos das frases do 3.º modelo)
    python ferramentas/benchmark/exp_fusao.py fusao  <pasta ESC-50> [variante]
    python ferramentas/benchmark/exp_fusao.py nome   <pasta ESC-50>   (peso do nome do ficheiro)

Escolhe sempre nos folds 1-3 e só depois lê os folds 4-5. Atenção: o ESC-50 só tem 44 das ~750
categorias como resposta certa, por isso aqui nunca se afina nada categoria a categoria.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

KEYS = ["nome", "explic", "sinon_lista", "sinon_cada", "sfx", "familia"]


def norm(m):
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


def soft(z):
    z = z - np.nanmax(np.where(np.isinf(z), np.nan, z), axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def textos(esc: Path) -> None:
    from msclap import CLAP
    from sfxtagger import ucs
    from sfxtagger.ai import category_texts
    clap = CLAP(version="2023", use_cuda=False)
    texts = [category_texts(c) for c in ucs.load_ucs()]
    out = {}
    for key in KEYS:
        rows = [norm(norm(clap.get_text_embeddings(t[key]).detach().cpu().numpy()).mean(0)) for t in texts]
        out[key] = np.stack(rows).astype(np.float32)
        print("  ", key, flush=True)
    np.savez_compressed(esc.parent / "msclap_textos.npz", **out)


class Dados:
    def __init__(self, esc: Path, variante: str | None = None):
        from bench import Data
        from sfxtagger import precisao
        from sfxtagger.ai import HELPER_MODEL, Tagger
        self.d = Data(esc)
        d2 = Data(esc, HELPER_MODEL)
        t = Tagger()
        t.ensure_loaded()
        self.t = t
        names = [r["filename"] for r in self.d.meta]

        def load(f):
            with np.load(f) as z:
                return np.stack([z[n] for n in names])
        X = [self.d.X, d2.X, load(esc.parent / "emb_msclap_app.npz")]
        if variante:
            X = [load(esc.parent / f"sinal_{variante}_{m}.npz") for m in (1, 2, 3)]
        self.X = X
        self.C = [t.cat_matrix, t.helper.cat_matrix, np.load(precisao.CATEGORIES)]
        self.scales = [t.logit_scale, t.helper.logit_scale, precisao.SCALE]
        self.S = [x @ c.T for x, c in zip(X, self.C)]
        self.sugg = t.suggestable
        self.esc = esc


def prob(d: Dados, i: int, a: float, bias=None):
    z = a * (d.S[i] - (0 if bias is None else bias))
    z = z.copy()
    z[:, ~d.sugg] = -np.inf
    return soft(z)


def frases(esc: Path) -> None:
    from bench import metrics
    d = Dados(esc)
    with np.load(esc.parent / "msclap_textos.npz") as z:
        T = {k: z[k] for k in KEYS}
    X = d.X[2]
    best = None
    for w in itertools.product((0, 1, 2), repeat=len(KEYS)):
        if not any(w):
            continue
        C = norm(sum(wi * T[k] for wi, k in zip(w, KEYS)))
        z = 80 * (X[d.d.dev] @ C.T)
        z[:, ~d.sugg] = -np.inf
        full = np.full((len(X), C.shape[0]), -1.0)
        full[d.d.dev] = soft(z)
        m = metrics(full, d.d, d.d.dev)
        key = (m["top1"], m["top5"])
        if best is None or key > best[0]:
            best = (key, w)
    print("melhor (dev):", dict(zip(KEYS, best[1])), best[0], flush=True)
    C = norm(sum(wi * T[k] for wi, k in zip(best[1], KEYS)))
    S = X @ C.T
    orig = d.S[2]
    for label, s in (("atual", orig), ("afinado", S)):
        z = 80 * s
        z[:, ~d.sugg] = -np.inf
        P = soft(z)
        print(f"{label:8} dev {metrics(P, d.d, d.d.dev)}  teste {metrics(P, d.d, d.d.test)}")
    np.save(esc.parent / "cat_msclap_afinado.npy", C.astype(np.float32))


def fusao(esc: Path, variante: str | None) -> None:
    from bench import metrics
    from sfxtagger.ai import family_rescore
    d = Dados(esc, variante)
    afinado = esc.parent / "cat_msclap_afinado.npy"
    if afinado.exists():
        d.S[2] = d.X[2] @ np.load(afinado).T
    data, fam = d.d, d.t.family_index
    a1, a2 = d.scales[0], d.scales[1]
    memo = {}

    def cached(i, a, lam):
        if (i, a, lam) not in memo:
            bg = data.dev
            memo[i, a, lam] = prob(d, i, a, lam * (d.S[i][bg].mean(0) - d.S[i][bg].mean()) if lam else None)
        return memo[i, a, lam]

    def run(params):
        s1, s2, a3, w2, w3, mode, g, lam = params
        P = [cached(0, a1 * s1, lam), cached(1, a2 * s2, lam), cached(2, a3, lam)]
        if mode == "prob":
            Q = (1 - w2) * P[0] + w2 * P[1]
            if g:
                Q = family_rescore(Q, fam, g)
            Q = (1 - w3) * Q + w3 * P[2]
        else:
            Q = soft((1 - w3) * ((1 - w2) * np.log(P[0] + 1e-12) + w2 * np.log(P[1] + 1e-12)) + w3 * np.log(P[2] + 1e-12))
            if g:
                Q = family_rescore(Q, fam, g)
        return Q

    def search(w3s, nome):
        grid = list(itertools.product((0.5, 1.0), (0.5, 1.0), (80.0, 160.0) if w3s != (0.0,) else (80.0,),
                                      (0.45, 0.55, 0.65), w3s, ("prob", "log"), (0.0, 0.5), (0.0, 0.5, 1.0)))
        best = None
        for p in grid:
            Q = run(p)
            m = metrics(Q, data, data.dev)
            key = (m["top1"], m["top5"])
            if best is None or key > best[0]:
                best = (key, p, Q)
        base = (1.0, 1.0, 80.0, 0.55, 0.33 if w3s != (0.0,) else 0.0, "prob", 0.5, 0.0)
        Q0 = run(base)
        print(f"\n{nome}: {len(grid)} combinações")
        print("  atual  ", base, "dev", metrics(Q0, data, data.dev), "teste", metrics(Q0, data, data.test))
        print("  melhor ", best[1], "dev", metrics(best[2], data, data.dev), "teste", metrics(best[2], data, data.test), flush=True)

    search((0.0,), "2 modelos")
    search((0.2, 0.33, 0.45, 0.6), "3 modelos")


def nome(esc: Path) -> None:
    """Nome do ficheiro certo vs enganador: o ESC-50 tem o nome real de cada som (classe)."""
    from bench import metrics
    d = Dados(esc)
    data, t = d.d, d.t
    classes = sorted(set(data.cls))
    txt = {c: t.embed_text(["the sound of " + c.replace("_", " ")])[0] for c in classes}
    rng = np.random.default_rng(0)
    cls = np.array(data.cls)
    errado = np.array([rng.choice([c for c in classes if c != x]) for x in cls])
    P = np.stack([t.audio_probs(a, b) for a, b in zip(d.X[0], d.X[1])])
    lp = np.log(P + 1e-12)
    base = metrics(P, data, data.dev)["top1"], metrics(P, data, data.test)["top1"]
    print("sem nome: dev/teste", base)

    def run(names, w, scale, mask):
        out = np.full_like(P, -1.0)
        idx = np.where(mask)[0]
        tx = {c: np.log(np.exp(scale * (t.cat_matrix @ v - (t.cat_matrix @ v).max())) /
                        np.exp(scale * (t.cat_matrix @ v - (t.cat_matrix @ v).max())).sum() + 1e-12) for c, v in txt.items()}
        z = (1 - w) * lp[idx] + w * np.stack([tx[n] for n in names[idx]])
        z[:, ~d.sugg] = -np.inf
        out[idx] = soft(z)
        return metrics(out, data, mask)["top1"]

    best = None
    for w, sc in itertools.product((0.15, 0.25, 0.4, 0.55), (10.0, 20.0, t.logit_scale)):
        good, bad = run(cls, w, sc, data.dev), run(errado, w, sc, data.dev)
        score = (good - base[0]) - (base[0] - bad)
        print(f"  peso {w:4} escala {sc:5.1f}: nome certo {good:5.1f}  nome enganador {bad:5.1f}", flush=True)
        if best is None or score > best[0]:
            best = (score, w, sc)
    _, w, sc = best
    print("melhor", best, "teste certo", run(cls, w, sc, data.test), "enganador", run(errado, w, sc, data.test))
    print("atual 0.4 teste certo", run(cls, 0.4, t.logit_scale, data.test), "enganador", run(errado, 0.4, t.logit_scale, data.test))


if __name__ == "__main__":
    cmd, esc = sys.argv[1], Path(sys.argv[2])
    if cmd == "fusao":
        fusao(esc, sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        {"textos": textos, "frases": frases, "nome": nome}[cmd](esc)
