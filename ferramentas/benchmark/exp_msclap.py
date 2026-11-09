"""Ponto 8: vale a pena juntar o CLAP da Microsoft (2023) aos dois modelos da app?

Passo 1, num ambiente à parte com o pacote msclap (não faz parte da app):
    python ferramentas/benchmark/exp_msclap.py embed <pasta ESC-50>
  -> guarda emb_msclap.npz (sons) e cat_msclap.npy (categorias UCS) ao lado do ESC-50.

Passo 2, no ambiente da app:
    python ferramentas/benchmark/exp_msclap.py eval <pasta ESC-50>
  -> compara: app atual, MS-CLAP sozinho, e os três juntos (peso afinado nos folds 1-3).

Passo 3, confirma que o código que vai na app (precisao.py) dá o mesmo que o pacote original:
    python ferramentas/benchmark/exp_msclap.py app <pasta ESC-50>
  -> precisão e calibração da confiança com os três modelos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))


def norm(m):
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


def embed(esc: Path) -> None:
    import csv
    from msclap import CLAP
    from sfxtagger import ucs
    from sfxtagger.ai import CATEGORY_TEXT_WEIGHTS, MODEL_NAME, category_texts

    clap = CLAP(version="2023", use_cuda=False)
    cats = ucs.load_ucs()
    weights = CATEGORY_TEXT_WEIGHTS[MODEL_NAME]
    texts = [category_texts(c) for c in cats]
    mat = 0
    for key, w in weights.items():
        if not w:
            continue
        rows = []
        for t in texts:
            e = clap.get_text_embeddings(t[key]).detach().cpu().numpy()
            rows.append(norm(norm(e).mean(0)))
        mat = mat + w * np.stack(rows)
        print("  textos", key, flush=True)
    np.save(esc.parent / "cat_msclap.npy", norm(mat).astype(np.float32))
    with open(esc / "meta" / "esc50.csv", newline="") as f:
        meta = list(csv.DictReader(f))
    out = {}
    files = [str(esc / "audio" / r["filename"]) for r in meta]
    for i in range(0, len(files), 32):
        e = clap.get_audio_embeddings(files[i:i + 32]).detach().cpu().numpy()
        for r, v in zip(meta[i:i + 32], norm(e)):
            out[r["filename"]] = v.astype(np.float32)
        print(f"  {i + 32}/{len(files)}", flush=True)
    np.savez_compressed(esc.parent / "emb_msclap.npz", **out)


def evaluate(esc: Path) -> None:
    from bench import Data, metrics
    from sfxtagger.ai import HELPER_MODEL, Tagger

    d1, d2 = Data(esc), Data(esc, HELPER_MODEL)
    t = Tagger()
    t.ensure_loaded()
    with np.load(esc.parent / "emb_msclap.npz") as z:
        X3 = np.stack([z[r["filename"]] for r in d1.meta])
    C3 = np.load(esc.parent / "cat_msclap.npy")
    P = np.stack([t.audio_probs(a, b) for a, b in zip(d1.X, d2.X)])

    def soft(S, scale):
        S = scale * S
        S[:, ~t.suggestable] = -np.inf
        S -= S.max(1, keepdims=True)
        e = np.exp(S)
        return e / e.sum(1, keepdims=True)

    print("app atual (2 modelos)       ", metrics(P, d1, d1.dev), metrics(P, d1, d1.test))
    best = None
    for scale in (20, 33, 50, 80):
        P3 = soft(X3 @ C3.T, scale)
        a = metrics(P3, d1, d1.dev)
        print(f"MS-CLAP sozinho escala {scale:3}  dev {a}")
        for w in (0.2, 0.33, 0.5, 0.66):
            Q = (1 - w) * P + w * P3
            m = metrics(Q, d1, d1.dev)
            if best is None or m["top1"] > best[0]["top1"]:
                best = (m, scale, w)
    m, scale, w = best
    Q = (1 - w) * P + w * soft(X3 @ C3.T, scale)
    print(f"3 modelos (escala {scale}, peso {w}): dev {m}  TESTE {metrics(Q, d1, d1.test)}")


def app(esc: Path) -> None:
    from bench import Data, metrics
    from sfxtagger import audio, precisao
    from sfxtagger.ai import HELPER_MODEL, Tagger

    d1, d2 = Data(esc), Data(esc, HELPER_MODEL)
    t = Tagger()
    t.ensure_loaded()
    precisao._load()
    cache = esc.parent / "emb_msclap_app.npz"
    if cache.exists():
        with np.load(cache) as z:
            X3 = np.stack([z[r["filename"]] for r in d1.meta])
    else:
        rows = []
        for i, r in enumerate(d1.meta):
            x = audio.load(esc / "audio" / r["filename"]).mono
            rows.append(precisao.embed(audio.best_windows(x)))
            if i % 200 == 0:
                print(f"  {i}/{len(d1.meta)}", flush=True)
        X3 = np.stack(rows)
        np.savez_compressed(cache, **{r["filename"]: v for r, v in zip(d1.meta, X3)})
    with np.load(esc.parent / "emb_msclap.npz") as z:
        orig = np.stack([z[r["filename"]] for r in d1.meta])
    cos = (X3 * orig).sum(1)
    print(f"semelhança com o pacote original: média {cos.mean():.4f}, mínimo {cos.min():.4f}")
    P = np.stack([t.audio_probs(a, b) for a, b in zip(d1.X, d2.X)])
    P3 = np.stack([precisao.probs(e, t.suggestable) for e in X3])
    Q = (1 - precisao.WEIGHT) * P + precisao.WEIGHT * P3
    print("2 modelos, teste:", metrics(P, d1, d1.test))
    print("3 modelos, teste:", metrics(Q, d1, d1.test))
    hit = d1.ok[np.arange(len(Q)), Q.argmax(1)]
    conf = Q.max(1)
    print("calibração (dev | teste):")
    for lo, hi in ((0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)):
        a = d1.dev & (conf >= lo) & (conf < hi)
        b = d1.test & (conf >= lo) & (conf < hi)
        print(f"  {lo:.1f}-{hi:.1f}: {100 * hit[a].mean():5.1f}% ({a.sum():3}) | {100 * hit[b].mean():5.1f}% ({b.sum():3})")


if __name__ == "__main__":
    {"embed": embed, "eval": evaluate, "app": app}[sys.argv[1]](Path(sys.argv[2]))
