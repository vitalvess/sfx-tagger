"""Duas ideias para o classificador treinado, medidas nos 6 testes:
  1. o 3.º modelo (MS-CLAP) também nas impressões digitais do classificador;
  2. exemplos extra misturados com fundos (ambientes), para reconhecer o som com ruído atrás.

    python ferramentas/benchmark/exp_aumento.py preparar    (ouve os sons; demora, fica em cache)
    python ferramentas/benchmark/exp_aumento.py medir

Só entram exemplos cujo som está neste computador (os do disco D: ficam de fora em todas as
versões, para a comparação ser justa). A avaliação reproduz o "Testar a IA" da app: sem nomes de
ficheiro, porta adaptada à pasta de teste, sons já aprendidos excluídos.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("EARMARK_HOME", str(Path(os.environ["LOCALAPPDATA"]) / "SFXTagger"))

from earmark import audio, avaliacao, classifier, precisao, suspeitos  # noqa: E402
from earmark.ai import get_tagger  # noqa: E402
from earmark.knowledge import example_id  # noqa: E402

CONHECIMENTO = Path(os.environ["EARMARK_HOME"]) / "conhecimento"
TESTES = Path.home() / "Documents" / "Earmark Treino" / "Testes"
CACHE = Path(os.environ["LOCALAPPDATA"]) / "SFXTagger-bench" / "aumento"
CENAS = {"TUT2018", "CochlScene"}        # já são ambientes: não se misturam com fundos
SNR_DB = (0.0, 12.0)                     # som principal 0 a 12 dB acima do fundo


def norm(m):
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


def carregar_conhecimento():
    meta = json.loads((CONHECIMENTO / "exemplos.json").read_text(encoding="utf-8"))
    with np.load(CONHECIMENTO / "embeddings.npz") as z:
        e1 = {k: z[k].astype(np.float32) for k in z.files}
    with np.load(CONHECIMENTO / "embeddings2.npz") as z:
        e2 = {k: z[k].astype(np.float32) for k in z.files}
    return meta, e1, e2


def emb3_de(path: Path, cache: dict, key: str):
    if key not in cache:
        cache[key] = precisao.embed(audio.best_windows(audio.load(path).mono))
    return cache[key]


def misturar(x: np.ndarray, fundo: np.ndarray, snr_db: float) -> np.ndarray:
    if len(fundo) < len(x):
        fundo = np.tile(fundo, len(x) // max(len(fundo), 1) + 1)
    fundo = fundo[: len(x)]
    rx, rf = np.sqrt(np.mean(x ** 2)) + 1e-9, np.sqrt(np.mean(fundo ** 2)) + 1e-9
    y = x + fundo * (rx / rf) / (10 ** (snr_db / 20))
    pico = np.abs(y).max()
    return (y / pico * 0.89 if pico > 0.99 else y).astype(np.float32)


def preparar() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    t = get_tagger()
    t.ensure_loaded()
    precisao._load()
    meta, e1, e2 = carregar_conhecimento()
    caminhos = {i: suspeitos.audio_path(m) for i, m in meta.items()}
    ids = [i for i, p in caminhos.items() if p is not None and i in e1 and i in e2]
    print(f"exemplos com som neste PC: {len(ids)} de {len(meta)}", flush=True)
    # 3.º modelo dos exemplos
    f3 = CACHE / "treino_emb3.npz"
    c3 = dict(np.load(f3)) if f3.exists() else {}
    for n, i in enumerate(ids):
        emb3_de(caminhos[i], c3, i)
        if n % 500 == 0:
            print(f"  3.º modelo, treino {n}/{len(ids)}", flush=True)
            np.savez_compressed(f3, **c3)
    np.savez_compressed(f3, **c3)
    # exemplos misturados com fundos (um por som isolado)
    fundos = [caminhos[i] for i in ids if meta[i].get("author") in CENAS]
    rng = np.random.default_rng(0)
    fm = CACHE / "misturas.npz"
    mix = dict(np.load(fm)) if fm.exists() else {}
    isolados = [i for i in ids if meta[i].get("author") not in CENAS]
    for n, i in enumerate(isolados):
        if f"{i}|1" in mix:
            continue
        x = audio.load(caminhos[i]).mono
        b = audio.load(fundos[rng.integers(len(fundos))]).mono
        y = misturar(x, b, float(rng.uniform(*SNR_DB)))
        w = audio.best_windows(y)
        mix[f"{i}|1"], mix[f"{i}|2"], mix[f"{i}|3"] = t.embed_audio(w), t.embed_audio_helper(w), precisao.embed(w)
        if n % 250 == 0:
            print(f"  misturas {n}/{len(isolados)}", flush=True)
            np.savez_compressed(fm, **mix)
    np.savez_compressed(fm, **mix)
    # 3.º modelo dos sons de teste
    for d in sorted(p for p in TESTES.iterdir() if p.is_dir()):
        ft = CACHE / f"teste_emb3_{d.name}.npz"
        ct = dict(np.load(ft)) if ft.exists() else {}
        for n, w in enumerate(sorted(d.glob("*.wav"))):
            emb3_de(w, ct, w.name)
            if n % 500 == 0:
                print(f"  3.º modelo, {d.name} {n}", flush=True)
        np.savez_compressed(ft, **ct)
    print("preparado", flush=True)


def testes():
    """[(nome, nomes dos ficheiros, X1, X2, X3, respostas aceites)] a partir do estado de cada pasta."""
    out = []
    for d in sorted(p for p in TESTES.iterdir() if p.is_dir()):
        st = d / ".earmark"
        recs = json.loads((st / "projeto.json").read_text(encoding="utf-8"))["records"]
        with np.load(st / "embeddings.npz") as z:
            e1 = {k: z[k] for k in z.files}
        with np.load(st / "embeddings2.npz") as z:
            e2 = {k: z[k] for k in z.files}
        with np.load(CACHE / f"teste_emb3_{d.name}.npz") as z:
            e3 = {k: z[k] for k in z.files}
        answers = avaliacao.load_answers(d)
        rows = []
        for rid, r in recs.items():
            name = Path(r["path"]).name
            truth = answers.get(name.lower())
            if truth and rid in e1 and rid in e2 and name in e3:
                rows.append((name, e1[rid], e2[rid], e3[name], truth))
        out.append((d.name, [r[0] for r in rows], np.stack([r[1] for r in rows]),
                    np.stack([r[2] for r in rows]), np.stack([r[3] for r in rows]), [r[4] for r in rows]))
    return out


def medir() -> None:
    t = get_tagger()
    t.ensure_loaded()
    precisao._matrix = np.load(precisao.CATEGORIES)
    precisao.enabled = lambda: True
    precisao._model = object()             # para precisao.active(): só se usa a matriz
    cats = [c.catid for c in t.categories]
    idx = {c: n for n, c in enumerate(cats)}
    meta, e1, e2 = carregar_conhecimento()
    with np.load(CACHE / "treino_emb3.npz") as z:
        e3 = {k: z[k] for k in z.files}
    with np.load(CACHE / "misturas.npz") as z:
        mix = {k: z[k] for k in z.files}
    ids = [i for i in e3 if i in e1 and i in e2 and meta[i].get("catid") in idx]
    y = np.array([idx[meta[i]["catid"]] for i in ids])
    X1, X2, X3 = (np.stack([e[i] for i in ids]) for e in (e1, e2, e3))
    mids = [i for i in ids if f"{i}|1" in mix]
    my = np.array([idx[meta[i]["catid"]] for i in mids])
    M1, M2, M3 = (np.stack([mix[f"{i}|{k}"] for i in mids]) for k in (1, 2, 3))
    print(f"treino: {len(ids)} exemplos, {len(set(y))} categorias | misturas: {len(mids)}", flush=True)
    aprendidos = {example_id(e1[i]) for i in ids}

    def feats(A, B, C, tres):
        F = [norm(A), norm(B)] + ([norm(C)] if tres else [])
        return np.concatenate(F, 1) / np.sqrt(len(F))

    tests = testes()
    versoes = {
        "atual (2 modelos)": dict(base3=False, clf3=False, aum=False),
        "Precisão máxima hoje": dict(base3=True, clf3=False, aum=False),
        "+ 3.º modelo no treino": dict(base3=True, clf3=True, aum=False),
        "+ fundos": dict(base3=False, clf3=False, aum=True),
        "as duas": dict(base3=True, clf3=True, aum=True),
    }
    res = {}
    for nome, v in versoes.items():
        F = feats(X1, X2, X3, v["clf3"])
        Y = y
        if v["aum"]:
            F = np.concatenate([F, feats(M1, M2, M3, v["clf3"])])
            Y = np.concatenate([y, my])
        t0 = time.time()
        model = classifier.train(F, None, Y, **dict(classifier.TRAIN, use_helper=False))
        res[nome] = {"treino_s": round(time.time() - t0)}
        for tname, names, T1, T2, T3, truth in tests:
            keep = [n for n in range(len(names)) if example_id(T1[n]) not in aprendidos]
            T1k, T2k, T3k = T1[keep], T2[keep], T3[keep]
            P = np.stack([t.audio_probs(a, b, c if v["base3"] else None) for a, b, c in zip(T1k, T2k, T3k)])
            Fq = feats(T1k, T2k, T3k, v["clf3"])
            m = classifier.adapt(model, Fq, gate_at=classifier.BLEND["gate_at"], **classifier.ADAPT)
            Q = classifier.blend(P, m, Fq, None, **classifier.BLEND)
            top = np.argsort(-Q, 1)[:, :5]
            tr = [truth[n] for n in keep]
            t1 = np.mean([cats[top[n, 0]] in tr[n] for n in range(len(tr))]) * 100
            t5 = np.mean([any(cats[j] in tr[n] for j in top[n]) for n in range(len(tr))]) * 100
            res[nome][tname] = (round(t1, 1), round(t5, 1), len(tr))
        print(nome, {k: v for k, v in res[nome].items()}, flush=True)
    nomes_t = [x[0] for x in tests]
    print("\n1.ª sugestão certa (nas 5)")
    print(f"{'':38}" + "".join(f"{v[:20]:>22}" for v in versoes))
    for tn in nomes_t:
        print(f"{tn[:38]:38}" + "".join(f"{res[v][tn][0]:>13} ({res[v][tn][1]:>5})" for v in versoes))
    print(f"{'média':38}" + "".join(f"{np.mean([res[v][tn][0] for tn in nomes_t]):>13.1f}{'':8}" for v in versoes))
    (Path(__file__).parent / "relatorios" / "aumento.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"preparar": preparar, "medir": medir}[sys.argv[1]]()
