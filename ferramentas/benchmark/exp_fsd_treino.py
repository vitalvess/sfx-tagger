"""Quanto sobe o acerto se a IA treinar com os sons de treino da FSD50K? Não toca na IA da pessoa.

    python ferramentas/benchmark/exp_fsd_treino.py embed <base fsd50k> <pasta teste> <n por classe>
    python ferramentas/benchmark/exp_fsd_treino.py eval  <base fsd50k> <pasta teste> <ficheiro.sfxk>

Treino: parte "dev" da FSD50K. Teste: parte "eval", de outros autores. O teste usa as impressões
digitais já calculadas pelo "Testar a IA".
"""
from __future__ import annotations

import csv
import io
import json
import struct
import sys
import tempfile
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preparar_fsd50k import Dividido, entradas  # noqa: E402

from sfxtagger import audio, avaliacao, classifier, ucs  # noqa: E402
from sfxtagger.ai import Tagger  # noqa: E402

# genéricas demais para ensinar uma só categoria (ex.: Speech seria sempre "voz de homem")
NO_TRAIN = {"Speech", "Singing", "Conversation", "Crowd", "Aircraft", "Bird", "Car", "Motor_vehicle_(road)"}


def mapping() -> dict[str, list[str]]:
    d = json.loads((Path(__file__).parent / "fsd50k_ucs.json").read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def seleccao(base: Path, m: dict) -> dict[str, list[tuple[str, int]]]:
    """Para cada classe, os sons por ordem fixa: primeiro os que só têm essa classe específica."""
    rows = list(csv.DictReader(open(base / "FSD50K.ground_truth" / "dev.csv", newline="", encoding="utf-8")))
    rng = np.random.default_rng(0)
    por = defaultdict(list)
    for r in rows:
        labs = [l for l in r["labels"].split(",") if l in m]
        if labs and labs[0] not in NO_TRAIN:
            por[labs[0]].append((len(labs), rng.random(), r["fname"]))
    return {k: [f for _, _, f in sorted(v)] for k, v in por.items()}


def embed(base: Path, n: int) -> None:
    m = mapping()
    sel = seleccao(base, m)
    wanted = {f: (lab, i) for lab, fs in sel.items() for i, f in enumerate(fs[:n])}
    cache = base / "treino_emb.npz"
    done = {}
    if cache.exists():
        with np.load(cache) as z:
            done = {k: z[k] for k in z.files}
    todo = {f for f in wanted if f"{f}.1" not in done}
    print(len(wanted), "sons escolhidos,", len(todo), "por ouvir", flush=True)
    t = Tagger()
    t.ensure_loaded()
    stem = "FSD50K.dev_audio"
    z = Dividido(sorted(base.glob(f"{stem}.z[0-9][0-9]")) + [base / f"{stem}.zip"])
    tmp = Path(tempfile.mkdtemp())
    k = 0
    for name, method, csize, loc in entradas(z):
        f = Path(name).stem
        if not name.endswith(".wav") or f not in todo:
            continue
        lh = z.read(loc, 30)
        data = z.read(loc + 30 + struct.unpack("<H", lh[26:28])[0] + struct.unpack("<H", lh[28:30])[0], csize)
        if method == 8:
            data = zlib.decompress(data, -15)
        p = tmp / "x.wav"
        p.write_bytes(data)
        w = audio.best_windows(audio.load(p).mono)
        done[f"{f}.1"], done[f"{f}.2"] = t.embed_audio(w), t.embed_audio_helper(w)
        k += 1
        if k % 500 == 0:
            np.savez_compressed(base / "treino_emb.tmp.npz", **done)
            (base / "treino_emb.tmp.npz").replace(cache)
            print(f"  {k}/{len(todo)}", flush=True)
    np.savez_compressed(base / "treino_emb.tmp.npz", **done)
    (base / "treino_emb.tmp.npz").replace(cache)
    print("feito", flush=True)


def evaluate(base: Path, teste: Path, sfxk: Path) -> None:
    m = mapping()
    sel = seleccao(base, m)
    t = Tagger()
    t.ensure_loaded()
    cats = t.categories
    idx = {c.catid: i for i, c in enumerate(cats)}
    fam = np.array([t.family_index[i] for i in range(len(cats))])

    st = teste / ".sfxtagger"
    rec = json.loads((st / "projeto.json").read_text(encoding="utf-8"))["records"]
    answers = avaliacao.load_answers(teste)
    with np.load(st / "embeddings.npz") as a, np.load(st / "embeddings2.npz") as b:
        ids = [r for r in a.files if r in b.files and rec[r]["path"].lower() in answers]
        X1 = np.stack([a[r] for r in ids]).astype(np.float32)
        X2 = np.stack([b[r] for r in ids]).astype(np.float32)
    truth = [[idx[c] for c in answers[rec[r]["path"].lower()]] for r in ids]
    print("teste:", len(ids), "sons", flush=True)
    P = np.stack([t.audio_probs(u, v) for u, v in zip(X1, X2)])
    F = classifier.features(X1, X2)

    def score(Q):
        top = np.argpartition(-Q, 5, axis=1)[:, :5]
        order = np.take_along_axis(top, np.argsort(-np.take_along_axis(Q, top, 1), axis=1), 1)
        conf = Q.max(1)
        t1 = np.array([o[0] in tr for o, tr in zip(order, truth)])
        t5 = np.array([bool(set(o) & set(tr)) for o, tr in zip(order, truth)])
        fm = np.array([fam[o[0]] in set(fam[tr]) for o, tr in zip(order, truth)])
        lv = {"alta": conf >= .6, "media": (conf >= .4) & (conf < .6), "baixa": conf < .4}
        return (f"1.a {100 * t1.mean():5.1f}  nas 5 {100 * t5.mean():5.1f}  principal {100 * fm.mean():5.1f} | "
                + "  ".join(f"{k} {100 * t1[v].mean():.0f}% ({100 * v.mean():.0f}% dos sons)" for k, v in lv.items()))

    def run(tr1, tr2, y, blend):
        c = classifier.train(tr1, tr2, y, **classifier.TRAIN)
        c = classifier.adapt(c, F, gate_at=blend["gate_at"], **classifier.ADAPT)
        return classifier.blend(P, c, X1, X2, **blend)

    print("sem treino       ", score(P), flush=True)
    with zipfile.ZipFile(sfxk) as z:
        items = json.loads(z.read("exemplos.json"))
        K1 = np.load(io.BytesIO(z.read("embeddings.npy"))).astype(np.float32)
        K2 = np.load(io.BytesIO(z.read("embeddings2.npy"))).astype(np.float32)
    keep = np.array([np.abs(v).sum() > 0 and it["catid"] in idx for it, v in zip(items, K2)])
    K1, K2 = K1[keep], K2[keep]
    Ky = np.array([idx[it["catid"]] for it, k in zip(items, keep) if k])
    print("só o teu .sfxk   ", score(run(K1, K2, Ky, classifier.BLEND)), flush=True)

    with np.load(base / "treino_emb.npz") as z:
        E = {k: z[k] for k in z.files}
    for n in (5, 10, 20, 30, 50, 100):
        rows = [(f, lab) for lab, fs in sel.items() for f in fs[:n] if f"{f}.1" in E]
        if n > 5 and sum(1 for lab, fs in sel.items() for f in fs[n // 2:n] if f"{f}.1" in E) == 0:
            continue
        A1 = np.stack([E[f"{f}.1"] for f, _ in rows])
        A2 = np.stack([E[f"{f}.2"] for f, _ in rows])
        y = np.array([idx[m[lab][0]] for _, lab in rows])
        all1, all2, ally = np.concatenate([K1, A1]), np.concatenate([K2, A2]), np.concatenate([Ky, y])
        print(f"{n:3} por classe ({len(rows):5} sons + .sfxk)  peso atual  ", score(run(all1, all2, ally, classifier.BLEND)), flush=True)
        for w in (0.8, 0.95):
            print(f"{'':31} peso {w:4}    ", score(run(all1, all2, ally, dict(classifier.BLEND, weight=w))), flush=True)


if __name__ == "__main__":
    cmd, base, teste = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if cmd == "embed":
        embed(base, int(sys.argv[4]))
    else:
        evaluate(base, teste, Path(sys.argv[4]))
