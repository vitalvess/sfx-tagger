"""Ferramentas de avaliação da precisão (ESC-50 -> UCS).

    python ferramentas/benchmark/bench.py embed  <pasta ESC-50> [modelo]
    python ferramentas/benchmark/bench.py eval   <pasta ESC-50> [modelo]

As impressões digitais (embeddings) dos 2000 sons são calculadas uma vez e guardadas
em cache, para que cada experiência demore segundos.
Divisão: folds 1-3 = desenvolvimento (afinar), folds 4-5 = teste (confirmar).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sfxtagger import audio, ucs  # noqa: E402
from sfxtagger.ai import MODEL_NAME, Tagger  # noqa: E402

HERE = Path(__file__).parent
MAPPING = {k: v for k, v in json.loads((HERE / "esc50_ucs.json").read_text()).items()
           if not k.startswith("_")}


def load_meta(esc: Path) -> list[dict]:
    with open(esc / "meta" / "esc50.csv", newline="") as f:
        return list(csv.DictReader(f))


def cache_file(esc: Path, model: str) -> Path:
    return esc.parent / f"emb_{model.replace('/', '_')}.npz"


def embed_all(esc: Path, model: str = MODEL_NAME, batch: int = 32) -> dict[str, np.ndarray]:
    out = cache_file(esc, model)
    if out.exists():
        with np.load(out) as z:
            return {k: z[k] for k in z.files}
    tagger = Tagger(model)
    tagger.ensure_loaded()
    meta = load_meta(esc)
    embs: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i in range(0, len(meta), batch):
        rows = meta[i:i + batch]
        waves = [audio.load(esc / "audio" / r["filename"]).mono for r in rows]
        torch = tagger._torch
        with torch.no_grad():
            feats = tagger._proc.feature_extractor(waves, sampling_rate=audio.ANALYSIS_SR,
                                                   return_tensors="pt")
            e = tagger._as_tensor(tagger._model.get_audio_features(**feats)).cpu().numpy()
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        for r, v in zip(rows, e):
            embs[r["filename"]] = v.astype(np.float32)
        print(f"  {i + len(rows)}/{len(meta)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, **embs)
    return embs


class Data:
    """Embeddings + etiquetas, prontos para experiências rápidas."""

    def __init__(self, esc: Path, model: str = MODEL_NAME):
        self.meta = load_meta(esc)
        embs = embed_all(esc, model)
        self.X = np.stack([embs[r["filename"]] for r in self.meta])
        self.cls = [r["category"] for r in self.meta]
        self.fold = np.array([int(r["fold"]) for r in self.meta])
        self.dev = self.fold <= 3
        self.test = ~self.dev
        self.cats = ucs.load_ucs()
        idx = {c.catid: i for i, c in enumerate(self.cats)}
        # máscara (n_sons x n_categorias) das respostas aceites
        self.ok = np.zeros((len(self.meta), len(self.cats)), bool)
        for n, c in enumerate(self.cls):
            for catid in MAPPING[c]:
                self.ok[n, idx[catid]] = True
        # mesma "família" UCS (ex.: DOORS) — acerto parcial
        fam = np.array([c.category for c in self.cats])
        self.ok_family = np.zeros_like(self.ok)
        for n in range(len(self.meta)):
            fams = set(fam[self.ok[n]])
            self.ok_family[n] = np.isin(fam, list(fams))


def metrics(scores: np.ndarray, data: Data, mask: np.ndarray, suggestable=None) -> dict:
    s = scores[mask].copy()
    if suggestable is not None:
        s[:, ~suggestable] = -np.inf
    order = np.argsort(-s, axis=1)
    ok = data.ok[mask]
    okf = data.ok_family[mask]
    rows = np.arange(len(s))[:, None]
    hit = ok[rows, order[:, :5]]
    return {
        "top1": round(float(hit[:, 0].mean()) * 100, 1),
        "top3": round(float(hit[:, :3].any(1).mean()) * 100, 1),
        "top5": round(float(hit.any(1).mean()) * 100, 1),
        "familia": round(float(okf[np.arange(len(s)), order[:, 0]].mean()) * 100, 1),
    }


def per_class(scores: np.ndarray, data: Data, mask: np.ndarray, suggestable=None) -> list:
    s = scores.copy()
    if suggestable is not None:
        s[:, ~suggestable] = -np.inf
    top = np.argmax(s, axis=1)
    res = []
    for c in sorted(set(data.cls)):
        m = mask & (np.array(data.cls) == c)
        idx = np.where(m)[0]
        acc = data.ok[idx, top[idx]].mean() * 100
        wrong = [data.cats[top[i]].catid for i in idx if not data.ok[i, top[i]]]
        common = max(set(wrong), key=wrong.count) if wrong else ""
        res.append((c, round(acc), common))
    return sorted(res, key=lambda r: r[1])


if __name__ == "__main__":
    cmd, esc = sys.argv[1], Path(sys.argv[2])
    model = sys.argv[3] if len(sys.argv) > 3 else MODEL_NAME
    if cmd == "embed":
        embed_all(esc, model)
        print("ok")
