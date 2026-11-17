"""Exemplos aprendidos que provavelmente têm a categoria errada ("Rever exemplos suspeitos").

Um exemplo é suspeito quando três sinais concordam contra a sua etiqueta:
  1. está mais perto do retrato (média) de outra categoria do que do retrato da sua, calculado
     sem ele próprio (leave-one-out), para um exemplo não "puxar" o seu próprio retrato;
  2. os modelos CLAP, só com o áudio, também põem essa outra categoria à frente da sua;
  3. a sua categoria tem pelo menos 3 exemplos (com menos, o retrato não diz nada).
É a ideia do "confident learning": encontrar etiquetas erradas onde os dados discordam delas.
Não altera nada: só lista. As decisões (manter, mudar, apagar) são da pessoa.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from . import ai, classifier, paths

MIN_PER_CLASS = 3

_cache: dict = {"version": None, "items": None}
_lock = threading.Lock()


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def _audio_probs(tagger, E1: np.ndarray, E2: np.ndarray | None) -> np.ndarray | None:
    """= Tagger.audio_probs para muitos sons de uma vez (sem o 3.º modelo)."""
    if tagger is None or tagger.cat_matrix is None:
        return None
    mask = np.where(tagger.suggestable, 0.0, -np.inf)
    p = _softmax_rows(tagger.logit_scale * (E1 @ tagger.cat_matrix.T) + mask)
    h = tagger.helper
    if E2 is not None and h is not None and h.cat_matrix is not None:
        p2 = _softmax_rows(h.logit_scale * (E2 @ h.cat_matrix.T) + mask)
        p = (1 - ai.HELPER_WEIGHT) * p + ai.HELPER_WEIGHT * p2
    if ai.FAMILY_GAMMA > 0:
        p = ai.family_rescore(p, tagger.family_index, ai.FAMILY_GAMMA)
    return p


def find(kb, tagger=None, limit: int = 200) -> dict:
    """Lista os exemplos suspeitos, do mais para o menos provável."""
    with _lock:
        if _cache["version"] == kb._version and _cache["items"] is not None:
            items = _cache["items"]
        else:
            items = _compute(kb, tagger)
            _cache.update(version=kb._version, items=items)
    return {"items": items[:limit], "total": len(items), "checked": _cache.get("checked", 0)}


def _compute(kb, tagger) -> list[dict]:
    with kb.lock:
        ids = [i for i, m in kb.meta.items()
               if m.get("catid") and i in kb.emb and not m.get("kept")]
        use2 = sum(1 for i in ids if i in kb.emb2) >= 0.8 * max(len(ids), 1)
        if use2:
            ids = [i for i in ids if i in kb.emb2]
        E1 = np.stack([kb.emb[i] for i in ids]).astype(np.float32) if ids else np.zeros((0, 512), np.float32)
        E2 = np.stack([kb.emb2[i] for i in ids]).astype(np.float32) if (ids and use2) else None
        metas = [dict(kb.meta[i]) for i in ids]
        # os "mantidos" continuam a contar para os retratos
        all_ids = [i for i, m in kb.meta.items() if m.get("catid") and i in kb.emb and (not use2 or i in kb.emb2)]
        A1 = np.stack([kb.emb[i] for i in all_ids]).astype(np.float32) if all_ids else E1
        A2 = np.stack([kb.emb2[i] for i in all_ids]).astype(np.float32) if (all_ids and use2) else None
        acat = [kb.meta[i]["catid"] for i in all_ids]
    _cache["checked"] = len(ids)
    if len(ids) < 10:
        return []
    F = classifier.features(E1, E2)
    FA = classifier.features(A1, A2)
    cats = sorted(set(acat))
    ci = {c: k for k, c in enumerate(cats)}
    ya = np.array([ci[c] for c in acat])
    sums = np.zeros((len(cats), F.shape[1]), np.float64)
    np.add.at(sums, ya, FA)
    counts = np.bincount(ya, minlength=len(cats)).astype(np.float64)
    protos = sums / np.maximum(counts, 1)[:, None]
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9

    y = np.array([ci[m["catid"]] for m in metas])
    sims = F @ protos.T
    # retrato da própria categoria sem o próprio exemplo
    own_sum = sums[y] - F
    own_n = counts[y] - 1
    own_proto = own_sum / np.maximum(own_n, 1)[:, None]
    own_proto /= np.linalg.norm(own_proto, axis=1, keepdims=True) + 1e-9
    own = (F * own_proto).sum(1)
    sims[np.arange(len(y)), y] = -np.inf
    other = sims.argmax(1)
    other_sim = sims[np.arange(len(y)), other]
    margin = other_sim - own

    P = _audio_probs(tagger, E1, E2)
    uidx = {c.catid: k for k, c in enumerate(tagger.categories)} if P is not None else {}

    out = []
    for k, m in enumerate(metas):
        if own_n[k] + 1 < MIN_PER_CLASS or margin[k] <= 0:
            continue
        sug = cats[other[k]]
        rank_own = rank_sug = None
        if P is not None and m["catid"] in uidx and sug in uidx:
            order = np.argsort(-P[k])
            pos = {int(j): r for r, j in enumerate(order[:50], 1)}
            rank_own = pos.get(uidx[m["catid"]])
            rank_sug = pos.get(uidx[sug])
            if P[k, uidx[sug]] <= P[k, uidx[m["catid"]]]:
                continue          # os modelos concordam com a etiqueta: não é suspeito
        out.append({"id": ids[k], "catid": m["catid"], "suggest": sug, "author": m.get("author") or "anónimo",
                    "file": m.get("file", ""), "own": round(float(own[k]), 3), "other": round(float(other_sim[k]), 3),
                    "margin": round(float(margin[k]), 3), "rank_own": rank_own, "rank_suggest": rank_sug})
    out.sort(key=lambda r: -r["margin"])
    return out


# --- onde está o áudio de cada exemplo (o conhecimento não guarda áudio) --------------------------------

_files: dict | None = None
_files_at = 0.0
_files_lock = threading.Lock()
REINDEX_AFTER = 60.0   # um disco ligado entretanto passa a contar sem reiniciar a app
EXTERNAL = Path("D:/SFX Testes")


def _index() -> dict:
    """{(autor, ficheiro): caminho} e {ficheiro: caminho}, a partir dos sítios conhecidos."""
    idx: dict = {}
    by_name: dict = {}

    def add(author, path: Path):
        idx.setdefault((author, path.name), path)
        by_name.setdefault(path.name, path)

    # 1) plano de redistribuição das bases FSD50K / BBC / ESC-50 (no disco externo, que pode faltar)
    for root in (EXTERNAL,):
        try:
            plan = json.loads((root / "redistribuicao.json").read_text(encoding="utf-8"))
            for s in plan.get("sons", []):
                p = s.get("path")
                if p:
                    add(str(s.get("base", "")).replace("-", ""), Path(p))
        except Exception:  # noqa: BLE001 — disco desligado ou ficheiro ausente
            pass
    # 2) pastas de treino em massa
    treino = Path.home() / "Documents" / "Earmark Treino"
    try:
        for d in treino.glob("Treino - *"):
            for p in d.glob("*.wav"):
                by_name.setdefault(p.name, p)
    except Exception:  # noqa: BLE001
        pass
    # 3) lotes carregados pelo browser (Documentos/SFX Tagger e Documentos/Earmark)
    for name in ("SFX Tagger", "Earmark"):
        try:
            for p in (Path.home() / "Documents" / name).rglob("*.wav"):
                by_name.setdefault(p.name, p)
        except Exception:  # noqa: BLE001
            pass
    # 4) pastas recentes (só as que respondem)
    try:
        s = json.loads((paths.app_dir() / "definicoes.json").read_text(encoding="utf-8"))
        for r in s.get("recent", [])[:8]:
            try:
                for n, p in enumerate(Path(r).rglob("*.wav")):
                    if n > 50000:
                        break
                    by_name.setdefault(p.name, p)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return {"pair": idx, "name": by_name}


def _lookup(meta: dict) -> Path | None:
    f = meta.get("file") or ""
    p = _files["pair"].get((meta.get("author") or "", f)) or _files["name"].get(f)
    try:
        return p if p is not None and p.exists() else None
    except OSError:
        return None


def audio_path(meta: dict) -> Path | None:
    global _files, _files_at
    with _files_lock:
        if _files is None:
            _files, _files_at = _index(), time.time()
        p = _lookup(meta)
        if p is None and time.time() - _files_at > REINDEX_AFTER:
            _files, _files_at = _index(), time.time()   # o disco pode ter sido ligado
            p = _lookup(meta)
        return p


_originals: dict | None = None


def original_class(meta: dict) -> str:
    """A etiqueta que a base de dados deu ao som (ex.: "dog_bark"), lida dos origem.csv que
    acompanham as pastas de treino. Vazio se não se souber."""
    global _originals
    with _files_lock:
        if _originals is None:
            _originals = {}
            roots = [Path.home() / "Documents" / "Earmark Treino", EXTERNAL]
            for root in roots:
                try:
                    for f in root.rglob("origem.csv"):
                        import csv
                        with open(f, newline="", encoding="utf-8-sig") as h:
                            for row in csv.DictReader(h):
                                if row.get("ficheiro") and row.get("classe original"):
                                    _originals.setdefault(row["ficheiro"], row["classe original"])
                except Exception:  # noqa: BLE001 — disco desligado
                    continue
    return _originals.get(meta.get("file") or "", "")


def external_connected() -> bool:
    try:
        return EXTERNAL.exists()
    except OSError:
        return False


def refresh_index() -> None:
    global _files
    with _files_lock:
        _files = None
