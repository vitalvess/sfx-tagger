"""Memória de exemplos: o que a IA aprende com os sons confirmados por pessoas.

Cada som confirmado vira um "exemplo": a impressão digital do som (embedding CLAP de
512 números de cada um dos dois modelos, sem o áudio) + a categoria certa + o que a IA
tinha sugerido. Os exemplos:
  * melhoram as sugestões logo a seguir: sons parecidos com exemplos confirmados (kNN) e
    um classificador treinado com todos eles (classifier.py);
  * podem ser exportados para um ficheiro .sfxk e importados noutro computador,
    para que o trabalho de cada colega melhore a IA de todos.

Tudo fica em <pasta da app>/conhecimento/ — independente das bibliotecas de som.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
import zipfile
from collections import Counter, defaultdict

import numpy as np

from . import __version__, classifier, paths, precisao

FORMAT = "sfxtagger-conhecimento"   # identificador interno dos .sfxk: fica igual para as versões antigas os lerem
FORMAT_VERSION = 1
MIN_SIM = 0.65        # abaixo disto um exemplo não conta como "parecido" (= ai.MEMORY)


def example_id(emb: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(emb, np.float16).tobytes()).hexdigest()[:16]


class Knowledge:
    def __init__(self) -> None:
        self.dir = paths.app_dir() / "conhecimento"
        self.dir.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.meta: dict[str, dict] = {}      # id -> dados do exemplo
        self.emb: dict[str, np.ndarray] = {}  # id -> embedding (float32 normalizado)
        self.emb2: dict[str, np.ndarray] = {}  # id -> embedding do segundo modelo (se houver)
        self.emb3: dict[str, np.ndarray] = {}  # id -> embedding do 3.º modelo (Precisão máxima)
        # cópias de cada exemplo misturado com um fundo (rua, sala…): só para treinar, nunca
        # partilhadas nem contadas. id -> (modelo 1, modelo 2, modelo 3). exp_aumento.py
        self.aug: dict[str, tuple] = {}
        self._clf_key = None
        self._clf = None
        self._clf_ready = False
        self._version = 0                              # sobe a cada alteração dos exemplos
        self._training: threading.Thread | None = None
        self._trained = threading.Condition(self.lock)
        self._matrix = None
        self._ids: list[str] = []
        self._timer: threading.Timer | None = None
        self._load()

    # --- persistência ------------------------------------------------------------
    def _load(self) -> None:
        mf, ef = self.dir / "exemplos.json", self.dir / "embeddings.npz"
        if mf.exists():
            self.meta = json.loads(mf.read_text(encoding="utf-8"))
        if ef.exists():
            with np.load(ef) as z:
                self.emb = {k: z[k].astype(np.float32) for k in z.files}
        ef2 = self.dir / "embeddings2.npz"
        if ef2.exists():
            with np.load(ef2) as z:
                self.emb2 = {k: z[k].astype(np.float32) for k in z.files if k in self.emb}
        ef3 = self.dir / "embeddings3.npz"
        if ef3.exists():
            with np.load(ef3) as z:
                self.emb3 = {k: z[k].astype(np.float32) for k in z.files if k in self.emb}
        fa = self.dir / "aumentos.npz"
        if fa.exists():
            with np.load(fa) as z:
                parts: dict = {}
                for k in z.files:
                    eid, m = k.rsplit("|", 1)
                    parts.setdefault(eid, {})[m] = z[k].astype(np.float32)
                self.aug = {e: (p["1"], p["2"], p["3"]) for e, p in parts.items() if len(p) == 3}
        self.meta = {k: v for k, v in self.meta.items() if k in self.emb}

    def save_extras(self) -> None:
        """3.º modelo e misturas com fundos (escritos à parte: são grandes e mudam pouco)."""
        with self.lock:
            tmp = self.dir / "embeddings3.tmp.npz"
            np.savez_compressed(tmp, **{k: v.astype(np.float16) for k, v in self.emb3.items()})
            os.replace(tmp, self.dir / "embeddings3.npz")
            tmp = self.dir / "aumentos.tmp.npz"
            np.savez_compressed(tmp, **{f"{e}|{m + 1}": v[m].astype(np.float16)
                                        for e, v in self.aug.items() for m in range(3)})
            os.replace(tmp, self.dir / "aumentos.npz")
            self._changed()

    def _save(self) -> None:
        tmp = self.dir / "exemplos.tmp"
        tmp.write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.dir / "exemplos.json")
        tmp = self.dir / "embeddings.tmp.npz"
        np.savez_compressed(tmp, **{k: v.astype(np.float16) for k, v in self.emb.items()})
        os.replace(tmp, self.dir / "embeddings.npz")
        tmp = self.dir / "embeddings2.tmp.npz"
        np.savez_compressed(tmp, **{k: v.astype(np.float16) for k, v in self.emb2.items()})
        os.replace(tmp, self.dir / "embeddings2.npz")
        self._changed()

    def _changed(self) -> None:
        self._matrix = None
        self._clf_ready = False
        self._version += 1

    # --- aprender ----------------------------------------------------------------
    def learn(self, emb: np.ndarray, review: dict, fields: dict, author: str,
              filename: str, save: bool = True, emb2: np.ndarray | None = None,
              emb3: np.ndarray | None = None) -> str:
        """Guarda (ou atualiza) o exemplo de um som confirmado."""
        eid = example_id(emb)
        with self.lock:
            catid = fields.get("catid", "")
            # decisões do ecrã "Rever exemplos suspeitos": reabrir a pasta de origem (learn_existing)
            # não pode desfazê-las; só uma confirmação NOVA, com outra categoria, conta
            gone = self._removed().get(eid)
            if gone is not None and gone == catid:
                return eid
            old = self.meta.get(eid, {})
            ov = old.get("relabeled") or {}
            if ov and catid == ov.get("from"):
                catid = ov.get("to", catid)
            self.meta[eid] = {
                "catid": catid,
                "ai_catid": review.get("ai_catid", ""),
                "ai_score": review.get("ai_score"),
                "verdict": review.get("verdict", ""),
                "heard": review.get("heard", ""),
                "fxname": fields.get("fxname", ""),
                "keywords": fields.get("keywords", ""),
                "author": old.get("author") or author or "anónimo",
                "file": filename,
                "created": old.get("created") or time.strftime("%Y-%m-%d %H:%M"),
                "updated": time.strftime("%Y-%m-%d %H:%M"),
                "origin": old.get("origin", "local"),
                "app": __version__,
            }
            for k in ("relabeled", "kept"):
                if k in old and catid == old.get("catid"):
                    self.meta[eid][k] = old[k]
            self.emb[eid] = np.asarray(emb, np.float32)
            if emb2 is not None:
                self.emb2[eid] = np.asarray(emb2, np.float32)
            if emb3 is not None:
                self.emb3[eid] = np.asarray(emb3, np.float32)
            self._changed()
            if save:
                self._schedule_save()
        return eid

    def _schedule_save(self, delay: float = 2.0) -> None:
        """Agrupa gravações: aprovar 200 sons de seguida grava o ficheiro uma só vez."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(delay, self.save)
        self._timer.daemon = True
        self._timer.start()

    def save(self) -> None:
        with self.lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._save()

    # --- usar --------------------------------------------------------------------
    def _mat(self):
        if self._matrix is None:
            self._ids = list(self.emb)
            self._matrix = (np.stack([self.emb[i] for i in self._ids])
                            if self._ids else np.zeros((0, 512), np.float32))
        return self._matrix

    def neighbours(self, emb: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        with self.lock:
            m = self._mat()
            if not len(m):
                return []
            sims = m @ np.asarray(emb, np.float32)
            order = np.argsort(sims)[::-1]
            out = []
            for i in order:
                s = float(sims[i])
                if s < MIN_SIM:
                    break
                # (o mesmo ficheiro já etiquetado por alguém também conta: é a resposta certa)
                out.append((self.meta[self._ids[i]]["catid"], s))
                if len(out) >= k:
                    break
            return out

    # com milhares de exemplos o treino demora segundos (9130 exemplos: ~25 s): a partir daqui
    # refaz-se em segundo plano e, entretanto, usa-se o classificador anterior (o som acabado de
    # confirmar já conta na memória kNN). wait=True (testes, recalcular) espera pelo novo.
    BACKGROUND_FROM = 2000

    def _training_data(self):
        """(X1, X2, y, use_helper) com os exemplos atuais, ou None se forem poucos."""
        ids = [i for i in self.emb if self.meta.get(i, {}).get("catid")]
        if len(ids) < classifier.MIN_EXAMPLES:
            return None
        from .ucs import load_ucs
        index = {c.catid: n for n, c in enumerate(load_ucs())}
        ids = [i for i in ids if self.meta[i]["catid"] in index]
        # um exemplo antigo sem o segundo modelo não pode desligá-lo para os outros todos
        # (ESC-50: 1 exemplo em 502 custava 3 pontos); só se desiste se faltar em muitos
        with2 = [i for i in ids if i in self.emb2]
        use2 = classifier.TRAIN.get("use_helper", False) and len(with2) >= 0.8 * len(ids)
        if use2:
            ids = with2
        # o 3.º modelo só conta com a Precisão máxima ligada e se quase todos os exemplos o tiverem
        with3 = [i for i in ids if i in self.emb3]
        use3 = (use2 and classifier.TRAIN.get("use_third", False) and precisao.active()
                and len(with3) >= 0.8 * len(ids))
        if use3:
            ids = with3
        if not ids:
            return None
        X1 = [self.emb[i] for i in ids]
        X2 = [self.emb2[i] for i in ids] if use2 else None
        X3 = [self.emb3[i] for i in ids] if use3 else None
        y = [index[self.meta[i]["catid"]] for i in ids]
        if use2 and classifier.TRAIN.get("use_aug", False):   # cópias com fundos, com a categoria atual do exemplo (segue as correções)
            for i in ids:
                a = self.aug.get(i)
                if a is not None:
                    X1.append(a[0])
                    X2.append(a[1])
                    if use3:
                        X3.append(a[2])
                    y.append(index[self.meta[i]["catid"]])
        return (np.stack(X1), np.stack(X2) if use2 else None, np.stack(X3) if use3 else None,
                np.array(y), use2, use3)

    @staticmethod
    def _fit(data):
        if data is None:
            return None
        X1, X2, X3, y, use2, use3 = data
        return classifier.train(X1, X2, y, X3=X3, **dict(classifier.TRAIN, use_helper=use2, use_third=use3))

    def _train_background(self, data, version: int) -> None:
        try:
            clf, failed = self._fit(data), False
        except Exception:   # noqa: BLE001 — fica o anterior (sem repetir em ciclo)
            clf, failed = None, True
        with self.lock:
            self._training = None
            if failed:
                self._clf_ready = True
            elif version == self._version:
                self._clf, self._clf_ready = clf, True
            self._trained.notify_all()

    def warm(self) -> None:
        """Começa já o treino (ao abrir a app), para a primeira sugestão não esperar."""
        threading.Thread(target=self.classifier, daemon=True, name="aquecer-classificador").start()

    def classifier(self, wait: bool = False) -> classifier.Model | None:
        """Classificador treinado com todos os exemplos (refeito quando estes mudam)."""
        with self.lock:
            key = precisao.active()          # ligar/desligar a Precisão máxima muda o classificador
            if key != self._clf_key:
                self._clf_key, self._clf_ready = key, False
            if self._clf_ready:
                return self._clf
            if self._training is None:
                data, version = self._training_data(), self._version
                if data is None or len(data[3]) < self.BACKGROUND_FROM:
                    self._clf, self._clf_ready = self._fit(data), True   # rápido: aqui mesmo
                    return self._clf
                self._training = threading.Thread(target=self._train_background, args=(data, version),
                                                  daemon=True, name="treino-classificador")
                self._training.start()
            if not wait and self._clf is not None:
                return self._clf        # o anterior, até o novo estar pronto
            while not self._clf_ready:
                if self._training is None:      # outra alteração entretanto: treina-se de novo
                    return self.classifier(wait=True)
                self._trained.wait(timeout=1.0)
            return self._clf

    # --- partilhar ---------------------------------------------------------------
    def export_bytes(self, author: str) -> bytes:
        with self.lock:
            ids = list(self.meta)
            mat = (np.stack([self.emb[i] for i in ids]).astype(np.float16)
                   if ids else np.zeros((0, 512), np.float16))
            manifest = {"format": FORMAT, "version": FORMAT_VERSION, "app": __version__,
                        "exported_by": author or "anónimo",
                        "exported_at": time.strftime("%Y-%m-%d %H:%M"),
                        "count": len(ids)}
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("manifesto.json", json.dumps(manifest, ensure_ascii=False, indent=1))
                z.writestr("exemplos.json", json.dumps([dict(self.meta[i], id=i) for i in ids],
                                                        ensure_ascii=False))
                npy = io.BytesIO()
                np.save(npy, mat)
                z.writestr("embeddings.npy", npy.getvalue())
                # segundo modelo (v0.7+): linhas a zero onde não existe; versões antigas ignoram
                mat2 = np.zeros_like(mat)
                for n, i in enumerate(ids):
                    if i in self.emb2:
                        mat2[n] = self.emb2[i]
                npy = io.BytesIO()
                np.save(npy, mat2)
                z.writestr("embeddings2.npy", npy.getvalue())
            return buf.getvalue()

    def import_bytes(self, data: bytes) -> dict:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                manifest = json.loads(z.read("manifesto.json"))
                if manifest.get("format") != FORMAT:
                    raise ValueError
                items = json.loads(z.read("exemplos.json"))
                mat = np.load(io.BytesIO(z.read("embeddings.npy"))).astype(np.float32)
                mat2 = (np.load(io.BytesIO(z.read("embeddings2.npy"))).astype(np.float32)
                        if "embeddings2.npy" in z.namelist() else None)
        except Exception:
            raise ValueError("Este ficheiro não é um ficheiro de conhecimento do Earmark (.sfxk)")
        if len(items) != len(mat) or (len(mat) and mat.shape[1] != 512):
            raise ValueError("O ficheiro de conhecimento está danificado")
        new = updated = same = 0
        authors = Counter()
        with self.lock:
            for n, (item, vec) in enumerate(zip(items, mat)):
                if not item.get("catid"):   # ficheiro editado ou de outra versão
                    continue
                eid = item.pop("id", None) or example_id(vec)
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                item["origin"] = "importado"
                if item.get("author") in (None, "", "anónimo") and manifest.get("exported_by"):
                    item["author"] = manifest["exported_by"]   # quem exportou é quem o ensinou
                authors[item.get("author", "anónimo")] += 1
                old = self.meta.get(eid)
                if old is None:
                    new += 1
                elif old.get("updated", "") < item.get("updated", ""):
                    updated += 1
                else:
                    same += 1
                    continue
                if old and old.get("origin") == "local":
                    item["origin"] = "local"
                self.meta[eid] = item
                self.emb[eid] = vec.astype(np.float32)
                if mat2 is not None and len(mat2) == len(mat) and np.abs(mat2[n]).sum() > 0:
                    v2 = mat2[n]
                    self.emb2[eid] = (v2 / (np.linalg.norm(v2) + 1e-9)).astype(np.float32)
            self._save()
        return {"new": new, "updated": updated, "same": same,
                "from": manifest.get("exported_by", ""), "authors": dict(authors)}

    def forget_author(self, author: str) -> int:
        with self.lock:
            ids = [i for i, m in self.meta.items() if m.get("author") == author]
            for i in ids:
                self.meta.pop(i, None)
                self.emb.pop(i, None)
                self.emb2.pop(i, None)
                self.emb3.pop(i, None)
            self._save()
            return len(ids)

    # --- rever exemplos (ecrã "Rever exemplos suspeitos") -----------------------
    def _removed(self) -> dict:
        """Exemplos apagados na revisão: {id: categoria que tinham}. Ficam registados para que
        reabrir a pasta de origem não os volte a ensinar."""
        if not hasattr(self, "_removed_cache"):
            f = self.dir / "removidos.json"
            try:
                self._removed_cache = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            except (OSError, ValueError):
                self._removed_cache = {}
        return self._removed_cache

    def relabel(self, eid: str, catid: str) -> bool:
        with self.lock:
            m = self.meta.get(eid)
            if m is None:
                return False
            if m.get("catid") != catid:
                m["relabeled"] = {"from": m.get("catid", ""), "to": catid, "at": time.strftime("%Y-%m-%d %H:%M")}
                m["catid"] = catid
            m["kept"] = True        # revisto por uma pessoa: não volta à lista
            m["updated"] = time.strftime("%Y-%m-%d %H:%M")
            self._changed()
            self._schedule_save()
            return True

    def keep(self, eid: str) -> bool:
        with self.lock:
            m = self.meta.get(eid)
            if m is None:
                return False
            m["kept"] = True        # a etiqueta está certa: não volta à lista (não muda o treino)
            from . import suspeitos
            suspeitos._cache["version"] = None
            self._schedule_save()
            return True

    def remove(self, ids: list[str]) -> int:
        with self.lock:
            gone = self._removed()
            n = 0
            for i in ids:
                m = self.meta.pop(i, None)
                if m is None:
                    continue
                gone[i] = m.get("catid", "")
                self.emb.pop(i, None)
                self.emb2.pop(i, None)
                self.emb3.pop(i, None)
                n += 1
            if n:
                tmp = self.dir / "removidos.tmp"
                tmp.write_text(json.dumps(gone, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, self.dir / "removidos.json")
                self._changed()
                self._schedule_save()
            return n

    # --- painel ------------------------------------------------------------------
    def stats(self) -> dict:
        with self.lock:
            items = list(self.meta.values())
        by_author = defaultdict(lambda: {"count": 0, "certo": 0, "medidos": 0})
        confusions = Counter()
        cats = Counter()
        right = measured = 0
        for m in items:
            a = by_author[m.get("author") or "anónimo"]
            a["count"] += 1
            cats[m.get("catid", "")] += 1
            if m.get("verdict"):
                measured += 1
                a["medidos"] += 1
                if m["verdict"] == "certo":
                    right += 1
                    a["certo"] += 1
                elif m.get("ai_catid") and m["ai_catid"] != m["catid"]:
                    confusions[(m["ai_catid"], m["catid"])] += 1
        return {
            "total": len(items),
            "categories": len(cats),
            "measured": measured,
            "right": right,
            "authors": [{"author": k, **v} for k, v in
                        sorted(by_author.items(), key=lambda kv: -kv[1]["count"])],
            "confusions": [{"ai": a, "real": b, "n": n} for (a, b), n in confusions.most_common(10)],
            "top_categories": [{"catid": c, "n": n} for c, n in cats.most_common(10)],
        }


_kb: Knowledge | None = None


def get_knowledge() -> Knowledge:
    global _kb
    if _kb is None:
        _kb = Knowledge()
    return _kb
