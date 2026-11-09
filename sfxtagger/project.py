"""Uma biblioteca = uma pasta de sons. O estado fica em <pasta>/.sfxtagger/."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import threading
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from . import audio, avaliacao, classifier, metadata, precisao, ucs
from .ai import GROUP_BETA, get_tagger, name_words
from .knowledge import example_id, get_knowledge

STATE_DIR = ".sfxtagger"
AUDIO_EXT = {".wav"}

FIELD_KEYS = ["catid", "category", "subcategory", "category_full", "fxname", "description",
              "keywords", "notes", "library", "designer", "microphone", "rec_medium",
              "location", "creator_id", "source_id"]
# campos que se aplicam a toda a biblioteca (definidos uma vez)
DEFAULT_KEYS = ["library", "designer", "microphone", "rec_medium", "location",
                "creator_id", "source_id"]

FXNAME_MAX = 30
SEGMENT_MIN_DURATION = 15.0   # gravações mais longas são ouvidas também por partes
SEGMENT_MAX = 10
SEGMENT_SECONDS = 10.0


def _now() -> float:
    return time.time()


class Project:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.state_dir = self.root / STATE_DIR
        self.lock = threading.RLock()
        self.records: dict[str, dict] = {}
        self.defaults: dict[str, str] = {k: "" for k in DEFAULT_KEYS}
        self.embeddings: dict[str, np.ndarray] = {}
        self.embeddings2: dict[str, np.ndarray] = {}   # segundo modelo CLAP
        self.embeddings3: dict[str, np.ndarray] = {}   # terceiro modelo (Precisão máxima)
        self.waveforms: dict[str, np.ndarray] = {}     # formas de onda (para o ecrã)
        self._arrays_dirty = False   # embeddings/formas de onda mudaram desde a última gravação
        self.writing = False         # a gravar nos WAV (não pode começar uma análise)
        self.evaluation: dict | None = None   # último "Testar a IA" (com o detalhe por som)
        self.ignored: set[str] = set()        # sons tirados da lista (o ficheiro continua na pasta)
        self._stop = threading.Event()        # pedido para parar a análise ou o teste
        self.warning = ""
        self.job = {"running": False, "phase": "", "done": 0, "total": 0,
                    "current": "", "errors": 0, "message": ""}
        self._load()

    # --- persistência ------------------------------------------------------
    @property
    def state_file(self) -> Path:
        return self.state_dir / "projeto.json"

    @property
    def emb_file(self) -> Path:
        return self.state_dir / "embeddings.npz"

    @property
    def emb2_file(self) -> Path:
        return self.state_dir / "embeddings2.npz"

    @property
    def emb3_file(self) -> Path:
        return self.state_dir / "embeddings3.npz"

    @property
    def wave_file(self) -> Path:
        return self.state_dir / "formas.npz"

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.records = data.get("records", {})
                self.defaults.update(data.get("defaults", {}))
                self.ignored = set(data.get("ignored", []))
            except (ValueError, OSError):
                # ficheiro danificado (ex.: falha de energia): guarda-o à parte e recomeça
                bad = self.state_file.with_name(f"projeto.danificado-{int(_now())}.json")
                os.replace(self.state_file, bad)
                self.warning = ("O ficheiro de progresso desta pasta estava danificado e foi guardado "
                                f"à parte ({bad.name}). Os metadados já gravados nos WAV não se perderam.")
        self.embeddings = self._load_arrays(self.emb_file)
        self.embeddings2 = self._load_arrays(self.emb2_file)
        self.embeddings3 = self._load_arrays(self.emb3_file)
        self.waveforms = self._load_arrays(self.wave_file)
        # versões antigas guardavam a forma de onda dentro do projeto.json
        for rid, rec in self.records.items():
            wf = (rec.get("tech") or {}).pop("waveform", None)
            if wf is not None:
                self.waveforms[rid] = np.asarray(wf, np.float16)
                self._arrays_dirty = True

    def _load_arrays(self, f: Path) -> dict[str, np.ndarray]:
        """Um ficheiro danificado (ex.: conflito do OneDrive) não pode impedir de abrir a pasta."""
        if not f.exists():
            return {}
        try:
            with np.load(f) as z:
                return {k: z[k] for k in z.files}
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            try:
                os.replace(f, f.with_name(f"{f.stem}.danificado-{int(_now())}.npz"))
            except OSError:
                pass
            self.warning = ("Parte do progresso desta pasta estava danificada. Se as sugestões "
                            "parecerem incompletas, usa Analisar > Reanalisar tudo.")
            return {}

    def save(self) -> None:
        try:
            self._save()
        except OSError as e:
            print(f"  Não foi possível guardar o progresso: {e}", flush=True)
            self.warning = ("Não é possível guardar o progresso nesta pasta (pode ser só de leitura). "
                            "O que fizeres pode perder-se ao fechar a app.")

    def _save(self) -> None:
        with self.lock:
            self.state_dir.mkdir(exist_ok=True)
            if os.name == "nt":  # pasta escondida no Windows
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetFileAttributesW(str(self.state_dir), 0x02)
                except Exception:
                    pass
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": 2, "defaults": self.defaults, "records": self.records,
                                       "ignored": sorted(self.ignored)}, ensure_ascii=False,
                                      separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self.state_file)
            # os ficheiros grandes só são regravados quando mudam (não a cada edição)
            if self._arrays_dirty:
                for data, target in ((self.embeddings, self.emb_file), (self.embeddings2, self.emb2_file),
                                     (self.embeddings3, self.emb3_file), (self.waveforms, self.wave_file)):
                    if data:
                        tmp = target.with_name(target.stem + ".tmp.npz")
                        np.savez_compressed(tmp, **data)
                        os.replace(tmp, target)
                    else:
                        target.unlink(missing_ok=True)   # todos os sons foram removidos
                self._arrays_dirty = False

    # --- ficheiros ---------------------------------------------------------
    def abspath(self, rec: dict) -> Path:
        return self.root / rec["path"]

    def scan(self) -> dict:
        """Procura WAVs na pasta (e subpastas). Mantém o trabalho já feito."""
        found: dict[str, Path] = {}
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for fn in sorted(filenames):
                if fn.startswith(".") or Path(fn).suffix.lower() not in AUDIO_EXT:
                    continue
                p = Path(dirpath) / fn
                rel = p.relative_to(self.root).as_posix()
                if rel not in self.ignored:
                    found[rel] = p
        with self.lock:
            by_path = {r["path"]: rid for rid, r in self.records.items()}
            added = removed = 0
            for rel, p in found.items():
                st = p.stat()
                rid = by_path.get(rel)
                if rid is None:
                    rid = hashlib.sha1(rel.encode()).hexdigest()[:12]
                    while rid in self.records:
                        rid = hashlib.sha1((rid + rel).encode()).hexdigest()[:12]
                    self.records[rid] = self._new_record(rid, rel, p, st)
                    added += 1
                else:
                    rec = self.records[rid]
                    rec.pop("missing", None)
                    if rec.get("mtime") != st.st_mtime or rec.get("size") != st.st_size:
                        # alterado fora da app: volta a ler metadados, mantém sugestões
                        rec.update(size=st.st_size, mtime=st.st_mtime)
                        if not rec.get("dirty"):
                            self._read_existing(rec, p)
            for rid, rec in self.records.items():
                if rec["path"] not in found and not rec.get("missing"):
                    rec["missing"] = True
                    removed += 1
            self._compute_groups()
            self.save()
        return {"added": added, "missing": removed, "total": len(found)}

    def _new_record(self, rid: str, rel: str, p: Path, st) -> dict:
        rec = {"id": rid, "path": rel, "size": st.st_size, "mtime": st.st_mtime,
               "tech": None, "suggestion": None, "fields": {k: "" for k in FIELD_KEYS},
               "status": "novo", "had_metadata": False, "dirty": False, "group": None,
               "error": None}
        self._read_existing(rec, p)
        return rec

    def _read_existing(self, rec: dict, p: Path) -> None:
        fields = {}
        try:
            fields.update(metadata.read_metadata(p)["fields"])
        except Exception as e:
            rec["error"] = f"Não foi possível ler metadados: {e}"
        for k, v in ucs.parse_filename(p.stem).items():
            fields.setdefault(k, v)
        cat = ucs.lookup(fields.get("catid", ""))
        if cat:
            fields.setdefault("category", cat.category)
            fields.setdefault("subcategory", cat.subcategory)
        rec["fields"].update({k: v for k, v in fields.items() if k in FIELD_KEYS and v})
        rec["had_metadata"] = bool(fields.get("catid") or fields.get("description"))

    # --- grupos (séries de takes: "Door Creak 01", "Door Creak 02"...) -----
    def _compute_groups(self) -> None:
        buckets: dict[str, list[str]] = {}
        for rid, rec in self.records.items():
            p = Path(rec["path"])
            # num nome UCS o número da série está no FXName, a meio do nome do ficheiro
            uname = ucs.parse_filename(p.stem)
            name = f"{uname['catid']} {uname['fxname']}" if uname else p.stem
            stem = re.sub(r"[\s_\-.]*\(?\d+\)?$", "", name).strip(" _-.").lower()
            # "ZOOM0001", "ZOOM0002" não são uma série: o nome não diz nada sobre o som
            if not stem or not name_words(stem):
                continue
            buckets.setdefault(f"{p.parent.as_posix()}/{stem}", []).append(rid)
        for rec in self.records.values():
            rec["group"] = None
        for key, ids in buckets.items():
            if len(ids) >= 2:
                gid = hashlib.sha1(key.encode()).hexdigest()[:8]
                for rid in ids:
                    self.records[rid]["group"] = gid

    def full_record(self, rid: str) -> dict:
        """Registo completo para o painel de detalhe (inclui a forma de onda)."""
        rec = dict(self.records[rid])
        wf = self.waveforms.get(rid)
        if rec.get("tech") and wf is not None:
            rec["tech"] = {**rec["tech"], "waveform": [round(float(x), 3) for x in wf]}
        return rec

    def similar(self, rid: str, n: int = 20, min_sim: float = 0.8) -> list[dict]:
        v = self.embeddings.get(rid)
        if v is None:
            return []
        with self.lock:
            ids = [k for k in self.embeddings if k != rid and k in self.records]
            if not ids:
                return []
            mat = np.stack([self.embeddings[k] for k in ids])
        sims = mat @ v
        order = np.argsort(sims)[::-1][:n]
        return [{"id": ids[i], "sim": round(float(sims[i]), 3)} for i in order
                if sims[i] >= min_sim]

    # --- tirar sons da lista ---------------------------------------------------------
    def remove(self, ids: list[str], delete_files: bool) -> int | None:
        """Tira os sons da lista. delete_files só para lotes carregados pelo browser, que são
        cópias da app; numa pasta da pessoa o ficheiro nunca é apagado. None = há trabalho a decorrer."""
        with self.lock:
            if self.job["running"] or self.writing:
                return None
            n = 0
            for rid in ids:
                rec = self.records.pop(rid, None)
                if rec is None:
                    continue
                n += 1
                for store in (self.embeddings, self.embeddings2, self.embeddings3, self.waveforms):
                    store.pop(rid, None)
                (self.state_dir / "preview" / f"{rid}.wav").unlink(missing_ok=True)
                if delete_files:
                    self.abspath(rec).unlink(missing_ok=True)
                else:
                    self.ignored.add(rec["path"])
            if n:
                self._arrays_dirty = True
                self._compute_groups()
                self.save()
            return n

    def restore_ignored(self) -> int:
        with self.lock:
            n = len(self.ignored)
            self.ignored.clear()
        self.scan()
        return n

    def stop_job(self) -> bool:
        """Pede para parar: o som que está a ser ouvido termina e o trabalho feito fica guardado."""
        if not self.job["running"]:
            return False
        self._stop.set()
        self.job["phase"] = "a parar"
        return True

    # --- análise -----------------------------------------------------------
    def start_analysis(self, ids: list[str] | None, force: bool) -> bool:
        with self.lock:
            if self.job["running"] or self.writing:
                return False
            if ids:
                todo = [i for i in ids if i in self.records]
            else:
                todo = [rid for rid, r in self.records.items()
                        if not r.get("missing") and (force or r.get("suggestion") is None)]
            self.job = {"running": True, "phase": "a preparar", "done": 0, "total": len(todo),
                        "current": "", "errors": 0, "message": ""}
            self._stop.clear()
        threading.Thread(target=self._run_analysis, args=(todo,), daemon=True).start()
        return True

    def _run_analysis(self, todo: list[str]) -> None:
        tagger = get_tagger()
        try:
            self.job["phase"] = tagger.status
            tagger.ensure_loaded()
            precisao.wait_if_starting()
        except Exception as e:
            traceback.print_exc()
            self.job.update(running=False, phase="erro",
                            message=f"Falha ao carregar o modelo de IA: {e}")
            return
        self.job["phase"] = "a analisar"
        last_save = _now()
        save_cost = 0.0
        processed: list[str] = []
        for rid in todo:
            if self._stop.is_set():
                break
            rec = self.records.get(rid)
            if rec is None:
                continue
            self.job["current"] = rec["path"]
            try:
                self._analyze_one(rec, tagger)
            except Exception as e:
                traceback.print_exc()
                rec["error"] = f"Erro na análise: {e}"
                self.job["errors"] += 1
            self.job["done"] += 1
            processed.append(rid)
            # em bibliotecas grandes guardar demora: nunca mais de ~3% do tempo total
            if _now() - last_save > max(10.0, 30 * save_cost):
                t0 = _now()
                self.save()
                last_save = _now()
                save_cost = last_save - t0
        try:
            with self.lock:
                self._compute_groups()
            stopped = self._stop.is_set()
            if not stopped:
                self.job["phase"] = "a comparar takes da mesma série"
            self._apply_group_consensus(tagger, processed)
            self.save()
            self.job.update(running=False, phase="parado" if stopped else "concluído", current="")
        except Exception as e:  # noqa: BLE001 — sem isto a app ficava presa em "a analisar"
            traceback.print_exc()
            self.job.update(running=False, phase="erro", current="", message=f"A análise terminou com um erro: {e}")

    def _analyze_one(self, rec: dict, tagger) -> None:
        p = self.abspath(rec)
        loaded = audio.load(p)
        tech = audio.analyze(loaded)
        wave = np.asarray(audio.waveform_peaks(loaded.mono, 800), np.float16)
        parents = list(Path(rec["path"]).parent.parts)[-2:]
        # num nome UCS (CatID_FXName_Creator_Source) só o FXName descreve o som
        words = name_words(ucs.parse_filename(p.stem).get("fxname") or p.stem)
        folder = name_words(*parents)
        windows = audio.best_windows(loaded.mono)
        emb = tagger.embed_audio(windows)
        emb2 = tagger.embed_audio_helper(windows)
        emb3 = precisao.embed(windows) if precisao.active() else None
        kb = get_knowledge()
        sug = tagger.suggest(emb, words, folder, memory=kb.neighbours(emb),
                             helper_emb=emb2, clf=self._classifier(), third_emb=emb3)
        sug["name_words"] = words
        sug["folder_words"] = folder
        with self.lock:
            self.embeddings[rec["id"]] = emb
            if emb2 is not None:
                self.embeddings2[rec["id"]] = emb2
            if emb3 is not None:
                self.embeddings3[rec["id"]] = emb3
            self.waveforms[rec["id"]] = wave
            self._arrays_dirty = True
            rec["tech"] = tech
            rec["suggestion"] = sug
            # as partes são calculadas quando alguém abre o som (segments_for): analisar a
            # biblioteca inteira por partes deixava a análise ~2x mais lenta
            rec.pop("segments", None)
            rec.pop("mixed", None)
            rec["error"] = None
            self._redraft(rec)
            if rec["status"] == "novo":
                rec["status"] = "sugerido"

    # --- testar a IA com sons já classificados --------------------------------------
    def start_evaluation(self) -> bool:
        with self.lock:
            if self.job["running"] or self.writing:
                return False
            todo = [rid for rid, r in self.records.items() if not r.get("missing")]
            self.job = {"running": True, "phase": "a testar a IA", "done": 0, "total": len(todo),
                        "current": "", "errors": 0, "message": "", "kind": "teste"}
            self._stop.clear()
        threading.Thread(target=self._run_evaluation, args=(todo,), daemon=True).start()
        return True

    def _embeddings_for_test(self, rec: dict, tagger) -> None:
        """Ouve o som, se ainda não tiver sido ouvido. Não mexe nos campos do som."""
        rid = rec["id"]
        third = precisao.active()
        if rid in self.embeddings and rid in self.embeddings2 and (rid in self.embeddings3 or not third):
            return
        windows = audio.best_windows(audio.load(self.abspath(rec)).mono)
        emb, emb2 = tagger.embed_audio(windows), tagger.embed_audio_helper(windows)
        emb3 = precisao.embed(windows) if third else None
        with self.lock:
            self.embeddings[rid] = emb
            self.embeddings2[rid] = emb2
            if emb3 is not None:
                self.embeddings3[rid] = emb3
            self._arrays_dirty = True

    def _run_evaluation(self, todo: list[str]) -> None:
        try:
            tagger = get_tagger()
            tagger.ensure_loaded()
            precisao.wait_if_starting()
            answers = avaliacao.load_answers(self.root)
            truths, no_label = {}, 0
            for rid in todo:
                rec = self.records[rid]
                truth = avaliacao.truth_of(self.abspath(rec), answers)
                if truth:
                    truths[rid] = truth
                else:
                    no_label += 1
            self.job["total"] = len(truths)
            if not truths:
                self.evaluation = None
                self.job.update(running=False, phase="erro",
                                message="Esta pasta não tem sons classificados: o teste precisa do CatID nos ficheiros ou de um respostas.csv.")
                return
            last_save, save_cost = _now(), 0.0
            for rid in truths:   # primeiro ouve tudo, para o limiar do classificador ser estável
                if self._stop.is_set():
                    self.save()
                    self.job.update(running=False, phase="parado", current="")
                    return
                self.job["current"] = self.records[rid]["path"]
                try:
                    self._embeddings_for_test(self.records[rid], tagger)
                except Exception:  # noqa: BLE001 — um ficheiro mau não estraga o teste
                    traceback.print_exc()
                    self.job["errors"] += 1
                self.job["done"] += 1
                if _now() - last_save > max(30.0, 30 * save_cost):   # testes em massa demoram horas
                    t0 = _now()
                    self.save()
                    last_save = _now()
                    save_cost = last_save - t0
            self.job["phase"] = "a comparar com as respostas"
            kb = get_knowledge()
            clf = self._classifier(fresh=True)
            rows, known = [], 0
            for rid, truth in truths.items():
                emb = self.embeddings.get(rid)
                if emb is None:
                    continue
                if example_id(emb) in kb.meta:   # a IA já aprendeu este som: não seria um teste justo
                    known += 1
                    continue
                sug = tagger.suggest(emb, [], [], memory=kb.neighbours(emb), helper_emb=self.embeddings2.get(rid),
                                     clf=clf, third_emb=self.embeddings3.get(rid))
                rows.append({"file": self.records[rid]["path"], "truth": truth, "ai": sug["categories"][0]["catid"],
                             "confidence": sug["confidence"]["value"], "level": sug["confidence"]["level"],
                             **avaliacao.score(sug["categories"], truth)})
            summary = avaliacao.summarize(rows, no_label, known, self.job["errors"])
            models = 3 if precisao.active() else 2
            summary.update(folder=self.root.name, examples=len(kb.meta), models=models)
            self.evaluation = {"summary": summary, "rows": rows}
            if rows:
                avaliacao.remember(summary, self.root.name, len(kb.meta), models)
            self.save()
            self.job.update(running=False, phase="concluído", current="")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.job.update(running=False, phase="erro", current="", message=f"O teste terminou com um erro: {e}")

    def _redraft(self, rec: dict) -> None:
        """Rascunhos só da IA (ninguém mexeu, nada vinha no ficheiro) refazem-se."""
        if (rec["status"] in ("novo", "sugerido") and not rec.get("dirty")
                and not rec.get("had_metadata")):
            for k in ("catid", "category", "subcategory", "category_full",
                      "fxname", "description", "keywords"):
                rec["fields"][k] = ""
        self._fill_draft(rec)

    def segments_for(self, rid: str) -> dict:
        """Partes de uma gravação longa, calculadas na primeira vez que o som é aberto."""
        rec = self.records[rid]
        if "segments" not in rec and rec.get("tech"):
            if (rec["tech"].get("duration") or 0) < SEGMENT_MIN_DURATION:
                segments = []
            else:
                tagger = get_tagger()
                tagger.ensure_loaded()
                loaded = audio.load(self.abspath(rec))
                segments = self._segments(loaded.mono, rec["tech"], tagger)
            with self.lock:
                rec["segments"] = segments
                rec["mixed"] = _is_mixed(segments)
                self.save()
        return {"segments": rec.get("segments") or [], "mixed": rec.get("mixed", False)}

    def _segments(self, x: np.ndarray, tech: dict, tagger) -> list[dict]:
        """O que a IA ouve em cada parte de uma gravação longa (ambientes, vários takes)."""
        sr = audio.ANALYSIS_SR
        dur = len(x) / sr
        if dur < SEGMENT_MIN_DURATION:
            return []
        # o modelo ouve sempre blocos de 10 s (um bloco de 2 s custa o mesmo): takes curtos
        # seguidos juntam-se no mesmo bloco, e os ambientes vão em blocos de 10 s
        if tech.get("kind") == "takes" and tech.get("event_times"):
            spans: list[tuple[float, float]] = []
            for s0, e0 in tech["event_times"]:
                if e0 - s0 < 0.3:
                    continue
                if spans and e0 - spans[-1][0] <= SEGMENT_SECONDS:
                    spans[-1] = (spans[-1][0], e0)
                else:
                    spans.append((s0, min(e0, s0 + SEGMENT_SECONDS)))
        else:
            spans = [(t, min(t + SEGMENT_SECONDS, dur)) for t in np.arange(0, dur - 2.0, SEGMENT_SECONDS)]
        if len(spans) > SEGMENT_MAX:   # gravação muito longa: partes espalhadas por todo o ficheiro
            pick = np.linspace(0, len(spans) - 1, SEGMENT_MAX).round().astype(int)
            spans = [spans[i] for i in pick]
        if len(spans) < 2:
            return []
        _, probs = tagger.window_probs([x[int(a * sr):int(b * sr)] for a, b in spans])
        out: list[dict] = []
        for (a, b), pr in zip(spans, probs):
            i = int(pr.argmax())
            catid = tagger.categories[i].catid
            if out and out[-1]["catid"] == catid and a - out[-1]["end"] < 1.0:
                out[-1]["end"] = round(float(b), 2)
                out[-1]["score"] = round(max(out[-1]["score"], float(pr[i])), 3)
            else:
                out.append({"start": round(float(a), 2), "end": round(float(b), 2),
                            "catid": catid, "score": round(float(pr[i]), 3)})
        return out

    # --- classificador treinado, com a "porta" ajustada a esta biblioteca --------------
    def _classifier(self, fresh: bool = False):
        """O classificador da memória, com o limiar de cada categoria acima do que é típico
        NESTA biblioteca (gravações da mesma sala/gravador parecem-se todas)."""
        kb = get_knowledge()
        base = kb.classifier()
        if base is None:
            return None
        with self.lock:   # a análise pode estar a acrescentar sons noutra thread
            embeddings = dict(self.embeddings)
            embeddings2 = dict(self.embeddings2)
        n = len(embeddings)
        cached = getattr(self, "_clf_cache", None)
        if (not fresh and cached and cached[0] is base
                and n < max(cached[1] * 1.25, cached[1] + 25)):
            return cached[2]
        X1, X2 = [], []
        for rid, e in embeddings.items():
            e2 = embeddings2.get(rid)
            if base.use_helper and e2 is None:
                continue
            X1.append(e)
            X2.append(e2)
        if len(X1) < 20:   # biblioteca pequena: junta os exemplos confirmados (mais prudente)
            with kb.lock:
                for i, e in kb.emb.items():
                    e2 = kb.emb2.get(i)
                    if base.use_helper and e2 is None:
                        continue
                    X1.append(e)
                    X2.append(e2)
        bg = (classifier.features(np.stack(X1), np.stack(X2) if base.use_helper else None)
              if X1 else None)
        clf = classifier.adapt(base, bg, gate_at=classifier.BLEND["gate_at"], **classifier.ADAPT)
        self._clf_cache = (base, n, clf)
        return clf

    # --- séries: takes do mesmo som ajudam-se uns aos outros ------------------------
    def _group_priors(self, tagger) -> dict[str, np.ndarray]:
        groups: dict[str, list[str]] = {}
        for rid, rec in self.records.items():
            if rec.get("group") and rid in self.embeddings and not rec.get("missing"):
                groups.setdefault(rec["group"], []).append(rid)
        priors = {}
        for ids in groups.values():
            if len(ids) < 2:
                continue
            P = np.stack([tagger.audio_probs(self.embeddings[r], self.embeddings2.get(r),
                                             self.embeddings3.get(r)) for r in ids])
            for n, r in enumerate(ids):
                # a opinião dos OUTROS takes (sem contar o próprio duas vezes)
                priors[r] = np.delete(P, n, axis=0).mean(0)
        return priors

    def _apply_group_consensus(self, tagger, ids: list[str]) -> None:
        priors = self._group_priors(tagger)
        if not priors:
            return
        kb = get_knowledge()
        clf = self._classifier(fresh=True)
        with self.lock:
            for rid in ids:
                rec = self.records.get(rid)
                if rec is None or rid not in priors or rec.get("suggestion") is None:
                    continue
                old = rec["suggestion"]
                emb = self.embeddings[rid]
                sug = tagger.suggest(emb, old.get("name_words") or [], old.get("folder_words") or [],
                                     memory=kb.neighbours(emb), helper_emb=self.embeddings2.get(rid),
                                     clf=clf, prior=priors[rid], prior_weight=GROUP_BETA,
                                     third_emb=self.embeddings3.get(rid),
                                     heard=bool(old.get("from_heard")))
                for k in ("name_words", "folder_words", "from_heard"):
                    if k in old:
                        sug[k] = old[k]
                rec["suggestion"] = sug
                self._redraft(rec)

    def _fill_draft(self, rec: dict) -> None:
        """Preenche só os campos vazios — nunca apaga o trabalho de uma pessoa."""
        f = rec["fields"]
        sug = rec["suggestion"] or {}
        tech = rec["tech"] or {}
        cats = sug.get("categories") or []
        words = sug.get("name_words") or []
        ai_kw = sug.get("keywords") or []
        if not f.get("catid") and cats:
            self.apply_catid(rec, cats[0]["catid"])
        cat = ucs.lookup(f.get("catid", ""))
        if not f.get("fxname"):
            f["fxname"] = (_title(words[:4], FXNAME_MAX) if words
                           else _fxname_from_category(cat, ai_kw))
        if not f.get("keywords"):
            kws = []
            if cat:
                kws.append(cat.category.lower())
                if cat.subcategory.lower() not in ("misc", "general"):
                    kws.append(cat.subcategory.lower())
            kws += words[:6] + ai_kw
            f["keywords"] = ", ".join(dict.fromkeys(kws))
        if not f.get("description"):
            # sem nome útil, a descrição parte do FXName (coerente com a categoria escolhida)
            subject = words or (f.get("fxname") or "").lower().split()
            attrs = dict(sug.get("attributes") or {})
            if (sug.get("confidence") or {}).get("level") == "baixa":
                # a IA não sabe bem o que é: não inventa fonte nem ação, fica no que se ouve
                attrs.pop("source", None)
                attrs.pop("action", None)
            f["description"] = _draft_description(subject, attrs, tech)
        for k in DEFAULT_KEYS:
            if not f.get(k) and self.defaults.get(k):
                f[k] = self.defaults[k]

    # --- edição ------------------------------------------------------------
    @staticmethod
    def apply_catid(rec: dict, catid: str) -> None:
        cat = ucs.lookup(catid)
        f = rec["fields"]
        f["catid"] = catid
        f["category"] = cat.category if cat else ""
        f["subcategory"] = cat.subcategory if cat else ""
        f["category_full"] = cat.full if cat else ""

    def update_fields(self, ids: list[str], fields: dict, status: str | None = None) -> None:
        with self.lock:
            for rid in ids:
                rec = self.records.get(rid)
                if rec is None:
                    continue
                for k, v in fields.items():
                    if k == "catid":
                        self.apply_catid(rec, v)
                    elif k in FIELD_KEYS:
                        rec["fields"][k] = v
                if fields:
                    rec["dirty"] = True
                    if rec["status"] == "gravado":
                        rec["status"] = "revisto"
                if status:
                    rec["status"] = status
                    if status == "revisto":
                        self._record_review(rec)
                elif "catid" in fields and (rec.get("review") or {}).get("verdict"):
                    self._record_review(rec)   # corrigido depois de confirmar: a IA aprende a certa
            self.save()

    # --- confirmação humana (mede se a IA acertou) ----------------------------
    @staticmethod
    def _remember_ai(rec: dict) -> dict:
        """Fixa as sugestões originais da IA, antes de qualquer pista escrita pela pessoa."""
        review = rec.setdefault("review", {})
        cats = (rec.get("suggestion") or {}).get("categories") or []
        if cats and "ai_catid" not in review:
            review.update(ai_catid=cats[0]["catid"], ai_score=cats[0]["score"],
                          ai_top5=[c["catid"] for c in cats])
        return review

    def _record_review(self, rec: dict) -> None:
        review = self._remember_ai(rec)
        if "ai_catid" not in review:  # som não analisado: nada a medir
            return
        final = rec["fields"].get("catid", "")
        if final == review["ai_catid"]:
            verdict = "certo"
        elif final in review.get("ai_top5", []):
            verdict = "top5"
        else:
            verdict = "errado"
        review.update(final_catid=final, verdict=verdict, at=_now())
        self._learn(rec)

    # --- memória de exemplos (o que a IA aprende com as confirmações) -----------------
    @property
    def author(self) -> str:
        return self.defaults.get("creator_id") or self.defaults.get("designer") or ""

    def _learn(self, rec: dict, save: bool = True) -> None:
        emb = self.embeddings.get(rec["id"])
        review = rec.get("review") or {}
        if emb is None or not review.get("verdict") or not rec["fields"].get("catid"):
            return
        get_knowledge().learn(emb, review, rec["fields"], self.author,
                              Path(rec["path"]).name, save=save,
                              emb2=self.embeddings2.get(rec["id"]))

    def learn_existing(self) -> int:
        """Ensina à memória os sons desta biblioteca que já tinham sido confirmados."""
        n = 0
        with self.lock:
            for rec in self.records.values():
                if (rec.get("review") or {}).get("verdict") and rec["id"] in self.embeddings:
                    self._learn(rec, save=False)
                    n += 1
        if n:
            get_knowledge().save()
        return n

    def resuggest(self) -> int:
        """Recalcula as sugestões dos sons ainda por rever, com o conhecimento atual.
        Usa as impressões digitais já guardadas: não volta a ouvir o áudio."""
        tagger = get_tagger()
        tagger.ensure_loaded()
        kb = get_knowledge()
        clf = self._classifier(fresh=True)
        priors = self._group_priors(tagger)
        changed = 0
        with self.lock:
            for rec in self.records.values():
                emb = self.embeddings.get(rec["id"])
                old = rec.get("suggestion")
                if emb is None or old is None or rec["status"] not in ("sugerido", "novo"):
                    continue
                # só mexe na categoria se foi a IA a escolhê-la (não a que já vinha no ficheiro)
                old_top = (old.get("categories") or [{}])[0].get("catid")
                ai_chosen = rec["fields"].get("catid") == old_top
                words = old.get("name_words") or []
                if old.get("from_heard"):
                    words = name_words((rec.get("review") or {}).get("heard", ""))
                sug = tagger.suggest(emb, words, old.get("folder_words") or [],
                                     memory=kb.neighbours(emb),
                                     helper_emb=self.embeddings2.get(rec["id"]), clf=clf,
                                     prior=priors.get(rec["id"]), prior_weight=GROUP_BETA,
                                     third_emb=self.embeddings3.get(rec["id"]),
                                     heard=bool(old.get("from_heard")))
                sug["name_words"] = words
                sug["folder_words"] = old.get("folder_words") or []
                sug["from_heard"] = old.get("from_heard", False)
                rec["suggestion"] = sug
                # o som ainda não foi confirmado: a "sugestão original" passa a ser esta
                for k in ("ai_catid", "ai_score", "ai_top5"):
                    (rec.get("review") or {}).pop(k, None)
                if ai_chosen and not rec.get("dirty") and sug["categories"]:
                    new_cat = sug["categories"][0]["catid"]
                    if new_cat != rec["fields"].get("catid"):
                        if rec.get("had_metadata"):
                            self.apply_catid(rec, new_cat)
                        else:   # rascunho só da IA: FXName e Description seguem a nova categoria
                            self._redraft(rec)
                        changed += 1
            self.save()
        return changed

    def set_heard(self, rid: str, text: str) -> dict:
        """'O que ouves?' — guarda a descrição da pessoa e usa-a como pista para a IA."""
        with self.lock:
            rec = self.records[rid]
            review = self._remember_ai(rec)
            review["heard"] = text.strip()
            emb = self.embeddings.get(rid)
        if emb is not None and text.strip():
            tagger = get_tagger()
            tagger.ensure_loaded()
            kb = get_knowledge()
            sug = tagger.suggest(emb, name_words(text), memory=kb.neighbours(emb),
                                 helper_emb=self.embeddings2.get(rid), clf=self._classifier(),
                                 third_emb=self.embeddings3.get(rid), heard=True)
            with self.lock:
                old = rec.get("suggestion") or {}
                sug["name_words"] = name_words(text)
                sug["folder_words"] = []
                sug["from_heard"] = True
                sug["keywords"] = old.get("keywords") or sug["keywords"]
                rec["suggestion"] = sug
        self.save()
        return rec

    REPORT_COLUMNS = ["Ficheiro", "O que é (pessoa)", "Sugestão IA", "Confiança IA",
                      "CatID final", "Resultado"]

    def report_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.REPORT_COLUMNS)
        label = {"certo": "IA acertou", "top5": "estava nas 5 sugestões", "errado": "IA errou"}
        with self.lock:
            for rec in sorted(self.records.values(), key=lambda r: r["path"].lower()):
                rv = rec.get("review") or {}
                if rec.get("missing") or not rv.get("verdict"):
                    continue
                score = rv.get("ai_score")
                w.writerow([Path(rec["path"]).name, _csv_safe(rv.get("heard", "")), rv.get("ai_catid", ""),
                            f"{round(score * 100)}%" if score is not None else "",
                            rv.get("final_catid", ""), label.get(rv["verdict"], "")])
        return buf.getvalue()

    def set_defaults(self, defaults: dict, apply_to_all: bool) -> None:
        with self.lock:
            for k in DEFAULT_KEYS:
                if k in defaults:
                    self.defaults[k] = defaults[k]
            if apply_to_all:
                for rec in self.records.values():
                    for k in DEFAULT_KEYS:
                        if self.defaults.get(k) and not rec["fields"].get(k):
                            rec["fields"][k] = self.defaults[k]
                            # rascunhos por rever não contam como editados: senão deixavam de
                            # ser atualizados ao reanalisar
                            rec["dirty"] = rec.get("dirty") or rec["status"] not in ("novo", "sugerido")
            self.save()

    # --- gravar nos ficheiros ----------------------------------------------
    def write(self, ids: list[str], rename: bool, backup: bool) -> dict:
        ok, errors = 0, []
        with self.lock:
            if self.job["running"] or self.writing:
                return {"ok": 0, "errors": [{"path": "", "error": "Já há uma análise ou gravação a decorrer"}]}
            self.writing = True
            numbering = self._series_numbers(ids) if rename else {}
        written: list[tuple[dict, dict]] = []
        try:
            for rid in ids:
                with self.lock:
                    rec = self.records.get(rid)
                    if rec is None or rec.get("missing"):
                        continue
                    p = self.abspath(rec)
                    fields = dict(rec["fields"])
                    want_backup = backup and not rec.get("backup")
                cat = ucs.lookup(fields.get("catid", ""))
                if cat:
                    fields["category_full"] = cat.full
                try:
                    # só se guarda o original verdadeiro (a primeira vez que a app mexe no ficheiro)
                    bdir = self.state_dir / "backup" / Path(rec["path"]).parent if want_backup else None
                    # fora do lock: em bibliotecas de vários GB a interface não pode ficar parada
                    copy = metadata.write_metadata(p, fields, backup_dir=bdir)
                    with self.lock:
                        if copy is not None:
                            rec["backup"] = copy.relative_to(self.root).as_posix()
                        rec.update(status="gravado", dirty=False, error=None)
                    written.append((rec, fields))
                    ok += 1
                except Exception as e:
                    rec["error"] = f"Erro ao gravar: {e}"
                    errors.append({"path": rec["path"], "error": str(e)})
        finally:
            self.writing = False
        with self.lock:
            if rename:
                errors += self._rename_all(written, numbering)
            for rec, _ in written:
                p = self.abspath(rec)
                if p.exists():
                    st = p.stat()
                    rec.update(size=st.st_size, mtime=st.st_mtime)
            self._compute_groups()
            self.save()
        return {"ok": ok, "errors": errors}

    def _rename_all(self, items: list[tuple[dict, dict]], numbering: dict[str, int]) -> list[dict]:
        """Renomeia em duas fases (nome temporário -> nome final) para que os sons de uma
        série possam trocar de nome entre si sem colisões."""
        errors = []
        staged = []
        for rec, fields in items:
            p = self.abspath(rec)
            tmp = p.with_name(f".sfxtagger-rename-{rec['id']}{p.suffix.lower()}")
            try:
                os.rename(p, tmp)
                rec.setdefault("orig_path", rec["path"])
                staged.append((rec, fields, p, tmp))
            except OSError as e:
                errors.append({"path": rec["path"], "error": f"Não foi possível renomear: {e}"})
        for rec, fields, original, tmp in staged:
            try:
                final = self._unique_name(original.parent, fields, numbering.get(rec["id"]),
                                          original.suffix.lower())
                os.rename(tmp, final)
            except OSError as e:
                os.rename(tmp, original)  # repõe o nome antigo
                final = original
                errors.append({"path": rec["path"], "error": f"Não foi possível renomear: {e}"})
            rec["path"] = final.relative_to(self.root).as_posix()
        return errors

    def _series_numbers(self, ids: list[str]) -> dict[str, int]:
        """Sons com o mesmo nome UCS na mesma pasta recebem 01, 02, 03… (por ordem do nome original)."""
        buckets: dict[tuple, list[dict]] = {}
        for rid in ids:
            rec = self.records.get(rid)
            if rec is None or rec.get("missing"):
                continue
            f = rec["fields"]
            key = (str(Path(rec["path"]).parent), f.get("catid", ""),
                   (f.get("fxname") or "").strip().lower(), f.get("creator_id", ""),
                   f.get("source_id", ""))
            buckets.setdefault(key, []).append(rec)
        out = {}
        for recs in buckets.values():
            if len(recs) > 1:
                # ordem do nome original da gravação, para manter a ordem dos takes
                order = sorted(recs, key=lambda r: (r.get("orig_path") or r["path"]).lower())
                for i, rec in enumerate(order, 1):
                    out[rec["id"]] = i
        return out

    @staticmethod
    def _unique_name(folder: Path, fields: dict, number: int | None, ext: str) -> Path:
        base_fx = fields.get("fxname") or "Untitled"
        if number is not None:
            base_fx = f"{base_fx} {number:02d}"
        fx, n = base_fx, 2
        while True:
            target = folder / ucs.build_filename(fields.get("catid", ""), fx,
                                                 fields.get("creator_id", ""),
                                                 fields.get("source_id", ""), ext=ext)
            if not target.exists():
                return target
            fx = f"{base_fx} {n:02d}"
            n += 1

    # --- exportar ----------------------------------------------------------
    CSV_COLUMNS = [
        ("Filename", lambda r: Path(r["path"]).name), ("Path", lambda r: r["path"]),
        ("CatID", "catid"), ("Category", "category"), ("SubCategory", "subcategory"),
        ("CategoryFull", "category_full"), ("FXName", "fxname"), ("Description", "description"),
        ("Keywords", "keywords"), ("Notes", "notes"), ("Library", "library"),
        ("Designer", "designer"), ("Microphone", "microphone"), ("RecMedium", "rec_medium"),
        ("Location", "location"), ("CreatorID", "creator_id"), ("SourceID", "source_id"),
        ("Duration", lambda r: (r.get("tech") or {}).get("duration", "")),
        ("SampleRate", lambda r: (r.get("tech") or {}).get("samplerate", "")),
        ("BitDepth", lambda r: (r.get("tech") or {}).get("bit_depth", "")),
        ("Channels", lambda r: (r.get("tech") or {}).get("channels", "")),
        ("LUFS", lambda r: (r.get("tech") or {}).get("lufs", "")),
        ("PeakdBFS", lambda r: (r.get("tech") or {}).get("peak_dbfs", "")),
        ("Status", lambda r: r.get("status", "")),
    ]

    def export_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([c for c, _ in self.CSV_COLUMNS])
        with self.lock:
            for rec in sorted(self.records.values(), key=lambda r: r["path"].lower()):
                if rec.get("missing"):
                    continue
                row = []
                for _, getter in self.CSV_COLUMNS:
                    if callable(getter):
                        v = getter(rec)
                    else:
                        v = rec["fields"].get(getter, "")
                    row.append("" if v is None else _csv_safe(v))
                w.writerow(row)
        return buf.getvalue()

    def summary(self) -> dict:
        with self.lock:
            return {"root": str(self.root), "defaults": self.defaults,
                    "records": [r for r in self.records.values() if not r.get("missing")],
                    "job": dict(self.job), "warning": self.warning, "ignored": len(self.ignored)}


def _csv_safe(v):
    """Texto vindo dos ficheiros não pode ser executado como fórmula ao abrir o CSV no Excel."""
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v


def _title(words: list[str], limit: int) -> str:
    out = ""
    for w in words:
        nxt = (out + " " + w.title()).strip()
        if len(nxt) > limit:
            break
        out = nxt
    return out


# Famílias UCS cujo nome é genérico demais para abrir um FXName (usa-se a subcategoria)
_GENERIC_FAMILIES = {"AMBIENCE", "OBJECTS", "DESIGNED", "MOVEMENT", "MECHANICAL", "MACHINES",
                     "EQUIPMENT", "FOLEY", "CARTOON", "SCIFI", "MAGIC", "USER INTERFACE", "HUMAN"}


def _singular(w: str) -> str:
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("shes") or w.endswith("ches"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _fxname_from_category(cat, ai_kw: list[str]) -> str:
    """FXName à maneira UCS (substantivo + especificação + ação/qualidade), ex.: 'Door Prison Heavy'."""
    if cat is None:
        return _title(ai_kw[:3], FXNAME_MAX)
    family = cat.category.split(" & ")[0].split()[-1].lower()
    sub = cat.subcategory.split(" & ")[0].lower()
    parts = []
    if cat.category not in _GENERIC_FAMILIES:
        parts.append(_singular(family))
    if sub not in ("misc", "general") and sub not in parts:
        parts.append(sub)
    used = set(" ".join(parts).split())
    for k in ai_kw:
        if len(parts) >= 3:
            break
        if not set(k.split()) & used and _singular(k) not in used:
            parts.append(k)
            used |= set(k.split())
    return _title(parts, FXNAME_MAX)


_MATERIAL_ADJ = {"wood": "wooden", "metal": "metallic", "stone": "stone", "glass": "glass",
                 "plastic": "plastic", "paper": "paper", "cardboard": "cardboard", "cloth": "cloth",
                 "fabric": "fabric", "leather": "leather", "rubber": "rubber", "concrete": "concrete",
                 "gravel": "gravel", "sand": "sand", "dirt": "dirt", "mud": "muddy", "snow": "snow",
                 "ice": "ice", "water": "water", "liquid": "liquid", "ceramic": "ceramic",
                 "porcelain": "porcelain", "electric": "electric", "electronic": "electronic",
                 "mechanical": "mechanical", "hydraulic": "hydraulic", "pneumatic": "pneumatic"}


_OPPOSITES = {frozenset(p) for p in (("wet", "dry"), ("loud", "soft"), ("loud", "quiet"), ("heavy", "light"),
                                      ("small", "large"), ("small", "big"), ("tiny", "big"), ("tiny", "large"),
                                      ("bright", "dark"), ("deep", "high pitched"), ("low", "high pitched"),
                                      ("sharp", "dull"), ("harsh", "smooth"), ("distant", "close"),
                                      ("noisy", "clean"), ("aggressive", "gentle"), ("muffled", "bright"))}


def _is_mixed(segments: list[dict]) -> bool:
    """A gravação tem sons de tipos claramente diferentes? (categorias principais UCS distintas,
    cada uma ouvida com alguma segurança em pelo menos 25% do tempo analisado)."""
    total = sum(sg["end"] - sg["start"] for sg in segments) or 1.0
    by_family: dict[str, float] = {}
    for sg in segments:
        cat = ucs.lookup(sg["catid"])
        if cat and sg.get("score", 0) >= 0.4:
            by_family[cat.category] = by_family.get(cat.category, 0.0) + sg["end"] - sg["start"]
    return sum(1 for d in by_family.values() if d / total >= 0.25) >= 2


def _draft_description(words: list[str], attrs: dict, tech: dict) -> str:
    """Rascunho curto em inglês: o quê (fonte, ação, material), como soa e a estrutura.
    O designer completa o contexto (onde, como foi gravado)."""
    subject = " ".join(words[:6])
    used = set(subject.split())
    first = [subject] if subject else []
    for group in ("source", "action"):
        for w in attrs.get(group, []):
            if not set(w.split()) & used:
                first.append(w)
                used |= set(w.split())
    for w in attrs.get("material", []):
        adj = _MATERIAL_ADJ.get(w, w)
        if w not in used and adj not in used:
            first.append(adj)
            used |= {w, adj}
    qualities = []
    for q in attrs.get("quality", []):
        if q not in used and not any(frozenset((q, o)) in _OPPOSITES for o in qualities):
            qualities.append(q)
    kind = tech.get("kind")
    n = tech.get("events") or 0
    dur = tech.get("duration") or 0
    tail = tech.get("tail_s")
    structure = {
        "takes": f"{n} takes" if n > 1 else "",
        "one-shot": "single one-shot",
        "ambiente": f"continuous ambience, {dur:.0f} s",
    }.get(kind, "")
    if kind in ("one-shot", "evento") and tail:
        structure = ", ".join(x for x in (structure, "long tail" if tail >= 1.5 else
                                          "short tail" if tail < 0.4 else "") if x)
    parts = [", ".join(first), " and ".join(qualities), structure]
    sentences = [x[:1].upper() + x[1:] for x in parts if x]
    return ". ".join(sentences) + "." if sentences else ""
