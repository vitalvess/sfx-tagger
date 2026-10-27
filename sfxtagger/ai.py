"""Sugestões automáticas com modelos CLAP (áudio <-> texto).

O CLAP coloca sons e frases no mesmo espaço vetorial. Comparamos cada som com:
  * as ~750 categorias UCS, descritas por várias frases (nome, explicação oficial,
    sinónimos, família) — combinação afinada no benchmark ESC-50;
  * um vocabulário de keywords (vocab.py);
  * o próprio nome do ficheiro/pasta, quando este diz alguma coisa;
  * a memória de sons já confirmados por pessoas (knowledge.py).

São usados dois modelos CLAP e as opiniões são combinadas: no benchmark isto acerta
mais ~4 pontos na 1.ª sugestão do que o melhor modelo sozinho (ferramentas/benchmark).
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path

import numpy as np

from . import paths, ucs, vocab
from .audio import ANALYSIS_SR

MODEL_NAME = "laion/larger_clap_general"      # modelo principal (keywords, memória)
HELPER_MODEL = "laion/clap-htsat-unfused"     # segundo modelo, só para as categorias
HELPER_WEIGHT = 0.55                          # peso do segundo modelo na combinação
PROMPTS_VERSION = "2"
DATA_DIR = Path(__file__).parent / "data"

# Peso de cada forma de descrever uma categoria (afinado no ESC-50, folds 1-3)
CATEGORY_TEXT_WEIGHTS = {
    "laion/larger_clap_general": {"nome": 2, "explic": 1, "sinon_lista": 1,
                                  "sinon_cada": 2, "sfx": 2, "familia": 2},
    "laion/clap-htsat-unfused": {"nome": 2, "explic": 1, "sinon_lista": 2,
                                 "sinon_cada": 0, "sfx": 2, "familia": 0},
}

# Memória de exemplos (afinado no ESC-50 com metade dos tipos de som desconhecidos,
# penalizando 3x os estragos em sons de tipos que ninguém confirmou)
MEMORY = {"min_sim": 0.65, "tau": 0.05, "start": 0.65, "full": 0.85, "max_weight": 0.5}

# Palavras que aparecem em nomes de ficheiro mas não descrevem o som
NAME_STOPWORDS = {
    "zoom", "tascam", "sound", "devices", "mixpre", "take", "tk", "track", "trk", "audio",
    "sfx", "fx", "wav", "wave", "bwf", "stereo", "mono", "ms", "xy", "ortf", "ab", "lr",
    "final", "edit", "edited", "copy", "new", "mix", "master", "rec", "recording", "file",
    "untitled", "export", "bounce", "render", "version", "the", "and", "with", "of", "sample",
    "samples", "clip", "region", "glued", "processed", "proc", "norm", "normalized", "khz",
    "bit", "bits", "left", "right", "ch", "channel", "mic", "mics", "tr", "sr", "sd",
}

NON_SOUND_CATEGORIES = {"ARCHIVED"}


def name_words(*texts: str) -> list[str]:
    """Palavras úteis de nomes de ficheiro/pastas ('DoorWood_creak-03' -> door, wood, creak)."""
    words: list[str] = []
    for t in texts:
        t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t or "")
        for w in re.split(r"[^A-Za-z]+", t):
            w = w.lower()
            if len(w) >= 3 and w not in NAME_STOPWORDS and w not in words:
                words.append(w)
    return words


def category_texts(c: ucs.UCSCategory) -> dict[str, list[str]]:
    """As várias formas de descrever uma categoria UCS ao modelo."""
    cat, sub = c.category.lower(), c.subcategory.lower()
    name = cat if sub in ("misc", "general") else f"{sub} {cat}"
    return {
        "nome": [f"the sound of {name}"],
        "explic": [c.explanation or f"the sound of {name}"],
        "sinon_lista": (["the sound of " + ", ".join(c.synonyms[:6])] if c.synonyms
                        else [f"the sound of {name}"]),
        "sinon_cada": [f"the sound of {s}" for s in c.synonyms[:8]] or [f"the sound of {name}"],
        "sfx": [f"{name} sound effect"],
        "familia": [f"the sound of {cat}"],
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _norm(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


class Tagger:
    def __init__(self, model_name: str = MODEL_NAME, helper: str | None = HELPER_MODEL) -> None:
        self.model_name = model_name
        self.status = "não carregado"
        self._lock = threading.Lock()
        self._model = None
        self._proc = None
        self._torch = None
        self.cat_matrix: np.ndarray | None = None
        self.kw_matrix: np.ndarray | None = None
        self.categories = ucs.load_ucs()
        # categorias organizacionais (ADR, mixes, gravações brutas…) não descrevem o som
        self.suggestable = np.array([c.category not in NON_SOUND_CATEGORIES
                                     for c in self.categories])
        self.keywords = vocab.all_keywords()
        self.logit_scale = 33.0
        self.helper = Tagger(helper, helper=None) if helper else None

    # --- carregar ----------------------------------------------------------
    def ensure_loaded(self) -> None:
        with self._lock:
            if self._model is None:
                self.status = "a carregar o modelo de IA (na primeira vez descarrega ~1,4 GB)…"
                self._load_model()
                self.status = "a preparar categorias UCS…"
                self.cat_matrix = self._category_matrix()
                self.kw_matrix = self._cached_matrix(
                    "kw", [[vocab.prompt_for(w)] for w in self.keywords])
            if self.helper is not None and self.helper._model is None:
                self.status = "a carregar o segundo modelo de IA…"
                self.helper._load_model()
                self.helper.cat_matrix = self.helper._category_matrix()
            self.status = "pronto"

    def _load_model(self) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import ClapModel, ClapProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self._torch = torch
        try:  # já descarregado: arranca sem internet
            self._proc = ClapProcessor.from_pretrained(self.model_name, local_files_only=True)
            self._model = ClapModel.from_pretrained(self.model_name, local_files_only=True)
        except OSError:
            self._proc = ClapProcessor.from_pretrained(self.model_name)
            self._model = ClapModel.from_pretrained(self.model_name)
        self._model.eval()
        try:
            self.logit_scale = float(self._model.logit_scale_a.exp().item())
        except Exception:
            pass

    def _category_matrix(self) -> np.ndarray:
        weights = CATEGORY_TEXT_WEIGHTS.get(self.model_name,
                                            CATEGORY_TEXT_WEIGHTS[MODEL_NAME])
        texts = [category_texts(c) for c in self.categories]
        groups = {k: [t[k] for t in texts] for k, w in weights.items() if w}
        sig = hashlib.sha1((self.model_name + PROMPTS_VERSION + repr(weights) + repr(groups))
                           .encode()).hexdigest()[:16]
        name = f"cat_{self.model_name.replace('/', '_')}_{sig}.npy"
        for f in (DATA_DIR / name, paths.cache_dir() / name):   # pré-calculada no pacote?
            if f.exists():
                return np.load(f)
        mat = sum(w * self._mean_rows(groups[k]) for k, w in weights.items() if w)
        mat = _norm(mat).astype(np.float32)
        np.save(paths.cache_dir() / name, mat)
        return mat

    def _mean_rows(self, prompt_groups: list[list[str]]) -> np.ndarray:
        flat = [p for g in prompt_groups for p in g]
        emb = self.embed_text(flat)
        out, i = [], 0
        for g in prompt_groups:
            out.append(_norm(emb[i:i + len(g)].mean(axis=0)))
            i += len(g)
        return np.stack(out)

    def _cached_matrix(self, tag: str, prompt_groups: list[list[str]]) -> np.ndarray:
        h = hashlib.sha1(
            (self.model_name + PROMPTS_VERSION + repr(prompt_groups)).encode()).hexdigest()[:16]
        name = f"{tag}_{self.model_name.replace('/', '_')}_{h}.npy"
        for f in (DATA_DIR / name, paths.cache_dir() / name):
            if f.exists():
                return np.load(f)
        mat = self._mean_rows(prompt_groups).astype(np.float32)
        np.save(paths.cache_dir() / name, mat)
        return mat

    # --- embeddings --------------------------------------------------------
    @staticmethod
    def _as_tensor(out):
        # versões recentes do transformers podem devolver um objeto em vez do tensor
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def embed_text(self, texts: list[str], batch: int = 64) -> np.ndarray:
        torch = self._torch
        res = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                tok = self._proc.tokenizer(texts[i:i + batch], padding=True, truncation=True,
                                           max_length=77, return_tensors="pt")
                e = self._as_tensor(self._model.get_text_features(**tok)).cpu().numpy()
                res.append(_norm(e))
        return np.concatenate(res).astype(np.float32)

    def embed_audio(self, windows: list[np.ndarray]) -> np.ndarray:
        """Embedding médio (normalizado) de várias janelas do mesmo som."""
        torch = self._torch
        windows = [w if len(w) >= ANALYSIS_SR // 10 else np.pad(w, (0, ANALYSIS_SR // 10 - len(w)))
                   for w in windows]
        with torch.no_grad():
            feats = self._proc.feature_extractor(
                [w.astype(np.float32) for w in windows], sampling_rate=ANALYSIS_SR,
                return_tensors="pt")
            e = self._as_tensor(self._model.get_audio_features(**feats)).cpu().numpy()
        return _norm(_norm(e).mean(axis=0)).astype(np.float32)

    def embed_audio_helper(self, windows: list[np.ndarray]) -> np.ndarray | None:
        return self.helper.embed_audio(windows) if self.helper is not None else None

    # --- sugestões ---------------------------------------------------------
    def _text_logp(self, words: list[str]) -> np.ndarray:
        emb = self.embed_text(["the sound of " + " ".join(words[:8])])[0]
        return np.log(_softmax(self.logit_scale * (self.cat_matrix @ emb)) + 1e-12)

    def audio_probs(self, audio_emb: np.ndarray, helper_emb: np.ndarray | None = None) -> np.ndarray:
        """Probabilidade de cada categoria só pelo áudio (combinando os dois modelos)."""
        p = _softmax(np.where(self.suggestable, self.logit_scale * (self.cat_matrix @ audio_emb), -np.inf))
        if helper_emb is not None and self.helper is not None and self.helper.cat_matrix is not None:
            h = self.helper
            p2 = _softmax(np.where(self.suggestable, h.logit_scale * (h.cat_matrix @ helper_emb), -np.inf))
            p = (1 - HELPER_WEIGHT) * p + HELPER_WEIGHT * p2
        return p

    def suggest(self, audio_emb: np.ndarray, words: list[str], folder_words: list[str] = (),
                top: int = 5, memory: list[tuple[str, float]] | None = None,
                helper_emb: np.ndarray | None = None) -> dict:
        """words: do nome do ficheiro (peso forte); folder_words: das pastas (peso fraco,
        porque muitas vezes são nomes de projeto, de gravador ou noutra língua);
        memory: exemplos confirmados parecidos com este som [(catid, semelhança)];
        helper_emb: embedding do segundo modelo (se existir)."""
        logp = np.log(self.audio_probs(audio_emb, helper_emb) + 1e-12)
        used_name = False
        if words:
            logp = 0.6 * logp + 0.4 * self._text_logp(words)
            used_name = True
        folder_words = [w for w in folder_words if w not in words]
        if folder_words:
            logp = 0.85 * logp + 0.15 * self._text_logp(folder_words)
            used_name = True
        logp = np.where(self.suggestable, logp, -np.inf)
        probs = _softmax(logp)
        used_memory = []
        if memory:
            probs, used_memory = self._apply_memory(probs, memory)
        order = np.argsort(probs)[::-1][:top]
        cats = [{"catid": self.categories[i].catid,
                 "category": self.categories[i].category,
                 "subcategory": self.categories[i].subcategory,
                 "score": round(float(probs[i]), 3)} for i in order]

        # keywords: as que se destacam neste som E fazem sentido com a categoria escolhida
        # (só o áudio dá muitos falsos positivos genéricos, ex.: "game", "user interface")
        def z(v):
            return (v - v.mean()) / (v.std() + 1e-9)
        cat_vec = _norm((probs[order, None] * self.cat_matrix[order]).sum(axis=0))
        score = 0.55 * z(self.kw_matrix @ audio_emb) + 0.45 * z(self.kw_matrix @ cat_vec)
        kw_order = np.argsort(score)[::-1]
        kws = [self.keywords[i] for i in kw_order[:8] if score[i] > 1.5][:6]
        return {"categories": cats, "keywords": kws, "used_name": used_name,
                "memory": used_memory}

    def _apply_memory(self, probs: np.ndarray, memory: list[tuple[str, float]]):
        """Mistura a opinião do modelo com a dos exemplos confirmados mais parecidos.
        Quanto mais parecido for o exemplo mais próximo, mais peso tem a memória."""
        index = {c.catid: i for i, c in enumerate(self.categories)}
        votes = np.zeros_like(probs)
        summary: dict[str, dict] = {}
        memory = [(c, s) for c, s in memory if s >= MEMORY["min_sim"]]
        for catid, sim in memory:
            i = index.get(catid)
            if i is None:
                continue
            votes[i] += np.exp((sim - 1.0) / MEMORY["tau"])
            s = summary.setdefault(catid, {"catid": catid, "n": 0, "best": 0.0})
            s["n"] += 1
            s["best"] = max(s["best"], round(sim, 3))
        if votes.sum() <= 0:
            return probs, []
        best = max(sim for _, sim in memory)
        weight = float(np.clip((best - MEMORY["start"]) / (MEMORY["full"] - MEMORY["start"]),
                               0.0, MEMORY["max_weight"]))
        mixed = (1 - weight) * probs + weight * votes / votes.sum()
        used = sorted(summary.values(), key=lambda s: (-s["n"], -s["best"]))
        return mixed / mixed.sum(), used


_tagger: Tagger | None = None


def get_tagger() -> Tagger:
    global _tagger
    if _tagger is None:
        _tagger = Tagger()
    return _tagger
