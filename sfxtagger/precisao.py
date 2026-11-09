"""Modo "Precisão máxima": um terceiro modelo a ouvir (CLAP 2023 da Microsoft).

Opcional, porque custa mais ~700 MB de download e torna a análise mais lenta. Só é preciso
o ficheiro de pesos: o codificador de áudio vem na app (msclap_audio/) e as categorias UCS
vêm pré-calculadas (data/cat_msclap_2023.npy), por isso não se instala nada de novo.
No benchmark (ESC-50, folds 4-5) a 1.ª sugestão passa de 76,9% para ~80%.
"""
from __future__ import annotations

import json
import threading
import time
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from . import paths
from .audio import ANALYSIS_SR

REPO = "microsoft/msclap"
WEIGHTS_FILE = "CLAP_weights_2023.pth"
WEIGHTS_MB = 690
CATEGORIES = Path(__file__).parent / "data" / "cat_msclap_2023.npy"
MODEL_SR = 44100
MODEL_SECONDS = 7
WEIGHT = 0.33      # peso na combinação com os outros dois modelos (afinado nos folds 1-3)
SCALE = 80.0

STATE = {"state": "desligado", "step": "", "error": ""}
_lock = threading.Lock()
_model = None
_matrix: np.ndarray | None = None
_enabled_seen: tuple[int, bool] | None = None   # (data do ficheiro, valor): evita reler o JSON


def enabled() -> bool:
    global _enabled_seen
    try:
        f = paths.settings_file()
        stamp = f.stat().st_mtime_ns
        if _enabled_seen is None or _enabled_seen[0] != stamp:
            _enabled_seen = (stamp, bool(json.loads(f.read_text(encoding="utf-8")).get("max_precision")))
        return _enabled_seen[1]
    except (OSError, ValueError):
        return False


def _download_dir() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE) / "models--microsoft--msclap"


def status() -> dict:
    s = dict(STATE, enabled=enabled(), total_mb=WEIGHTS_MB)
    if s["state"] == "a instalar":
        try:
            s["done_mb"] = sum(f.stat().st_size for f in (_download_dir() / "blobs").iterdir()) // 2 ** 20
        except OSError:
            s["done_mb"] = 0
    return s


def _load() -> None:
    global _model, _matrix
    import torch
    from huggingface_hub import hf_hub_download
    from torch import nn

    from .msclap_audio.htsat import HTSATWrapper

    try:  # já descarregado: funciona sem internet
        path = hf_hub_download(REPO, WEIGHTS_FILE, local_files_only=True)
    except Exception:  # noqa: BLE001
        STATE["step"] = f"A descarregar o terceiro modelo (cerca de {WEIGHTS_MB} MB, só desta vez)…"
        path = hf_hub_download(REPO, WEIGHTS_FILE)
    STATE["step"] = "A carregar o terceiro modelo…"

    class Projection(nn.Module):
        def __init__(self, d_in: int, d_out: int) -> None:
            super().__init__()
            self.linear1 = nn.Linear(d_in, d_out, bias=False)
            self.linear2 = nn.Linear(d_out, d_out, bias=False)
            self.layer_norm = nn.LayerNorm(d_out)

        def forward(self, x):
            a = self.linear1(x)
            return self.layer_norm(a + self.linear2(nn.functional.gelu(a)))

    class AudioEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = HTSATWrapper(MODEL_SR, 1024, 320, 64, 50, 8000, 527, 768)
            self.projection = Projection(768, 1024)

        def forward(self, x):
            return self.projection(self.base(x)["embedding"])

    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    prefix = "audio_encoder."
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model = AudioEncoder()
    result = model.load_state_dict(state, strict=False)
    # os pesos da STFT do original não fazem falta: aqui é a torch.stft
    missing = [k for k in result.missing_keys if "spectrogram_extractor" not in k]
    if missing:
        raise RuntimeError(f"ficheiro de pesos incompatível: {missing[:3]}")
    model.eval()
    _matrix = np.load(CATEGORIES)
    _model = model


def _run() -> None:
    try:
        _load()
        STATE.update(state="ok", step="Pronto", error="")
    except Exception as e:  # noqa: BLE001 — mostrado na interface; a app continua com dois modelos
        print(f"  Precisão máxima falhou: {e}", flush=True)
        STATE.update(state="erro", error=str(e))


def ensure_started() -> None:
    """Descarrega e carrega o modelo em segundo plano (uma vez)."""
    with _lock:
        if _model is not None or STATE["state"] == "a instalar":
            return
        STATE.update(state="a instalar", step="A preparar…", error="")
    threading.Thread(target=_run, daemon=True).start()


def active() -> bool:
    return _model is not None and enabled()


def wait_if_starting(timeout: float = 120.0) -> None:
    """Se o modelo já está no disco e a carregar, a análise espera por ele (só não espera
    por um download, que pode demorar muito)."""
    t0 = time.time()
    while (enabled() and STATE["state"] == "a instalar" and not STATE["step"].startswith("A descarregar")
           and time.time() - t0 < timeout):
        time.sleep(0.2)


def embed(windows: list[np.ndarray]) -> np.ndarray:
    """Embedding médio (normalizado) das janelas (a 48 kHz) que a app escolheu ouvir."""
    import torch
    g = gcd(ANALYSIS_SR, MODEL_SR)
    n = MODEL_SR * MODEL_SECONDS
    clips = []
    for w in windows:
        x = resample_poly(w, MODEL_SR // g, ANALYSIS_SR // g).astype(np.float32)
        if len(x) == 0:
            x = np.zeros(n, np.float32)
        if len(x) <= n:                       # curto: repete, como no modelo original
            x = np.tile(x, n // len(x) + 1)[:n]
        else:                                 # longo: a parte central da janela
            start = (len(x) - n) // 2
            x = x[start:start + n]
        clips.append(x)
    with torch.inference_mode():
        e = _model(torch.from_numpy(np.stack(clips))).numpy()
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    m = e.mean(0)
    return (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)


def probs(emb: np.ndarray, suggestable: np.ndarray) -> np.ndarray:
    z = np.where(suggestable, SCALE * (_matrix @ emb), -np.inf)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()
