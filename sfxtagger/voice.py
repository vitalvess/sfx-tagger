"""Ditado por voz: transforma fala em texto com o Whisper (OpenAI), no próprio computador.

O browser grava o microfone e envia um WAV mono de 16 kHz. O Whisper pode:
  * transcrever no idioma falado ("transcribe"), ou
  * traduzir diretamente para inglês ("translate") — útil para a Description,
    que nas bibliotecas de SFX é quase sempre em inglês.
O modelo (~1 GB) só é descarregado na primeira vez que alguém usa o microfone.
"""
from __future__ import annotations

import io
import os
import threading

import numpy as np
import soundfile as sf

from .audio import _resample

WHISPER_MODEL = "openai/whisper-small"
SR = 16000
MAX_SECONDS = 120


class Dictation:
    def __init__(self) -> None:
        self.status = "não carregado"
        self._pipe = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._pipe is not None:
            return
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        self.status = "a carregar o modelo de voz"

        def load(offline: bool):
            proc = AutoProcessor.from_pretrained(WHISPER_MODEL, local_files_only=offline)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL, local_files_only=offline)
            return pipeline("automatic-speech-recognition", model=model,
                            tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor)
        try:  # já descarregado: funciona sem internet
            self._pipe = load(True)
        except OSError:
            self.status = "a descarregar o modelo de voz (cerca de 1 GB, só na primeira vez)"
            self._pipe = load(False)
        self.status = "pronto"

    def transcribe(self, wav: bytes, translate: bool) -> str:
        try:
            data, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
        except Exception:
            raise ValueError("a gravação não chegou num formato de áudio válido")
        x = data.mean(axis=1)
        x = _resample(x, sr, SR)[: SR * MAX_SECONDS]
        if len(x) < SR * 0.3 or float(np.abs(x).max(initial=0)) < 1e-3:
            return ""  # silêncio ou gravação vazia
        x = x / max(float(np.abs(x).max()), 1e-6) * 0.9
        with self._lock:
            self._load()
            out = self._pipe({"raw": x.astype(np.float32), "sampling_rate": SR},
                             generate_kwargs={"task": "translate" if translate else "transcribe"},
                             chunk_length_s=30)
        text = (out.get("text") if isinstance(out, dict) else str(out)) or ""
        return " ".join(text.split())


_dictation: Dictation | None = None


def get_dictation() -> Dictation:
    global _dictation
    if _dictation is None:
        _dictation = Dictation()
    return _dictation
