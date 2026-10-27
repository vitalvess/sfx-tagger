"""Leitura de áudio e análise técnica (sem IA)."""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ANALYSIS_SR = 48000          # CLAP trabalha a 48 kHz
MAX_ANALYSIS_SECONDS = 20 * 60

BIT_DEPTH = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
             "FLOAT": 32, "DOUBLE": 64}


@dataclass
class LoadedAudio:
    mono: np.ndarray       # float32 mono a ANALYSIS_SR
    info: dict             # dados técnicos do ficheiro original
    peak: float            # pico absoluto (todas as canais, taxa original)
    clipped: int           # amostras >= 0.999


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def load(path: str | Path) -> LoadedAudio:
    info = sf.info(str(path))
    peak = 0.0
    clipped = 0
    parts = []
    frames_limit = int(MAX_ANALYSIS_SECONDS * info.samplerate)
    read = 0
    block = info.samplerate * 30
    for blk in sf.blocks(str(path), blocksize=block, dtype="float32", always_2d=True):
        if read >= frames_limit:
            break
        a = np.abs(blk)
        peak = max(peak, float(a.max(initial=0.0)))
        clipped += int((a >= 0.999).sum())
        parts.append(_resample(blk.mean(axis=1), info.samplerate, ANALYSIS_SR))
        read += len(blk)
    mono = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    meta = {
        "samplerate": info.samplerate,
        "channels": info.channels,
        "bit_depth": BIT_DEPTH.get(info.subtype),
        "subtype": info.subtype,
        "duration": info.frames / info.samplerate if info.samplerate else 0.0,
        "format": info.format,
    }
    return LoadedAudio(mono=mono, info=meta, peak=peak, clipped=clipped)


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-9)))


def frame_rms_db(x: np.ndarray, sr: int, frame_s: float = 0.01) -> np.ndarray:
    n = max(1, int(sr * frame_s))
    usable = len(x) // n * n
    if usable == 0:
        return np.array([-180.0])
    fr = x[:usable].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-18)
    return 20 * np.log10(rms + 1e-9)


def detect_events(x: np.ndarray, sr: int, min_gap_s: float = 0.35) -> list[tuple[float, float]]:
    """Encontra regiões com som separadas por silêncio (takes / one-shots)."""
    db = frame_rms_db(x, sr)
    if len(db) < 3:
        return []
    floor = np.percentile(db, 10)
    top = np.percentile(db, 99)
    if top - floor < 12:  # som contínuo: sem silêncios relevantes
        return [(0.0, len(x) / sr)]
    thr = max(floor + 0.35 * (top - floor), -70.0)
    active = db > thr
    events = []
    gap_frames = int(min_gap_s / 0.01)
    start = None
    last_on = None
    for i, on in enumerate(active):
        if on:
            if start is None:
                start = i
            last_on = i
        elif start is not None and i - last_on > gap_frames:
            events.append((start, last_on))
            start = None
    if start is not None:
        events.append((start, last_on))
    # ignora "eventos" minúsculos (cliques, ruído)
    return [(s * 0.01, (e + 1) * 0.01) for s, e in events if (e - s) >= 3]


def analyze(audio: LoadedAudio) -> dict:
    x = audio.mono
    sr = ANALYSIS_SR
    dur = audio.info["duration"]
    res = dict(audio.info)
    res["peak_dbfs"] = round(_db(audio.peak), 1)
    res["clipped_samples"] = audio.clipped
    if len(x) == 0:
        res.update(kind="vazio", events=0)
        return res

    rms = float(np.sqrt(np.mean(x ** 2)))
    res["rms_dbfs"] = round(_db(rms), 1)
    try:
        import pyloudnorm as pyln
        if len(x) >= int(0.5 * sr):
            lufs = pyln.Meter(sr).integrated_loudness(x.astype(np.float64))
            res["lufs"] = round(float(lufs), 1) if np.isfinite(lufs) else None
    except Exception:
        res["lufs"] = None

    db = frame_rms_db(x, sr)
    res["noise_floor_dbfs"] = round(float(np.percentile(db, 10)), 1)
    res["dc_offset"] = round(float(np.mean(x)), 5)

    # brilho (centróide espectral médio) — útil como pista para keywords
    seg = x[: sr * 30]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) if len(seg) > 64 else None
    if spec is not None and spec.sum() > 0:
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        res["centroid_hz"] = int((freqs * spec).sum() / spec.sum())

    events = detect_events(x, sr)
    res["events"] = len(events)
    res["event_times"] = [(round(s, 2), round(e, 2)) for s, e in events[:200]]

    crest = _db(audio.peak) - _db(rms)
    res["crest_db"] = round(crest, 1)
    dyn = float(np.percentile(db, 95) - np.percentile(db, 10))
    if len(events) >= 2:
        kind = "takes"
    elif dur >= 20 and dyn < 18:
        kind = "ambiente"
    elif dur <= 4 or crest > 18:
        kind = "one-shot"
    else:
        kind = "evento"
    res["kind"] = kind
    return res


def best_windows(x: np.ndarray, sr: int = ANALYSIS_SR, win_s: float = 10.0,
                 max_windows: int = 3) -> list[np.ndarray]:
    """Escolhe até max_windows janelas de 10 s com mais energia para a IA ouvir."""
    win = int(win_s * sr)
    if len(x) <= win:
        return [x]
    hop = sr  # passo de 1 s
    n = (len(x) - win) // hop + 1
    csum = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(n) * hop
    energy = csum[starts + win] - csum[starts]
    chosen: list[int] = []
    for idx in np.argsort(energy)[::-1]:
        if all(abs(int(idx) - c) * hop >= win for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= max_windows:
            break
    return [x[i * hop: i * hop + win] for i in sorted(chosen)]


def waveform_peaks(x: np.ndarray, n: int = 1000) -> list[float]:
    if len(x) == 0:
        return []
    n = min(n, len(x))
    edges = np.linspace(0, len(x), n + 1, dtype=int)
    peaks = np.maximum.reduceat(np.abs(x), edges[:-1])
    m = float(peaks.max()) or 1.0
    return [round(float(p) / m, 3) for p in peaks]
