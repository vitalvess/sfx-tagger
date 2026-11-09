"""O que entra nos modelos: cortar o silêncio e normalizar o nível ajudam?

    python ferramentas/benchmark/exp_sinal.py embed <pasta ESC-50>   (demora: 3 modelos x 3 variantes)
    python ferramentas/benchmark/exp_sinal.py eval  <pasta ESC-50>

Variantes: original, trim (corta o silêncio), norm (pico a -1 dB), trimnorm (as duas).
A decisão é tomada nos folds 1-3; os folds 4-5 só confirmam.
Resultado (v0.7.3): trimnorm piora em todos os casos (2 modelos 76,9 -> 75,8; 3 modelos 80,0 -> 78,5). Não entrou na app.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench import Data, load_meta, metrics  # noqa: E402

from sfxtagger import audio, precisao  # noqa: E402
from sfxtagger.ai import HELPER_MODEL, Tagger  # noqa: E402

def trim_silence(x: np.ndarray, sr: int = audio.ANALYSIS_SR, below_db: float = 45.0,
                 margin_s: float = 0.05) -> np.ndarray:
    """Corta o silêncio antes e depois do som (relativo ao momento mais forte do ficheiro)."""
    db = audio.frame_rms_db(x, sr)
    on = np.where(db > max(float(db.max()) - below_db, -90.0))[0]
    if not len(on):
        return x
    margin = int(margin_s * sr)
    frame = int(0.01 * sr)
    y = x[max(0, int(on[0]) * frame - margin): min(len(x), (int(on[-1]) + 1) * frame + margin)]
    return y if len(y) >= sr // 10 else x


def normalize_peak(x: np.ndarray, peak: float = 0.89) -> np.ndarray:
    """Pico a -1 dB: gravações muito baixas não devem soar diferentes aos modelos."""
    p = float(np.abs(x).max(initial=0.0))
    return x if p < 1e-5 else (x / p * peak).astype(np.float32)


VARIANTS = {
    "trimnorm": lambda x: normalize_peak(trim_silence(x)),
    "norm": normalize_peak,
    "trim": trim_silence,
}


def cache(esc: Path, variant: str, model: int) -> Path:
    return esc.parent / f"sinal_{variant}_{model}.npz"


def embed(esc: Path) -> None:
    t = Tagger()
    t.ensure_loaded()
    precisao._load()
    meta = load_meta(esc)
    waves = {r["filename"]: audio.load(esc / "audio" / r["filename"]).mono for r in meta}
    fns = {1: t.embed_audio, 2: t.helper.embed_audio, 3: precisao.embed}
    for name, fn in VARIANTS.items():
        for m, emb in fns.items():
            f = cache(esc, name, m)
            if f.exists():
                continue
            out = {}
            for i, (k, x) in enumerate(waves.items()):
                out[k] = emb([fn(x)])
                if i % 500 == 0:
                    print(f"  {name} modelo {m}: {i}/2000", flush=True)
            np.savez_compressed(f, **out)
    print("feito", flush=True)


def evaluate(esc: Path) -> None:
    d1, d2 = Data(esc), Data(esc, HELPER_MODEL)
    t = Tagger()
    t.ensure_loaded()
    precisao._matrix = np.load(precisao.CATEGORIES)
    names = [r["filename"] for r in d1.meta]

    def load(f):
        with np.load(f) as z:
            return np.stack([z[n] for n in names])

    base = (d1.X, d2.X, load(esc.parent / "emb_msclap_app.npz"))

    def report(label, X):
        P2 = np.stack([t.audio_probs(a, b) for a, b in zip(X[0], X[1])])
        P3 = np.stack([precisao.probs(e, t.suggestable) for e in X[2]])
        Q = (1 - precisao.WEIGHT) * P2 + precisao.WEIGHT * P3
        print(f"{label:10} 2 modelos dev {metrics(P2, d1, d1.dev)['top1']:5.1f}/{metrics(P2, d1, d1.dev)['top5']:5.1f}"
              f" teste {metrics(P2, d1, d1.test)['top1']:5.1f}/{metrics(P2, d1, d1.test)['top5']:5.1f}"
              f" | 3 modelos dev {metrics(Q, d1, d1.dev)['top1']:5.1f}/{metrics(Q, d1, d1.dev)['top5']:5.1f}"
              f" teste {metrics(Q, d1, d1.test)['top1']:5.1f}/{metrics(Q, d1, d1.test)['top5']:5.1f}", flush=True)

    report("original", base)
    for v in VARIANTS:
        if all(cache(esc, v, m).exists() for m in (1, 2, 3)):
            report(v, tuple(load(cache(esc, v, m)) for m in (1, 2, 3)))


if __name__ == "__main__":
    {"embed": embed, "eval": evaluate}[sys.argv[1]](Path(sys.argv[2]))
