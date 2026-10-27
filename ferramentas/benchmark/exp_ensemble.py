"""Vale a pena combinar dois modelos CLAP?  (média das probabilidades)

    python ferramentas/benchmark/exp_ensemble.py <pasta ESC-50>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench import Data, metrics  # noqa: E402
from exp_memory import KEYS, softmax  # noqa: E402
from exp_prompts import norm, template_matrices  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sfxtagger.ai import NON_SOUND_CATEGORIES, Tagger  # noqa: E402

MODELS = ["laion/larger_clap_general", "laion/clap-htsat-unfused"]


def probs(esc: Path, model: str):
    data = Data(esc, model)
    tagger = Tagger(model)
    tagger.ensure_loaded()
    T = template_matrices(tagger)
    w = np.load(esc.parent / f"best_weights_{model.replace('/', '_')}.npy")
    M = norm(sum(wi * T[k] for wi, k in zip(w, KEYS)))
    sugg = np.array([c.category not in NON_SOUND_CATEGORIES for c in tagger.categories])
    logits = tagger.logit_scale * (data.X @ M.T)
    logits[:, ~sugg] = -1e9
    return data, softmax(logits)


def main(esc: Path):
    (data, P1), (_, P2) = probs(esc, MODELS[0]), probs(esc, MODELS[1])
    for a in (0.6, 0.5, 0.45, 0.4, 0.3):
        P = a * P1 + (1 - a) * P2
        d, t = metrics(P, data, data.dev), metrics(P, data, data.test)
        print(f"peso {MODELS[0].split('/')[1]} = {a:.1f}:  dev top1 {d['top1']:5} top5 {d['top5']:5} | "
              f"teste top1 {t['top1']:5} top5 {t['top5']:5}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
