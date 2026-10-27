"""Confirma a precisão usando o código real da app (Tagger.suggest), não o avaliador.

    python ferramentas/benchmark/verificar_app.py <pasta ESC-50>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench import Data, embed_all  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sfxtagger.ai import HELPER_MODEL, Tagger  # noqa: E402

esc = Path(sys.argv[1])
data = Data(esc)
helper = embed_all(esc, HELPER_MODEL)
X2 = np.stack([helper[r["filename"]] for r in data.meta])
tagger = Tagger()
tagger.ensure_loaded()
idx = {c.catid: i for i, c in enumerate(tagger.categories)}

for label, use_helper in (("só o modelo principal", False), ("dois modelos (app)", True)):
    for name, mask in (("dev", data.dev), ("teste", data.test)):
        top1 = top5 = 0
        rows = np.where(mask)[0]
        for n in rows:
            s = tagger.suggest(data.X[n], [], helper_emb=X2[n] if use_helper else None)
            ids = [idx[c["catid"]] for c in s["categories"]]
            top1 += data.ok[n, ids[0]]
            top5 += data.ok[n, ids].any()
        print(f"{label:24} {name:6} top1 {100 * top1 / len(rows):5.1f}  top5 {100 * top5 / len(rows):5.1f}")
