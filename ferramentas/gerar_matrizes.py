"""Pré-calcula as matrizes de texto (categorias UCS e keywords) e guarda-as no pacote,
para que a app não tenha de as calcular no computador de cada pessoa.

    python ferramentas/gerar_matrizes.py

Correr de novo sempre que mudarem as frases/pesos em ai.py ou a lista UCS.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sfxtagger import paths  # noqa: E402
from sfxtagger.ai import DATA_DIR, Tagger  # noqa: E402

for old in DATA_DIR.glob("cat_*.npy"):
    old.unlink()
for old in DATA_DIR.glob("kw_*.npy"):
    old.unlink()
for f in paths.cache_dir().glob("*.npy"):   # força o cálculo a partir do zero
    f.unlink()

t = Tagger()
t.ensure_loaded()
for f in paths.cache_dir().glob("*.npy"):
    if f.name.startswith(("cat_", "kw_")):
        shutil.copy2(f, DATA_DIR / f.name)
        print("->", f.name, f"{f.stat().st_size // 1024} KB")
