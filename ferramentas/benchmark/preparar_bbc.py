"""Prepara as pastas BBC Sound Effects: uma de treino (já com o CatID gravado) e uma de teste (com respostas.csv).

    python ferramentas/benchmark/preparar_bbc.py <fichas.json> <seleccao.json> <pasta de treino> <pasta de teste>

fichas.json: as fichas da API da BBC. seleccao.json: {"train": [[id, [catids]]...], "test": [...]}, feita com
bbc_ucs.catids. O áudio vem em MP3 e é gravado como WAV mono de 16 bits (a app só abre WAV e analisa em mono).
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sfxtagger import metadata, ucs  # noqa: E402
from sfxtagger.project import FIELD_KEYS  # noqa: E402

MEDIA = "https://sound-effects-media.bbcrewind.co.uk/mp3/{}.mp3"
LICENCE = "RemArc (BBC): uso pessoal, educativo ou de investigacao. Uso comercial proibido."


def fetch(name: str, dest: Path) -> None:
    for t in range(5):
        try:
            req = urllib.request.Request(MEDIA.format(name), headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=120).read()
            tmp = dest.with_suffix(".mp3.part")
            tmp.write_bytes(data)
            y, sr = sf.read(tmp, dtype="float32", always_2d=True)
            sf.write(dest, y.mean(axis=1), sr, subtype="PCM_16")
            tmp.unlink()
            return
        except Exception:  # noqa: BLE001 — rede instável: tenta outra vez
            if t == 4:
                raise
            time.sleep(3 * (t + 1))


def fxname(desc: str) -> str:
    s = re.split(r"[.,;(]", desc)[0]
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s)
    return " ".join(s.split()[:5]).title()[:30] or "BBC"


def main(fichas: Path, sel: Path, train_dir: Path, test_dir: Path) -> None:
    recs = {r["id"]: r for r in json.loads(fichas.read_text(encoding="utf-8"))}
    s = json.loads(sel.read_text(encoding="utf-8"))
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rid, ids in s["test"]:
        jobs.append((recs[rid], ids, test_dir / f"BBC_{rid}.wav", False))
    for rid, ids in s["train"]:
        cat = ucs.lookup(ids[0])
        r = recs[rid]
        name = ucs.build_filename(cat.catid, f"{fxname(r['description'])} {rid[-5:]}", "BBC", "BBC")
        jobs.append((r, ids, train_dir / name, True))

    todo = [j for j in jobs if not j[2].exists()]
    print(len(jobs), "sons,", len(todo), "por descarregar", flush=True)
    done = [0]
    errors = []

    def one(job):
        r, ids, dest, is_train = job
        try:
            fetch(r["file"]["small"]["name"], dest)
            if is_train:
                cat = ucs.lookup(ids[0])
                f = {k: "" for k in FIELD_KEYS}
                tags = [t for t in r.get("tags", []) if len(t) > 2][:6]
                f.update(catid=cat.catid, category=cat.category, subcategory=cat.subcategory, category_full=cat.full,
                         fxname=fxname(r["description"]), description=r["description"][:250],
                         keywords=", ".join(dict.fromkeys([*tags, *cat.synonyms[:4]])),
                         library="BBC Sound Effects (treino)", creator_id="BBC", source_id="BBC")
                metadata.write_metadata(dest, f)
        except Exception as e:  # noqa: BLE001
            errors.append((r["id"], str(e)[:80]))
            dest.unlink(missing_ok=True)
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"  {done[0]}/{len(todo)}  erros {len(errors)}", flush=True)

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, todo))
    print("descarga feita, erros:", len(errors), errors[:5], flush=True)

    with open(test_dir / "respostas.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["ficheiro", "catid"])
        for rid, ids in s["test"]:
            if (test_dir / f"BBC_{rid}.wav").exists():
                w.writerow([f"BBC_{rid}.wav", "|".join(ids)])
    for folder, part in ((train_dir, "train"), (test_dir, "test")):
        with open(folder / "fontes.csv", "w", newline="", encoding="utf-8-sig") as h:
            w = csv.writer(h)
            w.writerow(["id BBC", "descricao", "catid", "licenca"])
            for rid, ids in s[part]:
                w.writerow([rid, recs[rid]["description"], ids[0], LICENCE])
    note = ("Sons da BBC Sound Effects (sound-effects.bbcrewind.co.uk).\r\n" + LICENCE + "\r\n"
            "A categoria UCS foi deduzida da descricao da BBC por regras fixas (nao pela IA); cerca de 1 em 15 a 20\r\n"
            "pode estar errada ou ser discutivel.\r\n")
    (train_dir / "LEIA-ME.txt").write_text(
        "SONS DE TREINO BBC PARA O SFX TAGGER\r\n\r\n" + note +
        "\r\nCOMO TREINAR: Abrir pasta > esta pasta > filtro \"Por rever\" > marcar tudo > \"Aprovar selecionados\".\r\n"
        "Os exemplos ficam com o autor BBC; o botao \"remover\" em Treinar a IA apaga-os todos.\r\n", encoding="utf-8")
    (test_dir / "LEIA-ME.txt").write_text(
        "SONS DE TESTE BBC PARA O SFX TAGGER\r\n\r\n" + note +
        "\r\nSERVE SO PARA \"Analisar > Testar a IA com sons ja classificados\". NUNCA APROVES ESTES SONS:\r\n"
        "se a IA aprender com eles, o teste deixa de ser justo.\r\n", encoding="utf-8")
    gb = lambda p: sum(f.stat().st_size for f in p.iterdir()) / 1e9
    print(f"treino {len(list(train_dir.glob('*.wav')))} sons {gb(train_dir):.1f} GB | "
          f"teste {len(list(test_dir.glob('*.wav')))} sons {gb(test_dir):.1f} GB", flush=True)


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:5]))
