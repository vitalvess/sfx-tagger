"""Treino em massa: transforma bases públicas (cópias em parquet do Hugging Face) em pastas de treino e
de teste para o Earmark, com respostas.csv (CatIDs UCS aceites, o primeiro é o principal).

    python ferramentas/benchmark/preparar_massa.py <base> <pasta das bases originais> <pasta de saída>

Corre num ambiente com pyarrow e soundfile (não precisa da app). Grava WAV mono de 16 bits.
Regras: o teste nunca partilha local/série/sessão de gravação com o treino; sons do Freesound que
estejam nos testes FSD50K ou ESC-50 ficam de fora (as bases derivadas do Freesound repetem sons).
"""
from __future__ import annotations

import collections
import csv
import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

# --- tradução das classes de cada base para CatIDs UCS ---------------------------------------------
MAPS: dict[str, dict[str, list[str]]] = {
    "tut2018": {
        "airport": ["AMBTran", "AMBPubl"],
        "bus": ["VEHBus", "VEHInt"],
        "metro": ["TRNSbwy", "TRNInt", "TRNElec"],
        "metro_station": ["AMBTran", "AMBUndr"],
        "park": ["AMBPark", "AMBSubn"],
        "public_square": ["AMBUrbn", "AMBTown", "AMBPubl"],
        "shopping_mall": ["AMBPubl", "CRWDWalla"],
        "street_pedestrian": ["AMBUrbn", "AMBTown", "CRWDWalla"],
        "street_traffic": ["AMBTraf", "AMBUrbn"],
        "tram": ["TRNTram", "TRNInt"],
    },
    "cochlscene": {
        "Bus": ["VEHBus", "VEHInt"],
        "Cafe": ["AMBRest", "CRWDConv"],
        "Car": ["VEHInt", "VEHCar"],
        "CrowdedIndoor": ["AMBPubl", "CRWDWalla", "CRWDConv"],
        "Elevator": ["MACHElev"],
        "Kitchen": ["AMBHome", "FOODCook", "FOODKware"],
        "Park": ["AMBPark", "AMBSubn"],
        "ResidentialArea": ["AMBSubn", "AMBHome", "AMBTown"],
        "Restaurant": ["AMBRest", "CRWDConv"],
        "Restroom": ["AMBPubl", "WATRPlmb", "AMBRoom"],
        "Street": ["AMBUrbn", "AMBTraf", "AMBTown"],
        "Subway": ["TRNSbwy", "TRNInt"],
        "SubwayStation": ["AMBTran", "AMBUndr"],
    },
    "urbansound8k": {
        "air_conditioner": ["MACHHvac", "MACHFan"],
        "car_horn": ["VEHHorn"],
        "children_playing": ["CRWDChld", "VOXChld", "AMBSchl"],
        "dog_bark": ["ANMLDog"],
        "drilling": ["TOOLPowr", "TOOLPneu"],
        "engine_idling": ["MOTRComb", "VEHMisc", "VEHTruck", "VEHCar", "VEHBus"],
        "gun_shot": ["GUNMisc", "GUNPis", "GUNRif"],
        "jackhammer": ["TOOLPneu", "MACHCnst"],
        "siren": ["VEHSirn", "ALRMSirn"],
        "street_music": ["MUSCPrfm", "MUSCInst"],
    },
    "nonspeech7k": {
        "breath": ["HMNBrth"],
        "cough": ["HMNCough"],
        "crying": ["VOXCry", "VOXBaby"],
        "laugh": ["VOXLaff"],
        "screaming": ["VOXScrm"],
        "sneeze": ["HMNSneez"],
        "yawn": ["VOXEfrt", "HMNBrth", "HMNMisc"],
    },
    "vocalsound": {
        "cough": ["HMNCough"],
        "laughter": ["VOXLaff"],
        "sigh": ["HMNBrth", "VOXEfrt"],
        "sneeze": ["HMNSneez"],
        "sniff": ["HMNSniff"],
        "throatclearing": ["HMNCough", "HMNMisc", "VOXEfrt"],
    },
}
REPOS = {"tut2018": "TUT-urban-acoustic-scenes-2018-development", "cochlscene": "CochlScene",
         "urbansound8k": "urbansound8K", "nonspeech7k": "Nonspeech7k", "vocalsound": "VocalSound"}
LABEL_COL = {"tut2018": "scene_label", "cochlscene": "class", "urbansound8k": "class", "nonspeech7k": "classname",
             "vocalsound": "classname"}
N_TRAIN = 200   # por classe
N_TEST = 100    # por classe


def group_of(base: str, row: dict) -> str:
    """Grupo que nunca se divide entre treino e teste (mesmo local, mesma gravação original)."""
    if base == "tut2018":   # identifier: "barcelona-0" (cidade-local)
        return str(row.get("identifier"))
    if base == "urbansound8k":
        return str(row.get("fold"))      # as 10 partes oficiais já separam gravações
    if base == "vocalsound":    # "f0003_0_cough.wav": o grupo é o falante
        return str(row.get("speaker"))
    if base == "nonspeech7k":   # filename: "<id Freesound>[-n]_<classe>_<aug>.wav": mesmo som original = mesmo grupo
        return freesound_id(row)
    if base == "cochlscene":   # audio_id: ".../Bus_user0618_14892164_001": o grupo é quem gravou
        a = str(row.get("audio_id") or "")
        m = re.search(r"user\d+", a)
        return m.group(0) if m else a
    return ""


def freesound_id(row: dict) -> str:
    m = re.match(r"\d+", str(row.get("filename") or ""))
    return m.group(0) if m else str(row.get("filename"))


def freesound_excluded() -> set[str]:
    """IDs do Freesound de TODA a FSD50K e do ESC-50: o Teste fixo é tirado delas, por isso nenhum
    destes sons pode entrar no treino das bases novas (as bases do Freesound repetem sons)."""
    out: set[str] = set()
    r = Path(r"D:\SFX Testes\Bases originais")
    for name in ("eval.csv", "dev.csv"):
        for p in (r / "FSD50K" / "FSD50K.ground_truth" / name, r / "FSD50K" / name):
            if p.exists():
                out |= {row["fname"] for row in csv.DictReader(open(p, encoding="utf-8"))}
    esc = r / "ESC-50" / "ESC-50-master" / "meta" / "esc50.csv"
    if esc.exists():
        out |= {row["src_file"] for row in csv.DictReader(open(esc, encoding="utf-8"))}
    assert len(out) > 50000, f"lista de exclusão incompleta ({len(out)}): o disco D: está ligado?"
    return out


def rows(base: str, src: Path):
    if base == "vocalsound":
        import zipfile
        with zipfile.ZipFile(src / "VocalSound.zip") as z:
            for n in sorted(x for x in z.namelist() if x.endswith(".wav") and "/audios/" in x):
                stem = n.rsplit("/", 1)[1][:-4]
                speaker, _, cls = stem.split("_", 2)
                yield {"audio": {"bytes": z.read(n)}, "classname": cls, "speaker": speaker, "filename": stem}
        return
    files = sorted(src.rglob("*.parquet"))
    for f in files:
        t = pq.ParquetFile(f)
        for batch in t.iter_batches(batch_size=64):
            for r in batch.to_pylist():
                yield r


def to_wav(audio: dict, dest: Path) -> float:
    data = audio.get("bytes")
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    sf.write(dest, y.mean(axis=1), sr, subtype="PCM_16")
    return len(y) / sr


def main(base: str, src_root: Path, out_root: Path) -> None:
    mp = MAPS[base]
    src = src_root / REPOS[base]
    excl = freesound_excluded() if base in ("urbansound8k", "nonspeech7k") else set()
    # 1.ª passagem: só metadados, para escolher treino/teste por grupo
    meta = []
    for i, r in enumerate(rows(base, src)):
        lab = str(r.get(LABEL_COL[base]))
        if lab not in mp:
            continue
        if base == "urbansound8k" and str(r.get("fsID")) in excl:
            continue
        if base == "nonspeech7k" and freesound_id(r) in excl:
            continue
        meta.append((i, lab, group_of(base, r)))
    print(base, len(meta), "sons com classe traduzível", collections.Counter(m[1] for m in meta), flush=True)
    rng = random.Random(0)
    pick: dict[int, str] = {}
    for lab in mp:
        groups = collections.defaultdict(list)
        for i, l, g in meta:
            if l == lab:
                groups[g].append(i)
        gs = list(groups.values())
        rng.shuffle(gs)
        te, tr = [], []
        for k, g in enumerate(gs):   # um grupo em cada 4 vai para o teste, até ter N_TEST sons
            (te if (k % 4 == 0 and len(te) < N_TEST) else tr).extend(g)
        rng.shuffle(te)
        rng.shuffle(tr)
        for i in te[:N_TEST]:
            pick[i] = "teste"
        for i in tr[:N_TRAIN]:
            pick[i] = "treino"
    name = {"tut2018": "TUT 2018 (cidades)", "cochlscene": "CochlScene (ambientes)",
            "urbansound8k": "UrbanSound8K (rua)", "nonspeech7k": "Nonspeech7k (sons humanos)",
            "vocalsound": "VocalSound (sons de voz)"}[base]
    dirs = {"treino": out_root / f"Treino - {name}", "teste": out_root / "Testes" / f"Teste - {name}"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    answers = {"treino": [], "teste": []}
    info = {"treino": [], "teste": []}
    n = 0
    for i, r in enumerate(rows(base, src)):
        part = pick.get(i)
        if not part:
            continue
        lab = str(r.get(LABEL_COL[base]))
        fn = f"{base}_{i:06d}.wav"
        try:
            dur = to_wav(r["audio"], dirs[part] / fn)
        except Exception as e:  # noqa: BLE001
            print("  erro", fn, str(e)[:80], flush=True)
            continue
        answers[part].append((fn, "|".join(mp[lab])))
        extra = {k: r.get(k) for k in ("identifier", "fsID", "fold", "audio_id", "source", "filename", "speaker") if r.get(k) is not None}
        info[part].append((fn, lab, mp[lab][0], round(dur, 2), json.dumps(extra, ensure_ascii=False)))
        n += 1
        if n % 1000 == 0:
            print(f"  {n}/{len(pick)}", flush=True)
    for part, d in dirs.items():
        with open(d / "respostas.csv", "w", newline="", encoding="utf-8") as h:
            w = csv.writer(h)
            w.writerow(["ficheiro", "catid"])
            w.writerows(answers[part])
        with open(d / "origem.csv", "w", newline="", encoding="utf-8-sig") as h:
            w = csv.writer(h)
            w.writerow(["ficheiro", "classe original", "catid principal", "duracao (s)", "origem"])
            w.writerows(info[part])
        print(part, d, len(answers[part]), "sons", collections.Counter(a[1].split("|")[0] for a in answers[part]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
