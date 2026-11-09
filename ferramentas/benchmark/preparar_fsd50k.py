"""Prepara a FSD50K para o botão "Testar a IA": extrai os sons com classe traduzível e cria respostas.csv.

    python ferramentas/benchmark/preparar_fsd50k.py <pasta dos downloads> <pasta de saída> [eval|dev]

Os zips da FSD50K estão divididos (.z01, .zip), o que o zipfile do Python não lê; aqui o índice
é lido à mão e cada posição é somada ao tamanho das partes anteriores.
"""
from __future__ import annotations

import csv
import json
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sfxtagger import ucs  # noqa: E402


class Dividido:
    def __init__(self, partes: list[Path]):
        self.files = [open(p, "rb") for p in partes]
        self.sizes = [p.stat().st_size for p in partes]
        self.start = [sum(self.sizes[:i]) for i in range(len(partes))]

    def read(self, pos: int, n: int) -> bytes:
        out = b""
        for f, st, sz in zip(self.files, self.start, self.sizes):
            if n and st <= pos < st + sz:
                f.seek(pos - st)
                b = f.read(min(n, st + sz - pos))
                out += b
                pos += len(b)
                n -= len(b)
        return out


def entradas(z: Dividido):
    tail_at = sum(z.sizes) - 70000
    tail = z.read(tail_at, 70000)
    i = tail.rfind(b"PK\x05\x06")
    _, _, cd_disk, _, count, _, off = struct.unpack("<IHHHHII", tail[i:i + 20])
    pos = z.start[cd_disk] + off
    n64 = tail.rfind(b"PK\x06\x06")
    if count == 0xFFFF or off == 0xFFFFFFFF:
        r = struct.unpack("<IQHHIIQQQQ", tail[n64:n64 + 56])
        count, cd_disk, off = r[7], r[5], r[9]
        pos = z.start[cd_disk] + off
    for _ in range(count):
        h = z.read(pos, 46)
        method, csize, name_len, extra_len, comment_len, disk, loc = (
            struct.unpack("<H", h[10:12])[0], struct.unpack("<I", h[20:24])[0],
            struct.unpack("<H", h[28:30])[0], struct.unpack("<H", h[30:32])[0],
            struct.unpack("<H", h[32:34])[0], struct.unpack("<H", h[34:36])[0], struct.unpack("<I", h[42:46])[0])
        name = z.read(pos + 46, name_len).decode()
        yield name, method, csize, z.start[disk] + loc
        pos += 46 + name_len + extra_len + comment_len


def main(base: Path, out: Path, parte: str) -> None:
    mapping = {k: v for k, v in json.loads((Path(__file__).parent / "fsd50k_ucs.json").read_text(encoding="utf-8")).items()
               if not k.startswith("_")}
    for k, v in mapping.items():
        assert v and all(ucs.lookup(c) for c in v), (k, v)
    with open(base / "FSD50K.ground_truth" / f"{parte}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    truth = {}
    for r in rows:
        ids = list(dict.fromkeys(c for l in r["labels"].split(",") for c in mapping.get(l, [])))
        if ids:
            truth[r["fname"]] = ids
    print(f"{len(truth)} de {len(rows)} sons têm classe traduzível", flush=True)
    stem = f"FSD50K.{parte}_audio"
    partes = sorted(base.glob(f"{stem}.z[0-9][0-9]")) + [base / f"{stem}.zip"]
    z = Dividido(partes)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, method, csize, loc in entradas(z):
        fname = Path(name).stem
        if not name.endswith(".wav") or fname not in truth:
            continue
        lh = z.read(loc, 30)
        data = z.read(loc + 30 + struct.unpack("<H", lh[26:28])[0] + struct.unpack("<H", lh[28:30])[0], csize)
        if method == 8:
            data = zlib.decompress(data, -15)
        assert data[:4] == b"RIFF", name
        (out / f"{fname}.wav").write_bytes(data)
        n += 1
        if n % 2000 == 0:
            print(f"  {n}", flush=True)
    with open(out / "respostas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ficheiro", "catid"])
        for fname, ids in truth.items():
            if (out / f"{fname}.wav").exists():
                w.writerow([f"{fname}.wav", "|".join(ids)])
    print(n, "sons extraídos para", out, sum(p.stat().st_size for p in out.iterdir()) // 2 ** 20, "MB", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "eval")
