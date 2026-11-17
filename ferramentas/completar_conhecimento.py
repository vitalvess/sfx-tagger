"""Completa o conhecimento da IA: 3.º modelo e cópias misturadas com fundos.

    python ferramentas/completar_conhecimento.py

Com a app DESLIGADA (escreve na pasta conhecimento) e, de preferência, com o disco externo
ligado (é lá que está o som de muitos exemplos). Pode ser interrompido: na vez seguinte continua
onde parou. Exemplos cujo som não se encontre ficam como estavam.
  * embeddings3.npz: cada exemplo ouvido pelo 3.º modelo (MS-CLAP), usado pelo classificador
    quando a Precisão máxima está ligada;
  * aumentos.npz: uma cópia de cada som isolado misturada com um ambiente (0 a 12 dB abaixo),
    só para treino. Os ambientes não se misturam (já são fundo).
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earmark import audio, precisao, suspeitos  # noqa: E402
from earmark.ai import get_tagger  # noqa: E402
from earmark.knowledge import get_knowledge  # noqa: E402

CENAS = {"TUT2018", "CochlScene"}
SNR_DB = (0.0, 12.0)
GUARDAR_A_CADA = 300


def app_ligada(port: int = 8765) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def misturar(x: np.ndarray, fundo: np.ndarray, snr_db: float) -> np.ndarray:
    if len(fundo) < len(x):
        fundo = np.tile(fundo, len(x) // max(len(fundo), 1) + 1)
    fundo = fundo[: len(x)]
    rx, rf = np.sqrt(np.mean(x ** 2)) + 1e-9, np.sqrt(np.mean(fundo ** 2)) + 1e-9
    y = x + fundo * (rx / rf) / (10 ** (snr_db / 20))
    pico = np.abs(y).max()
    return (y / pico * 0.89 if pico > 0.99 else y).astype(np.float32)


def main() -> None:
    if app_ligada():
        raise SystemExit("Desliga a app primeiro: esta ferramenta escreve no conhecimento da IA.")
    kb = get_knowledge()
    t = get_tagger()
    t.ensure_loaded()
    precisao._load()
    print(f"exemplos: {len(kb.meta)} | já com 3.º modelo: {len(kb.emb3)} | com fundo: {len(kb.aug)}", flush=True)
    caminhos = {i: suspeitos.audio_path(m) for i, m in kb.meta.items()}
    fundos = [p for i, p in caminhos.items() if p is not None and kb.meta[i].get("author") in CENAS]
    print(f"com som encontrado: {sum(p is not None for p in caminhos.values())} | ambientes para fundo: {len(fundos)}",
          flush=True)
    rng = np.random.default_rng(0)
    feitos = falhas = 0
    t0 = time.time()
    for n, (i, p) in enumerate(caminhos.items()):
        m = kb.meta[i]
        isolado = m.get("author") not in CENAS and not (m.get("catid") or "").startswith("AMB")
        precisa3, precisa_aug = i not in kb.emb3, isolado and i not in kb.aug and bool(fundos)
        if p is None or not (precisa3 or precisa_aug):
            continue
        try:
            x = audio.load(p).mono
            if precisa3:
                kb.emb3[i] = precisao.embed(audio.best_windows(x))
            if precisa_aug:
                y = misturar(x, audio.load(fundos[rng.integers(len(fundos))]).mono, float(rng.uniform(*SNR_DB)))
                w = audio.best_windows(y)
                kb.aug[i] = (t.embed_audio(w), t.embed_audio_helper(w), precisao.embed(w))
            feitos += 1
        except Exception as e:  # noqa: BLE001 — disco que falha, ficheiro estragado: segue
            falhas += 1
            if falhas <= 5:
                print(f"  falhou {p}: {e}", flush=True)
        if feitos and feitos % GUARDAR_A_CADA == 0:
            kb.save_extras()
            print(f"  {n + 1}/{len(caminhos)} vistos, {feitos} feitos, {falhas} falhas, "
                  f"{(time.time() - t0) / 60:.0f} min", flush=True)
    kb.save_extras()
    print(f"concluído: 3.º modelo em {len(kb.emb3)} de {len(kb.meta)} exemplos | com fundo: {len(kb.aug)} | "
          f"falhas: {falhas}", flush=True)


if __name__ == "__main__":
    main()
