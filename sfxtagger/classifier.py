"""Classificador treinado com os sons confirmados por pessoas (o "nível 2" da memória).

A memória kNN (ai._apply_memory) procura o exemplo mais parecido. Este classificador
aprende, para cada categoria já confirmada, um "retrato" que resume TODOS os seus
exemplos: com vários exemplos por categoria generaliza melhor do que o vizinho mais
próximo, e aprende o vosso estilo de gravação (microfones, salas, gravadores).

* Só conhece as categorias que alguém já confirmou. Para não estragar os sons de outros
  tipos, só intervém quando o som se parece com uma categoria conhecida ("porta"). O que é
  "parecido" depende da biblioteca: gravações da mesma sala e do mesmo gravador parecem-se
  todas umas com as outras, por isso o limiar de cada categoria sobe até ficar acima do
  típico da própria biblioteca (adapt()).
* Treina em menos de um segundo com milhares de exemplos (só numpy, sem PyTorch).
* Afinado no ESC-50 com metade dos tipos de som desconhecidos (ferramentas/benchmark/
  exp_otimizacao.py treino).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# parâmetros afinados (exp_otimizacao.py treino)
TRAIN = {"temperature": 10.0, "use_helper": True}
BLEND = {"weight": 0.6, "gate_at": 0.55, "gate_width": 0.06}
ADAPT = {"quantile": 0.75, "margin": 0.35}  # limiar >= sons típicos da biblioteca + margem
MIN_EXAMPLES = 3          # abaixo disto fica só a memória kNN


def _norm(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


def features(X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
    """Impressão digital usada pelo classificador (dos dois modelos, se houver)."""
    X1 = _norm(np.atleast_2d(np.asarray(X1, np.float32)))
    if X2 is None:
        return X1
    return np.concatenate([X1, _norm(np.atleast_2d(np.asarray(X2, np.float32)))], 1) / np.sqrt(2)


@dataclass
class Model:
    classes: np.ndarray      # índice UCS de cada classe aprendida
    W: np.ndarray            # (classes x dim), linhas normalizadas
    protos: np.ndarray       # média normalizada dos exemplos de cada classe
    temperature: float
    use_helper: bool
    thresholds: np.ndarray | None = None   # limiar da "porta" de cada classe (adapt())


def train(X1: np.ndarray, X2: np.ndarray | None, y: np.ndarray, temperature: float = 20.0,
          use_helper: bool = False, iters: int = 60, lr: float = 0.5, l2: float = 1.0) -> Model:
    """Classificador de cossenos: parte da média de cada categoria e afina com descida de
    gradiente (puxado para essa média, para não decorar poucos exemplos)."""
    F = features(X1, X2 if use_helper else None)
    y = np.asarray(y)
    classes, yi = np.unique(y, return_inverse=True)
    onehot = np.eye(len(classes), dtype=np.float32)[yi]
    protos = _norm(onehot.T @ F)
    W = protos.copy()
    if len(classes) >= 2:
        for _ in range(iters):
            logits = temperature * F @ _norm(W).T
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            grad = temperature * (p - onehot).T @ F / len(F) + l2 * (W - protos)
            W -= lr * grad
    return Model(classes=classes, W=_norm(W).astype(np.float32), protos=protos.astype(np.float32),
                 temperature=temperature, use_helper=use_helper)


def adapt(model: Model, background: np.ndarray, gate_at: float = 0.55, quantile: float = 0.75,
          margin: float = 0.35) -> Model:
    """Ajusta a "porta" de cada classe à biblioteca. background: features() dos sons da
    biblioteca (ou de todos os exemplos). Um som só passa se estiver bem mais perto do
    retrato da classe (margem) do que um som típico da biblioteca (quantil 75%). Não se usa
    um quantil alto: em bibliotecas pequenas seria o próprio som confirmado e o seu take
    vizinho (FOLDER04: com 0,95 o take gémeo de um som confirmado nunca era ajudado)."""
    if background is None or len(background) < 5:
        return model
    typical = np.quantile(background @ model.protos.T, quantile, axis=0)
    return Model(**{**model.__dict__, "thresholds": np.maximum(gate_at, typical + margin).astype(np.float32)})


def blend(P: np.ndarray, model: Model, X1: np.ndarray, X2: np.ndarray | None = None,
          weight: float = 0.8, gate_at: float = 0.65, gate_width: float = 0.03) -> np.ndarray:
    """Mistura as probabilidades da IA (P: sons x categorias UCS) com as do classificador."""
    P = np.atleast_2d(P)
    F = features(X1, X2 if model.use_helper else None)
    logits = model.temperature * F @ model.W.T
    logits -= logits.max(1, keepdims=True)
    q = np.exp(logits)
    q /= q.sum(1, keepdims=True)
    # parece-se com alguma categoria conhecida? (semelhança ao "retrato" mais próximo)
    sims = F @ model.protos.T
    nearest = sims.argmax(1)
    near = sims[np.arange(len(F)), nearest]
    at = model.thresholds[nearest] if model.thresholds is not None else gate_at
    gate = 1 / (1 + np.exp(-(near - at) / gate_width))
    w = (weight * gate)[:, None]
    Q = np.zeros_like(P)
    Q[:, model.classes] = q
    out = (1 - w) * P + w * Q
    return out / out.sum(1, keepdims=True)
