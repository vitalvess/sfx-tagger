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
# use_third: 3.º modelo (Precisão máxima) também no classificador. 6 testes, 6049 sons:
# 1.ª sugestão média 66,3 -> 68,7 (ferramentas/benchmark/exp_aumento.py)
# use_aug: cópias com fundos (aumentos.npz). Medido com o "Testar a IA": normal 64,7 -> 64,4,
# Precisão máxima 70,0 -> 69,8 (exp_validar.py). Não ajudam: ficam guardadas mas desligadas.
TRAIN = {"temperature": 10.0, "use_helper": True, "head": "linear", "use_third": True, "use_aug": False}
BLEND = {"weight": 0.6, "gate_at": 0.55, "gate_width": 0.06}
ADAPT = {"quantile": 0.75, "margin": 0.35,   # limiar >= sons típicos da biblioteca + margem
         "ceiling": 0.65}                    # ... mas nunca acima disto (ver adapt())
MIN_EXAMPLES = 3          # abaixo disto fica só a memória kNN


def _norm(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


def features(X1: np.ndarray, X2: np.ndarray | None = None, X3: np.ndarray | None = None) -> np.ndarray:
    """Impressão digital usada pelo classificador (de cada modelo que houver)."""
    parts = [_norm(np.atleast_2d(np.asarray(X, np.float32))) for X in (X1, X2, X3) if X is not None]
    return parts[0] if len(parts) == 1 else np.concatenate(parts, 1) / np.sqrt(len(parts))


@dataclass
class Model:
    classes: np.ndarray      # índice UCS de cada classe aprendida
    W: np.ndarray            # (classes x dim), linhas normalizadas
    protos: np.ndarray       # média normalizada dos exemplos de cada classe
    temperature: float
    use_helper: bool
    thresholds: np.ndarray | None = None   # limiar da "porta" de cada classe (adapt())
    b: np.ndarray | None = None            # cabeça linear: termo constante (None = cossenos)
    use_third: bool = False                # também o 3.º modelo (Precisão máxima)

    def probs(self, F: np.ndarray) -> np.ndarray:
        """Probabilidades do classificador nas suas classes (F = features())."""
        logits = F @ self.W.T + self.b if self.b is not None else self.temperature * F @ self.W.T
        logits = logits - logits.max(1, keepdims=True)
        q = np.exp(logits)
        return q / q.sum(1, keepdims=True)


def _linear_head(F: np.ndarray, yi: np.ndarray, K: int, epochs: int = 60, batch: int = 256,
                 lr: float = 2e-3, wd: float = 1e-3, smooth: float = 0.05, noise: float = 0.02,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Regressão softmax (AdamW) com pesos 1/sqrt(n) por categoria, label smoothing e ruído
    nas impressões. Igual a torch.nn.Linear + CrossEntropyLoss(weight, label_smoothing)."""
    rng = np.random.default_rng(seed)
    n, d = F.shape
    cw = 1 / np.sqrt(np.bincount(yi, minlength=K))
    cw = (cw / cw.mean()).astype(np.float32)
    lim = 1 / np.sqrt(d)
    W = rng.uniform(-lim, lim, (K, d)).astype(np.float32)
    b = rng.uniform(-lim, lim, K).astype(np.float32)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            j = perm[s:s + batch]
            x = F[j] + noise * rng.standard_normal((len(j), d)).astype(np.float32)
            z = x @ W.T + b
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            a = np.broadcast_to(smooth * cw / K, (len(j), K)).copy()   # peso de cada -log p
            a[np.arange(len(j)), yi[j]] += (1 - smooth) * cw[yi[j]]
            g = (p * a.sum(1, keepdims=True) - a) / cw[yi[j]].sum()
            gW, gb = g.T @ x, g.sum(0)
            step += 1
            for P_, G, M, V in ((W, gW, mW, vW), (b, gb, mb, vb)):
                P_ *= 1 - lr * wd
                M *= b1
                M += (1 - b1) * G
                V *= b2
                V += (1 - b2) * G * G
                P_ -= lr * (M / (1 - b1 ** step)) / (np.sqrt(V / (1 - b2 ** step)) + eps)
    return W, b


def train(X1: np.ndarray, X2: np.ndarray | None, y: np.ndarray, temperature: float = 20.0,
          use_helper: bool = False, iters: int = 60, lr: float = 0.5, l2: float = 1.0,
          head: str = "cos", X3: np.ndarray | None = None, use_third: bool = False) -> Model:
    """head "cos": classificador de cossenos; parte da média de cada categoria e afina com
    descida de gradiente (puxado para essa média, para não decorar poucos exemplos).
    head "linear": regressão softmax (_linear_head). A porta usa sempre as médias (protos)."""
    F = features(X1, X2 if use_helper else None, X3 if use_third else None)
    y = np.asarray(y)
    classes, yi = np.unique(y, return_inverse=True)
    onehot = np.eye(len(classes), dtype=np.float32)[yi]
    protos = _norm(onehot.T @ F)
    if head == "linear" and len(classes) >= 2:
        W, b = _linear_head(F.astype(np.float32), yi, len(classes))
        return Model(classes=classes, W=W, protos=protos.astype(np.float32),
                     temperature=1.0, use_helper=use_helper, b=b, use_third=use_third)
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
                 temperature=temperature, use_helper=use_helper, use_third=use_third)


def adapt(model: Model, background: np.ndarray, gate_at: float = 0.55, quantile: float = 0.75,
          margin: float = 0.35, ceiling: float | None = None) -> Model:
    """Ajusta a "porta" de cada classe à biblioteca. background: features() dos sons da
    biblioteca (ou de todos os exemplos). Um som só passa se estiver bem mais perto do
    retrato da classe (margem) do que um som típico da biblioteca (quantil 75%). Não se usa
    um quantil alto: em bibliotecas pequenas seria o próprio som confirmado e o seu take
    vizinho (FOLDER04: com 0,95 o take gémeo de um som confirmado nunca era ajudado).

    ceiling: teto do limiar. Sem ele, numa pasta só de sons parecidos de uma categoria que a IA
    conhece bem (ex.: só ambientes de cidade) o limiar subia acima dos próprios sons e o
    classificador calava-se onde mais ajudava. Medido em 6 pastas de teste (2026-11, fundo = a
    pasta, como na app): 1.ª sugestão média 49,5 -> 58,4 (TUT 2018 10,5 -> 33,8); sons de
    categorias desconhecidas 1,2 -> 2,9 pontos abaixo de sem classificador."""
    if background is None or len(background) < 5:
        return model
    typical = np.quantile(background @ model.protos.T, quantile, axis=0)
    thr = typical + margin
    if ceiling is not None:
        thr = np.minimum(thr, ceiling)
    return Model(**{**model.__dict__, "thresholds": np.maximum(gate_at, thr).astype(np.float32)})


def blend(P: np.ndarray, model: Model, X1: np.ndarray, X2: np.ndarray | None = None,
          weight: float = 0.8, gate_at: float = 0.65, gate_width: float = 0.03,
          X3: np.ndarray | None = None) -> np.ndarray:
    """Mistura as probabilidades da IA (P: sons x categorias UCS) com as do classificador."""
    P = np.atleast_2d(P)
    F = features(X1, X2 if model.use_helper else None, X3 if model.use_third else None)
    q = model.probs(F)
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
