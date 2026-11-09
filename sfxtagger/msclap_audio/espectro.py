"""Substitui o torchlibrosa (que puxa o librosa) só com o que o HTSAT usa em inferência.

Os filtros mel vêm do próprio ficheiro de pesos (melW), por isso não é preciso calculá-los.
"""
from __future__ import annotations

import torch
from torch import nn


class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, window="hann", center=True,
                 pad_mode="reflect", freeze_parameters=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.register_buffer("janela", torch.hann_window(win_length or n_fft), persistent=False)

    def forward(self, x):
        s = torch.stft(x, self.n_fft, self.hop_length, window=self.janela, center=self.center,
                       pad_mode=self.pad_mode, return_complex=True)
        return (s.real ** 2 + s.imag ** 2).transpose(1, 2).unsqueeze(1)   # (lote, 1, tempo, freq)


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=22050, n_fft=2048, n_mels=64, fmin=0.0, fmax=None, ref=1.0, amin=1e-10,
                 top_db=80.0, freeze_parameters=True):
        super().__init__()
        self.amin, self.ref, self.top_db = amin, ref, top_db
        self.melW = nn.Parameter(torch.zeros(n_fft // 2 + 1, n_mels), requires_grad=False)

    def forward(self, x):
        db = 10.0 * torch.log10(torch.clamp(x @ self.melW, min=self.amin))
        db = db - 10.0 * torch.log10(torch.tensor(max(self.amin, self.ref)))
        return db if self.top_db is None else torch.clamp(db, min=db.max().item() - self.top_db)


class SpecAugmentation(nn.Module):
    """Só se usa a treinar; aqui nunca é chamada."""

    def __init__(self, *args, **kwargs):
        super().__init__()
