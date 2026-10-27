"""Gera 'SFX Tagger (Mac).zip' com a app 'SFX Tagger.app' pronta a abrir.

Pode correr em qualquer sistema (não precisa de um Mac):
    python ferramentas/build_mac_app.py [pasta_de_saida]

A app não leva o Python nem a IA lá dentro (seria >1 GB e diferente para Mac Intel
e Apple Silicon): na primeira abertura instala tudo sozinha a partir da internet.
"""
from __future__ import annotations

import struct
import sys
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sfxtagger import __version__  # noqa: E402

APP = "SFX Tagger.app"
EXE = "sfxtagger"

INFO_PLIST = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>SFX Tagger</string>
  <key>CFBundleDisplayName</key><string>SFX Tagger</string>
  <key>CFBundleIdentifier</key><string>local.sfxtagger.app</string>
  <key>CFBundleVersion</key><string>{__version__}</string>
  <key>CFBundleShortVersionString</key><string>{__version__}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>{EXE}</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"""

LEIA_ME = """SFX TAGGER PARA MAC
===================

1. Arrasta "SFX Tagger" para a pasta Aplicacoes.

2. Abre-o com duplo clique.

   Na PRIMEIRA vez o macOS bloqueia-o, porque a app nao foi comprada na
   App Store nem assinada pela Apple. Para a autorizar (so uma vez):

   macOS 15 (Sequoia) ou mais recente:
     - Aparece "SFX Tagger nao foi aberto". Clica em "Concluido".
     - Abre Definicoes do Sistema > Privacidade e Seguranca.
     - Desce ate ao fim: "SFX Tagger foi bloqueado..." > "Abrir na mesma".
     - Confirma com a tua palavra-passe e depois em "Abrir na mesma".

   macOS 14 ou anterior:
     - Clica com o botao direito em SFX Tagger > Abrir > Abrir.

3. Na primeira abertura a app instala tudo sozinha. Precisa de internet e
   de cerca de 3 GB livres:
     - aparece uma notificacao e, 1 a 2 minutos depois, abre o browser;
     - no browser ve-se o progresso da instalacao da IA (5 a 15 minutos).
   Nas vezes seguintes abre em poucos segundos, e funciona sem internet.

COMO USAR
---------
- Arrasta os teus sons (WAV, AIFF, FLAC, OGG, MP3) para a janela do browser.
- Ouve cada som. Se a IA acertou: Cmd+Enter (Confirmar e seguinte).
  Se nao: escolhe outra sugestao (teclas 1 a 5) ou procura no campo CatID.
- Opcional: escreve em "O que ouves?" o que o som realmente e.
- Ditado por voz: clica no microfone ao lado de Description, Location, Notes
  ou "O que ouves?", fala (pode ser em portugues) e clica de novo. A
  Description e traduzida para ingles automaticamente. Na primeira vez o
  browser pede acesso ao microfone e e descarregado o modelo de voz (~1 GB).
- Gravar nos ficheiros: escreve os metadados dentro dos WAV.
- Exportar > Relatorio da IA: lista dos acertos e erros da IA.
- Conhecimento > "Exportar o meu conhecimento": cria um ficheiro .sfxk com o
  que a IA aprendeu com as tuas confirmacoes (sem audio). Envia-o a quem te
  deu a app, que o importa para a IA dele tambem melhorar.
- Para fechar a app: botao "Sair" (canto superior direito). Tambem fecha
  sozinha 20 minutos depois de fechares o separador do browser.

Os sons arrastados ficam numa copia em Documentos/SFX Tagger.
Os teus ficheiros originais nunca sao alterados.

PROBLEMAS
---------
- Se algo correr mal aparece uma janela com o botao "Abrir registo".
  Envia esse ficheiro a quem te deu a app.
- Para desinstalar: apaga a app e a pasta
  ~/Library/Application Support/SFXTagger
"""


# --- ícone (gerado aqui: quadrado arredondado com forma de onda) ---------------

def _png(rgba: np.ndarray) -> bytes:
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _icon(size: int) -> bytes:
    s = size
    y, x = np.mgrid[0:s, 0:s].astype(np.float32) + 0.5
    m = s * 0.09                      # margem à volta (estilo ícone macOS)
    r = s * 0.2                       # raio dos cantos
    cx = np.clip(x, m + r, s - m - r)
    cy = np.clip(y, m + r, s - m - r)
    inside = np.clip(r - np.hypot(x - cx, y - cy) + 0.5, 0, 1)
    t = (y - m) / (s - 2 * m)        # gradiente vertical laranja -> âmbar escuro
    top, bot = np.array([246, 176, 72]), np.array([214, 112, 22])
    rgb = top * (1 - t[..., None]) + bot * t[..., None]
    # barras da forma de onda
    bars = np.zeros((s, s), np.float32)
    heights = [0.18, 0.34, 0.55, 0.30, 0.72, 0.46, 0.62, 0.26, 0.40, 0.16]
    n = len(heights)
    area_w = s - 2 * m - s * 0.22
    bw = area_w / n
    x0 = m + s * 0.11
    for i, hgt in enumerate(heights):
        bx = x0 + i * bw + bw * 0.22
        half = hgt * (s - 2 * m) * 0.5
        sel = (x >= bx) & (x <= bx + bw * 0.56) & (np.abs(y - s / 2) <= half)
        bars[sel] = 1
    rgb = rgb * (1 - bars[..., None]) + np.array([255, 255, 255]) * bars[..., None]
    alpha = inside * 255
    img = np.dstack([rgb, alpha]).clip(0, 255).astype(np.uint8)
    return _png(img)


def _icns() -> bytes:
    entries = b""
    for code, size in ((b"ic07", 128), (b"ic08", 256), (b"ic09", 512)):
        data = _icon(size)
        entries += code + struct.pack(">I", len(data) + 8) + data
    return b"icns" + struct.pack(">I", len(entries) + 8) + entries


# --- empacotar -------------------------------------------------------------------

def _add(z: zipfile.ZipFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = zipfile.ZipInfo(name, time.localtime()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100000 | mode) << 16
    info.create_system = 3  # Unix: o macOS respeita as permissões (executável)
    z.writestr(info, data)


def _add_dir(z: zipfile.ZipFile, name: str) -> None:
    info = zipfile.ZipInfo(name.rstrip("/") + "/", time.localtime()[:6])
    info.external_attr = (0o40755 << 16) | 0x10
    info.create_system = 3
    z.writestr(info, b"")


def build(out_dir: Path) -> Path:
    out = out_dir / f"SFX Tagger (Mac) v{__version__}.zip"
    out.unlink(missing_ok=True)
    launcher = (ROOT / "ferramentas" / "mac_launcher.sh").read_bytes().replace(b"\r\n", b"\n")
    c = f"{APP}/Contents"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for d in (APP, c, f"{c}/MacOS", f"{c}/Resources", f"{c}/Resources/app"):
            _add_dir(z, d)
        _add(z, f"{c}/Info.plist", INFO_PLIST.encode())
        _add(z, f"{c}/PkgInfo", b"APPL????")
        _add(z, f"{c}/MacOS/{EXE}", launcher, 0o755)
        _add(z, f"{c}/Resources/AppIcon.icns", _icns())
        _add(z, f"{c}/Resources/a-preparar.html",
             (ROOT / "ferramentas" / "a-preparar.html").read_bytes())
        files = [ROOT / "run.py"] + sorted((ROOT / "sfxtagger").rglob("*"))
        for f in files:
            if f.is_dir() or "__pycache__" in f.parts or f.suffix == ".pyc":
                continue
            rel = f.relative_to(ROOT).as_posix()
            _add(z, f"{c}/Resources/app/{rel}", f.read_bytes())
        _add(z, "LEIA-ME.txt", LEIA_ME.replace("\n", "\r\n").encode("utf-8"))
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent
    path = build(dest)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        print(path, f"{path.stat().st_size // 1024} KB, {len(z.namelist())} entradas")
