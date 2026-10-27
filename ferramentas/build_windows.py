"""Gera 'SFX Tagger (Windows) vX.zip': um único ficheiro para abrir, instalação automática.

    python ferramentas/build_windows.py [pasta_de_saida]

Conteúdo do zip (pasta "SFX Tagger"):
    Abrir SFX Tagger.bat        <- a única coisa em que a pessoa clica
    Desinstalar SFX Tagger.bat
    LEIA-ME.txt
    app/                        <- código da app + ícone
"""
from __future__ import annotations

import struct
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from build_mac_app import _icon  # noqa: E402
from sfxtagger import __version__  # noqa: E402

LEIA_ME = """SFX TAGGER PARA WINDOWS
=======================

INSTALAR E ABRIR (nao precisas de instalar mais nada)
-----------------------------------------------------
1. Extrai este zip: botao direito > "Extrair tudo...".
2. Na pasta extraida, faz duplo clique em "Abrir SFX Tagger".

   Se aparecer "O Windows protegeu o seu PC":
   clica em "Mais informacoes" e depois em "Executar mesmo assim".
   (Aparece porque a app nao e comprada nem assinada. So na primeira vez.)

3. Na primeira vez abre uma janela que prepara tudo sozinha (1 a 2 minutos,
   precisa de internet). NAO a feches. No fim, o SFX Tagger abre no browser e
   termina a instalacao da IA (mais alguns minutos, ves o progresso no ecra).

4. E criado um atalho "SFX Tagger" no Ambiente de Trabalho. Das proximas
   vezes usa esse atalho. Ja podes apagar o zip e a pasta extraida.

COMO USAR
---------
- Arrasta os teus sons (WAV, AIFF, FLAC, OGG, MP3) para a janela do browser.
- "Treinar a IA" > "Comecar a treinar": ouve cada som; se a categoria estiver
  certa carrega Ctrl+Enter, se nao escolhe a certa (teclas 1-5) e Ctrl+Enter.
- Microfone ao lado de Description/Location/Notes: dita por voz (podes falar
  em portugues; a Description e traduzida para ingles).
- "Gravar nos ficheiros": escreve os metadados dentro dos WAV.
- "Treinar a IA" > "Exportar o que a IA aprendeu": cria um ficheiro .sfxk
  para enviares a quem coordena o projeto.
- Botao "Ajuda" na app: explicacao completa.
- Para fechar: botao "Sair" no canto superior direito (tambem fecha sozinha
  20 minutos depois de fechares o browser).

Os sons arrastados ficam numa copia em Documentos\\SFX Tagger.
Os teus ficheiros originais nunca sao alterados.

DESINSTALAR
-----------
Duplo clique em "Desinstalar SFX Tagger" (tambem fica uma copia em
%LOCALAPPDATA%\\SFXTagger). Exporta antes o que a IA aprendeu, se quiseres.

PROBLEMAS
---------
Se algo falhar, envia o ficheiro %LOCALAPPDATA%\\SFXTagger\\registo.log
a quem te deu a app.
"""


def _ico() -> bytes:
    """Ícone .ico com imagens PNG (formato aceite pelo Windows Vista e posteriores)."""
    images = [(s, _icon(s)) for s in (256, 64, 48, 32, 16)]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, data = b"", b""
    for size, png in images:
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset + len(data))
        data += png
    return header + entries + data


def _crlf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def _add(z: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, time.localtime()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data)


def build(out_dir: Path) -> Path:
    out = out_dir / f"SFX Tagger (Windows) v{__version__}.zip"
    out.unlink(missing_ok=True)
    win = ROOT / "ferramentas" / "windows"
    base = "SFX Tagger"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for bat in ("Abrir SFX Tagger.bat", "Desinstalar SFX Tagger.bat"):
            _add(z, f"{base}/{bat}", _crlf((win / bat).read_bytes()))
        _add(z, f"{base}/LEIA-ME.txt", _crlf(LEIA_ME.encode("utf-8")))
        _add(z, f"{base}/app/icone.ico", _ico())
        files = [ROOT / "run.py"] + sorted((ROOT / "sfxtagger").rglob("*"))
        for f in files:
            if f.is_dir() or "__pycache__" in f.parts or f.suffix == ".pyc":
                continue
            _add(z, f"{base}/app/{f.relative_to(ROOT).as_posix()}", f.read_bytes())
    return out


if __name__ == "__main__":
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent
    path = build(dest)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        print(path, f"{path.stat().st_size // 1024} KB, {len(z.namelist())} entradas")
