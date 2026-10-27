#!/bin/bash
# SFX Tagger — instalação para macOS (só é preciso uma vez)
cd "$(dirname "$0")" || exit 1

falhou() {
  echo
  echo "  [ERRO] A instalação falhou (ver mensagens acima)."
  echo "  Causas habituais: sem internet, pouco espaço em disco, ou versão do Python."
  read -r -p "  Carrega Enter para fechar."
  exit 1
}

echo
echo "  ============================================"
echo "    SFX Tagger - instalação (só é preciso 1x)"
echo "  ============================================"
echo

# Mac com chip Intel (ou Terminal a correr em Rosetta): o PyTorch mais recente para
# x86_64 é o 2.2.2, que só existe até ao Python 3.12
if [ "$(uname -m)" = "x86_64" ]; then
  INTEL=1
  CANDIDATOS="python3.12 python3.11 python3.10 python3"
  MAXPY="(3,12)"
else
  INTEL=0
  CANDIDATOS="python3.12 python3.13 python3.11 python3.14 python3.10 python3"
  MAXPY="(3,14)"
fi

PYEXE=""
for v in $CANDIDATOS; do
  if command -v "$v" >/dev/null 2>&1 && \
     "$v" -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=$MAXPY else 1)" 2>/dev/null; then
    PYEXE="$v"; break
  fi
done
if [ -z "$PYEXE" ]; then
  if [ "$INTEL" = 1 ]; then
    echo "  [ERRO] Este Mac tem chip Intel e precisa do Python 3.10, 3.11 ou 3.12."
  else
    echo "  [ERRO] Não encontrei o Python 3.10 a 3.14."
  fi
  echo "  Instala o Python 3.12 em https://www.python.org/downloads/ e volta a correr este ficheiro."
  read -r -p "  Carrega Enter para fechar."
  exit 1
fi
echo "  A usar: $PYEXE ($($PYEXE --version 2>&1))$([ "$INTEL" = 1 ] && echo ' · Mac Intel')"

VENV="$HOME/Library/Application Support/SFXTagger/venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "  A criar ambiente Python…"
  mkdir -p "$(dirname "$VENV")"
  "$PYEXE" -m venv "$VENV" || falhou
fi
VPY="$VENV/bin/python"

echo
echo "  [1/3] A instalar o PyTorch (pode demorar alguns minutos)…"
"$VPY" -m pip install --upgrade pip -q || falhou
if [ "$INTEL" = 1 ]; then
  "$VPY" -m pip install -q "torch==2.2.2" "numpy<2" || falhou
else
  "$VPY" -m pip install -q torch || falhou
fi

echo "  [2/3] A instalar as restantes bibliotecas…"
if [ "$INTEL" = 1 ]; then
  # versões compatíveis com o PyTorch 2.2
  "$VPY" -m pip install -q "numpy<2" "transformers>=4.40,<4.47" -r requirements.txt || falhou
else
  "$VPY" -m pip install -q -r requirements.txt || falhou
fi

echo "  [3/3] A descarregar o modelo de IA (cerca de 1,4 GB, só na primeira vez)…"
"$VPY" -c "import sys; sys.path.insert(0, '.'); from sfxtagger.ai import get_tagger; get_tagger().ensure_loaded(); print('  Modelo pronto.')" || falhou

# permite abrir o iniciar_mac.command com duplo clique (ficheiros vindos de um zip
# ficam "em quarentena" pelo macOS)
chmod +x iniciar_mac.command instalar_mac.command 2>/dev/null
xattr -dr com.apple.quarantine . 2>/dev/null

echo
echo "  Instalação concluída!"
echo "  Para abrir a app: duplo clique em \"iniciar_mac.command\"."
read -r -p "  Carrega Enter para fechar."
