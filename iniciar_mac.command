#!/bin/bash
cd "$(dirname "$0")" || exit 1
VPY="$HOME/Library/Application Support/SFXTagger/venv/bin/python"
if [ ! -x "$VPY" ]; then
  echo "  A app ainda não está instalada. Corre primeiro \"instalar_mac.command\"."
  read -r -p "  Carrega Enter para fechar."
  exit 1
fi
echo "  A abrir o SFX Tagger… (fecha esta janela para terminar)"
"$VPY" run.py || read -r -p "  Ocorreu um erro. Carrega Enter para fechar."
