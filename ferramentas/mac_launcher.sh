#!/bin/bash
# SFX Tagger — arranque da app de Mac.
# Na primeira vez instala tudo sozinho (Python, bibliotecas e IA) — sem Terminal.
# Tudo fica em ~/Library/Application Support/SFXTagger (apagar essa pasta desinstala).

RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
SUP="$HOME/Library/Application Support/SFXTagger"
LOG="$SUP/registo.log"
UVDIR="$SUP/uv"
UV="$UVDIR/uv"
VENV="$SUP/venv"
VPY="$VENV/bin/python"
STAMP="$VENV/.base-instalada"
URL="http://127.0.0.1:8765"
LOCK="$SUP/a-correr.pid"
PREP="$RES/a-preparar.html"

mkdir -p "$SUP"
# mantém o registo pequeno
[ -f "$LOG" ] && [ "$(wc -c <"$LOG")" -gt 2000000 ] && mv -f "$LOG" "$LOG.antigo"
exec >>"$LOG" 2>&1
echo "=== $(date) · arranque ==="

export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export UV_PYTHON_INSTALL_DIR="$SUP/python"
export UV_CACHE_DIR="$SUP/uv-cache"
export HF_HOME="$SUP/modelo"

aviso() {
  osascript -e "display notification \"$1\" with title \"SFX Tagger\"" >/dev/null 2>&1
}

erro() {
  echo "ERRO: $1"
  local r
  r=$(osascript -e "display dialog \"$1\n\nSe o problema continuar, envia o ficheiro de registo a quem te deu a app.\" with title \"SFX Tagger\" buttons {\"Abrir registo\", \"OK\"} default button \"OK\" with icon caution" -e "button returned of result" 2>/dev/null)
  [ "$r" = "Abrir registo" ] && open -e "$LOG"
  exit 1
}

# Já está aberta? Basta mostrar o browser.
if curl -s --max-time 2 "$URL/api/status" | grep -q '"version"'; then
  open "$URL"
  exit 0
fi

# Ainda a preparar-se (a pessoa clicou outra vez)? Mostra a página de progresso, não começa outra.
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  open "$PREP"
  exit 0
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

if [ ! -x "$VPY" ] || [ ! -f "$STAMP" ]; then
  # dá sinal de vida logo: uma página local que passa sozinha para a app quando estiver pronta
  open "$PREP"
  aviso "Primeira utilização: a preparar o SFX Tagger. Acompanha no browser."

  if [ ! -x "$UV" ]; then
    echo "A instalar o uv…"
    curl -LsSf https://astral.sh/uv/install.sh \
      | env UV_UNMANAGED_INSTALL="$UVDIR" INSTALLER_NO_MODIFY_PATH=1 sh \
      || erro "Não foi possível descarregar o instalador. Confirma que estás ligado à internet e abre a app outra vez."
  fi

  # Mac Intel: o PyTorch para Intel só existe até ao 2.2.2, que precisa de numpy 1.x
  if [ "$(uname -m)" = "x86_64" ]; then
    BASE=(fastapi uvicorn "numpy<2" scipy soundfile pyloudnorm)
  else
    BASE=(fastapi uvicorn numpy scipy soundfile pyloudnorm)
  fi

  echo "A criar o ambiente Python…"
  rm -rf "$VENV"
  "$UV" venv --python 3.12 "$VENV" \
    || erro "Não foi possível instalar o Python. Confirma a ligação à internet e abre a app outra vez."
  echo "A instalar as bibliotecas base…"
  "$UV" pip install --python "$VPY" "${BASE[@]}" \
    || erro "Não foi possível instalar as bibliotecas. Confirma a ligação à internet e abre a app outra vez."
  touch "$STAMP"
fi

export SFXTAGGER_APP=1
export SFXTAGGER_UV="$UV"
export PYTHONDONTWRITEBYTECODE=1
echo "A abrir a app…"
"$VPY" "$RES/app/run.py" || erro "A app fechou com um erro inesperado."
