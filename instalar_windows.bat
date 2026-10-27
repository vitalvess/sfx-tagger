@echo off
setlocal
chcp 65001 >nul
title SFX Tagger - Instalacao
cd /d "%~dp0"

echo.
echo  ============================================
echo    SFX Tagger - instalacao (so e preciso 1x)
echo  ============================================
echo.

set "PYEXE="
for %%V in (3.12 3.11 3.13 3.10 3.14) do (
  if not defined PYEXE (
    py -%%V -c "import sys" >nul 2>&1 && set "PYEXE=py -%%V"
  )
)
if not defined PYEXE (
  python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,14) else 1)" >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
  echo  [ERRO] Nao encontrei o Python 3.10 a 3.14.
  echo  Instala o Python 3.12 em https://www.python.org/downloads/
  echo  e marca a opcao "Add python.exe to PATH" durante a instalacao.
  echo.
  pause
  exit /b 1
)
echo  A usar: %PYEXE%

set "VENV=%LOCALAPPDATA%\SFXTagger\venv"
if not exist "%VENV%\Scripts\python.exe" (
  echo  A criar ambiente Python em %VENV% ...
  %PYEXE% -m venv "%VENV%" || goto :erro
)
set "VPY=%VENV%\Scripts\python.exe"

echo.
echo  [1/3] A instalar o PyTorch (cerca de 200 MB)...
"%VPY%" -m pip install --upgrade pip -q || goto :erro
"%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q || goto :erro

echo  [2/3] A instalar as restantes bibliotecas...
"%VPY%" -m pip install -r requirements.txt -q || goto :erro

echo  [3/3] A descarregar o modelo de IA (cerca de 1,4 GB, so na primeira vez)...
"%VPY%" -c "import sys; sys.path.insert(0, '.'); from sfxtagger.ai import get_tagger; get_tagger().ensure_loaded(); print('  Modelo pronto.')" || goto :erro

echo.
echo  Instalacao concluida! Para abrir a app usa o ficheiro "iniciar_windows.bat".
echo.
pause
exit /b 0

:erro
echo.
echo  [ERRO] A instalacao falhou. Ve as mensagens acima (falta de internet ou de espaco em disco?).
pause
exit /b 1
