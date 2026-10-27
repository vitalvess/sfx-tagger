@echo off
setlocal EnableExtensions
title SFX Tagger
rem ------------------------------------------------------------------------
rem  SFX Tagger - arrancador para Windows.
rem  Na primeira vez instala tudo sozinho (Python, bibliotecas e IA) e cria
rem  um atalho no Ambiente de Trabalho. Tudo fica em %LOCALAPPDATA%\SFXTagger.
rem ------------------------------------------------------------------------

if not defined SFXTAGGER_HOME set "SFXTAGGER_HOME=%LOCALAPPDATA%\SFXTagger"
if not defined SFXTAGGER_PORT set "SFXTAGGER_PORT=8765"
set "SUP=%SFXTAGGER_HOME%"
set "SELF=%~f0"
set "SRC=%~dp0"
set "UVDIR=%SUP%\uv"
set "UV=%SUP%\uv\uv.exe"
set "VENV=%SUP%\venv-app"
set "STAMP=%SUP%\venv-app\.base-instalada"
set "LOG=%SUP%\registo.log"
set "URL=http://127.0.0.1:%SFXTAGGER_PORT%"
set "UV_PYTHON_INSTALL_DIR=%SUP%\python"
set "UV_CACHE_DIR=%SUP%\uv-cache"
set "HF_HOME=%SUP%\modelo"
if not exist "%SUP%" mkdir "%SUP%"

rem Ja esta aberta? Basta mostrar o browser.
powershell -NoProfile -Command "try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%URL%/api/status').Content -match 'version') { exit 0 } } catch { }; exit 1" >nul 2>&1
if not errorlevel 1 (
  start "" "%URL%"
  exit /b 0
)

rem Copia a app para a pasta do utilizador (o zip pode ser apagado depois).
if /i not "%SRC%"=="%SUP%\" (
  robocopy "%SRC%app" "%SUP%\app" /MIR /NFL /NDL /NJH /NJS /NP >nul
  copy /y "%SELF%" "%SUP%\Abrir SFX Tagger.bat" >nul
  if exist "%SRC%Desinstalar SFX Tagger.bat" copy /y "%SRC%Desinstalar SFX Tagger.bat" "%SUP%\" >nul
)

if exist "%STAMP%" goto :abrir

echo.
echo   ================================================
echo     SFX Tagger - a preparar, so na primeira vez
echo   ================================================
echo.
echo   Isto demora 1 a 2 minutos. NAO feches esta janela.
echo   No fim, o SFX Tagger abre sozinho no browser.
echo.
echo ==== %date% %time% instalacao ==== >>"%LOG%"

if exist "%UV%" goto :python
echo   [1/3] A descarregar o instalador...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol='Tls12'; $z=Join-Path $env:TEMP 'sfxtagger-uv.zip'; Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $z; Expand-Archive -Force $z '%UVDIR%'; Remove-Item $z; $u=Get-ChildItem -Recurse -Filter uv.exe '%UVDIR%' | Select-Object -First 1; if ($u.FullName -ne '%UV%') { Copy-Item $u.FullName '%UV%' }" >>"%LOG%" 2>&1
if not exist "%UV%" goto :erro

:python
echo   [2/3] A instalar o Python...
if exist "%VENV%" rmdir /s /q "%VENV%"
"%UV%" venv --python 3.12 "%VENV%" >>"%LOG%" 2>&1 || goto :erro
echo   [3/3] A instalar as bibliotecas...
"%UV%" pip install --python "%VENV%\Scripts\python.exe" fastapi uvicorn numpy scipy soundfile pyloudnorm >>"%LOG%" 2>&1 || goto :erro
echo ok>"%STAMP%"
if not defined SFXTAGGER_NO_SHORTCUT call :atalho
echo.
echo   Pronto! A abrir o SFX Tagger no browser...
echo   Da proxima vez usa o atalho "SFX Tagger" no Ambiente de Trabalho.
ping -n 5 127.0.0.1 >nul

:abrir
set "SFXTAGGER_APP=1"
set "SFXTAGGER_UV=%UV%"
set "PYTHONDONTWRITEBYTECODE=1"
set "EXTRA="
if defined SFXTAGGER_NO_BROWSER set "EXTRA=--no-browser"
start "" "%VENV%\Scripts\pythonw.exe" "%SUP%\app\run.py" --port %SFXTAGGER_PORT% %EXTRA%
exit /b 0

:atalho
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\SFX Tagger.lnk'); $s.TargetPath='%SUP%\Abrir SFX Tagger.bat'; $s.WorkingDirectory='%SUP%'; $s.IconLocation='%SUP%\app\icone.ico'; $s.WindowStyle=7; $s.Description='SFX Tagger - metadados de efeitos sonoros'; $s.Save()" >>"%LOG%" 2>&1
exit /b 0

:erro
echo.
echo   Nao foi possivel preparar o SFX Tagger.
echo   Confirma que estas ligado a internet e volta a abrir este ficheiro.
echo   Se continuar a falhar, envia este ficheiro a quem te deu a app:
echo   %LOG%
echo.
pause
exit /b 1
