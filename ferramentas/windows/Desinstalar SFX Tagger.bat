@echo off
setlocal EnableExtensions
title Desinstalar SFX Tagger
if not defined SFXTAGGER_HOME set "SFXTAGGER_HOME=%LOCALAPPDATA%\SFXTagger"
if not defined SFXTAGGER_PORT set "SFXTAGGER_PORT=8765"
set "TMPCOPY=%TEMP%\sfxtagger-desinstalar.bat"

rem Se este ficheiro esta dentro da pasta que vai ser apagada, corre a partir de
rem uma copia temporaria (o Windows nao gosta que um .bat apague a sua propria pasta).
if /i "%~dp0"=="%SFXTAGGER_HOME%\" (
  copy /y "%~f0" "%TMPCOPY%" >nul
  "%TMPCOPY%"
)

echo.
echo   Isto apaga o SFX Tagger deste computador: a app, a IA e o que ela
echo   aprendeu com os teus sons. Se quiseres guardar o que a IA aprendeu,
echo   cancela e usa primeiro "Treinar a IA" - "Exportar" dentro da app.
echo.
echo   Os teus ficheiros de som e os metadados ja gravados neles NAO sao apagados.
echo.
choice /c SN /m "  Desinstalar agora"
if errorlevel 2 goto :fim

rem fecha a app se estiver aberta
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Method Post -TimeoutSec 3 'http://127.0.0.1:%SFXTAGGER_PORT%/api/quit' | Out-Null } catch { }" >nul 2>&1
ping -n 3 127.0.0.1 >nul
powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue ([Environment]::GetFolderPath('Desktop') + '\SFX Tagger.lnk')" >nul 2>&1
cd /d "%TEMP%"
rmdir /s /q "%SFXTAGGER_HOME%" 2>nul
echo.
echo   SFX Tagger desinstalado.
ping -n 5 127.0.0.1 >nul

:fim
rem a copia temporaria apaga-se a si propria
if /i "%~f0"=="%TMPCOPY%" (goto) 2>nul & del "%~f0"
exit /b 0
