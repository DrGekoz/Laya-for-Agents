@echo off
setlocal
cd /d "%~dp0"
title Laya for Agents

if not exist ".venv\Scripts\python.exe" (
  echo [FAIL] No .venv in this folder.
  echo        Create it with:
  echo          uv venv --python 3.12
  echo          uv pip install --python .venv\Scripts\python.exe -e ".[serve]"
  pause
  exit /b 1
)

echo ================================================================
echo   Laya for Agents
echo   Local typed decisions on the TypeSafe Jev wire protocol
echo ================================================================
echo.
echo   endpoint : http://127.0.0.1:8000/v1/systemone
echo   health   : http://127.0.0.1:8000/health
echo.
echo   To use it from Hermes, set in the Hermes .env:
echo     TYPESAFE_BASE_URL=http://127.0.0.1:8000
echo.
echo   Ctrl+C stops the server.
echo.

".venv\Scripts\python.exe" -m laya_for_agents.cli serve

echo.
echo [stopped] Laya for Agents is no longer listening.
pause
