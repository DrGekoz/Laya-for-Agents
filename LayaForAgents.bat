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
echo   To use it from Hermes, it needs one .env entry and one hook:
echo     TYPESAFE_BASE_URL=http://127.0.0.1:8000
echo   Do both at once, and make Hermes start this server whenever the
echo   gateway starts, with:
echo     .venv\Scripts\python.exe -m laya_for_agents.cli setup-hermes
echo   [already wired up?  install-gateway-hook refreshes the hook alone,
echo    uninstall-gateway-hook removes it, doctor reports whether it is there]
echo.
echo   Ctrl+C stops the server.
echo.

".venv\Scripts\python.exe" -m laya_for_agents.cli serve

echo.
echo [stopped] Laya for Agents is no longer listening.
pause
