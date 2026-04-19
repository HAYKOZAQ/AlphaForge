@echo off
echo ============================================================
echo AI-Augmented Market Development Research Pipeline
echo ============================================================
echo.
echo [%date% %time%] Starting Prefect Orchestrated Pipeline...
echo.

set PYTHONPATH=%~dp0
cd /d %~dp0

echo Running src\prefect_flow.py
python src\prefect_flow.py
if %errorlevel% neq 0 goto error

echo.
echo [%date% %time%] SUCCESS: Pipeline completed successfully.
exit /b 0

:error
echo.
echo [%date% %time%] ERROR: Pipeline halted due to script failure.
pause
exit /b %errorlevel%
