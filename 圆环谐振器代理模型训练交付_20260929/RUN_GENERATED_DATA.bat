@echo off
setlocal
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo Running the reviewed auto-generated designs through COMSOL, mode identification, append, and retraining.
echo.
python --version >nul 2>nul
if errorlevel 1 goto use_py
python -u "%~dp0auto_find_and_run.py" --use-existing
goto finished
:use_py
py -3 -u "%~dp0auto_find_and_run.py" --use-existing
:finished
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Generated-design run failed with exit code %EXIT_CODE%.
if "%EXIT_CODE%"=="0" echo Generated-design run finished.
pause
exit /b %EXIT_CODE%
