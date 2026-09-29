@echo off
setlocal
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo One closed-loop round: find designs, run COMSOL, append valid data, retrain MLP.
echo This can take a long time because COMSOL and model training will run.
echo.
python --version >nul 2>nul
if errorlevel 1 goto use_py
python -u "%~dp0auto_find_and_run.py"
goto finished
:use_py
py -3 -u "%~dp0auto_find_and_run.py"
:finished
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Closed-loop run failed with exit code %EXIT_CODE%.
if "%EXIT_CODE%"=="0" echo Closed-loop run finished.
pause
exit /b %EXIT_CODE%
