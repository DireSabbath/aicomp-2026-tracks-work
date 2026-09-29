@echo off
setlocal
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo Generating a new mixed design batch from the current training data and MLP...
echo.
python --version >nul 2>nul
if errorlevel 1 goto use_py
python -u "%~dp0generate_new_designs.py"
goto finished
:use_py
py -3 -u "%~dp0generate_new_designs.py"
:finished
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Candidate generation failed with exit code %EXIT_CODE%.
if "%EXIT_CODE%"=="0" echo Candidate generation finished. Review the auto CSV before COMSOL.
pause
exit /b %EXIT_CODE%
