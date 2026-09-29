@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo ============================================================
echo Starting final MLP training
echo Package: %~dp0
echo Output:  see README.md, section Training output
echo ============================================================
echo.
python --version >nul 2>nul
if errorlevel 1 goto use_py
echo Python command: python
python -u "%~dp0train.py" %*
goto finished
:use_py
py -3 --version >nul 2>nul
if errorlevel 1 goto no_python
echo Python command: py -3
py -3 -u "%~dp0train.py" %*
goto finished
:no_python
echo ERROR: Python 3 was not found.
echo Install Python 3.11 or 3.12, then run this file again.
set "EXIT_CODE=9009"
goto report
:finished
set "EXIT_CODE=%ERRORLEVEL%"
:report
echo.
if not "%EXIT_CODE%"=="0" echo Training failed with exit code %EXIT_CODE%.
if "%EXIT_CODE%"=="0" echo Training finished. See README.md for the output location.
pause
exit /b %EXIT_CODE%
