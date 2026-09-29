@echo off
setlocal
chcp 65001 >nul
where python >nul 2>nul
if errorlevel 1 goto use_py
python "%~dp0predict.py" %*
goto finished
:use_py
py -3 "%~dp0predict.py" %*
:finished
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Prediction failed with exit code %EXIT_CODE%.
if "%EXIT_CODE%"=="0" echo Prediction finished. See README.md for the output location.
pause
exit /b %EXIT_CODE%
