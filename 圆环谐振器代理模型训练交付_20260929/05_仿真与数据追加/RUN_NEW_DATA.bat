@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo This command will run COMSOL for the example width-6 design.
echo Edit input\designs_width6_example.csv before using it for new designs.
echo.
python --version >nul 2>nul
if errorlevel 1 goto use_py
python -u "%~dp0python\simulate_and_score.py" --order 6 --input "%~dp0input\designs_width6_example.csv" --label user_batch --append
goto finished
:use_py
py -3 -u "%~dp0python\simulate_and_score.py" --order 6 --input "%~dp0input\designs_width6_example.csv" --label user_batch --append
:finished
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Simulation pipeline failed with exit code %EXIT_CODE%.
if "%EXIT_CODE%"=="0" echo Simulation, mode identification, and data append finished.
pause
exit /b %EXIT_CODE%
