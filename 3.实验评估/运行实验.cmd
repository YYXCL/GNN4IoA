@echo off
setlocal
chcp 65001 >nul
if defined TC_MDSM_PYTHON (
    set "experiment_python=%TC_MDSM_PYTHON%"
) else (
    set "experiment_python=%~dp0..\.venv-tc\Scripts\python.exe"
)
if not exist "%experiment_python%" (
    echo Python environment was not found. Set TC_MDSM_PYTHON to your project Python executable.
    pause
    exit /b 1
)
"%experiment_python%" -B -u "%~dp0tc_mdsm_paper_runner.py" %*
set "experiment_exit=%errorlevel%"
if not "%experiment_exit%"=="0" echo Experiment stopped. Read the error above; saved results are retained.
pause
exit /b %experiment_exit%
