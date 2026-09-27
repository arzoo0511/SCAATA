@echo off
cd /d "%~dp0"
rem Same double-launch guard as run_dashboard.bat: if the SCAATA Signals
rem API is already listening on 8766, don't start a second instance.
netstat -ano | findstr ":8766" | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo SCAATA Signals API is already running at http://localhost:8766
) else (
    echo Starting SCAATA Signals API at http://localhost:8766 ...
    echo   Interactive docs: http://localhost:8766/docs
    uvicorn scaata.product.api:app --host 0.0.0.0 --port 8766
)
