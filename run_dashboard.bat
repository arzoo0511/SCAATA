@echo off
cd /d "%~dp0"
rem The SCAATA-Dashboard Windows Task Scheduler job already starts this
rem dashboard headlessly at every logon, so it's normally already running.
rem Without this check, double-clicking this file while that instance is up
rem made Streamlit silently bind a SECOND server on the next free port and
rem open a second browser tab -- this is why the dashboard was "coming
rem twice". Now: if something's already listening on 8765, just open a
rem browser to it instead of starting a redundant second server.
netstat -ano | findstr ":8765" | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo SCAATA dashboard is already running -- opening it in your browser.
    start http://localhost:8765
) else (
    echo Starting SCAATA dashboard at http://localhost:8765 ...
    start http://localhost:8765
    streamlit run scaata/dashboard/app.py --server.port 8765 --server.headless true
)
