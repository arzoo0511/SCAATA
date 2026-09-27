@echo off
rem HANSEI now trades on GitHub Actions (.github/workflows/hansei-daily.yml),
rem not on this laptop. This only pulls its latest state so the local
rem dashboard (run_hansei_dashboard.bat) matches hansei.streamlit.app.
rem Fast-forward only: it never merges or overwrites local work.
rem
rem To run the agent locally again (e.g. if Actions is down), run
rem   python -m scaata.agent.daily
rem by hand -- but not while the workflow is also running, or the two copies
rem of the paper book will diverge.
cd /d "%~dp0"
if not exist logs mkdir logs
set GIT_TERMINAL_PROMPT=0
set GCM_INTERACTIVE=never
echo [%date% %time%] pulling HANSEI state from GitHub >> logs\hansei_daily.log
git pull -q --ff-only origin main >> logs\hansei_daily.log 2>&1 || echo [%date% %time%] pull failed (local commits or changes in the way) >> logs\hansei_daily.log
