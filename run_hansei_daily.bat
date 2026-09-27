@echo off
rem HANSEI's daily run: fills yesterday's decisions, learns, reads news, decides.
rem Safe to run any number of times -- a session already processed is skipped.
cd /d "%~dp0"
if not exist logs mkdir logs
"C:\Users\ICG\AppData\Local\Programs\Python\Python312\python.exe" -m scaata.agent.daily >> logs\hansei_daily.log 2>&1

rem Publish the new state so the public dashboard (Streamlit Cloud) shows it.
rem Commits only these three files, and only if they changed. The push always
rem runs, so a push that failed last time goes out on the next run.
rem Never prompts for credentials, so a scheduled run can't hang here.
set GIT_TERMINAL_PROMPT=0
set GCM_INTERACTIVE=never
set STATE=paper_book_india.json forward_test\hansei_journal.json forward_test\hansei_memory.json
git add -- %STATE% >> logs\hansei_daily.log 2>&1
git diff --cached --quiet -- %STATE% || git commit -q -m "HANSEI daily update" -- %STATE% >> logs\hansei_daily.log 2>&1
git push -q origin HEAD:main >> logs\hansei_daily.log 2>&1 || echo [%date% %time%] push failed, will retry next run >> logs\hansei_daily.log
