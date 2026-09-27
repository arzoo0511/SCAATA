@echo off
rem HANSEI's daily run: fills yesterday's decisions, learns, reads news, decides.
rem Safe to run any number of times -- a session already processed is skipped.
cd /d "%~dp0"
if not exist logs mkdir logs
"C:\Users\ICG\AppData\Local\Programs\Python\Python312\python.exe" -m scaata.agent.daily >> logs\hansei_daily.log 2>&1
