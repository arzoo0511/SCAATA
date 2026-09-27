@echo off
cd /d "%~dp0"
streamlit run scaata\dashboard\hansei_app.py --server.port 8502
