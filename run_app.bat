@echo off
cd /d "D:\SkinCancerApp"   :: Change this path if your folder is different
call venv\Scripts\activate.bat
streamlit run app.py
pause
