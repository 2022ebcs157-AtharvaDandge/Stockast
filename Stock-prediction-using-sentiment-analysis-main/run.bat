@echo off
REM This script runs the Stock Prediction Streamlit application

REM Change to the project directory
cd /d "%~dp0"

REM Activate virtual environment
call myenv\Scripts\activate.bat

REM Run the Streamlit app
python -m streamlit run app.py

REM Keep the window open if there's an error
pause
