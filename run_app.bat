@echo off
REM Stock Trend Prediction & Sentiment Analysis - Quick Launch Script
REM This batch file runs the Streamlit application

echo ========================================
echo Stock Trend Prediction & Sentiment Analysis
echo ========================================
echo.
echo Installing/Updating dependencies...
echo.

py -3 -m pip install -q --upgrade streamlit pandas numpy matplotlib xgboost scikit-learn yfinance nltk beautifulsoup4 requests google-news textblob

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Make sure Python 3.10+ is installed
    pause
    exit /b 1
)

echo.
echo Starting the application...
echo.
echo ========================================
echo You can now view your app in your browser at:
echo http://localhost:8501
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "c:\Users\dandg\OneDrive\Desktop\BITS Project\Stock-prediction-using-sentiment-analysis-main"
py -3 -m streamlit run app.py

pause