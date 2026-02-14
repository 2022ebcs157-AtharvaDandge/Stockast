# Stock Trend Prediction & Sentiment Analysis - Quick Launch Script
# Run this file with: powershell -ExecutionPolicy Bypass -File run_app.ps1

Write-Host "========================================"
Write-Host "Stock Trend Prediction & Sentiment Analysis"
Write-Host "========================================"
Write-Host ""

Write-Host "Installing/Updating dependencies..."
Write-Host ""

# Use Python launcher (py) which is more reliable
py -3 -m pip install -q --upgrade streamlit pandas numpy matplotlib xgboost scikit-learn yfinance nltk beautifulsoup4 requests google-news textblob

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Write-Host "Make sure Python 3.10+ is installed" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Starting the application..."
Write-Host ""
Write-Host "========================================"
Write-Host "You can now view your app in your browser at:"
Write-Host "http://localhost:8501"
Write-Host "========================================"
Write-Host ""
Write-Host "Press Ctrl+C to stop the server"
Write-Host ""

# Navigate to app directory and run
Set-Location "c:\Users\dandg\OneDrive\Desktop\BITS Project\Stock-prediction-using-sentiment-analysis-main"
py -3 -m streamlit run app.py

Write-Host ""
Write-Host "Application stopped."
Read-Host "Press Enter to exit"
