@echo off
REM BGE-M3 Embedding Server Startup Script
REM This script starts the BGE-M3 embedding API server

echo ============================================================
echo BGE-M3 Embedding Server
echo ============================================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
    echo Using system Python
)

REM Set environment variables
set MODEL_NAME=BAAI/bge-m3
set DEVICE=cuda
set CACHE_DIR=./data/models
set PORT=8080

echo Starting BGE-M3 server on port %PORT%...
echo Model: %MODEL_NAME%
echo Device: %DEVICE%
echo.
echo Server will be available at: http://localhost:%PORT%
echo Press Ctrl+C to stop the server
echo.

REM Start the server
python bge_m3_server.py

pause
