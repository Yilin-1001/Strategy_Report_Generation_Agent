@echo off
REM Stop Milvus Script for Windows

echo ========================================
echo Stopping Milvus...
echo ========================================
echo.

docker-compose down

echo.
echo ========================================
echo Milvus has been stopped
echo ========================================
echo.
echo To restart Milvus, run: start_milvus.bat
echo.

pause
