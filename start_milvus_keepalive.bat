@echo off
echo ============================================================
echo Milvus Keep-Alive Service for LangFlow
echo ============================================================
echo.
echo This script keeps the Milvus collection loaded in memory.
echo Keep this window OPEN while using LangFlow.
echo.
echo Press Ctrl+C to stop the service.
echo ============================================================
echo.

python preload_milvus_for_langflow.py

pause
