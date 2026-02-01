@echo off
REM Milvus Quick Start Script for Windows
REM This script requires Docker Desktop to be installed and running

echo ========================================
echo Milvus Quick Start Script
echo ========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed!
    echo.
    echo Please install Docker Desktop first:
    echo https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo [OK] Docker is installed
echo.

REM Check if Docker is running
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running!
    echo.
    echo Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [OK] Docker is running
echo.

REM Check if Milvus containers already exist
docker ps -a | findstr "milvus-standalone" >nul 2>&1
if %errorlevel% equ 0 (
    echo Milvus containers already exist.
    echo.
    echo Stopping existing containers...
    docker-compose down
    echo.
)

echo Starting Milvus...
echo.
docker-compose up -d

echo.
echo ========================================
echo Waiting for Milvus to be ready...
echo ========================================
echo.

REM Wait for containers to start
timeout /t 5 /nobreak >nul

REM Check if containers are running
docker ps | findstr "milvus-standalone" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Milvus container failed to start!
    echo.
    echo Check logs with: docker logs milvus-standalone
    pause
    exit /b 1
)

echo [OK] Milvus containers are running
echo.

echo ========================================
echo Container Status:
echo ========================================
docker ps --filter "name=milvus"
echo.

echo ========================================
echo Milvus is now running!
echo ========================================
echo.
echo Port 19530: Milvus gRPC port
echo Port 9091: Metrics port
echo.
echo Next steps:
echo   1. Run tests: pytest rag_project/tests/ -v
echo   2. Stop Milvus: docker-compose down
echo   3. View logs: docker logs milvus-standalone
echo.

pause
