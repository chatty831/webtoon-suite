@echo off
setlocal enabledelayedexpansion

:: Configuration
set BACKEND_PORT=9374
set FRONTEND_PORT=8273
set TIMEOUT_HOURS=2
set /a TIMEOUT_SECONDS=%TIMEOUT_HOURS%*3600

:: Create unique session ID
set SESSION_ID=%RANDOM%%RANDOM%

echo ==========================================
echo         SERVICE MANAGER
echo ==========================================
echo.

:: Start Backend Service in background (same window)
echo Starting backend...
set CUDA_VISIBLE_DEVICES=0
start /B cmd /c uvicorn src.app:app --host 0.0.0.0 --port %BACKEND_PORT% 2^>^&1

:: Small delay to let backend initialize
timeout /t 2 /nobreak >nul

:: Start Frontend Service in background (same window)
echo Starting frontend...
start /B cmd /c streamlit run src/frontend/st_app.py --server.port %FRONTEND_PORT% 2^>^&1

:: Small delay to let frontend initialize
timeout /t 2 /nobreak >nul

:: Display service information
echo.
echo Backend running on port: %BACKEND_PORT%
echo Frontend running on port: %FRONTEND_PORT%
echo Services will automatically stop after %TIMEOUT_HOURS% hours
echo Press Ctrl+C to stop both services
echo ==========================================
echo.

:: Store start time for timeout calculation
for /f "tokens=1-3 delims=:." %%a in ("%TIME%") do (
    set /a START_SECONDS=%%a*3600+%%b*60+%%c
)

:MONITOR_LOOP
:: Check every 30 seconds
timeout /t 30 /nobreak >nul 2>&1
if errorlevel 1 goto :CLEANUP

:: Check if processes are still running
netstat -an | findstr :%BACKEND_PORT% >nul 2>&1
if errorlevel 1 (
    echo Backend service has stopped.
    goto :CLEANUP
)

netstat -an | findstr :%FRONTEND_PORT% >nul 2>&1
if errorlevel 1 (
    echo Frontend service has stopped.
    goto :CLEANUP
)

:: Calculate elapsed time
for /f "tokens=1-3 delims=:." %%a in ("%TIME%") do (
    set /a CURRENT_SECONDS=%%a*3600+%%b*60+%%c
)

:: Handle day rollover
if !CURRENT_SECONDS! lss %START_SECONDS% (
    set /a ELAPSED_SECONDS=86400-%START_SECONDS%+!CURRENT_SECONDS!
) else (
    set /a ELAPSED_SECONDS=!CURRENT_SECONDS!-%START_SECONDS%
)

:: Check timeout
if !ELAPSED_SECONDS! geq %TIMEOUT_SECONDS% (
    echo.
    echo Timeout reached ^(%TIMEOUT_HOURS% hours^). Shutting down services...
    goto :CLEANUP
)

goto :MONITOR_LOOP

:CLEANUP
echo.
echo Stopping services...

:: Kill processes by executable name
taskkill /F /IM uvicorn.exe >nul 2>&1
if not errorlevel 1 (
    echo Backend service stopped.
) else (
    echo Backend service was already stopped or not found.
)

taskkill /F /IM streamlit.exe >nul 2>&1
if not errorlevel 1 (
    echo Frontend service stopped.
) else (
    echo Frontend service was already stopped or not found.
)

:: Also try to kill Python processes running these specific scripts
for /f "tokens=2" %%a in ('wmic process where "CommandLine like '%%uvicorn%%src.app:app%%'" get ProcessId /value 2^>nul ^| find "="') do (
    taskkill /PID %%a