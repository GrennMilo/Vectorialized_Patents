@echo off
title Patent Search Web App
echo =====================================
echo     PATENT SEARCH WEB APP
echo =====================================
echo.
echo This will start the patent search web application.
echo.

rem Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python first.
    pause
    exit /b 1
)

rem Check if the database files exist
if not exist "Results\embeddings.npy" (
    echo Vector database not found!
    echo.
    echo Please run process_all_patents.py first to create the database.
    echo.
    choice /C YN /M "Do you want to run process_all_patents.py now?"
    if errorlevel 2 goto END
    if errorlevel 1 (
        echo.
        echo Running process_all_patents.py...
        python process_all_patents.py
    )
)

echo.
echo Starting the web application...
echo The application will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the application when done.
echo.

python app.py

:END
echo.
echo Application stopped. Press any key to exit...
pause > nul 