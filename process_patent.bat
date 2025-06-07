@echo off
title Patent Processing Tool
echo =====================================
echo     PATENT PROCESSING TOOL
echo =====================================
echo.

if "%~1"=="" (
    echo Drag and drop a PDF file onto this batch file to process it.
    echo.
    echo Usage: process_patent.bat [pdf_file]
    echo.
    pause
    exit /b 1
)

echo Processing: %~nx1
echo.

python process_patent.py "%~1"

echo.
echo Press any key to exit...
pause > nul 