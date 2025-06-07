@echo off
echo Patent Processing Tool
echo -----------------------
echo This tool will process a patent PDF file and extract components.
echo.

REM Check if Python is installed
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in the PATH.
    echo Please install Python 3.7 or higher.
    pause
    exit /b 1
)

REM Check if a file was provided
if "%~1"=="" (
    echo Usage: process_patent.bat [path to PDF file]
    echo.
    echo Please specify a PDF file to process.
    pause
    exit /b 1
)

REM Check if the file exists
if not exist "%~1" (
    echo ERROR: File not found: %~1
    pause
    exit /b 1
)

REM Check if the file is a PDF
if /I not "%~x1"==".pdf" (
    echo ERROR: The file must be a PDF: %~1
    pause
    exit /b 1
)

echo Processing: %~1
echo.

REM Run the patent processing script
python process_patent.py "%~1" --output Results

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Patent processing failed.
    pause
    exit /b 1
)

echo.
echo Patent processing completed successfully!
echo.
echo You can now explore the extracted components in the Results folder.
echo.
pause 