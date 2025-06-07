@echo off
echo Patent Batch Processing Tool
echo --------------------------
echo This tool will process all patent PDF files in the Patents directory.
echo.

REM Check if Python is installed
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in the PATH.
    echo Please install Python 3.7 or higher.
    pause
    exit /b 1
)

echo Starting batch processing of all patents...
echo.

REM Run the patent processing script for all patents
python process_patents.py --input Patents --output Results

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Patent batch processing failed.
    pause
    exit /b 1
)

echo.
echo Patent processing completed successfully!
echo.
echo You can now explore the extracted components in the Results folder.
echo.
pause 