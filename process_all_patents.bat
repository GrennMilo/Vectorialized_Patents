@echo off
title Patent Processing System
echo =====================================
echo     PATENT PROCESSING SYSTEM
echo =====================================
echo.
echo This will process all PDF files in the Patents folder,
echo extract images, apply OCR, and create a vector database.
echo.

set OPTIONS=

:MENU
echo Options:
echo [1] Process all patents
echo [2] Process with cleanup (delete previous results)
echo [3] Process limited number of patents
echo [4] Exit
echo.
set /p CHOICE="Enter your choice (1-4): "

if "%CHOICE%"=="1" (
    set OPTIONS=
    goto PROCESS
)
if "%CHOICE%"=="2" (
    set OPTIONS=--clean
    goto PROCESS
)
if "%CHOICE%"=="3" (
    set /p LIMIT="Enter the number of patents to process: "
    set OPTIONS=--limit %LIMIT%
    goto PROCESS
)
if "%CHOICE%"=="4" (
    echo Exiting...
    exit /b 0
)

echo Invalid choice, please try again.
echo.
goto MENU

:PROCESS
echo.
echo Starting patent processing...
echo.

python process_all_patents.py %OPTIONS%

echo.
echo Process completed. Press any key to exit...
pause > nul 