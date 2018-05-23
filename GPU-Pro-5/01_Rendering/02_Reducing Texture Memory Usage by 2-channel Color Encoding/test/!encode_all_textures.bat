@echo off

setlocal ENABLEDELAYEDEXPANSION

for %%F in (*.bmp) do call _encode.bat %%F

echo Done.
echo.
echo.
pause
