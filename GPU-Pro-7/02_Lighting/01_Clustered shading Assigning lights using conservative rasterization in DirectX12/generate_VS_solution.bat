@echo off
premake\premake5.exe vs2015
echo.
xcopy external\_BIN_COPY_\*.* bin\ /Y
echo.
echo.
echo Visual Studio 2015 Solution has been created in build/.
echo Relevant binaries have been copied to bin/.
echo.
pause
