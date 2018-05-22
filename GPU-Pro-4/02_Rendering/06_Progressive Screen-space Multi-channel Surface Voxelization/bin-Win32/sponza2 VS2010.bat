@echo off
xcopy VS2010\* . /y/q
cls
cd ..\examples
..\bin-Win32\eazd_VS2010.exe sponza2_articulated.scene
