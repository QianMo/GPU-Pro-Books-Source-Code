@echo off
xcopy VS2008\* . /y/q
cls
cd ..\examples
..\bin-Win32\eazd_VS2008.exe sponza2_articulated.scene
