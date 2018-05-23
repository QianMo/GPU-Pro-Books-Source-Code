@echo off
if "%1" == "" goto ERROR

set FILE=%1
echo Processing %FILE%
..\Release\tex2.exe -encode %FILE% %FILE:.bmp=_encoded.bmp% %FILE:.bmp=_encoded.txt%


goto END


:ERROR
echo No file name given.

:END
