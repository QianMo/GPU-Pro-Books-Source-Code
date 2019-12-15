@echo off

cl GenerateData.cpp d3dx9.lib

set TestParams=Test.cpp /DEXTERNAL_SSE_CONTROL /EHsc
cl %TestParams% /o Test_SSE1.exe
cl %TestParams% /o Test_SSE2.exe /DENABLE_SSE2
cl %TestParams% /o Test_SSE4.exe /DENABLE_SSE4

echo.
echo Generating data...
GenerateData.exe
echo.

echo Testing...
Test_SSE1.exe
if ERRORLEVEL 1 goto Error
echo SSE1 done.

Test_SSE2.exe
if ERRORLEVEL 1 goto Error
echo SSE2 done.

Test_SSE4.exe
if ERRORLEVEL 1 goto Error
echo SSE4 done.

goto End

:Error
echo EPIC FAIL!
pause

:End
call Clean.bat
