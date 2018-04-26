call "C:/Program Files (x86)/Microsoft Visual Studio 9.0/VC/bin/vcvars32.bat"
cd /d %~dp0
qmake -config release
nmake
robocopy /NP /NJH /NJS "%QTDIR%/bin" release QtCore4.dll QtGui4.dll QtOpenGL4.dll
robocopy /NP /NJH /NJS "%VCINSTALLDIR%/redist/x86/Microsoft.VC90.CRT" release/Microsoft.VC90.CRT
del /Q kyprianidis-demo-win.zip
cd release
"C:/Program Files/7-zip/7z.exe" a ../kyprianidis-demo-win.zip *.exe *.dll Microsoft.VC90.CRT
pause