@echo off
if exist %1\glew32.dll goto glut
copy .\External\glew\bin\glew32.dll /B /Y %1
:glut
if exist %1\glut32.dll goto cg
copy .\External\glut\bin\glut32.dll /B /Y %1
:cg
if exist %1\cg.dll goto cggl
copy .\External\cg\bin\cg.dll /B /Y %1
:cggl
if exist %1\cggl.dll goto end
copy .\External\cg\bin\cggl.dll /B /Y %1
:end