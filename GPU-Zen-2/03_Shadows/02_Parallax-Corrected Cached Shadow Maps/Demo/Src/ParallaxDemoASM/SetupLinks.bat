@echo off
if exist Shaders\_Shaders goto end
mklink /J Shaders\_Shaders ..\Core\Rendering\_Shaders
mklink /J UI ..\ExternalLibraries\DXUT11\RunTime\UI
:end
