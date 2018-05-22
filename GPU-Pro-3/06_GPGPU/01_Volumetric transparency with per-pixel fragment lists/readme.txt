GpuPro3 - Volumetric transparency demo
by Laszlo Szecsi (szecsi@iit.bme.hu)

To launch the demo, start VolumetricTransparency11.exe. Please allow some time for loading the scene.

Required SW:
D3D11 runtime (in Windows Vista and later)

Required HW:
D3D11, Shader Model 5 compatible

=============================================================================================

To compile, use Visual Studio 2010.

Required:
DirectX SDK (June 2010)
Boost libray (1.42 used)	- http://www.boost.org/

Please make sure you set include and library paths accordingly.

Included with the demo source (with licencing conditions):
Open Asset Importer library ()	- http://assimp.sourceforge.net/

Model used:
CryTek Sponza scene - created and donated by M. Dabrovic and F. Meinl, CryTek

=============================================================================================
Source code highlights:

Study the .fx files for the shaders. voltrasmain.fx includes all others. volparticles.fx contains the ray-casting technique, the others belong to the fragment list demo.

DxaVolTransRaster is the main application file, which handles scene and resource management, and performs rendering passes in the render method.

Particle system simulation is handled by the Particle class and the animate method of DxaVolTransRaster.
