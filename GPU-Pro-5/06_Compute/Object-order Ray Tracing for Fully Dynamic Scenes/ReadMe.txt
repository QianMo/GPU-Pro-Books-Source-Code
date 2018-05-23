GPU Pro 5: Object-order Ray Tracing for Fully Dynamic Scenes Demo
---------
 Tobias Zirr (tobias.zirr AT alphanew DOT net)
 Hauke Rehfeld (hauke.rehfeld AT kit DOT edu)
 Carsten Dachsbacher (dachsbacher AT kit DOT edu)

The code is provided as is under the MIT License. If you
use this code or part of it, please acknowledge the source
and its authors by citing their work.

Depending on the time you read this, you may or may not
find updated versions of this example at:
http://www.alphanew.net/index.php?section=articles

Requirements:
------------
The sample application was built against
the Windows 8 SDK: http://msdn.microsoft.com/en-us/library/windows/desktop/hh852363.aspx
and CUDA 5.5: https://developer.nvidia.com/cuda-toolkit

The provided Visual Studio solution file should work
with Visual C++ 2010 or higher.

If you're using VC++ 2010, you might have to install the
Windows 8 SDK manually.
In case you want to build against different versions of
the SDKs, simply change the Platform.Cpp.{Win32|x64}.user
or CUDA.Cpp.{Win32|x64}.user property sheets accordingly,
either by editing the respecitve files in the './global'
directory or by using the property sheet manager provided
by Visual Studio.

Known Issues:
------------
1) Since the sample application makes use of the state-of-the-art
radix sort implementation that is provided by the CUDA-based B40C
library, the sample only runs on NVIDIA hardware.

2) A whole range of NVIDIA drivers had severe CUDA/DirectCompute
interop issues on several graphics cards earlier this year. At the
time of writing, the drivers 320.49 and later seem to have fixed
these issues, so you will likely need to update your drivers if
you're still running an older version.

Guide:
-----
The demo features three scenes with reflection rays. The demo
automatically reloads all shaders and assets that are modified
during its execution, so feel free to play with the shaders.

The 'RayTracing' Visual Studio project contains a project folder
'Shaders'/'RayTracing' that references all shaders relevant to
the ray tracing pipeline.

The 'Hardwired' subfolder contains the shader files 'RayGen.fx'
(a good place to start experimenting) and 'RayGrid.fx' (contains
everything relevant to ray grid construction).

The 'Prototypes' subfolder contains 'SceneApprox.fx', which
centrally defines the voxelization shader code that is required
for building the conservative voxel representation of the scene.

The 'Materials' subfolder contains 'Textured.fx', which enhances
the 'Rasterization'/'Materials'/'Textured.fx' shader with the
necessary intersection testing functionality to compute hit points
with arbitrary rays. This includes voxelization into the ray grid
(first pass in the technique at the bottom) and cell-triangle pair
processing (second pass).

The 'Lights' and 'Atmospherics' subfolders contain examplary shaders
for the shading of hit points and the addition of sky light for
unoccluded rays.

Libraries:
---------
The sample code comes with the following open-source libraries:

breeze 2 (MIT): https://code.google.com/p/breeze-2/
back40computing (New BSD): https://code.google.com/p/back40computing/
AntTweakBar (zlib/png): http://anttweakbar.sourceforge.net/doc/
Effects11 (Microsoft Public): https://fx11.codeplex.com/
DirectXTex (Microsoft Public): https://directxtex.codeplex.com/
d3d-effects-lite (MIT): https://code.google.com/p/d3d-effects-lite/
lean (MIT): https://code.google.com/p/lean-cpp-lib/
utf8-cpp (MIT): http://utfcpp.sourceforge.net/
rapidxml (MIT): http://rapidxml.sourceforge.net/
boost (Boost): http://www.boost.org/

Assets:
------
The sample application uses the Crytek Sponza scene created by
Frank Meinl at Crytek, based on a model by Marko Dabrovic. It
also uses the Sibenik Cathedral scene that was created by Marko
Dabrovic.

Contact:
-------
See above
 + KIT Computer Graphics Group: http://cg.ibds.kit.edu/english/publikationen.php

 - Tobias Zirr, August 2013
   [Twitter: @alphanew]
   [Website: http://www.alphanew.net]