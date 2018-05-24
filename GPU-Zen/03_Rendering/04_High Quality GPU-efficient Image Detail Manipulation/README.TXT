
README.TXT for demo code of the chapter
'High Quality GPU-efficient Image Detail Manipulation'
in GPU Zen


[ 1. Folder contents ]
<bin> contains pre-built executable idm.exe and files necessary to run
	- idm.exe (executable)
	- nanogui.dll (gui library dll)
	- man.bmp (example bitmap)
	- <shaders> (all glsl shaders required to run idm.exe)
	
<source> contains all source files and the required 3rd party libraries
	- idm.cpp, bmp.h and bmp.cpp (our source code)
	- CMakeLists.txt (CMake file for creating VS2015 solution)
	- <ext> all required 3rd party libraries


	
[ 2. Build instructions for idm.exe ]

Use CMakeLists.txt in <source> folder to generate a Visual Studio 2015 solution using CMake.

We only tested x64 build on VS 2015.

After you have built the executable, please make sure you place
both 'nanogui.dll' and <shaders> folder into the Release folder.



[ 3. Using idm.exe ]

Usage:
C:\> idm <bitmap>
e.g. 
C:\> idm man.bmp

idm.exe supports only basic 24-bit uncompressed Windows .bmp bitmap with max. size 1024x1024.



[ 4. 3rd Party Libraries ]

Minimalistic GUI library for OpenGL by Wenzel Jakob
https://github.com/wjakob/nanogui

nanogui uses:

NanoVG by Mikko Mononen
https://github.com/memononen/nanovg

Eigen
http://eigen.tuxfamily.org/

GLAD (via online generator)
http://glad.dav1d.de/

GLFW
http://www.glfw.org/
