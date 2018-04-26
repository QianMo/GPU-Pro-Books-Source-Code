Realistic Depth-of-Field demo by David Illes, Peter Horvath

--------------------
build:
--------------------
The GPU layer was programmed using the NVIDIA CUDA SDK: http://www.nvidia.com/object/cuda_home.html
The Graphical User Interface was implemented using the cross-platform Qt toolkit by Nokia: http://qt.nokia.com/products

The project was developed in Microsoft Visual Studio 2005.

LensBlur solution contains two project:
- LensBlurGPU: the GPU layer
- LensBlur: CPU calculations, applicaion framework and GUI

First build the LensBlurGPU project which is a static library. Dependencies are cudart.lib and cutil32D.lib from the CUDA SDK.
To compile you have to use the nvcc compiler on the main file (gpudof.cu) and add the kernel file (lensblur.cu) as a dependency. In Visual Studio it looks like the following:

"$(CUDA_BIN_PATH)\nvcc.exe" --shared -ccbin "$(VCInstallDir)bin" -c -D_DEBUG -DWIN32 -D_CONSOLE -D_MBCS -Xcompiler /EHsc,/W3,/nologo,/Wp64,/Od,/Zi,/RTC1,/MTd -I./src -I../LensBlur/src -I"$(NVSDKCUDA_ROOT)/common/inc" -I"$(CUDA_INC_PATH)" -L"$(CUDA_LIB_PATH)" -L"$(NVSDKCUDA_ROOT)\common\lib" -lcudart -lcutil32D -o build\lensblurgpu.obj src\gpudof.cu

After that you can build the LensBlur project with the CUDA and Qt headers on the include path. Additional dependencies are cudart.lib cutil32D.lib from CUDA SDK, qtmain.lib QtCore4.lib QtGui4.lib from the Qt SDK and LensBlurGPU.lib. 
The Qt files need a custom build step using the moc compiler shipped with the Qt SDK:

"$(QTDIR)\bin\moc.exe"  "$(InputPath)" -o ".\build\moc_$(InputName).cpp" -DQT_CORE_LIB -DQT_GUI_LIB -DQT_THREAD_SUPPORT -DUNICODE -DWIN32 -I"$(QTDIR)\include\." -I"$(QTDIR)\include\QtCore\." -I"$(QTDIR)\include\QtGui\." -I".\." -I".\build"

Than you need to add the compiled moc files to the project and build.

--------------------
run:
--------------------
The cutil32D.dll from the CUDA SDK have to be in the same folder as the executable. To display the kernel icons you need them in the resource folder next to the executable. Simply run the LensBlur.exe and the program will prompt you to load the source and depth map image from the file system.

