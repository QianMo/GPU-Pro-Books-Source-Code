
Progressive Voxelization for Global Illumination Demo
demostrating both dynamic geometry and dynamic illumination
  Athanasios Gaitatzes (gaitat at yahoo dot com), 2012
  Georgios Papaioannou (gepap at aueb dot gr), 2012

If you use this code as is or any part of it in any kind
of project or product, please acknowledge the source and its authors
by citing their work:

  A. Gaitatzes, G. Papaioannou,
  "Progressive Screen-space Multi-channel Surface Voxelization"

You can find the latest version of the code at:
http://www.virtuality.gr/gaitat/en/publications.html


The project is build against the:
1. Microsoft DirectX SDK (I have used the June 2010 version)

Solution Files are provided both for:
1. Visual Studio 2008
2. Visual Studio 2010

To run the demo:
Go into the bin-Win32 directory and run either of the .bat files depending  
if you have compiled with Visual Studio 2008 or 2010. 
Executables are provided for both Visual Studio 2008 and 2010.

Errors:
When running the *VS2008.bat and you get the error message:
"The application has failed to start because its side-by-side
configuration is incorrect. Please see the application event 
log or use the command-line sxstrace.exe tool for more detail"

It means that you dont have the Visual Studio 2008 Redistributable installed.
Have a look at:
http://answers.microsoft.com/en-us/windows/forum/windows_7-pictures/error-the-application-has-failed-to-start-because/df019c0d-746e-42d0-ad68-465e18e3f3ef
or use the *VS2010.bat version.

Note:
There is a peculiarity with the cal3d library.
That is why in bin-Win32 there are versions compiled for VS2008 and VS2010.
Make sure you are using the correct one otherwise you will see a segmentation fault.
