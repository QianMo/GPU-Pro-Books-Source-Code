***************************************************************************
* Fast, Stencil-Based Multiresolution Splatting for Indirect Illumination *
*                        Demo for GPU Pro Article                         *
*                Chris Wyman, Greg Nichols and Jeremy Shopf               *
*                           University of Iowa                            *
*      More Information: http://www.cs.uiowa.edu/~cwyman/pubs.html        *
***************************************************************************

This directory contains executables and code for a demo for our article
entitled:  "Fast, Stencil-Based Multiresolution Splatting for Indirect 
Illumination."

This demo requires:
   1) GLUT (a .dll is included)
   2) GLEW (a .dll is included)
   3) A DirectX-10 class graphics card under Windows XP or Vista
   4) We have only tested on NVIDIA cards, though there is nothing in
      this demo that requires an NVIDIA card
   
The demo was developed, at various times, on GeForce 8800 GTX, 9800 GTX, 
and GTX 280 graphics cards.  It *should* work on older generation cards,
though this has not been tested and probably requires a bit of cleaning
(I think some shaders specify they require SM 4.0, even though they don't)

Most recent nVidia drivers should word.  The demo does not scale on 
multiple-GPU configurations.

The project file is for Microsoft Visual Studio .NET 2008. 

The codebase relies on a "standard" framework I have used for most of my 
recent OpenGL-based research.  The idea is to use a scenegraph-like 
construct that hides most of the rendering details from me when I want to 
render the whose scene at once, but still lets me easily pick out
individual objects to render as glass, with caustics, or casting volumetric
shadows.

Most of the code (inside the directories DataTypes, Interface, Materials,
Objects, Scenes, and Utils) is thus framework code that is not particularly
relevant to this demo.  While this code is mostly straightforward and
"self-commenting," I have not spent time making sure it is particularly
legible.

Most of the interesting stuff goes on inside the directory 
"RenderingTechniques/" which includes all the code for the various 
"rendering modes" mentiond above.  For example, one mode (in 
RSM_Splatting.cpp) draws the scene using a simplistic reflective
shadow map approach for comparison.  The code inside this directory 
is quite well commented.  The "stencilMultiResSplatting.cpp" code in 
the main directory keeps everything together (main() routine, display 
callback, etc).

Shaders are located in "bin/shaders/" and a README in that directory
describes which are particularly relevant to this demo.  


NOTES and CAVEATS:
---------------------

  1) It appears that during simplification to a web-ready, clean, 
     commented demo we did not transfer one of the implementation 
     details.  In particular, our interpolation scheme does not work
     particularly well when interpolating between adjacent
     multiresolution texels that differ by more than one resolution.
     To fix this, we usually add additional subdivde/refinement to the
     larger texel in this case.  This detail is not implemented in 
     this demo (it may be added when we find time).  This results in
     flickering stair-stepping artifacts along sharp boundaries as
     multiresolution texels abruptly change from very fine to very coarse.

  2) Some included scenes are thanks to Google 3D Warehouse.  Thanks!
     
