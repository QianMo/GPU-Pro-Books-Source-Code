Deferred Coarse Pixel Shading Demo
===================================

This application demonstrates deferred coarse pixel shading. The article accompanying
this demo is published in GPU Pro-7.
Article Abstract
================
Deferred shading has become a popular rendering technique in the games industry to avoid redundant shading 
of the occluded pixels. With the increasing screen resolutions and the DPI, one can reduce the  frequency 
of slowly varying components of the rendering equation to something lower than once per  pixel without a 
perceivable difference in visual quality. Recent research addresses this issue and proposes hardware solutions 
like Coarse Pixel Shading (https://software.intel.com/en-us/articles/coarse-pixel-shading). 
Although an elegant solution, Coarse Pixel Shader doesn’t fit well into the deferred shading pipeline. 
Lauritzen (https://software.intel.com/en-us/articles/deferred-rendering-for-current-and-future-rendering-pipelines) 
proposed a solution for deferred shading engines that identifies the pixels where pixel rate shading 
is not enough and shades them at the sample rate using a compute shader. We  extend the later idea 
further, but in the reverse direction by identifying the regions of the screen where one can reduce 
the shading to a rate lower than pixel frequency (e.g. 2x2 pixel sized blocks). With our technique we are 
able show about 40-50% reduction in shading time with slight increase  in the G-buffer generation time.



User Interaction
================

- As with other DXSDK samples, use WASD with the mouse (click and drag) to fly
  around the scene.


Requirements
============

For compilation:

1) Visual C++ 2010
2) DirectX SDK (June 2010) for Windows Vista or Windows 7

For execution:

1) Windows Vista or Windows 7 or Windows 10
2) DirectX 11 compatible video card
3) Appropriate Visual C++ Redistributable Package (2016)


