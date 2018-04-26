***************************************************************************
* Fast, Stencil-Based Multiresolution Splatting for Indirect Illumination *
*                        Demo for GPU Pro Article                         *
*                Chris Wyman, Greg Nichols and Jeremy Shopf               *
*                           University of Iowa                            *
*      More Information: http://www.cs.uiowa.edu/~cwyman/pubs.html        *
***************************************************************************

This directory contains GLSL shaders for our demo for the paper above.

There are five directories here:
   deferredLighting/         -> Shaders for simple deferred rendering and
                                basic (non-multires) reflective shadow mapping
   normalSurfaceShaders/     -> Shaders for standard scene surfaces (boring!)
   subsplatRefinement/       -> Shaders to create the depth & normal mipmaps
                                used to refine subsplats, and our shader that
                                performs our fast stencil-based subsplat refinement
   subSplatSplatting/        -> Shaders to render illumination into the multires
                                buffer and those to upsample and display the final
                                upsampled result.
   utilityShaders/           -> Utilities (copy FBO, merge tex array, store frag position)

The two directories that contain shaders most relevant to this paper are
"subsplatRefinement/", which contains all the refinement related code, 
and "subSplatSplatting/", which contains code to compute the indirect light
and upsample it for display.  Shaders in these directories have the most
in-code documentation.

However, "deferredLighting/" may also be of interest since it contains our
implementations for simple deferred shading and simple reflective shadow mapping.
Some of these shaders are used in the multires code, and will at the least give
some insight into how the more complex multiresolution shader work.

"normalSurfaceShaders/" contains boring code to render diffuse and Phong
shaders.  

Shaders in the "utilityShaders/" directory should be self explanatory 
based upon their names or the comments in the code.  Most shaders are 2-4 
lines and should need little explanation.

