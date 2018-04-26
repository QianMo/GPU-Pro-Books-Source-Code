
Alpha-Blending As A Post-Process

To accompany the article on the alpha blending technique developed for Pure, there are two RenderMonkey scenes 
 & several color images to help demonstrate the effect - and how it may be integrated into your project.

Requirements:

- DirectX 9c
- RenderMonkey Version 1.81 upward  ( http://developer.amd.com/GPU/RENDERMONKEY/Pages/default.aspx )
- 3D Graphics card supporting at least shader model 2

Contents:

.\RenderMonkey Projects
   |
   *-- Screen Space Alpha Masking - Single Tree.rfx
   *-- Screen Space Alpha Masking - Forest.rfx
   |
   *-- Single Pine Tree.3ds
   *-- Pine Forest.3ds
   *-- Aspen Forest.3ds
   *-- Aspen 2 Forest.3ds
   *-- Winter Forest.3ds
   *-- BackDrop_Hills.3ds
   *-- Opaque Word.3ds
   *-- ScreenAlignedQuad.3ds
   *-- Sphere.3ds
   |
   *-- Pine Forest.tga
   *-- Aspen Forest.tga
   *-- Aspen 2 Forest.tga
   *-- Backdrop_Hills_Color.tga
   *-- Backdrop_Hills_DetailMask.tga
   *-- Backdrop_Hills_Lightmap.tga
   *-- DetailPass_1.tga
   *-- DetailPass_2.tga
   *-- Fractal Cloud.tga
   *-- SkyDome.dds

.\Color Plates
   |
   *-- Screen Space Alpha Masking - Artifact Reduction.psd
   |
   *-- Pure 001 - Opaque Color Image.bmp
   *-- Pure 002 - Opaque Depth Image.bmp
   *-- Pure 003 - Foliage Mask ADD.bmp
   *-- Pure 004 - Foliage Mask MAX.bmp
   *-- Pure 005 - Foliage Color.bmp
   *-- Pure 006 - Foliage Color Depth.bmp
   *-- Pure 007 - Combined Image.bmp
   *-- Pure 008 - Final Image.bmp

* Screen Space Alpha Masking - Single Tree.rfx

- A RenderMonkey scene that contains 3 examples of a single tree rendered using either: Alpha-Blended (unsorted), Alpha-Tested (z-buffered) & Screen Space Alpha Masking.
  The intension of this scene is to allow a visual-quality comparison of the various techniques.

* Screen Space Alpha Masking - Forest.rfx

- A RenderMonkey scene that demonstrates the rendering of a large-scale forest scene.
  This scene renders hundreds of SSAM-blended trees using just 2 passes of 4 geometry batches (8 draw calls!).
  The scene elements have been positioned to be visible by the default camera position: 0, 0, -200
   and before spinning the camera around, the user may want to set the camera-target position to: 0, 0, -160

* Screen Space Alpha Masking - Artifact Reduction.psd

- This .PSD file demonstrates both the clear-screen-color & squared-alpha fixes mentioned in the article.
  The image contains several layers which can be enabled / disabled to highlight the contributions made by each fix.

* Pure XXX - Blah.bmp

- Color plates presented in the article, showing the various stages of the SSAM pipeline.

I would like to say an extra special thank you to Black Rock art team, especially: 
Julia Friedl, Trevor Moore & Mark Knowles, for their wonderful contribution towards the art assets used in the demo.

Benjamin Hathaway
Black Rock Studio 2007-2009
ben.hathaway@disney.com
