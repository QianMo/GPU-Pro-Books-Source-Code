// Copyright 2010 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.


Adaptive Volumetric Shadow Maps Demo (July 21th, 2010)
======================================================
This application demonstrates the basic Adaptive Volumetric Shadow Maps (AVSM) algorithm.

User Interaction
================
As with other DXSDK samples, use WASD with the mouse (click and drag) to fly around the scene.

Requirements for compilation
============================

1) Visual C++ 2008 SP1 (makes some use of tr1::shared_ptr)
2) DirectX SDK (June 2010 or newer)

For execution
=============

1) Windows Vista or Windows 7
2) DirectX 11 compatible video card (the demo has not been tested on NVIDIA DX11-class hardware yet)
3) Visual C++ 2008 SP1 Redistributable Package

User Interface
==============
There are several Heads Up Displays (HUDs) that let you control various aspects of the AVSM demo.

In addition to the main HUD which is on by default, please use F7 and F8 to turn on the hair HUD and expert HUD respectively. The hair HUD shows the options specific to rendering of hair. 

You can use F5 to turn on a 2D visualization of the AVSM.

The expert HUD has additional controls that control AVSM creation/lookup and debug display of the transmittance curve for a particular visibility ray in the AVSM. To use this feature, hit F5 to turn on the visualization of the AVSM and use the left mouse button to pick a point on the AVSM. The red curve in the bottom left corner displays the transmittance curve for the picked visibility ray at the given point.

Known Issues
============

1) The current capture-all-fragments DX11 implementation exhibits artifacts/temporal noise if fragments are not sorted due to the fact that we employ a lossy compression scheme (pause the particle animation to observe some flickering in particle close-ups). It’s possible to enable via UI a per shadow map texel sort, which removes any noise introduced by inconsistent ordering. 
2) When rendering both hair and particles, the ordering in camera space is not correct for certain camera angles. This is because we turn off depth buffer writing when rendering both of those "objects." This is unrelated to AVSM and can be easily be fixed by rendering z only passes for both particles and hair and dumping those into the main depth buffer when rendering the hair and particles respectively.
3) At the time of writing, the demo crashes at startup on NVIDIA DirectX 11 cards with the latest drivers. The crash is within NVIDIA's user mode driver (nvwgf2um.dll) while compiling shaders. We expect this to be fixed in future driver releases.

AVSM Algorithm Overview
=======================
Adaptive Volumetric Shadow Maps (AVSMs) is a new technique for rendering self-shadowing volumetric. Like shadow map techniques for opaque occluders, AVSMs first render from the light to compute visibility information that is queried to determine shadowing when rendering from the eye. However, unlike standard shadow maps, AVSMs captures visibility information from many translucent occluders and creates a compressed representation of visibility along each light ray. AVSMs are similar in spirit to Opacity Shadow Maps but instead of slicing space into a regular grid, AVSMs stores an adaptive representation that delineates the extents of homogeneous opacity along a light ray. The result is a representation that uses the same amount of memory as Opacity Shadow Maps but provides much higher quality shadows (no banding artifacts, better capturing of high-frequency detail, etc).

The basic algorithm shares the same structure of standard shadow maps:

1) Render translucent geometry from the light position. Compute a compressed visibility function on-the-fly using a fixed number of samples,building an adaptive representation (i.e., we do not assume a regular slicing in depth from the light).
2) Render translucent geometry from the eye, compute volumetric shadows by looking up into the AVSM generated in the first step.

For more details please refer to the original paper:

Salvi M., Vidimce K., Lauritzen A., Lefohn A., "Adaptive Volumetric Shadow Maps", in "Rendering 2010" - Compter Graphics Forum - Volume 29, Number 4, pp. 1289-1296

http://www.eg.org/EG/DL/CGF/volume29/issue4

To obtain a copy of the paper please contact the authors:

marco.salvi@intel.com
kiril.vidimce@intel.com
andrew.t.lauritzen@intel.com
aaron.lefohn@intel.com

Demo And Code Overview
======================

This demo shows how to use AVSMs to cast shadows from volumes made of particles and/or hair, although the algorithm is general enough to handle  other types of light blockers (vegetation, etc.)

To render an AVSM we add contributions to  it via ‘segment insertion’. Segments are sections of light rays intersecting some object. In particular a single segment is specified as two 1D entry and exit points into a volumetric blocker. We also assume that the intersected volume is uniformly dense and we specify the amount of light absorbed by the volume computing a transmittance value and passing it to the AVSM insertion code  (1 = completely translucent, 0 = fully opaque).

The function used to insert a segment in the visibility representation is (see AVSM.hlsl):

void InsertSegmentAVSM(in float segmentDepth[2],  //entry, exit points
                       in float segmentTransmittance, 
                       inout AVSMData avsmData)


This code can be used to add per pixel light blockers in any order (it’s not required to sort the volume representation, although sorting can improve image quality and performance)

The demo renders particles in light and view space as billboards. At AVSM rendering time we intersect each particle (which is modeled as a sphere) on a per pixel basis with a light ray, computing entry and exit points over the ‘particle surface’. Assuming a uniform density for the particle we then compute a transmittance value which is directly proportional to the thickness of the particle along the light ray. 

Once the AVSM is ready we render the particles again in view space. A per pixel occlusion value is determined by sampling the AVSM over the surface of the particle at the entry point (one bi-linear sample per particle per pixel). The same process is used to cast particle shadows over opaque geometry.

AVSM sampling can be seen as a generalized version of PCF (see AVSM.hlsl):

float AVSMPointSample(in float2 uv, in float receiverDepth)
float AVSMBilinearSample(in float2 uv, in float receiverDepth)

main.cpp contains the majority of the DXUT and UI code, while App.[cpp/h]
contains the majority of the application code. In particular, the App constructor initializes all shaders and resources required by the algorithms, and the App::Render function (and children) handle rendering a frame.

Rendering generally proceeds as follows (see App::Render):

1) All opaque geometry is rendered to the G-buffer
2) Render particles in light space, capture all blockers in a per pixel linked list 
3) Walk over each list and insert all its blockers into an AVSM 
4) Accumulate lighting for all opaque light blockers
5) Accumulate lighting for all translucent light blockers

The AVSM insertion and sampling code is implemented in AVSM.hlsl, while AVSM_Resolve.hlsl contains the main functions used in step 3.
Particle rendering code is in Particle.hlsl, while code used for creating linked lists is in ListTexture.hlsl (used in step 2 and 3).
ParticleSystem.[cpp/h] implements a simple particle emitter.

AVSM_def.h defines some values that control some aspects of the AVSM algorithm (see code for more details).
