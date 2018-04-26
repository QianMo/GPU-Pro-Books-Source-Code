/***************************************************************************/
/* renderingData.h                                                         */
/* -----------------------                                                 */
/*                                                                         */
/* Stores a "class" that encapsulates global data used by the various      */
/*     rendering techniques described in this directory.  The internals of */
/*     this class are initialized by calling InitializeRenderingData()     */
/*     defined in initializeRenderingData.cpp.                             */
/*                                                                         */
/* Chris Wyman (02/23/2008)                                                */
/***************************************************************************/

#ifndef __RENDERING_DATA_H__
#define __REDNERING_DATA_H__

#include "Interface/UIVars/UIInt.h"
#include "Interface/UIVars/UIBool.h"
#include "Interface/UIVars/UIFloat.h"

class FrameBuffer;
class GLSLProgram;
class GLTexture;
class FrameBufferData;
class VertexBufferData;
class TextureData;
class ShaderData;
class ParameterData;
class OtherGLIDData;
class UIData;
class MovieMaker;
class Random;
class Point;

// Class encapsulating global rendering data
class RenderingData
{
public:
	FrameBufferData  *fbo;
	ShaderData       *shader;
	ParameterData    *param;
	TextureData      *tex;
	VertexBufferData *vbo;
	OtherGLIDData    *glID;
	UIData           *ui;
	Random			 *rng;
};



class FrameBufferData
{
public:
	// Framebuffers for simple rendering of direct illumination with shadow maps
	FrameBuffer *shadowMap[4], *mainWin, *helpScreen;

	// Framebuffer for a reflective shadow map (RSM)
	FrameBuffer *reflectiveShadowMap;
	
	// Framebuffer used for deferred rendering of the eye-view (i.e., a G-buffer)
	FrameBuffer *deferredGeometry;

	// Framebuffer used to compactly store sampled VPLs
	FrameBuffer *compactVPLTex;

	// Our multiresolution indirect illumination buffer (and an associated
	//    buffer needed during upsampling for "ping-ponging")
	FrameBuffer *multiResIllumBuffer, *pingPongBuffer;

	// Framebuffers that store our depth and normal mipmaps used to 
	//    select appropriate resoltion texels for rendering our multires illumination
	FrameBuffer *depthDeriv, *normalThreshMipMap;

};


class ShaderData
{
public:
	// Shaders for deferred lighting
	GLSLProgram *deferredShade;

	// This shader performs reflective shadow mapping using a single
	//    full-screen "splat" that gathers illumination from all of
	//    the selected VPLs.
	GLSLProgram *indirectGather;

	// This shader performs reflective shadow mapping using one
	//    full-screen "splat" for each VPL.  Each splat accumulates
	//    illumination from the appropriate VPL into the final image.
	GLSLProgram *indirectAccumulate;

	// A shader that creates a compact texture containing the selected
	//    VPL locations, surface normals, and reflected flux.  This 
	//    avoids having to sample from the full-resolution RSM textures
	//    and generally improves cache coherence during gathering.
	GLSLProgram *vplCompact;

	// Shaders for min-max mipmap creation
	GLSLProgram *depthDerivMax, *createRSMDerivativeMap;
	GLSLProgram *createNormalThresholdBuffer, *normalThresholdMipmap;

	// Shaders for 'refining' the splats to the appropriate resolution
	GLSLProgram *depthAndNormalStencilRefinement;

	// Gather illumination at each subsplat (i.e., each stenciled locations
	//     in the multiresolution buffer)
	GLSLProgram *perSubsplatStenciledGather;

	// Upsample the multiresolution buffer.  There's two flavors.  One
	//     does a simple nearest-neighbor interpolation.  The other uses
	//     the approach explained in our I3D paper.
	GLSLProgram *upsampleNoInterpolation, *upsampleI3DInterpolation;

	// This shader takes the multresolution buffer that has been upsampled
	//     using the upsample*Interpolation shaders, and multiplies the
	//     0-th level (i.e., full screen) with the surface colors to generate
	//     the final indirect illumination seen from the eye.
	// The basic idea is to take our whacky multires buffer and output the
	//     final computed and upsampled result to a *normal* image for viewing.
	GLSLProgram *modulateLevel0;
	
};


class ParameterData
{
public:
	// Parameters for simple rendering with shadow maps
	float shadowMatrixTrans[4][16];
	UIFloat *shadowMapBias, *lightFOV;

	// A simple matrix representing gluOrtho2D(0,1,0,1);
	float gluOrtho[16];

	// Variables for changing the # of VPLs.   Note:  These may not yet
	//    be changable at run-time.  However, between execution changes
	//    should be possible.
	int maxVplCount;
	int vplCountSqrt, vplCount;  // vplCount = vplCountSqrt*vplCountSqrt;
	float vplDelta, vplOffset;   // vplDelta = 1.0/vplCountSqrt;  vplOffset = 0.5/vplCountSqrt;
};

class TextureData
{
public:
	// Static exutres that might used throught the code
	GLTexture *spotlight; 
	GLTexture *iowaLogo;
};

class VertexBufferData
{
public:
	// Vertex buffers used when photons, photon buffers, or caustic splats.  
	//   Multiple buffers are used for various types of ping-ponging computation.
};

class OtherGLIDData
{
public:
	// Occlusion Query IDs
	GLuint primCountQuery, timerQuery[10];
};

class UIData
{
public:
	// Various data the user might control
	bool updateHelpScreen;
	bool displayHelp;
	bool captureScreen;
	UIBool *animation;
	UIBool *benchmark;
	UIInt *numLightsUsed;
	UIInt *uiVPLCount;
	UIFloat *depthThreshold;
	UIFloat *normThreshold;
	UIFloat *vplIntensityThreshold;
	UIFloat *translationSpeed;

	// This is simply a flag used by the help-screen to display
	//   the appropriate type of help information for how to move.
	//   The need for this is some demos will include scenes with
	//   type different key-mappings for movement...
	int movementType;
};



#endif

