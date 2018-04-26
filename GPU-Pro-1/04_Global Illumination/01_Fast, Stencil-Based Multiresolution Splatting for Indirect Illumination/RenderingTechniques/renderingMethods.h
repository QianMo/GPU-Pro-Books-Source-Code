/***************************************************************************/
/* renderingMethods.h                                                      */
/* -----------------------                                                 */
/*                                                                         */
/* Has prototypes and other data needed to use the various rendering       */
/*     techniques (i.e., DisplayCallback()'s) located in the .cpp files    */
/*     in this directory.                                                  */
/*                                                                         */
/* Chris Wyman (02/01/2008)                                                */
/***************************************************************************/

#ifndef __REDNERINGMETHODS_H__
#define __REDNERINGMETHODS_H__

class FrameBuffer;

// Called if the selected rendering mode detects something (in the scene) that would
//    cause the program to crash if the display mode were executed.
void InvalidRenderingMode( char *errorString );

// Display functions for more complex multiresolution indirect splatting
void CopyRSMLinearDepthToBuffer( FrameBuffer *output, GLSLProgram *shader, GLint texID );
void ComputeCompactVPLBuffer( GLSLProgram *shader, FrameBuffer *compactVPLTex );
void CreateRefinementDataBuffer( FrameBuffer *output, GLSLProgram *shader );
void UpsampleMultiresBuffer( FrameBuffer *fb, GLSLProgram *shader );
void ModulateUpsampledBuffer( FrameBuffer *multiResBuf, GLSLProgram *shader );
void GatherIntoStenciledSubsplatMap( FrameBuffer *fb, GLSLProgram *shader );
void StencilSubsplatRefinement( GLSLProgram *shader, FrameBuffer *depthMipmap, FrameBuffer *normMipmap );
void CreateNormalThresholdBuffer( FrameBuffer *output, GLSLProgram *shader );

// Display functions for rendering with a simple reflective shadow map
void AccumulateIndirect( void );
void DeferredShading( void );
void ComputeReflectiveShadowMap( FrameBuffer *RSM );
void ComputeDeferredShadingBuffers( FrameBuffer *deferred );


// The data structure in "renderingData.h" stores various data needed
//    by the rendering techniques in this directory.  This function 
//    initializes all this data.
void InitializeRenderingData( void );					     // See: initializeRenderingData.cpp
void FreeMemoryAndQuit( void );


// Math functions I ocassionally like to use
float log2( float x );                                             // See: utilRoutines.cpp									     


// Useful for shadow maps
void GenericShadowMapEnabler( GLSLProgram *shader, int lightNum, GLuint texID, float *matrix );
void GenericShadowMapDisabler( GLSLProgram *shader, int lightNum );
void GenericComputeShadowMapMatrix( int lightNum, float sMapAspectRatio, float *shadMapMatrixTranspose );

// Create various types of non-standard mipmaps
void CreateCustomMipmap( FrameBuffer *f, int colorAttachment, GLSLProgram *shader );

// Benchmarking functions
void BeginTimerQuery( GLuint queryID );
void EndTimerQuery( GLuint queryID );
void GetTimerQueryResults( float *resultArray );

#endif

