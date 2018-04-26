/***************************************************************************/
/* RSM_Splatting.cpp                                                       */
/* -----------------------                                                 */
/*                                                                         */
/* A basic implementation of reflective shadow maps, using splatting to    */
/*     accumulate indirect illumination.                                   */
/*                                                                         */
/* Chris Wyman (04/25/2008)                                                */
/***************************************************************************/

#include "stencilMultiResSplatting.h"
#include "renderingData.h"
#include "renderingMethods.h"

extern Scene *scene;
extern RenderingData *data;


// A display routine that renders a scene with indirect illumination using
//    the multiresolution splatting technique discussed in our I3D 2009
//    paper.  In this case, extremely naive interpolation (nearest neighbor)
//    is used to give the viewer a sense of what this multiresolution
//    indirect illumination buffer looks like.
void Display_WithMultiResGather_NoInterpolation( void )
{
	// Compute the reflective shadow map
	ComputeReflectiveShadowMap( data->fbo->reflectiveShadowMap );

	// Sample the actual VPLs into a smallish buffer for quicker accumulation
	ComputeCompactVPLBuffer( data->shader->vplCompact, data->fbo->compactVPLTex );

	// Compute the g-buffer for deferred shading
	ComputeDeferredShadingBuffers( data->fbo->deferredGeometry );

	// Compute the depth max-min mipmap of the RSM
	CopyRSMLinearDepthToBuffer( data->fbo->depthDeriv, 
								data->shader->createRSMDerivativeMap, 
								data->fbo->deferredGeometry->GetColorTextureID( 1 ) );
	CreateCustomMipmap( data->fbo->depthDeriv, FBO_COLOR0, data->shader->depthDerivMax );

	// Create the normal metric mipmap.  We cannot make this in the same pass as the depth, unfortunately
	//    due to the fact that we need to output > 4 values (4 for normal, 1 for depth) and the mipmapping
	//    seems to have problems when there are more than one color buffer attached to the FBO (???).
	CreateNormalThresholdBuffer( data->fbo->normalThreshMipMap, 
		                     	data->shader->createNormalThresholdBuffer );
	CreateCustomMipmap( data->fbo->normalThreshMipMap, FBO_COLOR0, data->shader->normalThresholdMipmap ); 

	// "Refine" the subsplats we'll use for this frame by turning on corresponding bits in
	//     a multi-resolution stencil buffer.
	StencilSubsplatRefinement( data->shader->depthAndNormalStencilRefinement, 
		                       data->fbo->depthDeriv,
							   data->fbo->normalThreshMipMap );

	// Render into our multiresolution illumination buffer.  This is done
	//    by drawing a full-screen quad that only affects those texels whose
	//    stencil bit was set above.  For each of these texels, illumination
	//    is gathered from all sampled VPLs (similar to the approach used
	//    in the RSM_Gathering.cpp code)
	GatherIntoStenciledSubsplatMap( data->fbo->multiResIllumBuffer, data->shader->perSubsplatStenciledGather );
 
	// Now that we have this whacky multiresolution buffer, we need to combine
	//    it to some normal 2D image we can actually use to render to screen.
	//    This function upsamples it (here with nearest neighbor interpolation)
	UpsampleMultiresBuffer( data->fbo->multiResIllumBuffer, data->shader->upsampleNoInterpolation );

		// OK, generate the final rendering
	data->fbo->mainWin->BindBuffer();
		DeferredShading();                                            // Draw the direct illumination
		ModulateUpsampledBuffer( data->fbo->multiResIllumBuffer,      // Copy the indirect light from the multires buffer
			                     data->shader->modulateLevel0 );
	data->fbo->mainWin->UnbindBuffer();
}

