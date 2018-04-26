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

// This function gathers illumination at all the texels marked via a stencil
//   bit in the multiresolution image buffer "fb".  The general idea is exactly 
//   equivalent to the working of the GatherIndirect() in RSM_Gathering.cpp,
//   except instead of gathering illumination at ALL pixels in a regular 2D
//   image, we're gathering illumination only at the stenciled pixels in our
//   multiresolution buffer.
void GatherIntoStenciledSubsplatMap( FrameBuffer *fb, GLSLProgram *shader )
{
	// Bind our multiresolution render target 
	fb->BindBuffer();

	// Enable our shader
	shader->EnableShader();
	shader->SetParameter( "lightIntensity", scene->GetLightIntensityModifier() );
	shader->SetParameter( "lightSamples", data->param->vplCount );
	shader->SetParameter( "offset", 1.0/data->param->maxVplCount );
	shader->SetParameter( "maxKValue", data->param->vplCount/(float)data->param->maxVplCount );

	// We're not modifying the depth in the stenciled regions, so disable depth writes
	glDepthMask( GL_FALSE );

	// This shader has already had stenciling automatically enabled (see initialization code)
	//   However, we have to modify the stencil operation (we do not want to modify the 
	//   stencil bits) and the stencil func (since we only want to draw where the stencil 
	//   is on, i.e. stencil = 255)
	glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
	glStencilFunc(GL_EQUAL, 255, 0xffffffff);

	// OK let's draw into our multires buffer
	glLoadIdentity();
	gluOrtho2D(0,1,0,1);

	// Here we draw one quad for each level of the multiresolution buffer.  The 
	//    vertices are offset oddly, because we are storing our multires buffer
	//    by flattening it into a single 2048 x 1024 image.
	glBegin( GL_QUADS );
	float offset=0, delta = 0.5;
	for (float i=0; i<=6; i+=1)
	{
		glTexCoord3f(0,0,i);	glVertex2f(offset,0);
		glTexCoord3f(1,0,i);	glVertex2f(offset+delta,0);
		glTexCoord3f(1,1,i);	glVertex2f(offset+delta,2.0*delta);
		glTexCoord3f(0,1,i);	glVertex2f(offset,2.0*delta);
		offset += delta;
		delta *= 0.5;
	}
	glEnd();

	// OK.  We're done gathering.  We'll probably need our depth buffer again.
	glDepthMask( GL_TRUE );

	// Clean up.  If we're doing a screen capture, also output a copy of our multires buffer
	shader->DisableShader();
	if (data->ui->captureScreen) data->fbo->multiResIllumBuffer->SaveColorImage( FBO_COLOR0, "color.ppm" );
	fb->UnbindBuffer();
}


// This function sets up the appropriate stencil bits in our multiresolution illumination 
//    buffer.  This is equivalent to the "refinement" process described in our I3D paper
//    but runs significantly faster.  Please see our EGSR 2009 paper for additional details 
//    about this approach.
void StencilSubsplatRefinement( GLSLProgram *shader, 
								FrameBuffer *depthMipmap, 
								FrameBuffer *normMipmap )
{
	// Bind our multires buffer, clear the screen.  Our interpolation during
	//   upsampling demands that invalid texels have an alpha value of zero, so
	//   we need to change our clear color here.
	data->fbo->multiResIllumBuffer->BindBuffer();
	glClearColor( 0.0, 0.0, 0.0, 0.0 );
	data->fbo->multiResIllumBuffer->ClearBuffers();
	glClearColor( 0.0, 0.0, 0.0, 1.0 );

	// This pass ONLY sets the stencil bit (i.e., determines which subsplats to render)
	//    so we turn of writes to the depth and color buffers, with the hope that the
	//    GPU will speed this pass up as much as possible.
	glDepthMask( GL_FALSE );
	glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );

	//  Sets a stencil to on everywhere (where a fragment is drawn).  The shader 
	//      controls where this occurs by discarding fragments that do not
	//      correspond to valid locations in the illumination buffer.
	glEnable(GL_STENCIL_TEST);
	glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);
	glStencilFunc(GL_ALWAYS, 255, 0xffffffff);

	// Setup our shader.  This shader will use our depth and normal mipmaps
	//     in order to determine if each texel in the multires buffer is valid
	//     so we need to pass both of these down, as well as the theshold values.
	shader->EnableShader();
	shader->BindAndEnableTexture( "depthMinMax", depthMipmap->GetColorTextureID(0), GL_TEXTURE0 );
	depthMipmap->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST );
	shader->BindAndEnableTexture( "normMinMax", normMipmap->GetColorTextureID(0), GL_TEXTURE1 );
	normMipmap->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST );
	shader->SetParameter( "depthThreshold", *(data->ui->depthThreshold) );
	shader->SetParameter( "normThreshold", *(data->ui->normThreshold) );
	shader->SetParameter( "maxMipLevel", 6 );
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D( 0, 1, 0, 1 );
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	// Here we draw one quad for each level of the multiresolution buffer.  The 
	//    vertices are offset oddly, because we are storing our multires buffer
	//    by flattening it into a single 2048 x 1024 image.
	glBegin( GL_QUADS );
	float offset=0, delta = 0.5;
	for (float i=0; i<=6; i+=1)
	{
		glTexCoord3f(0,0,i);	glVertex2f(offset,0);
		glTexCoord3f(1,0,i);	glVertex2f(offset+delta,0);
		glTexCoord3f(1,1,i);	glVertex2f(offset+delta,2.0*delta);
		glTexCoord3f(0,1,i);	glVertex2f(offset,2.0*delta);
		offset += delta;
		delta *= 0.5;
	}
	glEnd();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	// Clean up and disable our shader.
	normMipmap->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );
	shader->DisableTexture( GL_TEXTURE1 );
	depthMipmap->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );
	shader->DisableTexture( GL_TEXTURE0 );
	shader->DisableShader();

	// Disable our stencil text and re-enable writes to depth and color
	glDisable(GL_STENCIL_TEST);
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
	glDepthMask( GL_TRUE );

	// Before finishing, check if we're doing a screen capture, in which case
	//    we'd also like to save a copy of the stencil buffer for our viewing pleasure.
	if (data->ui->captureScreen) data->fbo->multiResIllumBuffer->SaveStencilImage( "stencilBuf.pgm" );
	data->fbo->multiResIllumBuffer->UnbindBuffer();
}

// This function creates the base image for our normal-mipmap buffer.  Because
//    creating a "mipmap" of normals doesn't make much sense, we actually store
//    the minimum and maximum x&y components of all normals in the mipmap texel.
//    (i.e., our buffer is (min-x, min-y, max-x, max-y))  This can easily be
//    mipmapped, and because we don't actually *need* the normal, but just a
//    measure of its variance in the region, this gives us enough information.
void CreateNormalThresholdBuffer( FrameBuffer *output, GLSLProgram *shader )
{
	// Bind our buffer
	output->BindBuffer();

	// Enable the shader and appropriate images from our G-buffer.  For this
	//   demo, only one of these is needed, but this normal buffer can be used
	//   for some more advanced things later, so to keep code consistant with
	//   future demos, we're binding both the normal and position g-buffers.
	shader->EnableShader();
	shader->BindAndEnableTexture( "rsmBuffer1", data->fbo->deferredGeometry->GetColorTextureID( 1 ), GL_TEXTURE0 );
	shader->BindAndEnableTexture( "rsmBuffer2", data->fbo->deferredGeometry->GetColorTextureID( 2 ), GL_TEXTURE1 );
	shader->SetParameter( "offset", 1.0/output->GetWidth() );

	// Draw a screen-size quad to store our base level of the normal mipmap
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0,1,0,1);
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glBegin( GL_QUADS );
		glTexCoord2f(0,0);	glVertex2f(0,0);
		glTexCoord2f(1,0);	glVertex2f(1,0);
		glTexCoord2f(1,1);	glVertex2f(1,1);
		glTexCoord2f(0,1);	glVertex2f(0,1);
	glEnd();
	
	// Clean up and disable the shader
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();
	shader->DisableTexture( GL_TEXTURE0 );
	shader->DisableTexture( GL_TEXTURE1 );
	shader->DisableTexture( GL_TEXTURE2 );
	shader->DisableShader();
	output->UnbindBuffer();
}

// A display routine that renders a scene with indirect illumination using
//    the multiresolution splatting technique discussed in our I3D 2009 paper  
void Display_WithMultiResGather_StencilRefinement( void )
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
	//    This function upsamples it (here with the approach described in our
	//    I3D paper).
	UpsampleMultiresBuffer( data->fbo->multiResIllumBuffer, data->shader->upsampleI3DInterpolation );

	// OK, generate the final rendering
	data->fbo->mainWin->BindBuffer();
		DeferredShading();                                            // Draw the direct illumination
		ModulateUpsampledBuffer( data->fbo->multiResIllumBuffer,      // Copy the indirect light from the multires buffer
			                     data->shader->modulateLevel0 );
	data->fbo->mainWin->UnbindBuffer();
}

