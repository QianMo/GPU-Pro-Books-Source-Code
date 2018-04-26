/***************************************************************************/
/* RSM_Gathering.cpp                                                       */
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

// This function draws a single full-screen quad, which gathers the 
//    indirect illumination from ALL VPLs at each pixel on screen.
//    This is faster (probably due to fill rate and texture caching 
//    issues) than rendering one quad per VPL by a significant margin.
void GatherIndirect( void )
{
	// Select the appropriate shader.  The parameter "offset" is used
	//    to march through the columns of compactVPLTex.  In that 
	//    texture each column (3 pixels high) represents a single VPL
	GLSLProgram *shader = data->shader->indirectGather;
	shader->EnableShader();
	shader->SetParameter( "lightIntensity", scene->GetLightIntensityModifier() );
	shader->SetParameter( "lightSamples", data->param->vplCount );
	shader->SetParameter( "offset", 1.0/data->param->maxVplCount );
	shader->SetParameter( "maxKValue", data->param->vplCount/(float)data->param->maxVplCount );
	
	// We need to blend this one quad in with the direct illumination
	//   we already drew into the buffer.
	glBlendFunc( GL_ONE, GL_ONE );

	// In this shader, the GL_MODELVIEW matrix is not used to transfor
	//   geometry (i.e., the quad).  Instead a simple glOrtho matrix
	//   is used.  Instead, this modelview matrix is used to transform
	//   the light-space position of each VPL (stored in the RSM) back
	//   into eye-space so it can be used with the data stored in the
	//   G-buffers created for deferred shading.
	glPushMatrix();
	glLoadIdentity();
	scene->LookAtMatrix( );
	scene->LightLookAtInverseMatrix( 0 );

	// Ok, actually draw the quad.  (A bit anticlimatic)
	glBegin( GL_QUADS );
			glVertex2f(0,0);
			glVertex2f(1,0);
			glVertex2f(1,1);
			glVertex2f(0,1);
	glEnd();

	// Clean up and disable the shader
	glPopMatrix();
	shader->DisableShader();
}



// A display routine to draw the scene with reflective shadow maps using
//   a single full-screen splat that (for each pixel) gathers illumination
//   from all of the sampled VPLs.
void Display_WithReflectiveShadowMapGathering( void )
{	
	// Compute the reflective shadow map
	ComputeReflectiveShadowMap( data->fbo->reflectiveShadowMap );

	// Sample the actual VPLs into a smallish buffer for quicker accumulation
	ComputeCompactVPLBuffer( data->shader->vplCompact, data->fbo->compactVPLTex );

	// Compute the g-buffer for deferred shading
	ComputeDeferredShadingBuffers( data->fbo->deferredGeometry );

	// OK, draw the scene with the reflective shadow maps
	data->fbo->mainWin->BindBuffer();
	data->fbo->mainWin->ClearBuffers();
		DeferredShading();  // Draw the direct illumination
		GatherIndirect();   // Draw the indirect illumination	
	data->fbo->mainWin->UnbindBuffer();
}
