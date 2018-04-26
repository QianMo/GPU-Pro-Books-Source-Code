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

// This function draws one full-screen quad (aka "splat") per VPL.  For
//   each quad it computes the VPL's illumination contribution for each pixel
//   on screen and blends this into the current buffer (i.e., the final image).
void AccumulateIndirect( void )
{
	// Setup the correct GLSL shader.
	GLSLProgram *shader = data->shader->indirectAccumulate;
	shader->EnableShader();
	shader->SetParameter( "lightIntensity", scene->GetLightIntensityModifier() );
	shader->SetParameter( "lightSamples", data->param->vplCount );

	// Make sure we're blending with the correct function
	glBlendFunc( GL_ONE, GL_ONE );

	// Note that the GL_MODELVIEW matrix is not used to transform the quad
	//    in this shader (a simple glOrtho matrix is used).  Instead, the
	//    modelview matrix is used to transform the light-space position of
	//    the VPL sample (stored in the compactVPLTex texture) into eye-space
	//    so it is comparable with the data stored in the eye-space G-buffer.
	glPushMatrix();
	glLoadIdentity();
	scene->LookAtMatrix( );
	scene->LightLookAtInverseMatrix( 0 );

	// Ok, actually draw the quads
	glBegin( GL_QUADS );
	for (int k=0; k<data->param->vplCount; k++)
		{
			// The texture coordinate is used to pass down the column index
			//    into the compactVPLTex (i.e., k/vplCount gives a [0..1]
			//    value used as the x coordinate into compactVPLTex).
			glTexCoord2f(0,k/(float)data->param->maxVplCount);	glVertex2f(0,0);
			glTexCoord2f(1,k/(float)data->param->maxVplCount);	glVertex2f(1,0);
			glTexCoord2f(1,k/(float)data->param->maxVplCount);	glVertex2f(1,1);
			glTexCoord2f(0,k/(float)data->param->maxVplCount);	glVertex2f(0,1);
		}
	glEnd();

	// Clean up and disable the shader
	glPopMatrix();
	shader->DisableShader();
}


// A display routine to draw the scene with reflective shadow maps using
//   the extremely slow method of drawing one full-screen splat for each VPL.
void Display_WithReflectiveShadowMap( void )
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
		DeferredShading();      // Draw the direct illumination
		AccumulateIndirect();   // Draw the indirect illumination
	data->fbo->mainWin->UnbindBuffer();
	
}
