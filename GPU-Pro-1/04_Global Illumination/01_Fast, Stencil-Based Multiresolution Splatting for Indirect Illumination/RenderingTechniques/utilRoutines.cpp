/***************************************************************************/
/* utilRoutines.cpp                                                        */
/* -----------------------                                                 */
/*                                                                         */
/* These are utility routines that may (or may not) be reused in multiple  */
/*    techniques in the various files in this directory.  However, these   */
/*    functions are not particularly relevant to the renderings, but       */
/*    instead encapsulate annoying OpenGL code that would otherwise make   */
/*    the rendering code difficult to read.                                */
/*                                                                         */
/* Chris Wyman (02/23/2008)                                                */
/***************************************************************************/

#include "stencilMultiResSplatting.h"
#include "renderingData.h"
#include "renderingMethods.h"

extern Scene *scene;
extern RenderingData *data;


// Three functions we have used in the past to benchmark individual rendering modes.
//    This is not currently implemented, but these functions are available if desired.
void BeginTimerQuery( GLuint queryID )
{
	if (*data->ui->benchmark)
		glBeginQuery( GL_TIME_ELAPSED_EXT, queryID );
}

void EndTimerQuery( GLuint queryID )
{
	if (*data->ui->benchmark)
		glEndQuery( GL_TIME_ELAPSED_EXT );
}

void GetTimerQueryResults( float *resultArray )
{
	GLuint timerTicks=0;
	for (int i=0; i<10; i++)
	{
		GLuint timerTicks=0;
		glGetQueryObjectuiv( data->glID->timerQuery[i], GL_QUERY_RESULT, &timerTicks );
		resultArray[i] = timerTicks/1.0e6;
	}
}



// A simple log-base2 function
float log2( float x )
{
	return log(x)/log(2.0);
}


// Sets up a prerendered help screen.  Drawing text with GLUT seems to be 
//    quite slow, so we avoid doing it every frame, and instead only do it
//    once (or once every time parameters change, in our case).
void SetupHelpScreen( FrameBuffer *helpFB )
{
	helpFB->BindBuffer();
	glClearColor( 0, 0, 0, 0 );
	helpFB->ClearBuffers();
	glClearColor( 0, 0, 0, 1 );
	
	glPushAttrib( GL_ALL_ATTRIB_BITS );
	glDisable( GL_DEPTH_TEST );
	glDisable( GL_LIGHTING );
	glDisable( GL_NORMALIZE );
	glDisable( GL_CULL_FACE );

	glColor4f(0.2, 0.2, 0.2, 0.8);
		glMatrixMode( GL_PROJECTION );
		glPushMatrix();
		glLoadIdentity();
		gluOrtho2D(0,1,0,1);
		glMatrixMode( GL_MODELVIEW );
		glPushMatrix();
		glLoadIdentity();
		glBegin( GL_QUADS );
			glTexCoord2f(0,0);	glVertex2f(0.05,0.05);
			glTexCoord2f(1,0);	glVertex2f(0.95,0.05);
			glTexCoord2f(1,1);	glVertex2f(0.95,0.95);
			glTexCoord2f(0,1);	glVertex2f(0.05,0.95);
		glEnd();

		glActiveTexture( GL_TEXTURE0 );
		glEnable( GL_TEXTURE_2D );
		glEnable( GL_BLEND );
		glBlendFuncSeparate( GL_ONE, GL_ONE, GL_ZERO, GL_ONE );
		glBindTexture( GL_TEXTURE_2D, data->tex->iowaLogo->TextureID() );
		glBegin( GL_QUADS );
			glTexCoord2f(0.01,0.01);	glVertex2f(0.8,0.07);
			glTexCoord2f(0.99,0.01);	glVertex2f(0.93,0.07);
			glTexCoord2f(0.99,0.99);	glVertex2f(0.93,0.2);
			glTexCoord2f(0.01,0.99);	glVertex2f(0.8,0.2);
		glEnd();
		glBindTexture( GL_TEXTURE_2D, 0 );
		glDisable( GL_BLEND );
		glDisable( GL_TEXTURE_2D );

		glMatrixMode( GL_PROJECTION );
		glPopMatrix();
		glMatrixMode( GL_MODELVIEW );
		glPopMatrix();
		
	glColor4f(1,1,0,1);
	    DisplayString( 75, 950, "User Interface Commands:", 1024, 1024 );
		DisplayString( 475, 60, "More Information: http://www.cs.uiowa.edu/~cwyman/pubs.html" );
		DisplayString( 455, 105, "\"Fast, Stencil-Based Multiresolution Splatting for Indirect Illumination\"" );
		DisplayString( 530, 90, "Chris Wyman, Greg Nichols, and Jeremy Shopf" );
		DisplayString( 575, 75, "Demo for 'GPU Pro' Article" );
	glColor4f(1,1,1,1);
		DisplayString( 120, 920, "Toggle on / off this help screen", 1024, 1024 );
		DisplayString( 120, 880, "Capture screenshot", 1024, 1024 );
		DisplayString( 120, 860, "Reload shaders (Try reloading if your driver had errors during shader compilation)", 1024, 1024 );
		DisplayString( 120, 840, "Quit demonstration program", 1024, 1024 );

	glColor4f(0,1,1,1);
		DisplayString( 75, 920, "[ h ]", 1024, 1024 );
		DisplayString( 75, 880, "[ f12 ]", 1024, 1024 );
		DisplayString( 75, 860, "[ r ]", 1024, 1024 );
		DisplayString( 75, 840, "[ q ]", 1024, 1024 );
		DisplayString( 75, 800, "[ + ] and [ - ]", 1024, 1024 );
		DisplayString( 75, 780, "[ ] ] and [ [ ]", 1024, 1024 );
		DisplayString( 75, 760, "[ . ] or [ , ]", 1024, 1024 );

		DisplayString( 75, 720, "[ up-arrow ]", 1024, 1024 );
		DisplayString( 75, 700, "[ down-arrow ]", 1024, 1024 );
		
		if (data->ui->movementType == 2)
		{
			DisplayString( 75, 660, "[ left-mouse ]", 1024, 1024 );
			DisplayString( 75, 640, "[ middle-mouse ]", 1024, 1024 );
			DisplayString( 75, 620, "[ right-mouse ]", 1024, 1024 );
		}
		else
		{
			DisplayString( 75, 660, "[ left-mouse ]", 1024, 1024 );
			DisplayString( 75, 640, "[ right-mouse ]", 1024, 1024 );
			DisplayString( 75, 620, "[ w ]", 1024, 1024 );
			DisplayString( 75, 600, "[ s ]", 1024, 1024 );
			DisplayString( 75, 580, "[ a ]", 1024, 1024 );
			DisplayString( 75, 560, "[ d ]", 1024, 1024 );
		}

		DisplayString( 275, 235, "[ NumPad / ] and [ NumPad * ]", 1024, 1024 );
		DisplayString( 620, 310, "(see Dachsbacher & Stamminger, I3D 2006)", 1024, 1024 );
		DisplayString( 620, 295, "(similar to Dachsbacher & Stamminger, I3D 2005)", 1024, 1024 );
		DisplayString( 620, 280, "(see article)", 1024, 1024 );
		DisplayString( 620, 265, "(see article)", 1024, 1024 );

	glColor4f(1,1,1,1);
		DisplayString( 180, 800, "Increase or decrease the depth threshold used during multiresolution splat refinement", 1024, 1024 );
		DisplayString( 180, 780, "Increase or decrease the normal threshold used during multiresolution splat refinement", 1024, 1024 );
		DisplayString( 180, 760, "Increase or decrease the light's intensity", 1024, 1024 );

		DisplayString( 180, 720, "Increase the number of virtual point lights used", 1024, 1024 );
		DisplayString( 180, 700, "Decrease the number of virtual point lights used", 1024, 1024 );

		if (data->ui->movementType == 2)
		{
			DisplayString( 200, 660, "Rotate eye point around dragon using a trackball", 1024, 1024 );
			DisplayString( 200, 640, "Rotate glass object using a trackball", 1024, 1024 );
			DisplayString( 200, 620, "Rotate light using a trackball", 1024, 1024 );
		}
		else
		{
			DisplayString( 200, 660, "Rotate viewing direction (up/down or left/right)", 1024, 1024 );
			DisplayString( 200, 640, "Rotate light using a trackball", 1024, 1024 );
			DisplayString( 200, 620, "Move the camera forwards along the viewing ray", 1024, 1024 );
			DisplayString( 200, 600, "Move the camera backwards along the viewing ray", 1024, 1024 );
			DisplayString( 200, 580, "Move the camera to the left", 1024, 1024 );
			DisplayString( 200, 560, "Move the camera to the right", 1024, 1024 );
		}

		DisplayString( 75, 340, "Five rendering modes are enabled in this demo program:", 1024, 1024 );
		DisplayString( 100, 325, "1) Standard diffuse OpenGL Rendering", 1024, 1024 );
		DisplayString( 100, 310, "2) Reflective shadow mapping using one full-screen splat per VPL..................................................", 1024, 1024 );
		DisplayString( 100, 295, "3) Reflective shadow mapping using one full-screen splat to gather from all VPLs.......................", 1024, 1024 );
		DisplayString( 100, 280, "4) Multiresolution splatting for indirect illumination (no interpolation during upsampling)..............", 1024, 1024 );
		DisplayString( 100, 265, "5) Multiresolution splatting for indirect illumination (with interpolation during upsampling)............", 1024, 1024 );

		DisplayString( 75, 235, "Switch between these modes using", 1024, 1024 );

	glColor4f(1,1,1,1);
		DisplayString( 75, 160, "Current Parameters:", 1024, 1024 );
		DisplayString( 100, 145, "Number of Lights Enabled:", scene->GetWidth(), scene->GetHeight() );
		DisplayString( 100, 130, "Depth Discontinuity Threshold:", scene->GetWidth(), scene->GetHeight() );
		DisplayString( 100, 115, "Normal Discontinuity Threshold:", scene->GetWidth(), scene->GetHeight() );
		DisplayString( 100, 100, "Number of Virtual Point Lights:", scene->GetWidth(), scene->GetHeight() );

	glColor4f(1,1,0,1);
		char buf[1024];
		sprintf( buf, "%d", scene->NumLightsEnabled() );
		DisplayString( 300, 145, buf, scene->GetWidth(), scene->GetHeight() );
		sprintf( buf, "%f", (float)(*data->ui->depthThreshold) );
		DisplayString( 300, 130, buf, scene->GetWidth(), scene->GetHeight() );	
		sprintf( buf, "%f", (float)(*data->ui->normThreshold) );
		DisplayString( 300, 115, buf, scene->GetWidth(), scene->GetHeight() );
		sprintf( buf, "%d (%d x %d)", data->param->vplCount, data->param->vplCountSqrt, data->param->vplCountSqrt );
		DisplayString( 300, 100, buf, scene->GetWidth(), scene->GetHeight() );

	glPopAttrib();
	glColor4f(1,1,1,1);
	helpFB->UnbindBuffer();

	// We've completed updating the help screen!
	data->ui->updateHelpScreen = false;
}

// This should never come up in the demo, unless users start modifying the
//    scene file.  However, some of our research using this framework utilizes
//    area light sources (instead of point lights).  Using code designed for 
//    a scene with area lights to display one with point lights (or vice versa)
//    usually causes a variety of nasty crashes.  When we detect that, we
//    call this function to simply render a slightly-more friendly error message.
void InvalidRenderingMode( char *errorString )
{
	data->fbo->mainWin->BindBuffer();
	data->fbo->mainWin->ClearBuffers();

		glMatrixMode( GL_PROJECTION );
		glPushMatrix();
		glLoadIdentity();
		gluOrtho2D(0,1,0,1);
		glMatrixMode( GL_MODELVIEW );
		glPushMatrix();
		glLoadIdentity();
	
		glColor4f(1,1,0,1);
			DisplayString( 300, 600, "ERROR: Invalid rendering mode!", 1024, 1024 );

		glColor4f(1,1,1,1);
			char buf[1024];
			sprintf( buf, "Reason: %s", errorString );
			DisplayString( 400, 512, buf, 1024, 1024 );

		glMatrixMode( GL_PROJECTION );
		glPopMatrix();
		glMatrixMode( GL_MODELVIEW );
		glPopMatrix();

	data->fbo->mainWin->UnbindBuffer();
}

// Displays the already created help screen
void DisplayHelpScreen( FrameBuffer *helpFB )
{
	glPushAttrib( GL_ALL_ATTRIB_BITS );
	glEnable(GL_BLEND);
	glBlendFunc( GL_ONE, GL_ONE_MINUS_SRC_ALPHA );
	glDisable( GL_DEPTH_TEST );
	helpFB->DisplayAsFullScreenTexture( FBO_COLOR0, true );
	glPopAttrib();
}

// Enabling shadow mapping always looks ugly.  We've encapsulated it here.
void GenericShadowMapEnabler( GLSLProgram *shader, int lightNum, GLuint texID, float *matrix )
{
	char mapName[32];
	GLenum texUnit = GL_TEXTURE7-lightNum;
	sprintf( mapName, "shadowMap%d", lightNum );  // 7 = shadowMap0, 6 = shadowMap1, 5 = shadowMap2
	shader->BindAndEnableTexture( mapName, texID, texUnit, GL_TEXTURE_2D );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE );
	glPushMatrix();
	glLoadIdentity();
	glTexGeni( GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR );
	glTexGeni( GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR );
	glTexGeni( GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR );
	glTexGeni( GL_Q, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR );
	glTexGenfv( GL_S, GL_EYE_PLANE, &matrix[0] );
	glTexGenfv( GL_T, GL_EYE_PLANE, &matrix[4] );
	glTexGenfv( GL_R, GL_EYE_PLANE, &matrix[8] );
	glTexGenfv( GL_Q, GL_EYE_PLANE, &matrix[12] );
	glEnable( GL_TEXTURE_GEN_S );
	glEnable( GL_TEXTURE_GEN_T );
	glEnable( GL_TEXTURE_GEN_R );
	glEnable( GL_TEXTURE_GEN_Q );
	glPopMatrix();
	shader->SetParameter( "useShadowMap", 1 );
}

// Disabling shadow mapping always looks ugly.  We've encapsulated it here.
void GenericShadowMapDisabler( GLSLProgram *shader, int lightNum )
{
	GLenum texUnit = GL_TEXTURE7-lightNum;
	glActiveTexture( texUnit );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE );
	glDisable( GL_TEXTURE_GEN_S );
	glDisable( GL_TEXTURE_GEN_T );
	glDisable( GL_TEXTURE_GEN_R );
	glDisable( GL_TEXTURE_GEN_Q );
	shader->DisableTexture( texUnit, GL_TEXTURE_2D );
}

// Computing a shadow map matrix always looks ugly.  We've encapsulated it here.
void GenericComputeShadowMapMatrix( int lightNum, float sMapAspectRatio, float *shadMapMatrixTranspose )
{
	// Compute Shadow Map Matrix
	glPushMatrix();
	glLoadIdentity();
	glTranslatef( 0.5f, 0.5f, 0.5f + *(data->param->shadowMapBias) );
	glScalef( 0.5f, 0.5f, 0.5f );
	scene->LightPerspectiveMatrix( lightNum, sMapAspectRatio );
	scene->LightLookAtMatrix( lightNum );
	scene->GetCamera()->InverseLookAtMatrix();
	glGetFloatv(GL_TRANSPOSE_MODELVIEW_MATRIX, shadMapMatrixTranspose);
	glPopMatrix(); 
}


// This function peels off the linear depth of a scene (stored in one of the alpha
//   channels of the RSM buffers) and outputs as a grayscale result into the 
//   specified buffer.  
void CopyRSMLinearDepthToBuffer( FrameBuffer *output, GLSLProgram *shader, GLint texID )
{
	output->BindBuffer();
	output->ClearBuffers();

	// Render with deferred shading
	shader->EnableShader();
	shader->BindAndEnableTexture( "rsmBuffer1", texID );
	shader->SetParameter( "offset", 1.0/output->GetWidth() );
	//shader->SetParameter( "farPlane", 50.0 );
	shader->SetParameter( "farPlane", scene->GetCamera()->GetFar() );

	// Draw the screen-size quad for deferred refraction
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

	shader->DisableTexture();
	shader->DisableShader();
	output->UnbindBuffer();
}


// Actually shade the scene using deferred geometry
void DeferredShading( void )
{
	// Compute Shadow Map Matrix
	float shadMapMatrixTranspose[16];
	scene->CreateShadowMap( data->fbo->shadowMap[0], 
							shadMapMatrixTranspose, 
							0, // Light Number
							*(data->param->shadowMapBias) );


	// Render with deferred shading
	data->shader->deferredShade->EnableShader();
	data->shader->deferredShade->SetParameter( "lightIntensity", scene->GetLightIntensityModifier() );
	GenericShadowMapEnabler( data->shader->deferredShade, 0,  
		                     data->fbo->shadowMap[0]->GetDepthTextureID(), 
							 shadMapMatrixTranspose );

	// Draw the screen-size quad for deferred refraction
	glPushMatrix();
	glLoadIdentity();
	glBegin( GL_QUADS );
		glTexCoord2f(0,0);	glVertex2f(0,0);
		glTexCoord2f(1,0);	glVertex2f(1,0);
		glTexCoord2f(1,1);	glVertex2f(1,1);
		glTexCoord2f(0,1);	glVertex2f(0,1);
	glEnd();
	
	// Clean up and disable the shader
	glPopMatrix();

	// Clean up and disable stuff
	GenericShadowMapDisabler( data->shader->deferredShade, 0 );
	data->shader->deferredShade->DisableShader();
}

// Generates a G-buffer from the light's point of view (i.e., a reflective shadow map)
//    Because this is used for sampling and is never visible, we allow our rendering
//    engine to use low-res models (if available) to save on cost.
void ComputeReflectiveShadowMap( FrameBuffer *RSM )
{
	RSM->BindBuffer();
	RSM->DrawBuffers( FBO_COLOR0, FBO_COLOR1, FBO_COLOR2 );
	RSM->ClearBuffers();
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	scene->LightPerspectiveMatrix( 0, RSM->GetAspectRatio() );
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	scene->LightLookAtMatrix( 0 );
	scene->SetupEnabledLightsWithCurrentModelview();
		scene->Draw( MATL_FLAGS_ENABLEONLYTEXTURES, OBJECT_OPTION_USE_LOWRES );
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	RSM->DrawBuffers( FBO_COLOR0 );
	RSM->UnbindBuffer();
	RSM->AutomaticallyGenerateMipmaps( FBO_COLOR0 );
}

// Generates a G-buffer from the eye's point of view.  This is used in the
//     function DeferredShading() to compute a final image.
void ComputeDeferredShadingBuffers( FrameBuffer *deferred )
{
	deferred->BindBuffer();
	deferred->DrawBuffers( FBO_COLOR0, FBO_COLOR1, FBO_COLOR2 );
	deferred->ClearBuffers();
	glLoadIdentity();
	scene->LookAtMatrix();
	scene->SetupEnabledLightsWithCurrentModelview();
		scene->Draw( MATL_FLAGS_ENABLEONLYTEXTURES );
	deferred->DrawBuffers( FBO_COLOR0 );
	deferred->UnbindBuffer();
}

// Take the multiresolution buffer in *fb and flatten it.  This is a destructive
//    process that modifies the multiresolution buffer.  The highest resolution
//    portion now contains the upsampled image.  The rest contains temporaries (junk).
void UpsampleMultiresBuffer( FrameBuffer *fb, GLSLProgram *shader )
{
	// Parameters for the various levels of the multiresolution buffer
	float lvHeight[9] =	 {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625 };
	float lvWidth[9] =	 {0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125 };
	float lvXOffset[9] = {0.0, 0.5,  0.75,  0.875,  0.9375,  0.96875,  0.984375,  0.9921875,  0.99609375 };
	float lvRes[9] =	 {1024., 512., 256., 128., 64., 32., 16., 8., 4. };

	// We'll be ping-ponging between two buffers during upsampling.  
	//    Introduce a nice pointer array to make coding this easy.
	FrameBuffer *fbs[2] = { fb, data->fbo->pingPongBuffer };
	
	// Make sure that our buffers have the right settings -- we'll be assuming nearest-neighbor interpolation.
	fbs[0]->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );
	fbs[1]->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );

	// We're going to ping pong back and forth between multires buffers, without drawing
	//    full screen quads each time, this means we need to first make a duplicate of the
	//    data in the original buffer.  I suspect something more intelligent couple be done
	//    here to save a bit of fillrate.
	fbs[1]->BindBuffer();
	fbs[0]->DisplayAsFullScreenTexture( FBO_COLOR0, false );
	fbs[1]->UnbindBuffer();

	// All right, we're ready to do the upsampling.  Enable the correct shader. 
	//    Setup some parameters that are constant over all the ping-pong passes.
	shader->EnableShader();
	shader->SetParameter( "hInc", 1.0/float(fb->GetWidth()) );
	shader->SetParameter( "vInc", 1.0/float(fb->GetHeight()) );

	// Ping pong back and forth, from coarsest to finest multires levels as we upsample.
	for( int iMip = 5; iMip >= 0; iMip-- )
	{
		// Select one buffer as output and the other buffer as input.
		fbs[iMip%2]->BindBuffer();
		shader->BindAndEnableTexture( "multiResBuf", fbs[(iMip+1)%2]->GetColorTextureID(0), GL_TEXTURE0 );		
		shader->SetParameter( "leftEdge",  lvXOffset[iMip], lvXOffset[iMip+1] );
		shader->SetParameter( "rightEdge", lvXOffset[iMip]+lvWidth[iMip], lvXOffset[iMip+1]+lvWidth[iMip+1] );
		shader->SetParameter( "topEdge",   lvHeight[iMip], lvHeight[iMip+1] );
		shader->SetParameter( "coarseRes",		 lvRes[iMip+1] ); 
		glBegin( GL_QUADS );
			glTexCoord4f(lvXOffset[iMip+1],0.0,				                 0.0,0.0);	
			glVertex3f(lvXOffset[iMip],0,0);
			
			glTexCoord4f(lvXOffset[iMip+1]+lvWidth[iMip+1],0.0,		         1.0,0.0);	
			glVertex3f(lvXOffset[iMip]+lvWidth[iMip],0,0);
			
			glTexCoord4f(lvXOffset[iMip+1]+lvWidth[iMip+1],lvHeight[iMip+1], 1.0,1.0);	
			glVertex3f(lvXOffset[iMip]+lvWidth[iMip],lvHeight[iMip],0);
			
			glTexCoord4f(lvXOffset[iMip+1],lvHeight[iMip+1],		         0.0,1.0);	
			glVertex3f(lvXOffset[iMip],lvHeight[iMip],0);
		glEnd();

		shader->DisableTexture( GL_TEXTURE0 );
		fbs[iMip%2]->UnbindBuffer();
	}

	// Clean up.
	shader->DisableShader();
}

// This is a crude little routine that takes a (flattened) multiresolution buffer
//    and modulates it with the eye-space deferred rendering's color/flux buffer
//    to generate correctly colored indirect illumination.  This should probably
//    be handled in a nicer manner.
void ModulateUpsampledBuffer( FrameBuffer *multiResBuf, GLSLProgram *shader )
{
	shader->EnableShader();
	shader->BindAndEnableTexture( "multiResBuf", multiResBuf->GetColorTextureID(0), GL_TEXTURE0 );
	shader->BindAndEnableTexture( "geomTexColor", data->fbo->deferredGeometry->GetColorTextureID(0), GL_TEXTURE1 );
	glPushMatrix();
	glLoadIdentity();
	glBegin( GL_QUADS );
		glTexCoord2f(0,0);	glVertex2f(0,0);
		glTexCoord2f(1,0);	glVertex2f(1,0);
		glTexCoord2f(1,1);	glVertex2f(1,1);
		glTexCoord2f(0,1);	glVertex2f(0,1);
	glEnd();
	glPopMatrix();
	shader->DisableTexture( GL_TEXTURE1 );
	shader->DisableTexture( GL_TEXTURE0 );
	shader->DisableShader();
}

// This takes the RSM and samples it a specified number of times (to generate VPLs).
//    These are stored in a compact texture (size of #VPLs x 3) allowing easy
//    access in a texture cache-friendly manner.  Note that the color (or reflected
//    flux) texture is sampled via a mipmap allowing an average of the color over a
//    large region of the light view to be used.  This helps (slightly) to reduce
//    flickering artifacts when using only a few VPLs.
void ComputeCompactVPLBuffer( GLSLProgram *shader, FrameBuffer *compactVPLTex )
{
	compactVPLTex->BindBuffer();
	compactVPLTex->ClearBuffers();

	// We're going to grab the color of the VPL based on a mipmap of the flux in the RSM
	//    so we need to determine the appropriate mip level
	float mipLevel = int( log2( data->fbo->reflectiveShadowMap->GetWidth() ) - log2( data->param->vplCountSqrt ) + 0.05 );

	shader->EnableShader();
	shader->SetParameter( "vplCountOffsetDelta", 
		                  data->param->vplCountSqrt,
						  data->param->vplOffset,
						  data->param->vplDelta );

	// Set the shader and texture to allow the access of the mipmap levels
	shader->SetParameter( "mipLevel", mipLevel );
	data->fbo->reflectiveShadowMap->SetAttachmentFiltering( FBO_COLOR0, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST );

	glPushMatrix();
	glLoadIdentity();
	scene->LookAtMatrix( );
	scene->LightLookAtInverseMatrix( 0 );

	glBegin( GL_QUADS );
		glTexCoord2f(0,0);						glVertex2f(0,0);
		glTexCoord2f(data->param->vplCount,0);	glVertex2f(data->param->vplCount/(float)data->param->maxVplCount, 0); 
		glTexCoord2f(data->param->vplCount,2);	glVertex2f(data->param->vplCount/(float)data->param->maxVplCount, 1); 
		glTexCoord2f(0,2);						glVertex2f(0,1);
	glEnd();

	glPopMatrix();

	// Disable RSM mipmapping
	data->fbo->reflectiveShadowMap->SetAttachmentFiltering( FBO_COLOR0, GL_NEAREST, GL_NEAREST );
	shader->DisableShader();

	compactVPLTex->UnbindBuffer();
}