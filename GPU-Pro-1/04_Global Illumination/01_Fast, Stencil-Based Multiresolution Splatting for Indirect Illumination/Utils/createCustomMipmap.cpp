

#include "framebufferObject.h"
#include "glslProgram.h"


// This function creates a custom mipmap based upon the image currently
//   residing inside the the framebuffer's specified color attachment.
//   The specified shader is applied once per level, using as input for
//   mipmap level i the (i-1)st level of the mipmap.
// To apply this function, the glGenerateMipmapEXT( GL_TEXTURE_2D ) must
//   have been called inside the FrameBuffer class initializer.  This is
//   done by setting the "enableAutomaticMipmaps" flag to "true" when
//   calling the constructor.
// BEWARE: Currently, this code ONLY works for square input textures.
void CreateCustomMipmap( FrameBuffer *f, int colorAttachment, GLSLProgram *shader )
{ 
	//GLenum error = glGetError();
	//while (error != GL_NO_ERROR) error = glGetError();

	//printf("0) %d, %s\n", error, gluErrorString( error ));

	// We need to change some state in pretty extensive ways for this.  To
	//   avoid a lengthy cleanup, we'll just pop back to the current state.
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0,1,0,1);
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_VIEWPORT_BIT | 
					GL_ENABLE_BIT | GL_TEXTURE_BIT );
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
	glDepthMask( GL_FALSE );

	int i, j;
	float mult=1;
	GLuint texType = f->GetTextureType( colorAttachment );
	GLuint texID   = f->GetColorTextureID( colorAttachment-FBO_COLOR0 );
	f->BindBuffer();
	f->TemporarilyUnattachAllBuffersExcept( colorAttachment );

	if (shader)
	{
		shader->EnableShader();
		shader->BindAndEnableTexture( "inputTex", texID, GL_TEXTURE0, texType );
	}
	else
	{
		glBindTexture( texType, texID );
		glEnable( texType );
	}	

	//f->CheckFramebufferStatus( 1 );

    for (i=1, j=f->GetWidth()/2;j>=1;i++,j/=2,mult*=2) // works only for square textures...
	{
		//printf("1+%d) %d, %s\n", i, error=glGetError(), gluErrorString( error ));
		glViewport( 0, 0, j, j );
		glFramebufferTextureEXT(GL_FRAMEBUFFER_EXT,
                                  GL_COLOR_ATTACHMENT0_EXT+(colorAttachment-FBO_COLOR0),
                                  texID, i);
        glTexParameteri( texType, GL_TEXTURE_BASE_LEVEL, i-1) ;
        glTexParameteri( texType, GL_TEXTURE_MAX_LEVEL, i-1 );
		glTexParameteri( texType, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri( texType, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		//printf("2+%d) %d, %s\n", i, error=glGetError(), gluErrorString( error ));
		//f->CheckFramebufferStatus( 1 );

		if (shader)
			shader->SetParameter( "offset", mult/f->GetWidth() );

		glColor3f(1,1,1);
		glBegin(GL_QUADS);
		glTexCoord2f( 0, 0 );  glVertex2f( 0, 0 );
		glTexCoord2f( 1, 0 );  glVertex2f( 1, 0 );
		glTexCoord2f( 1, 1 );  glVertex2f( 1, 1 );
		glTexCoord2f( 0, 1 );  glVertex2f( 0, 1 );
		glEnd();
		//printf("3+%d) %d, %s\n", i, error=glGetError(), gluErrorString( error ));
	}

	//printf("9) %d, %s\n", error=glGetError(), gluErrorString( error ));

	glTexParameteri( texType, GL_TEXTURE_BASE_LEVEL, 0 );
    glTexParameteri( texType, GL_TEXTURE_MAX_LEVEL, i-1 );
	if (shader)
	{
		shader->DisableTexture( GL_TEXTURE0, texType );
		shader->DisableShader();
	}
	else
		glDisable( texType );

	f->ReattachAllBuffers();
	f->UnbindBuffer();
	glDepthMask( GL_TRUE );

	// Return our state to normal
	glPopAttrib();	
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	//printf("End) %d, %s\n", error=glGetError(), gluErrorString( error ) );
}

