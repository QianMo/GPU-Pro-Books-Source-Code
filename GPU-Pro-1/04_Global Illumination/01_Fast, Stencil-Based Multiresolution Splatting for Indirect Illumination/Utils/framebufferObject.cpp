/************************************************
** framebufferObject.cpp                       **
** ---------------------                       **
**                                             **
** This is the frame-work for general purpose  **
**   initialization of a framebuffer object,   **
**   as specified in the OpenGL extension:     **
**       GL_EXT_FRAMEBUFFER_OBJECT             **
**                                             **
** Since this is an OpenGL extension, not WGL, **
**   it should be much more portable (and      **
**   supposedly) faster than p-buffers and     **
**   render-to-texture.                        **
**                                             **
** Chris Wyman (4/27/2005)                     **
************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "framebufferObject.h"
#include <GL/glu.h>

#pragma warning( disable: 4996 )

FrameBuffer::FrameBuffer( char *name ) : depth(-1), automaticMipmapsEnabled(0)
{
	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorBuffers );
	colorIDs = new GLuint[maxColorBuffers];
	colorType = new GLenum[maxColorBuffers];
	numColorAttachments = 0;
	depthID = 0;
	stencilID = 0;
	for (int i=0; i<maxColorBuffers; i++)
	{
		colorIDs[i] = 0;
		colorType[i] = GL_TEXTURE_2D;
	}
	prevFrameBuf = 0;
	width = height = 0;
	includedBuffers = 0;
	depthType = stencilType = GL_TEXTURE_2D;
	glGenFramebuffersEXT( 1, &ID );

	if (!name)
		sprintf( fbName, "Framebuffer %d", ID );
	else
		strncpy( fbName, name, 79 );
}

FrameBuffer::FrameBuffer( int width, int height, char *name ) 
: width( width ), height( height ), depth(-1), automaticMipmapsEnabled(0)
{
	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorBuffers );
	colorIDs = new GLuint[maxColorBuffers];
	colorType = new GLenum[maxColorBuffers];
	depthID = stencilID = 0;
	numColorAttachments = 0;
	for (int i=0; i<maxColorBuffers; i++)
	{
		colorIDs[i] = 0;
		colorType[i] = GL_TEXTURE_2D;
	}
	prevFrameBuf = 0;
	includedBuffers = 0;
	depthType = stencilType = GL_TEXTURE_2D;
	glGenFramebuffersEXT( 1, &ID );

	if (!name)
		sprintf( fbName, "Framebuffer %d", ID );
	else
		strncpy( fbName, name, 79 );
}

FrameBuffer::FrameBuffer( GLenum type, int width, int height, int depth, 
				 GLuint colorBufType, int numColorBufs, int hasZbuf, 
				 bool enableAutomaticMipmaps, char *name ) :
width( width ), height( height ), depth( depth ), 
automaticMipmapsEnabled(enableAutomaticMipmaps?1:0)
{
	if ( type == GL_TEXTURE_1D || type == GL_TEXTURE_3D )
		printf("Warning!  FrameBuffer constructor called with untested texture type!\n");

	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorBuffers );
	colorIDs = new GLuint[maxColorBuffers];
	colorType = new GLenum[maxColorBuffers];
	depthID = stencilID = prevFrameBuf = includedBuffers = 0;
	for (int i=0; i<maxColorBuffers; i++)
	{
		colorIDs[i] = 0;
		colorType[i] = type;
	}
	depthType = stencilType = type;

	if (!name) sprintf( fbName, "Framebuffer %d", ID );
	else strncpy( fbName, name, 79 );

	glGenFramebuffersEXT( 1, &ID );

	numColorAttachments = numColorBufs;
	if (numColorBufs > 0)
	{
		includedBuffers |= GL_COLOR_BUFFER_BIT;
		glGenTextures(numColorBufs, colorIDs);

		for (int i=0; i<numColorBufs; i++)
		{
			glBindTexture(type, colorIDs[i]);
			glTexParameteri( type, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri( type, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri( type, GL_GENERATE_MIPMAP, automaticMipmapsEnabled > 0 ? GL_TRUE : GL_FALSE );
			if (type == GL_TEXTURE_CUBE_MAP)
			{
				glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, colorBufType, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
				glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, colorBufType, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
				glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, colorBufType, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
				glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, colorBufType, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
				glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, colorBufType, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
				glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, colorBufType, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
			}
			else if (type == GL_TEXTURE_2D_ARRAY_EXT || type == GL_TEXTURE_3D)
				glTexImage3D(type, 0, colorBufType, width, height, depth, 0, GL_RGBA, GL_FLOAT, NULL);
			else if (type == GL_TEXTURE_2D || type == GL_TEXTURE_1D_ARRAY_EXT)
				glTexImage2D(type, 0, colorBufType, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
			else if (type == GL_TEXTURE_1D)
				glTexImage1D(type, 0, colorBufType, width, 0, GL_RGBA, GL_FLOAT, NULL);
			if (enableAutomaticMipmaps) glGenerateMipmapEXT( type );
			BindBuffer();
			glFramebufferTextureEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT+i, colorIDs[i], 0 );
			UnbindBuffer();
		}
	}
	if (hasZbuf > 0)
	{
		includedBuffers |= GL_DEPTH_BUFFER_BIT;
		glGenTextures(1, &depthID);

		if (hasZbuf > 1) includedBuffers |= GL_STENCIL_BUFFER_BIT;

		GLint internalBufFormat = (hasZbuf > 1 ? GL_DEPTH24_STENCIL8_EXT : GL_DEPTH_COMPONENT);

		glBindTexture(type, depthID);
		glTexParameteri( type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri( type, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri( type, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE );
		if (type == GL_TEXTURE_CUBE_MAP)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, internalBufFormat, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, internalBufFormat, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, internalBufFormat, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, internalBufFormat, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, internalBufFormat, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
			glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, internalBufFormat, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		}
		else if (type == GL_TEXTURE_2D_ARRAY_EXT || type == GL_TEXTURE_3D)
			glTexImage3D(type, 0, internalBufFormat, width, height, depth, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		else if (type == GL_TEXTURE_2D || type == GL_TEXTURE_1D_ARRAY_EXT)
			glTexImage2D(type, 0, internalBufFormat, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		else if (type == GL_TEXTURE_1D)
			glTexImage1D(type, 0, internalBufFormat, width, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		BindBuffer();
		glFramebufferTextureEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, depthID, 0 );
		if (hasZbuf > 1) glFramebufferTextureEXT( GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, depthID, 0 );
		UnbindBuffer();
	}

	glBindTexture( type, 0 );
}


FrameBuffer::FrameBuffer( int test ) :
	width( 512 ), height( 512 ), depth( -1 ), automaticMipmapsEnabled( 0 )
{
	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorBuffers );
	colorIDs = new GLuint[maxColorBuffers]; colorType = new GLenum[maxColorBuffers];
	depthID = stencilID = prevFrameBuf = includedBuffers = 0;
	for (int i=0; i<maxColorBuffers; i++)
	{
		colorIDs[i] = 0;
		colorType[i] = GL_TEXTURE_2D_ARRAY_EXT;
	}
	depthType = stencilType = GL_TEXTURE_2D_ARRAY_EXT;
	sprintf( fbName, "Test CubeMap Framebuffer" );

	numColorAttachments = 1;
	includedBuffers = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
	glGenFramebuffersEXT( 1, &ID );

	glGenTextures(1, colorIDs);
	glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, colorIDs[0] );
	glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA16F_ARB, 512, 512, 0, GL_RGBA, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA16F_ARB, 512, 512, 0, GL_RGBA, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA16F_ARB, 512, 512, 0, GL_RGBA, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA16F_ARB, 512, 512, 0, GL_RGBA, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA16F_ARB, 512, 512, 0, GL_RGBA, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA16F_ARB, 512, 512, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, GL_LUMINANCE, width, height, 6, 0, GL_RGBA, GL_FLOAT, NULL);

	glGenTextures(1, &depthID);
	glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, depthID );
	glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_DEPTH_TEXTURE_MODE, GL_LUMINANCE );	
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_DEPTH_COMPONENT, 512, 512, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_DEPTH_COMPONENT, 512, 512, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_DEPTH_COMPONENT, 512, 512, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_DEPTH_COMPONENT, 512, 512, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_DEPTH_COMPONENT, 512, 512, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_DEPTH_COMPONENT, 512, 512, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, GL_DEPTH_COMPONENT, width, height, 6, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);

	glBindTexture( GL_TEXTURE_2D_ARRAY_EXT, 0 );

	BindBuffer();
	glFramebufferTextureEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, colorIDs[0], 0 );
	glFramebufferTextureEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, depthID, 0 );
	UnbindBuffer();
	
}

GLint GetBoundTexture( GLenum texType )
{
	GLint retVal = 0;
	if (texType == GL_TEXTURE_2D) glGetIntegerv( GL_TEXTURE_BINDING_2D, &retVal );
	else if (texType == GL_TEXTURE_1D) glGetIntegerv( GL_TEXTURE_BINDING_1D, &retVal );
	else if (texType == GL_TEXTURE_3D) glGetIntegerv( GL_TEXTURE_BINDING_3D, &retVal );
	else if (texType == GL_TEXTURE_2D_ARRAY) glGetIntegerv( GL_TEXTURE_BINDING_2D_ARRAY, &retVal );
	else if (texType == GL_TEXTURE_1D_ARRAY) glGetIntegerv( GL_TEXTURE_BINDING_1D_ARRAY, &retVal );
	else if (texType == GL_TEXTURE_CUBE_MAP) glGetIntegerv( GL_TEXTURE_BINDING_CUBE_MAP, &retVal );
	return retVal;
}

void FrameBuffer::SetAttachmentFiltering( int attachment, GLint minFilter, GLint magFilter )
{
	GLuint texID = GetTextureID( attachment );
	GLenum texType = GetTextureType( attachment );
	GLint  lastTex = GetBoundTexture( texType );
	glBindTexture( texType, texID );
	glTexParameteri( texType, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri( texType, GL_TEXTURE_MAG_FILTER, magFilter);
	glBindTexture( texType, lastTex );
}

void FrameBuffer::SetAttachmentClamping( int attachment, GLint sWrap, GLint tWrap )
{
	GLuint texID = GetTextureID( attachment );
	GLenum texType = GetTextureType( attachment );
	GLint  lastTex = GetBoundTexture( texType );
	glBindTexture( texType, texID );
	glTexParameteri( texType, GL_TEXTURE_WRAP_S, sWrap);
	glTexParameteri( texType, GL_TEXTURE_WRAP_T, tWrap);
	glBindTexture( texType, lastTex );
}

void FrameBuffer::TemporarilyUnattachAllBuffersExcept( GLuint colorBuffer )
{
	BindBuffer();
	for (int i=0; i<numColorAttachments; i++)
	{
		if (colorBuffer-FBO_COLOR0 == i) continue;
		glFramebufferTextureEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT+i, 0, 0 );
	}
	glFramebufferTextureEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, 0, 0 );
	UnbindBuffer();
}

void FrameBuffer::ReattachAllBuffers( void )
{
	BindBuffer();
	for (int i=0; i<numColorAttachments; i++)
		glFramebufferTextureEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT+i, colorIDs[i], 0 );
	glFramebufferTextureEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, depthID, 0 );
	UnbindBuffer();
}

void FrameBuffer::AutomaticallyGenerateMipmaps( int attachment )
{
	assert( automaticMipmapsEnabled );

	GLenum type = GetTextureType( attachment );
	glBindTexture( type, GetTextureID( attachment ) );
	glGenerateMipmapEXT( type );
	glBindTexture( type, 0 );
}



FrameBuffer::~FrameBuffer( )
{
	// unbind this buffer, if bound
	GLint tmpFB;
	glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &tmpFB );
	if (tmpFB == ID)
		glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, prevFrameBuf );
	
	for (int i=0; i < maxColorBuffers; i++)
		if (colorIDs[i])
			glDeleteTextures( 1, &colorIDs[i] );

	// delete the stencil & depth renderbuffers
	if (depthID)
		glDeleteTextures( 1, &depthID );
		//glDeleteRenderbuffersEXT(1, &depthID);
	if (stencilID)
		glDeleteRenderbuffersEXT(1, &stencilID);

	// delete the framebuffer
	glDeleteFramebuffersEXT( 1, &ID );
	delete [] colorIDs;
	delete [] colorType; 
}



// check to see if the framebuffer 'fb' is complete (i.e., renderable) 
//    if fb==NULL, then check the currently bound framebuffer          
GLenum FrameBuffer::CheckFramebufferStatus( int printMessage )
{
	GLenum error;
	GLint oldFB = 0;
	glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &oldFB );

	// there may be some other framebuffer currently bound...  if so, save it 
	if ( oldFB != ID )
		glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, ID);
	
	// check the error status of this framebuffer */
	error = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

	// if error != GL_FRAMEBUFFER_COMPLETE_EXT, there's an error of some sort 
	if (printMessage)
	{
		switch(error)
		{
			case GL_FRAMEBUFFER_COMPLETE_EXT:
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
				printf("Error!  %s missing a required image/buffer attachment!\n", fbName);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
				printf("Error!  %s has no images/buffers attached!\n", fbName);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
				printf("Error!  %s has mismatched image/buffer dimensions!\n", fbName);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
				printf("Error!  %s's colorbuffer attachments have different types!\n", fbName);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
				printf("Error!  %s trying to draw to non-attached color buffer!\n", fbName);
				break;
			case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
				printf("Error!  %s trying to read from a non-attached color buffer!\n", fbName);
				break;
			case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
				printf("Error!  %s format is not supported by current graphics card/driver!\n", fbName);
				break;
			default:
				printf("*UNKNOWN ERROR* reported from glCheckFramebufferStatusEXT() for %s!\n", fbName);
				break;
		}
	}

	// if this was not the current framebuffer, switch back! 
	if ( oldFB != ID )
		glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, oldFB );

	return error;
}

// attach a texture (colorTexID) to one of the color buffer attachment points 
//    This function is not completely general, as it does not allow specification
//    of which MIPmap level to draw to (it uses the base, level 0).
int FrameBuffer::AttachColorTexture( GLuint colorTexID, int colorBuffer )
{
	// If the colorBuffer value is valid, then bind the texture to the color buffer.
	if (colorBuffer < maxColorBuffers)
	{
		BindBuffer();
		glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT+colorBuffer, 
								   GL_TEXTURE_2D, colorTexID, 0);
		includedBuffers |= GL_COLOR_BUFFER_BIT;
		UnbindBuffer();
		if ( colorIDs[colorBuffer]==0 && colorTexID>0  ) numColorAttachments++;
		if ( colorIDs[colorBuffer]!=0 && colorTexID==0 ) numColorAttachments--;
	}
	else
		return 0;
	colorIDs[colorBuffer] = colorTexID;
	return 1;
}


// attach a texture (depthTexID) to the depth buffer attachment point.
int FrameBuffer::AttachDepthTexture( GLuint depthTexID )
{
	BindBuffer();
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, 
							  GL_TEXTURE_2D, depthTexID, 0);
	depthID = depthTexID;
	includedBuffers |= GL_DEPTH_BUFFER_BIT;
	UnbindBuffer();
	return 1;
}

// attach a texture (stencilTexID) to the stencil buffer attachment point.
int FrameBuffer::AttachStencilTexture( GLuint stencilTexID )
{
	BindBuffer();
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, 
							  GL_TEXTURE_2D, stencilTexID, 0);
	stencilID = stencilTexID;
	includedBuffers |= GL_STENCIL_BUFFER_BIT;
	UnbindBuffer();
	return 1;
}


// attach a renderbuffer (colorBufID) to one of the color buffer attachment points 
int FrameBuffer::AttachColorBuffer( GLuint colorBufID, int colorBuffer )
{
	// If the colorBuffer value is valid, then bind the texture to the color buffer.
	if (colorBuffer < maxColorBuffers)
	{
		BindBuffer();
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, colorBufID);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGBA, 
		                         width, height);
		glFramebufferRenderbufferEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT+colorBuffer, 
								      GL_RENDERBUFFER_EXT, colorBufID);
		includedBuffers |= GL_COLOR_BUFFER_BIT;
		if ( colorIDs[colorBuffer]==0 && colorBufID>0  ) numColorAttachments++;
		if ( colorIDs[colorBuffer]!=0 && colorBufID==0 ) numColorAttachments--;
		UnbindBuffer();
	}
	else
		return 0;
	colorIDs[colorBuffer] = colorBufID;
	return 1;
}

// attach a renderbuffer (depthBufID) to the depth buffer attachment point.
int FrameBuffer::AttachDepthBuffer( GLuint depthBufID )
{
	BindBuffer();
	//glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthBufID);
    //glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, 
	//                         width, height);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, 
								   GL_RENDERBUFFER_EXT, depthBufID);
	includedBuffers |= GL_DEPTH_BUFFER_BIT;
	depthID = depthBufID;
	UnbindBuffer();
	return 1;
}

// attach a renderbuffer (stencilBufID) to the stencil buffer attachment point.
int FrameBuffer::AttachStencilBuffer( GLuint stencilBufID )
{
	BindBuffer();
	//glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, stencilBufID);
    //glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_STENCIL_INDEX8_EXT, 
	 //                        width, height);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, 
								   GL_RENDERBUFFER_EXT, stencilBufID);
	includedBuffers |= GL_STENCIL_BUFFER_BIT;
	stencilID = stencilBufID;
	UnbindBuffer();
	return 1;
}


// Bind this framebuffer as the current one.  Store the old one to reattach
//    when we unbind.  Also return the ID of the previous framebuffer.
GLuint FrameBuffer::BindBuffer( void )
{
	GLint tmp;
	glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &tmp );
	prevFrameBuf = tmp;
	glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, ID );
	//AllowUnclampedColors();
	glPushAttrib( GL_VIEWPORT_BIT );
	glViewport( 0, 0, width, height );
	return prevFrameBuf;
}


// This function unbinds this framebuffer to whatever buffer was attached
//     previously...  If for some reason the binding have changed so we're
//     no longer the current buffer, DO NOT unbind, return 0.  Else, unbind
//     and return 1.
int FrameBuffer::UnbindBuffer( void )
{
	GLint tmpFB;
	glGetIntegerv( GL_FRAMEBUFFER_BINDING_EXT, &tmpFB );
	if (tmpFB != ID) return 0;
	glPopAttrib();
	glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, prevFrameBuf );
	prevFrameBuf = 0;
	return 1;
}

void FrameBuffer::DrawToColorMipmapLevel( GLuint colorBuffer, GLuint level )
{
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
                              GL_COLOR_ATTACHMENT0_EXT+colorBuffer,
                              GL_TEXTURE_2D, GetColorTextureID( colorBuffer ), level);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, level-1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, level-1);

	glBindTexture( GL_TEXTURE_2D, GetColorTextureID( colorBuffer ) );
	glEnable(GL_TEXTURE_2D);
}

void FrameBuffer::DoneDrawingMipmapLevels( GLuint colorBuffer )
{
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
                              GL_COLOR_ATTACHMENT0_EXT+colorBuffer,
                              GL_TEXTURE_2D, GetColorTextureID( colorBuffer ), 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1000);
}


void FrameBuffer::DisplayAsFullScreenTexture( int attachment, bool blending, float lod )
{
	int minFilter, magFilter;
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0,1,0,1);
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glPushAttrib( GL_ALL_ATTRIB_BITS );
	if (blending) glEnable( GL_BLEND ); else glDisable( GL_BLEND );
	glDisable( GL_LIGHTING );
	glDisable( GL_DEPTH_TEST );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, GetTextureID(attachment) );
	glGetTexParameteriv( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, &minFilter );
	glGetTexParameteriv( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, &magFilter );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, lod > 0 ? GL_NEAREST_MIPMAP_NEAREST : GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, lod > 0 ? GL_LINEAR : GL_LINEAR );
	if (lod > 0) glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, lod );
	glEnable( GL_TEXTURE_2D );
	glColor3f( 1, 1, 1 );
	glBegin( GL_QUADS );
		glTexCoord2f(0,0);	glVertex2f(0,0);
		glTexCoord2f(1,0);	glVertex2f(1,0);
		glTexCoord2f(1,1);	glVertex2f(1,1);
		glTexCoord2f(0,1);	glVertex2f(0,1);
	glEnd();
	if (lod > 0) glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, 0 );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	glDisable( GL_TEXTURE_2D );
	glBindTexture( GL_TEXTURE_2D, 0 );
	glPopAttrib();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();
}

void FrameBuffer::DisplayAlphaAsFullScreenTexture( int attachment, float lod )
{
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0,1,0,1);
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glPushAttrib( GL_ALL_ATTRIB_BITS );
	glDisable( GL_LIGHTING );
	glEnable( GL_BLEND );
	glBlendFunc( GL_ZERO, GL_SRC_ALPHA );
	glClearColor( 1, 1, 1, 0 );
	glClear( GL_COLOR_BUFFER_BIT );
	glClearColor( 0, 0, 0, 1 );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, GetTextureID(attachment) );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, lod > 0 ? GL_NEAREST_MIPMAP_NEAREST : GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, lod > 0 ? GL_NEAREST_MIPMAP_NEAREST : GL_LINEAR );
	if (lod > 0) glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, lod );
	glEnable( GL_TEXTURE_2D );
	glColor3f( 1, 1, 1 );
	glBegin( GL_QUADS );
		glTexCoord2f(0,0);	glVertex2f(0,0);
		glTexCoord2f(1,0);	glVertex2f(1,0);
		glTexCoord2f(1,1);	glVertex2f(1,1);
		glTexCoord2f(0,1);	glVertex2f(0,1);
	glEnd();
	if (lod > 0) glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, 0 );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glDisable( GL_TEXTURE_2D );
	glBindTexture( GL_TEXTURE_2D, 0 );
	glPopAttrib();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();
}

GLuint FrameBuffer::GetTextureID( int attachment )
{
	if (attachment == FBO_DEPTH) return depthID;
	if (attachment == FBO_STENCIL) return stencilID;
	return GetColorTextureID( attachment - FBO_COLOR0 );
}

GLenum FrameBuffer::GetTextureType( int attachment )
{
	if (attachment == FBO_DEPTH) return depthType;
	if (attachment == FBO_STENCIL) return stencilType;
	assert( attachment-FBO_COLOR0 < maxColorBuffers );
	return colorType[attachment-FBO_COLOR0];
}


// This resizes a framebuffer object that already exists and has textures
//      already associated with it.
void FrameBuffer::ResizeExistingFBO( int newWidth, int newHeight )
{
	GLint format;

	// Resize the color buffers
	for ( int i = 0; i < maxColorBuffers; i++)
	{
		if (colorIDs[i] > 0)
		{
			glBindTexture( colorType[i], colorIDs[i] );
			glGetTexLevelParameteriv( colorType[i], 0, GL_TEXTURE_INTERNAL_FORMAT, &format );	
			if (colorType[i] == GL_TEXTURE_2D)
				glTexImage2D(GL_TEXTURE_2D, 0, format, newWidth, newHeight, 0, GL_RGBA, GL_FLOAT, NULL);
			else if (colorType[i] == GL_TEXTURE_2D_ARRAY_EXT)
				glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, format, newWidth, newHeight, depth, 0, GL_RGBA, GL_FLOAT, NULL);
		}
	}	

	if (depthID > 0)
	{
		glBindTexture( GL_TEXTURE_2D, depthID );
		glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &format );	
		glTexImage2D(GL_TEXTURE_2D, 0, format, newWidth, newHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	}

	if (stencilID > 0)
	{
		printf("**** Error:  Called FrameBuffer::ResizeExistingFBO() on FBO with stencil\n");
		printf("             buffer.  Resizing FBOs with stencil bufs is not supported yet!\n");
		exit(0);
	}

	glBindTexture( GL_TEXTURE_2D, 0 );
	width = newWidth; 
	height = newHeight;

	CheckFramebufferStatus( 1 );
}


void FrameBuffer::DrawBuffers( int cBuf1, int cBuf2, int cBuf3, int cBuf4 )
{
	GLenum bufs[4] = { cBuf1-FBO_COLOR0+GL_COLOR_ATTACHMENT0_EXT, 
		               cBuf2-FBO_COLOR0+GL_COLOR_ATTACHMENT0_EXT, 
					   cBuf3-FBO_COLOR0+GL_COLOR_ATTACHMENT0_EXT, 
					   cBuf4-FBO_COLOR0+GL_COLOR_ATTACHMENT0_EXT };
	int bufsToUse = (cBuf1+cBuf2+cBuf3+cBuf4)/20;
	glDrawBuffers( bufsToUse, bufs );
}


	
void FrameBuffer::SaveColorImage( int attachment, char *filename )
{
	// Note we will capture RGB (not RGBA) seeing as PPMs only support 3 channels per pixel.
	unsigned char *frameData = (unsigned char *)malloc( width * height * 3 * sizeof( unsigned char ) );
	if (!frameData)
	{
		fprintf( stderr, "***Error: Unable to allocate temporary memory during frame capture!\n");
		exit(0);
	}

	// Select the correct buffer to read from, then read from it.
	//    Note this read happens *without* permanently changing the state of the read buffer!
	glPushAttrib( GL_PIXEL_MODE_BIT );
	glReadBuffer( GL_COLOR_ATTACHMENT0_EXT+(attachment-FBO_COLOR0) );
	glReadPixels( 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frameData );
	glPopAttrib();

	FrameToPPM( filename, frameData, width, height ); 

	free( frameData );
}

void FrameBuffer::SaveDepthImage( char *filename )
{
	if ( !(includedBuffers & GL_DEPTH_BUFFER_BIT) ) 
		return;

	unsigned char *frameData = (unsigned char *)malloc( width * height * sizeof( unsigned char ) );
	if (!frameData)
	{
		fprintf( stderr, "***Error: Unable to allocate temporary memory during frame capture!\n");
		exit(0);
	}

	glReadPixels( 0, 0, width, height, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, frameData );
	FrameToPGM( filename, frameData, width, height ); 

	free( frameData );
}

void FrameBuffer::SaveStencilImage( char *filename )
{
	if ( !(includedBuffers & GL_STENCIL_BUFFER_BIT) ) 
		return;

	unsigned char *frameData = (unsigned char *)malloc( width * height * sizeof( unsigned char ) );
	if (!frameData)
	{
		fprintf( stderr, "***Error: Unable to allocate temporary memory during frame capture!\n");
		exit(0);
	}

	glReadPixels( 0, 0, width, height, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, frameData );
	FrameToPGM( filename, frameData, width, height ); 

	free( frameData );
}


void FrameBuffer::FrameToPGM( char *f, unsigned char *data, int width, int height )
{
  FILE *out = fopen(f, "wb");
  if (!out) 
  {
	  fprintf( stderr, "***Error: Unable to capture frame.  fopen() failed!!\n");
	  return;
  }
  
  fprintf(out, "P5\n# File captured by Chris Wyman's OpenGL framegrabber\n");  
  fprintf(out, "%d %d\n", width, height);
  fprintf(out, "%d\n", 255); 
  
  for ( int y = height-1; y >= 0; y-- )
	  fwrite( data+(y*width), 1, width, out );

  fprintf(out, "\n");
  fclose(out);
}

void FrameBuffer::FrameToPPM( char *f, unsigned char *data, int width, int height )
{
  FILE *out = fopen(f, "wb");
  if (!out) 
  {
	  fprintf( stderr, "***Error: Unable to capture frame.  fopen() failed!!\n");
	  return;
  }
  
  fprintf(out, "P6\n# File captured by Chris Wyman's OpenGL framegrabber\n");  
  fprintf(out, "%d %d\n", width, height);
  fprintf(out, "%d\n", 255); 
  
  for ( int y = height-1; y >= 0; y-- )
	  fwrite( data+(3*y*width), 1, 3*width, out );

  fprintf(out, "\n");
  fclose(out);
}



