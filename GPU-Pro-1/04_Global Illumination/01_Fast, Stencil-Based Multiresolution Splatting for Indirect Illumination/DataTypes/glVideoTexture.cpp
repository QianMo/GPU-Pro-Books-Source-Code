/******************************************************************/
/* glVideoTexture.cpp                                             */
/* -----------------------                                        */
/*                                                                */
/* The file defines an image class that stores and easily         */
/*     displays in OpenGL a texture fed that is by a video stream */
/*                                                                */
/* Chris Wyman (05/11/2009)                                       */
/******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "stencilMultiResSplatting.h"
#include "glVideoTexture.h"
#include "Utils/ImageIO/imageIO.h"


GLVideoTexture::GLVideoTexture( char *filename, float fps, unsigned int flags ):
	GLTexture()
{
	// 
	secPerFrame = 1.0/fps;

	// Load the video file, and set parameters based upon the video reader 
	reader = new VideoReader();
	reader->OpenAVI( filename );
	width = reader->GetWidth();
	height = reader->GetHeight();
	internalImageType = TEXTURE_TYPE_PPM;
	glPixelFormat = GL_RGB;
	glTextureType = GL_TEXTURE_2D;
	glPixelStorage = GL_UNSIGNED_BYTE; 
	texSize = width*height*3*1;
	this->fileName = strdup( filename );
	lastUpdated = -1;
	
	// Setup filtering and wrapping preferences based on the input flags
	minFilter = GL_LINEAR; 
	if (flags & TEXTURE_MIN_NEAREST)                  minFilter = GL_NEAREST;
	else if (flags & TEXTURE_MIN_NEAR_MIP_NEAR)       minFilter = GL_NEAREST_MIPMAP_NEAREST;
	else if (flags & TEXTURE_MIN_NEAR_MIP_LINEAR)     minFilter = GL_NEAREST_MIPMAP_LINEAR;
	else if (flags & TEXTURE_MIN_LINEAR_MIP_NEAR)     minFilter = GL_LINEAR_MIPMAP_NEAREST;
	else if (flags & TEXTURE_MIN_LINEAR_MIP_LINEAR)   minFilter = GL_LINEAR_MIPMAP_LINEAR;
	magFilter = GL_LINEAR;
	if (flags & TEXTURE_MAG_NEAREST)                  magFilter = GL_NEAREST;
	sWrap = GL_CLAMP;
	if (flags & TEXTURE_REPEAT_S)                     sWrap = GL_REPEAT;
	else if (flags & TEXTURE_MIRROR_REPEAT_S)         sWrap = GL_MIRRORED_REPEAT;
	else if (flags & TEXTURE_CLAMP_TO_BORDER_S)       sWrap = GL_CLAMP_TO_BORDER;
	else if (flags & TEXTURE_CLAMP_TO_EDGE_S)         sWrap = GL_CLAMP_TO_EDGE;
	tWrap = GL_CLAMP;
	if (flags & TEXTURE_REPEAT_T)                     tWrap = GL_REPEAT;
	else if (flags & TEXTURE_MIRROR_REPEAT_T)         tWrap = GL_MIRRORED_REPEAT;
	else if (flags & TEXTURE_CLAMP_TO_BORDER_T)       tWrap = GL_CLAMP_TO_BORDER;
	else if (flags & TEXTURE_CLAMP_TO_EDGE_T)         tWrap = GL_CLAMP_TO_EDGE;

	// Determine if this video texture will be mipmapped
	usingMipmaps = false;
	if ( (flags & TEXTURE_MIN_LINEAR_MIP_LINEAR) || (flags & TEXTURE_MIN_LINEAR_MIP_NEAR) ||
		 (flags & TEXTURE_MIN_NEAR_MIP_LINEAR)   || (flags & TEXTURE_MIN_NEAR_MIP_NEAR) )
		 usingMipmaps = true;

	// Unlike the static textures, we're going to assume Preprocess() will be called!
	initialized = false;
}

GLVideoTexture::~GLVideoTexture()
{
	imgData = 0;  // imgData isn't owned by us and ~GLTexture() tries to delete it...
	glDeleteBuffers( 1, &texBuffer );
	glDeleteTextures( 1, &texID );
	reader->CloseAVI();
}

void GLVideoTexture::Preprocess( void )
{
	if (initialized) return;

	// Grab the first video frame
	imgData = (void *)reader->ReadInitialFrame( 100 );
	
	// Bind the texture
	glGenTextures( 1, &texID );
	glBindTexture( glTextureType, texID );

	// Create our texture / mipmap
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
	glTexImage2D( glTextureType, 0, glPixelFormat, 
		          width, height, 0, glPixelFormat, glPixelStorage, NULL );

	// Setup the texture parameters
	glTexParameteri( glTextureType, GL_TEXTURE_MIN_FILTER, minFilter );
	glTexParameteri( glTextureType, GL_TEXTURE_MAG_FILTER, magFilter );
	glTexParameteri( glTextureType, GL_TEXTURE_WRAP_S, sWrap );
	glTexParameteri( glTextureType, GL_TEXTURE_WRAP_T, tWrap );
	glTexParameteri( glTextureType, GL_GENERATE_MIPMAP, usingMipmaps ? GL_TRUE : GL_FALSE );

	// Setup our streaming texture buffer
	glGenBuffers( 1, &texBuffer );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, texBuffer );
	glBufferData( GL_PIXEL_UNPACK_BUFFER, texSize, NULL, GL_STREAM_DRAW );
	bufferMemory = glMapBuffer( GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY );
	memcpy( bufferMemory, imgData, texSize );
	glUnmapBuffer( GL_PIXEL_UNPACK_BUFFER );

	// Stream the first frame into texture memory
	glTexSubImage2D( glTextureType, 0, 0, 0, width, height, GL_BGR, glPixelStorage, BUFFER_OFFSET(0) );
	if (usingMipmaps) glGenerateMipmapEXT( glTextureType );

	// Unbind the buffer and texture
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
	glBindTexture( glTextureType, 0 );

	//printf("preprocessing area light: min: %x, mag: %x\n", minFilter, magFilter);

	// Let everyone know we're ready to go!
	initialized = true;
}

void GLVideoTexture::Update( void )
{
	// Bind the correct textures and buffers
	glBindTexture( glTextureType, texID );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, texBuffer );
	
	// Grab the updated texture frame
	imgData = (void *)reader->ReadNextFrame();
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, texBuffer );
	//glBufferData( GL_PIXEL_UNPACK_BUFFER, texSize, NULL, GL_STREAM_DRAW );
	bufferMemory = glMapBuffer( GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY );
	memcpy( bufferMemory, imgData, texSize );
	glUnmapBuffer( GL_PIXEL_UNPACK_BUFFER );

	// Stream the updated frame into texture memory
	glTexSubImage2D( glTextureType, 0, 0, 0, width, height, glPixelFormat, glPixelStorage, BUFFER_OFFSET(0) );
	if (usingMipmaps) glGenerateMipmapEXT( glTextureType );

	// Unbind the texture and buffer
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
	glBindTexture( glTextureType, 0 );

}

void GLVideoTexture::Update( float frameTime )
{
	// Make sure we're not updating the frame too quickly.
	//    Ideally, we'd make sure it's not too slow...  But I ignore that for now.
	if (frameTime < secPerFrame+lastUpdated) 
		return;
	lastUpdated = frameTime;

	// Bind the correct textures and buffers
	glBindTexture( glTextureType, texID );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, texBuffer );
	
	// Grab the updated texture frame
	imgData = (void *)reader->ReadNextFrame();
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, texBuffer );
	//glBufferData( GL_PIXEL_UNPACK_BUFFER, texSize, NULL, GL_STREAM_DRAW );
	bufferMemory = glMapBuffer( GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY );
	memcpy( bufferMemory, imgData, texSize );
	glUnmapBuffer( GL_PIXEL_UNPACK_BUFFER );

	// Stream the updated frame into texture memory
	glTexSubImage2D( glTextureType, 0, 0, 0, width, height, GL_BGR, glPixelStorage, BUFFER_OFFSET(0) );
	if (usingMipmaps) glGenerateMipmapEXT( glTextureType );

	// Unbind the texture and buffer
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
	glBindTexture( glTextureType, 0 );

}

