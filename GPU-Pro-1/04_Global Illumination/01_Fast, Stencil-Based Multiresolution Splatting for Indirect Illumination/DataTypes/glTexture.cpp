/******************************************************************/
/* Texture.cpp                                                    */
/* -----------------------                                        */
/*                                                                */
/* The file defines an image class that stores a texture.         */
/*     This is very similar to the image class, but also defines  */
/*     access patterns for the texture with interpolation.        */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "glTexture.h"
#include "Utils/ImageIO/imageIO.h"


GLTexture::GLTexture( int width, int height, int depth ) : width(width), 
	height(height), depth(depth), fileName(0), imgData(0), texID(0),
	initialized(false), internalImageType(TEXTURE_TYPE_UNKNOWN), usingMipmaps(false)
{
	name = strdup( "<Unnamed Texture>" );
}


GLTexture::GLTexture( char *filename, unsigned int flags, bool processLater ) 
{
	usingMipmaps = false;
	if ( (flags & TEXTURE_MIN_LINEAR_MIP_LINEAR) || (flags & TEXTURE_MIN_LINEAR_MIP_NEAR) ||
		 (flags & TEXTURE_MIN_NEAR_MIP_LINEAR)   || (flags & TEXTURE_MIN_NEAR_MIP_NEAR) )
		 usingMipmaps = true;

	fileName = strdup( filename );
	name = strdup( "<Unnamed Texture>" );

	// Identify the type of file
	char *ptr = strrchr( filename, '.' );
	char buf[16];
	strncpy( buf, ptr, 16 );
	for (int i=0;i<16;i++)
		buf[i] = tolower( buf[i] );
	if (!strcmp(buf, ".ppm"))			internalImageType = TEXTURE_TYPE_PPM;
	else if (!strcmp(buf, ".rgb"))		internalImageType = TEXTURE_TYPE_RGB;
	else if (!strcmp(buf, ".rgba"))		internalImageType = TEXTURE_TYPE_RGBA;
	else if (!strcmp(buf, ".hdr"))		internalImageType = TEXTURE_TYPE_HDR;
	else if (!strcmp(buf, ".bmp"))		internalImageType = TEXTURE_TYPE_BMP;

	// All these functions have *fatal errors* given bad files.
	if (internalImageType == TEXTURE_TYPE_PPM) LoadPPM( fileName );
	else if (internalImageType == TEXTURE_TYPE_RGB) LoadRGB( fileName );
	else if (internalImageType == TEXTURE_TYPE_RGBA) LoadRGB( fileName );
	else if (internalImageType == TEXTURE_TYPE_BMP) LoadBMP( fileName );
	else if (internalImageType == TEXTURE_TYPE_HDR) 
	{
		printf("***Error: GLTexture class currently does not support .hdr files!\n");
		exit(0);
	}

	// At this point (due to fatal errors, above) we know we have a valid texture.
	glPixelFormat = ( internalImageType != TEXTURE_TYPE_RGBA ? GL_RGB : GL_RGBA );

	// The only type currently allowed...
	glTextureType  = GL_TEXTURE_2D;

	// Generate a GL structure for this texture
	if (!processLater)
	{
		glGenTextures( 1, &texID );
		glBindTexture( glTextureType, texID );
		switch (glTextureType) {
			case GL_TEXTURE_1D:
				glTexImage1D( glTextureType, 0, glPixelFormat, width, 0, 
							  glPixelFormat, glPixelStorage, imgData );
				break;
			case GL_TEXTURE_2D:
				if (!usingMipmaps)
					glTexImage2D( glTextureType, 0, (glPixelFormat==GL_RGB ? GL_COMPRESSED_RGB_S3TC_DXT1_EXT : GL_COMPRESSED_RGBA_S3TC_DXT1_EXT), 
							  width, height, 0, glPixelFormat, glPixelStorage, imgData );
				else
					gluBuild2DMipmaps( glTextureType, (glPixelFormat==GL_RGB ? GL_COMPRESSED_RGB_S3TC_DXT1_EXT : GL_COMPRESSED_RGBA_S3TC_DXT1_EXT), 
							  width, height, glPixelFormat, glPixelStorage, imgData );
				break;
			case GL_TEXTURE_3D:
				glTexImage3D( glTextureType, 0, glPixelFormat, width, height, depth, 0, 
							  glPixelFormat, glPixelStorage, imgData );
				break;
			default:
				printf("***Error: Unhandled texture type encountered when loading '%s'!\n", fileName );
				exit(0);
				break;
		}
	}
	else if ( (glTextureType != GL_TEXTURE_1D) && 
			  (glTextureType != GL_TEXTURE_2D) && 
			  (glTextureType != GL_TEXTURE_3D) )
	{
		printf("***Error: Unhandled texture type encountered when loading '%s'!\n", fileName );
			exit(0);
	}
	
	minFilter = GL_LINEAR; 
	if (flags & TEXTURE_MIN_NEAREST)                  minFilter = GL_NEAREST;
	else if (flags & TEXTURE_MIN_NEAR_MIP_NEAR)       minFilter = GL_NEAREST_MIPMAP_NEAREST;
	else if (flags & TEXTURE_MIN_NEAR_MIP_LINEAR)     minFilter = GL_NEAREST_MIPMAP_LINEAR;
	else if (flags & TEXTURE_MIN_LINEAR_MIP_NEAR)     minFilter = GL_LINEAR_MIPMAP_NEAREST;
	else if (flags & TEXTURE_MIN_LINEAR_MIP_LINEAR)   minFilter = GL_LINEAR_MIPMAP_LINEAR;
	if (!processLater) glTexParameteri( glTextureType, GL_TEXTURE_MIN_FILTER, minFilter );

	magFilter = GL_LINEAR;
	if (flags & TEXTURE_MAG_NEAREST)                  magFilter = GL_NEAREST;
    if (!processLater) glTexParameteri( glTextureType, GL_TEXTURE_MAG_FILTER, magFilter );

	sWrap = GL_CLAMP;
	if (flags & TEXTURE_REPEAT_S)                     sWrap = GL_REPEAT;
	else if (flags & TEXTURE_MIRROR_REPEAT_S)         sWrap = GL_MIRRORED_REPEAT;
	else if (flags & TEXTURE_CLAMP_TO_BORDER_S)       sWrap = GL_CLAMP_TO_BORDER;
	else if (flags & TEXTURE_CLAMP_TO_EDGE_S)         sWrap = GL_CLAMP_TO_EDGE;
	if (!processLater) glTexParameteri( glTextureType, GL_TEXTURE_WRAP_S, sWrap );

	if ( glTextureType != GL_TEXTURE_1D )
	{
		tWrap = GL_CLAMP;
		if (flags & TEXTURE_REPEAT_T)                     tWrap = GL_REPEAT;
		else if (flags & TEXTURE_MIRROR_REPEAT_T)         tWrap = GL_MIRRORED_REPEAT;
		else if (flags & TEXTURE_CLAMP_TO_BORDER_T)       tWrap = GL_CLAMP_TO_BORDER;
		else if (flags & TEXTURE_CLAMP_TO_EDGE_T)         tWrap = GL_CLAMP_TO_EDGE;
		if (!processLater) glTexParameteri( glTextureType, GL_TEXTURE_WRAP_T, tWrap );
	}

	// Currently there's no parameters defined for other types of wrap behavior in R.
	//    if this is a problem you should define some.
	if ( glTextureType == GL_TEXTURE_3D )
	{
		if (!processLater) glTexParameteri( glTextureType, GL_TEXTURE_WRAP_R, rWrap=GL_CLAMP );
	}

	if (!processLater) 
		initialized = true;
	else
		initialized = false;
}


GLTexture::~GLTexture()
{
	if (fileName) free(fileName);
	if (imgData) free(imgData);
	glDeleteTextures( 1, &texID );
}


void GLTexture::Preprocess( void )
{
	if (initialized) return;

	// We're going to try to compress the texture in memory, if possible.
	GLenum compressedFormat = glPixelFormat==GL_RGB ? 
                      GL_COMPRESSED_RGB_S3TC_DXT1_EXT : 
			          GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;

    // Check if it's possible
	if ( (width%4 !=0) || (height%4 !=0) )
		compressedFormat = glPixelFormat;

	glGenTextures( 1, &texID );
	glBindTexture( glTextureType, texID );
	if (glTextureType == GL_TEXTURE_1D)
		glTexImage1D( glTextureType, 0, glPixelFormat, width, 0, 
					  glPixelFormat, glPixelStorage, imgData );
	else if (glTextureType == GL_TEXTURE_2D)
		if (!usingMipmaps)
			glTexImage2D( glTextureType, 0, compressedFormat, 
			          width, height, 0, glPixelFormat, glPixelStorage, imgData );
		else
			gluBuild2DMipmaps( glTextureType, compressedFormat, 
					  width, height, glPixelFormat, glPixelStorage, imgData );
	else if (glTextureType == GL_TEXTURE_3D)
		glTexImage3D( glTextureType, 0, glPixelFormat, width, height, depth, 0, 
					  glPixelFormat, glPixelStorage, imgData );

	glTexParameteri( glTextureType, GL_TEXTURE_MIN_FILTER, minFilter );
	glTexParameteri( glTextureType, GL_TEXTURE_MAG_FILTER, magFilter );
	glTexParameteri( glTextureType, GL_TEXTURE_WRAP_S, sWrap );
	if (glTextureType!=GL_TEXTURE_1D) 
		glTexParameteri( glTextureType, GL_TEXTURE_WRAP_T, tWrap );
	if (glTextureType==GL_TEXTURE_3D)
		glTexParameteri( glTextureType, GL_TEXTURE_WRAP_R, rWrap=GL_CLAMP );

	initialized = true;
}


void GLTexture::LoadRGB( char *filename )
{
	int components;
	imgData = (void *)ReadRGB(filename, &width, &height, &components, true);
	glPixelStorage = GL_UNSIGNED_BYTE; 

	if (!imgData)
	{
		printf("Error in Texture::LoadRGB(): Unknown error in ReadRGB! (NULL return)\n");
		exit(0);
	}
}


void GLTexture::LoadPPM( char *filename )
{
	int mode;
	imgData = (void *)ReadPPM( filename, &mode, &width, &height, true );
	glPixelStorage = GL_UNSIGNED_BYTE; 

	if (!imgData)
	{
		printf("Error in Texture::LoadPPM(): Unknown error in ReadPPM! (NULL return)\n");
		exit(0);
	}
}


void GLTexture::LoadBMP( char *filename )
{
	imgData = (void *)ReadBMP( filename, &width, &height, true );
	glPixelStorage = GL_UNSIGNED_BYTE; 

	if (!imgData)
	{
		printf("Error in Texture::LoadBMP(): Unknown error in ReadBMP! (NULL return)\n");
		exit(0);
	}
}

