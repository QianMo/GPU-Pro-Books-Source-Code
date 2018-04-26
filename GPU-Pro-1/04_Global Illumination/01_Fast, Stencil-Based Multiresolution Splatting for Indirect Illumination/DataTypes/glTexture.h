/******************************************************************/
/* Texture.h                                                      */
/* -----------------------                                        */
/*                                                                */
/* The file defines an image class that stores a texture.         */
/*     This is very similar to the image class, but also defines  */
/*     access patterns for the texture with interpolation.        */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/

#ifndef TEXTURE_H
#define TEXTURE_H

#pragma warning( disable: 4996 )

#include <stdio.h>
#include <stdlib.h>
#include "Utils/GLee.h"
#include <GL/glut.h>


#define TEXTURE_TYPE_UNKNOWN  0
#define TEXTURE_TYPE_PPM      1
#define TEXTURE_TYPE_RGB      2
#define TEXTURE_TYPE_RGBA     3
#define TEXTURE_TYPE_HDR      4
#define TEXTURE_TYPE_BMP      5

class GLTexture
{
protected:
	char *fileName, *name;
	void *imgData;
    int width, height, depth, internalImageType;
	int glPixelFormat, glPixelStorage;
	int glTextureType;
	GLint minFilter, magFilter;
	GLint sWrap, tWrap, rWrap;
	GLuint texID;
	bool initialized, usingMipmaps;

	void LoadPPM( char *filename );
	void LoadRGB( char *filename );
	void LoadBMP( char *filename );
	//void LoadHDR( char *filename );
public:
	GLTexture( int width=-1, int height=-1, int depth=-1 );
    GLTexture( char *filename, unsigned int flags=0, bool processLater=false );
    virtual ~GLTexture();

	// Returns a boolean that notes if glTexImage*() or equivalent has been called
	inline bool IsValid() const    { return initialized; }

	// Unfortunately, the constructor may not be able to set everything up if
	//   it is called before OpenGL is initialized.  In this case, we need to do
	//   a later "preprocess" pass after GL has been initialized.
	virtual void Preprocess( void );

	// Does this texture need updates?  (Nope!  It's a static texture, stupid!)
	virtual void Update( void )								{}
	virtual void Update( float frameTime )                  {}
	virtual bool NeedsUpdates( void )						{ return false; }

	// Returns the GL handle for this texture
	inline GLint TextureID() const { return texID; }

	// Get size of the texture
	inline int GetWidth()  const   { return width;  }  // Valid for 1D, 2D, or 3D textures
	inline int GetHeight() const   { return height; }  // Valid for 2D or 3D textures
	inline int GetDepth()  const   { return depth;  }  // Valid for 3D textures
 
	// Return the filename (if any) of the texture
	inline char *GetFilename( void ) { return fileName; }

	// Set and/or get an internal, arbitrary process-specific texture name used for ID
	inline void SetName( char *newName ) { if (name) free(name); name = strdup( newName ); }
	inline char *GetName( void )		 { return name; }
};


// Flags to pass to the constructor that change the GL parameters of the texture.
#define TEXTURE_DEFAULT                                      0x00000
#define TEXTURE_MAG_NEAREST                                  0x00001
#define TEXTURE_MAG_LINEAR                                   0x00002
#define TEXTURE_REPEAT_S                                     0x00004
#define TEXTURE_MIRROR_REPEAT_S                              0x00008
#define TEXTURE_CLAMP_S                                      0x00010
#define TEXTURE_CLAMP_TO_EDGE_S                              0x00020
#define TEXTURE_CLAMP_TO_BORDER_S                            0x00040
#define TEXTURE_REPEAT_T                                     0x00080
#define TEXTURE_MIRROR_REPEAT_T				                 0x00100
#define TEXTURE_CLAMP_T                                      0x00200
#define TEXTURE_CLAMP_TO_EDGE_T                              0x00400
#define TEXTURE_CLAMP_TO_BORDER_T                            0x00800
#define TEXTURE_MIN_NEAREST                                  0x01000
#define TEXTURE_MIN_LINEAR                                   0x02000
#define TEXTURE_MIN_NEAR_MIP_NEAR                            0x04000
#define TEXTURE_MIN_NEAR_MIP_LINEAR                          0x08000
#define TEXTURE_MIN_LINEAR_MIP_NEAR                          0x10000
#define TEXTURE_MIN_LINEAR_MIP_LINEAR                        0x20000
#define TEXTURE_INTERNAL_RGB                                 0x40000
#define TEXTURE_INTERNAL_RGBA                                0x80000



#endif
