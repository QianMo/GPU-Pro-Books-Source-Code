/******************************************************************/
/* glVideoTexture.h                                               */
/* -----------------------                                        */
/*                                                                */
/* The file defines an image class that stores and easily         */
/*     displays in OpenGL a texture fed that is by a video stream */
/*                                                                */
/* Chris Wyman (05/11/2009)                                       */
/******************************************************************/

#ifndef VIDEOTEXTURE_H
#define VIDEOTEXTURE_H

#pragma warning( disable: 4996 )

#include <stdio.h>
#include <stdlib.h>
#include "Utils/GLee.h"
#include <GL/glut.h>
#include "glTexture.h"
#include "Utils/VideoIO/videoReader.h"
//#include "Utils/VideoIO/videoReaderOpenCV.h"

class GLVideoTexture : public GLTexture
{
private:
	//OCVVideoReader *reader;
	VideoReader *reader;
	GLuint texBuffer;
	GLuint texSize;
	void *bufferMemory;
	float lastUpdated;
	float secPerFrame;
public:
	//GLVideoTexture( int width=-1, int height=-1, int depth=-1 );
    GLVideoTexture( char *filename, float fps=30.0, unsigned int flags=0 );
    virtual ~GLVideoTexture();

	// Unfortunately, the constructor may not be able to set everything up if
	//   it is called before OpenGL is initialized.  In this case, we need to do
	//   a later "preprocess" pass after GL has been initialized.
	virtual void Preprocess( void );
	
	// Does this texture need updates?  (Yep!  It's a video, stupid!)
	virtual void Update( void );
	virtual void Update( float frameTime );
	virtual bool NeedsUpdates( void )						{ return true; }
};



#endif
