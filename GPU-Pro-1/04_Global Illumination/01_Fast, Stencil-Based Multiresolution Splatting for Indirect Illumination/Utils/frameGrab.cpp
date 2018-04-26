/***************************************************************************/
/* framegrab.cpp                                                           */
/* ------------                                                            */
/*                                                                         */
/* This is a simple class for grabbing frames from the OpenGL framebuffer. */
/*                                                                         */  
/* (See the header for more useful usage information.)                     */
/*                                                                         */
/* Chris Wyman (12/4/2007)                                                 */
/***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// If you are not using GLEW, feel free to remove the first include
//    It is here because GLEW must be included before GLUT when it is used.
#include "Utils/GLee.h"
#include <GL/glut.h>
#include "framegrab.h"

// Visual Studio 2005 arbitrarily decided all sorts of standard library calls
//    are obsolete, making this essential to avoid pages and pages of useless
//    warnings.
#pragma warning( disable: 4996 )



FrameGrab::FrameGrab( char *baseFileName ): nextFrameNum(0)
{
	captureBuffer = GL_BACK;
	baseName = strdup( baseFileName );
}

FrameGrab::~FrameGrab()
{
	free( baseName );
}

void FrameGrab::CaptureFrame( void )
{
	char outputFile[512];
	int width, height;
	unsigned char *frameData = GrabWholeFrame( &width, &height );
	sprintf( outputFile, "%s%d.ppm", baseName, nextFrameNum++ );
	FrameToPPM( outputFile, frameData, width, height );
}

void FrameGrab::CaptureFrame( char *outputFilename ) 
{ 
	int width, height;
	unsigned char *frameData = GrabWholeFrame( &width, &height );
	FrameToPPM( outputFilename, frameData, width, height ); 
}

void FrameGrab::CaptureFrameAsFloat( char *outputFilename )
{
	int capturedWidth = glutGet( GLUT_WINDOW_WIDTH );
	int capturedHeight = glutGet( GLUT_WINDOW_HEIGHT );
	float *frameData = (float *)malloc( capturedWidth * capturedHeight * 4 * sizeof( float ) );
	if (!frameData) {
		fprintf( stderr, "***Error: Unable to allocate temporary memory during frame capture!\n");
		exit(0);
	}
	glPushAttrib( GL_PIXEL_MODE_BIT );
	glReadBuffer( captureBuffer );
	glReadPixels( 0, 0, capturedWidth, capturedHeight, GL_RGBA, GL_FLOAT, frameData );
	glPopAttrib();

	FILE *out = fopen( outputFilename, "wb");
	if (!out) {
	  fprintf( stderr, "***Error: Unable to capture frame.  fopen() failed!!\n");
	  return;
	}

	int count=0;
	fprintf(out, "%d %d\n", capturedWidth, capturedHeight);
	for (int j=0; j<capturedHeight; j++)
		for (int i=0; i<capturedWidth; i++)
		{
			fprintf( out, "%.8f, %.8f, %.8f, %.8f\n", 
				frameData[count], frameData[count+1], frameData[count+2], frameData[count+3] );
			count+= 4;
		}
	fclose( out );
}

void FrameGrab::CaptureFrameRegion( int left, int bottom, int right, int top )
{
	char outputFile[512];
	unsigned char *frameData = GrabFrameRegion( left, bottom, right, top );
	sprintf( outputFile, "%s%d.ppm", baseName, nextFrameNum++ );
	FrameToPPM( outputFile, frameData, abs(right-left), abs(top-bottom) );
}

void FrameGrab::GetFilenameForNextFrame( char *nextName, int maxSize )
{
	if (!nextName) return;
	char outputFile[512];
	sprintf( outputFile, "%s%d.ppm", baseName, nextFrameNum );
	strncpy( nextName, outputFile, maxSize );
}

unsigned char *FrameGrab::GrabFrameRegion( int left, int bottom, int right, int top )
{
	unsigned char *frameData = (unsigned char *)malloc( abs(right-left) * abs(top-bottom) * 3 * sizeof( unsigned char ) );
	if (!frameData)
	{
		fprintf( stderr, "***Error: Unable to allocate temporary memory during frame capture!\n");
		exit(0);
	}

	// Select the correct buffer to read from, then read from it.
	//    Note this read happens *without* permanently changing the state of the read buffer!
	glPushAttrib( GL_PIXEL_MODE_BIT );
	glReadBuffer( captureBuffer );
	glReadPixels( left, bottom, right-left, top-bottom, GL_RGB, GL_UNSIGNED_BYTE, frameData );
	glPopAttrib();

	return frameData;
}


unsigned char *FrameGrab::GrabWholeFrame( int *capturedWidth, int *capturedHeight )
{
	*capturedWidth = glutGet( GLUT_WINDOW_WIDTH );
	*capturedHeight = glutGet( GLUT_WINDOW_HEIGHT );

	// Note we will capture RGB (not RGBA) seeing as PPMs only support 3 channels per pixel.
	unsigned char *frameData = (unsigned char *)malloc( (*capturedWidth) * (*capturedHeight) * 3 * sizeof( unsigned char ) );
	if (!frameData)
	{
		fprintf( stderr, "***Error: Unable to allocate temporary memory during frame capture!\n");
		exit(0);
	}

	// Select the correct buffer to read from, then read from it.
	//    Note this read happens *without* permanently changing the state of the read buffer!
	glPushAttrib( GL_PIXEL_MODE_BIT );
	glReadBuffer( captureBuffer );
	glReadPixels( 0, 0, *capturedWidth, *capturedHeight, GL_RGB, GL_UNSIGNED_BYTE, frameData );
	glPopAttrib();

	return frameData;
}

void FrameGrab::CaptureStencil( char *outputFilename )
{
	int width = glutGet( GLUT_WINDOW_WIDTH );
	int height = glutGet( GLUT_WINDOW_HEIGHT );

	unsigned char *frameData = (unsigned char *)malloc( width * height * sizeof( unsigned char ) );
	if (!frameData)
	{
		fprintf( stderr, "***Error: Unable to allocate temporary memory during frame capture!\n");
		exit(0);
	}

	glReadPixels( 0, 0, width, height, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, frameData );
	FrameToPGM( outputFilename, frameData, width, height ); 
}

void FrameGrab::CaptureDepth( char *outputFilename )
{
	int width = glutGet( GLUT_WINDOW_WIDTH );
	int height = glutGet( GLUT_WINDOW_HEIGHT );

	unsigned char *frameData = (unsigned char *)malloc( width * height * sizeof( unsigned char ) );
	if (!frameData)
	{
		fprintf( stderr, "***Error: Unable to allocate temporary memory during frame capture!\n");
		exit(0);
	}

	glReadPixels( 0, 0, width, height, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, frameData );
	FrameToPGM( outputFilename, frameData, width, height ); 
}


void FrameGrab::FrameToPGM( char *f, unsigned char *data, int width, int height )
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

void FrameGrab::FrameToPPM( char *f, unsigned char *data, int width, int height )
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

