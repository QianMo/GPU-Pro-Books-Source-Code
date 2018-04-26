/***************************************************************************/
/* framegrab.h                                                             */
/* ------------                                                            */
/*                                                                         */
/* This is a simple class for grabbing frames from the OpenGL framebuffer. */
/*     In basic usage, to grab an image displayed on the screen:           */
/*                                                                         */
/*     Initialization:                                                     */
/*          FrameGrab *grab = new FrameGrab();                             */
/*                                                                         */
/*     In display() routine right before glutSwapBuffers():                */
/*          grab->CaptureFrame();                                          */
/*                                                                         */
/* This outputs an image "screenCaptureXXX.ppm", where the first capture   */
/*     has a name screenCapture0.ppm, and it is incremented for every      */
/*     additional capture during the program's execution.                  */
/*                                                                         */
/* The base name ("screenCapture") can be changed based on the value given */
/*     to the constructor.  Giving an explicit filename to CaptureFrame()  */
/*     overrides this behavior and outputs a one-time (i.e., no number)    */
/*     PPM file with the name specified.                                   */
/*                                                                         */
/* Some of these assumptions can be changed by varying CaptureFrame calls. */
/*                                                                         */
/* Chris Wyman (12/4/2007)                                                 */
/***************************************************************************/

#ifndef __FRAMEGRAB_H
#define __FRAMEGRAB_H

class FrameGrab
{
private:
	GLenum captureBuffer;
	int nextFrameNum;
	char *baseName;

	// Actually grabs the data from the scene.  These function encapsulates
	//    _all_ the OpenGL code.  
	unsigned char *GrabWholeFrame( int *capturedWidth, int *capturedHeight );
	unsigned char *GrabFrameRegion( int left, int bottom, int right, int top );

	// Saves data with a certain width and height to the specified file.  
	//    This is the function containing all the explicit I/O code.  If you 
	//    want to output a different file type, this is the function to update.
	void FrameToPPM( char *f, unsigned char *data, int width, int height );

	// Some formats (depth & stencil) only need a grayscale image.  This func is used.
	void FrameToPGM( char *f, unsigned char *data, int width, int height );
public:
	// Setup the frame-grabber.  Images will be output as "<baseFileName><frameNumber>.ppm"
	FrameGrab( char *baseFileName="screenCapture" );
	~FrameGrab();

	// Actually capture the current frame
	void CaptureFrame( void );
	void CaptureFrame( char *outputFilename );
	void CaptureFrameAsFloat( char *outputFilename );

	// Capture a region of the frame (values specified in pixels, origin in lower left)
	void CaptureFrameRegion( int left, int bottom, int right, int top );

	// Change the buffer captured by CaptureFrame()
	inline void SetCaptureBuffer( GLenum buffer ) { captureBuffer=buffer; }

	// Capture the stencil buffer of the current frame
	void CaptureStencil( char *outputFilename );

	// Capture the depth buffer of the current frame
	void CaptureDepth( char *outputFilename );

	// Sometimes you'd like to know what the next file will be named.  This tells you.
	//    A string must be passed in, along with a maximal array size.
	void GetFilenameForNextFrame( char *nextName, int maxSize );
};


#endif