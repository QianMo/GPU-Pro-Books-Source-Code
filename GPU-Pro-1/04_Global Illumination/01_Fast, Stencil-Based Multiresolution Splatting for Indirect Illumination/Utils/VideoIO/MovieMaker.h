/**************************************************************************
** MovieMaker.h                                                          **
** ------------                                                          **
**                                                                       **
** This header defines a class for utilizing Video-for-Windows to        **
**   capture video files of your OpenGL programs.  Please note that as   **
**   this utilizes vfw, it *only* runs under Windows, unlike most of my  **
**   utility classes!   At this time, I do not have video capture code   **
**   that works under Linux.  Sorry.  This still compiles under Linux.   **
**                                                                       **
** This code has been more-or-less directly stolen from an nVidia SDK    **
**   demo program.  At this point, I forget which one.  I have modified  **
**   it to make usage a bit cleaner, more flexible, and less prone to    **
**   crashing, however it is still a fairly brittle interface, and it is **
**   not unusual to create non-standard video files (for instance, I had **
**   trouble recently using them in Adobe Premier).                      **
**                                                                       **
** Initialization:                                                       **
**   MovieMaker *m = new MovieMaker();                                   **
**                                                                       **
** Begin Capture:                                                        **
**   m->StartCapture( "myVideoFile.avi", videoFileFrameRate );           **
**                                                                       **
** Capture Each Frame:                                                   **
**   m->AddCurrentFrame();                                               **
**                                                                       **
** Finish Capture:                                                       **
**   m->EndCapture();                                                    **
**                                                                       **
**                                                                       **
** Chris Wyman (1/30/2008)                                               **
**************************************************************************/



#ifndef _MOVIEMAKER_H
#define _MOVIEMAKER_H

// You should be able to include this in non-Windows code, since this
//    #ifdef will essentially comment the whole class out!
#ifdef WIN32

#include <windows.h>
#include <vfw.h>

#define BUFSIZE 260

// Define the MovieMaker class...
class MovieMaker {
public:
	// Initialize the class and necessary internal variables
	MovieMaker();
    ~MovieMaker();

	// Start capturing a new video with a given filename and specific
	//    framerate.  Unfortunately, you cannot give a per-frame length,
	//    which means every frame in the video will be the same length.
	void StartCapture(const char *name, int framesPerSecond=30 );

	// Add the current frame to the video.
	inline bool AddCurrentFrame( void ) { return Snap(); }

	// Finish up capturing.  Close the file.  Other cleanup.
	void EndCapture( void );

	// Check if the StartCapture() worked correctly.
    inline bool IsOK( void ) const { return bOK; }

	// Returns the number of frames in the currently captured movie.
	inline int GetFrameCount( void ) const { return nFrames; }


/***************************************************************************/
/*                            PRIVATE STUFF                                */
/* --   should not need to modify, unless you know what you're doing!   -- */ 
/***************************************************************************/
public:
    char fname[64];
    int width, height, nFrames, ready, estFramesPerSecond;
	WORD wVer;

  	AVISTREAMINFO strhdr;
	PAVIFILE pfile;
	PAVISTREAM ps;
	PAVISTREAM psCompressed;
	PAVISTREAM psText;
	AVICOMPRESSOPTIONS opts;
	AVICOMPRESSOPTIONS FAR * aopts[1];
	DWORD dwTextFormat;
	char szText[BUFSIZE];
	bool bOK;

	HDC hdcScreen;
    HDC hdcCompatible; 
	HBITMAP hbmScreen; 

	bool Snap();
	void PrepareForCapture();
	HBITMAP LoadBMPFromFB( int w, int h );
	void PrintErrorMesage( void );
	bool CaptureError( char *errMsg, LPBITMAPINFOHEADER alpbi );
};


// #include <windows.h> pollutes the namespace (with MACROS!) clean up a bit
#undef CreateWindow
#undef AVIFileOpen

// We'll need to link in the library, too
#pragma comment(lib, "vfw32.lib")

// Define some macros used by the code
#define TEXT_HEIGHT	20
#define AVIIF_KEYFRAME	0x00000010L // this frame is a key frame.



// What happens if we're on a non-windows system?  
#else

// We're on a non-Windows system!  Ack!  We don't know how to capture video!!

// A dummy class that defines stubs for the methods to avoid compile errors.
class MovieMaker {
public:
	MovieMaker();
    ~MovieMaker();
	inline void StartCapture(const char *name, int framesPerSecond=30 ) {}
	inline bool AddCurrentFrame( void ) { return false; }
	inline void EndCapture( void ) {}
    inline bool IsOK( void ) const { return false; }
	inline int GetFrameCount( void ) const { return 0; }
};

#endif

#endif
