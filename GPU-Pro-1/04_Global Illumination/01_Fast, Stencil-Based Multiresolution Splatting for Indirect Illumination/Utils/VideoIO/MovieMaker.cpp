/*************************************************
** MovieMaker.cpp                               **
** -----------                                  **
**                                              **
** Code for capturing movies from the GL window **
**    This is from an nVidia demo, and it only  **
**    works under Windows systems.              **
**                                              **
** A bit messy, and I don't really understand   **
**    it all anyways, since I haven't looked at **
**    VFW manuals.                              **
**                                              **
** Chris Wyman (9/07/2006)                      **
*************************************************/

#ifdef WIN32

//#define _CRT_SECURE_NO_WARNINGS

#include "MovieMaker.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <windowsx.h>
#include <GL/glut.h>

HANDLE  MakeDib( HBITMAP hbitmap, UINT bits );
HBITMAP LoadBMPFromFB( int w, int h );


MovieMaker::MovieMaker()
{
    sprintf( fname, "movie.avi" );
    width = height = -1;
    bOK = true;
    nFrames = 0;
	ready = 1;
  	pfile = NULL;
	ps = psCompressed = psText = NULL;
	aopts[0] = &opts;

    // Check VFW version.
	wVer = HIWORD( VideoForWindowsVersion() );
	if ( wVer < 0x010A ) fprintf( stderr, "MovieMaker Error: VFW version is too old!\n" );
	else AVIFileInit();
}

void MovieMaker::PrepareForCapture( void )
{
	/* make sure everything from the last capture is clear */
	if (ps) AVIStreamClose(ps);
	if (psCompressed) AVIStreamClose(psCompressed);
	if (psText) AVIStreamClose(psText);
	if (pfile) AVIFileClose(pfile);
	if (wVer >= 0x010A) AVIFileExit();

	/* reset error bounds & screen res */
	bOK = true;
	ready = 1;
	nFrames = 0;
	estFramesPerSecond = 30;
	width = height = -1;

	/* reset internals */
	pfile = NULL;
	ps = psCompressed = psText = NULL;
	aopts[0] = &opts;

	/* initialize a new AVI file */
	if (wVer >= 0x010A)	AVIFileInit();
}


MovieMaker::~MovieMaker()
{
	if (ps) AVIStreamClose(ps);
	if (psCompressed) AVIStreamClose(psCompressed);
	if (psText) AVIStreamClose(psText);
	if (pfile) AVIFileClose(pfile);
	if (wVer >= 0x010A)	AVIFileExit();
}

void MovieMaker::StartCapture( const char *name, int framesPerSecond )
{
	if (wVer < 0x010A) return;
	if (!ready) PrepareForCapture();

    // Get the width and height.
    width = glutGet( GLUT_WINDOW_WIDTH );
    height = glutGet( GLUT_WINDOW_HEIGHT );

	strcpy( fname, name );
    fprintf( stderr, "(+) Starting %d x %d video capture to: %s\n", width, height, fname );
	estFramesPerSecond = framesPerSecond;
	ready = 0;

	hdcScreen = wglGetCurrentDC();
    hdcCompatible = CreateCompatibleDC( hdcScreen ); 
	hbmScreen = CreateCompatibleBitmap( hdcScreen, width, height ); 

	//printf("pfile: %d, name: %s\n", pfile, fname);
	//int output = AVIFileOpenA(&pfile, fname, OF_WRITE | OF_CREATE, NULL);
	//printf("pfile: %d, name: %s, ret: %d\n", pfile, fname, output);
	//printf("%d %d %d %d\n", AVIERR_BADFORMAT, AVIERR_MEMORY, AVIERR_FILEREAD, AVIERR_FILEOPEN );
}

void MovieMaker::EndCapture()
{
	if (ps)				AVIStreamClose(ps);
 	if (psCompressed)	AVIStreamClose(psCompressed);
	if (psText)			AVIStreamClose(psText);
 	if (pfile)			AVIFileClose(pfile);
	if (wVer >= 0x010A) AVIFileExit(); 

	ps = psCompressed = psText = NULL;
	pfile = NULL;
}

bool MovieMaker::Snap( void )
{
	// Make sure everyting is OK.  Otherwise, errors and crashes galore!
	if (!bOK) return false;

    // Get an image and stuff it into a bitmap.
    HBITMAP bmp = LoadBMPFromFB( width, height );
	LPBITMAPINFOHEADER alpbi = (LPBITMAPINFOHEADER)GlobalLock(MakeDib(bmp, 32));
    DeleteObject( bmp );

	// Check if the device-independent bitmap creation failed.
	if (!alpbi) return (bOK = false);

	// Make sure the width and height didn't get modified in the process.
	if ( ( width>=0 && width != alpbi->biWidth ) ||
		 ( height>=0 && height != alpbi->biHeight ) )
		return CaptureError( "DIB width/height changed!", alpbi );

	// Maybe this is the first image we've captured....
	if (nFrames == 0)
	{
		// If the AVIFileOpenA gives you compile errors, you can try using
		//    AVIFileOpen(), or AVIFileOpenW(), though I have found that 
		//    AVIFileOpenW gives me errors, and I explicitly #undef'd the
		//    AVIFileOpen macro in the header to avoid messing with this
		//    call.     -Chris
		if (AVIFileOpenA(&pfile, fname, OF_WRITE | OF_CREATE, NULL)) 
			return CaptureError( "Unable to open video file!", alpbi );

		_fmemset(&strhdr, 0, sizeof(strhdr));
		strhdr.fccType                = streamtypeVIDEO;	// stream type
		strhdr.fccHandler             = 0;
		strhdr.dwScale                = 1;					// frames per second = dwRate / dwScale
		strhdr.dwRate                 = estFramesPerSecond; 
		strhdr.dwSuggestedBufferSize  = alpbi->biSizeImage;
		SetRect(&strhdr.rcFrame, 0, 0,						// rectangle for stream
			(int) alpbi->biWidth,
			(int) alpbi->biHeight);

		// Create the stream
		if (AVIFileCreateStream(pfile, &ps, &strhdr))
			return CaptureError( "AVIFileCreateStream() failed!", alpbi );

		_fmemset(&opts, 0, sizeof(opts));

		if (!AVISaveOptions(NULL, 0, 1, &ps, (LPAVICOMPRESSOPTIONS FAR *) &aopts))
			return CaptureError( "AVISaveOptions() failed!", alpbi );
		if (AVIMakeCompressedStream(&psCompressed, ps, &opts, NULL) != AVIERR_OK)
			return CaptureError( "AVIMakeCompressedStream() failed!", alpbi );
		if (AVIStreamSetFormat(psCompressed, 0, alpbi, alpbi->biSize + alpbi->biClrUsed * sizeof(RGBQUAD)))
			return CaptureError( "AVIStreamSetFormat() failed!", alpbi );
	}

	// Now actual writing
	if(AVIStreamWrite(psCompressed, nFrames, 1,
 		                (LPBYTE) alpbi + alpbi->biSize + alpbi->biClrUsed * sizeof(RGBQUAD),
						alpbi->biSizeImage, AVIIF_KEYFRAME, NULL, NULL))
		CaptureError( "AVIStreamWrite() failed!", alpbi );

	// We've successfully added one more frame.  Cleanup, increment frame count
	GlobalFreePtr(alpbi);
	nFrames++;
	return true;
}

static HANDLE  MakeDib( HBITMAP hbitmap, UINT bits )
{
	HANDLE              hdib ;
	HDC                 hdc ;
	BITMAP              bitmap ;
	UINT                wLineLen ;
	DWORD               dwSize ;
	DWORD               wColSize ;
	LPBITMAPINFOHEADER  lpbi ;
	LPBYTE              lpBits ;
	
	GetObject(hbitmap,sizeof(BITMAP),&bitmap) ;

	//
	// DWORD align the width of the DIB
	// Figure out the size of the colour table
	// Calculate the size of the DIB
	//
	wLineLen = (bitmap.bmWidth*bits+31)/32 * 4;
	wColSize = sizeof(RGBQUAD)*((bits <= 8) ? 1<<bits : 0);
	dwSize = sizeof(BITMAPINFOHEADER) + wColSize +
		(DWORD)(UINT)wLineLen*(DWORD)(UINT)bitmap.bmHeight;

	//
	// Allocate room for a DIB and set the LPBI fields
	//
	hdib = GlobalAlloc(GHND,dwSize);
	if (!hdib)
		return hdib ;

	lpbi = (LPBITMAPINFOHEADER)GlobalLock(hdib) ;

	lpbi->biSize = sizeof(BITMAPINFOHEADER) ;
	lpbi->biWidth = bitmap.bmWidth ;
	lpbi->biHeight = bitmap.bmHeight ;
	lpbi->biPlanes = 1 ;
	lpbi->biBitCount = (WORD) bits ;
	lpbi->biCompression = BI_RGB ;
	lpbi->biSizeImage = dwSize - sizeof(BITMAPINFOHEADER) - wColSize ;
	lpbi->biXPelsPerMeter = 0 ;
	lpbi->biYPelsPerMeter = 0 ;
	lpbi->biClrUsed = (bits <= 8) ? 1<<bits : 0;
	lpbi->biClrImportant = 0 ;

	//
	// Get the bits from the bitmap and stuff them after the LPBI
	//
	lpBits = (LPBYTE)(lpbi+1)+wColSize ;

	hdc = CreateCompatibleDC(NULL) ;

	GetDIBits(hdc,hbitmap,0,bitmap.bmHeight,lpBits,(LPBITMAPINFO)lpbi, DIB_RGB_COLORS);

	// Fix this if GetDIBits messed it up....
	lpbi->biClrUsed = (bits <= 8) ? 1<<bits : 0;

	DeleteDC(hdc) ;
	GlobalUnlock(hdib);

	return hdib ;
}


void MovieMaker::PrintErrorMesage( void )
{  
	LPVOID lpMsgBuf;
	DWORD dw = GetLastError(); 
	
	FormatMessage(
			FORMAT_MESSAGE_ALLOCATE_BUFFER | 
			FORMAT_MESSAGE_FROM_SYSTEM |
			FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			dw,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			(LPTSTR) &lpMsgBuf,
			0, NULL );
    printf("Failed with error %d: %s\n", dw, lpMsgBuf); 
    LocalFree(lpMsgBuf);

}

HBITMAP MovieMaker::LoadBMPFromFB( int w, int h )
{
	SelectObject(hdcCompatible, hbmScreen);
	BitBlt( hdcCompatible, 0,0, w, h, 
            hdcScreen, 0, 0, SRCCOPY);
    return( hbmScreen );
}


bool MovieMaker::CaptureError( char *errMsg, LPBITMAPINFOHEADER alpbi )
{
	fprintf( stderr, "    (-) MovieMaker Error: %s\n", errMsg );
	GlobalFreePtr(alpbi);
	return (bOK = false);
}


#endif