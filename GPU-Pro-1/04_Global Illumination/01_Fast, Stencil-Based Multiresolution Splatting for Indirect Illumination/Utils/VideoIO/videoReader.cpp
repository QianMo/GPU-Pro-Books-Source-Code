/**********************************************
** videoReader.vpp                           **
** ---------------                           **
**                                           **
** A simplistic video reader using VFW to    **
**    open and grab RGB frames from a video  **
**    stream.  Thanks to Greg Nichols for a  **
**    initial implementation of this class.  **
**                                           **
** Chris Wyman (05/11/2009)                  **
**********************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "videoReader.h"

// Our standard Error() message prototype
void Error( char *formatStr, char *insertThisStr );


VideoReader::VideoReader()
{
	aviOpened = false;
	frameReadBase = 0;
	currentFrame = 0;
	milliSec = 0;
}


VideoReader::~VideoReader()
{
	if( aviOpened )
		CloseAVI();
}


void VideoReader::OpenAVI( char* szFile, int frameBase )
{
	if( aviOpened )
		CloseAVI();
	else
		AVIFileInit();									// Opens The AVIFile Library

	// Opens The AVI Stream
	if (AVIStreamOpenFromFile(&pavi, szFile, streamtypeVIDEO, 0, OF_READ, NULL) !=0)
		Error("VideoReader::OpenAVI() failed to open '%s'!\n", szFile);

	AVIStreamInfo(pavi, &psi, sizeof(psi));						// Reads Information About The Stream Into psi
	width=psi.rcFrame.right-psi.rcFrame.left;					// Width Is Right Side Of Frame Minus Left
	height=psi.rcFrame.bottom-psi.rcFrame.top;					// Height Is Bottom Of Frame Minus Top
	lastframe=AVIStreamLength(pavi);							// The Last Frame Of The Stream
	mpf=AVIStreamSampleToTime(pavi,lastframe)/lastframe;		// Calculate Rough Milliseconds Per Frame
	milliSec = mpf/1000.0;

	pgf=AVIStreamGetFrameOpen(pavi, NULL);				// Create The PGETFRAME Using Our Request Mode
	if (pgf==NULL)
	{
		Error("VideoReader::OpenAVI() failed to open AVI frame in '%s'!\n", szFile);
		exit(0);
	}

	// Information For The Title Bar (Width / Height / Last Frame)
	//printf ("NeHe's AVI Player: Width: %d, Height: %d, Frames: %d\n", width, height, lastframe);

	aviOpened = true;
	frameReadBase = frameBase;
	currentFrame = 0;
}



void VideoReader::CloseAVI()
{
	AVIStreamGetFrameClose(pgf);			// Deallocates The GetFrame Resources
	AVIStreamRelease(pavi);					// Release The Stream
	AVIFileExit();							// Release The File
	aviOpened = false;
}


unsigned char* VideoReader::ReadArbitraryFrame( int frameNum )
{
	if( !aviOpened )
		return NULL;

	lpbi = (LPBITMAPINFOHEADER)AVIStreamGetFrame(pgf, frameReadBase+frameNum);	// Grab Data From The AVI Stream
	pdata=(char *)lpbi+lpbi->biSize+lpbi->biClrUsed * sizeof(RGBQUAD);			// Pointer To Data Returned By AVIStreamGetFrame																				// (Skip The Header Info To Get To The Data)
	return (unsigned char*)pdata;
}


unsigned char* VideoReader::ReadInitialFrame( int startFrame )
{
	if( !aviOpened )
		return NULL;

	currentFrame = startFrame;
	lpbi = (LPBITMAPINFOHEADER)AVIStreamGetFrame(pgf, frameReadBase+currentFrame);	// Grab Data From The AVI Stream
	pdata=(char *)lpbi+lpbi->biSize+lpbi->biClrUsed * sizeof(RGBQUAD);				// Pointer To Data Returned By AVIStreamGetFrame
	return (unsigned char*)pdata;
}

unsigned char* VideoReader::ReadNextFrame( void )
{
	if( !aviOpened )
		return NULL;

	currentFrame = (currentFrame + 1) % (lastframe - frameReadBase);
	lpbi = (LPBITMAPINFOHEADER)AVIStreamGetFrame(pgf, frameReadBase+currentFrame);	// Grab Data From The AVI Stream
	pdata=(char *)lpbi+lpbi->biSize+lpbi->biClrUsed * sizeof(RGBQUAD);				// Pointer To Data Returned By AVIStreamGetFrame
	return (unsigned char*)pdata;
}

