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

#include "videoReaderOpenCV.h"

// Our standard Error() message prototype
void Error( char *formatStr, char *insertThisStr );


OCVVideoReader::OCVVideoReader()
{
	// Setup a blank video reader
	videoOpened = false;
	frameReadBase = 0;
	currentFrame = 0;
}


OCVVideoReader::~OCVVideoReader()
{
	// Before deleting, make sure we close the video
	if( videoOpened )
		CloseVideo();
}


void OCVVideoReader::OpenVideo( char* szFile, int frameBase )
{
	// If we're re-opening, first close
	if( videoOpened )
		CloseVideo();

	// Grab the specified video
	inputVideo = cvCaptureFromAVI( szFile );

	// Double check that worked
	if (!inputVideo)
	{
		Error("OCVVideoReader::OpenVideo() failed to open '%s'!\n", szFile);
		exit(0);
	}

	// All right, now gram the video properties
	lastframe  = (int)cvGetCaptureProperty( inputVideo, CV_CAP_PROP_FRAME_COUNT );
	width      = (int)cvGetCaptureProperty( inputVideo, CV_CAP_PROP_FRAME_WIDTH );
	height     = (int)cvGetCaptureProperty( inputVideo, CV_CAP_PROP_FRAME_HEIGHT );  
	mpf        = 1000 / ((int)cvGetCaptureProperty( inputVideo, CV_CAP_PROP_FPS ));

	// Print some debuffing info for now
	printf ("OpenCV Opened '%s': Width: %d, Height: %d, Frames: %d\n", szFile, width, height, lastframe);

	// Set some parameters for later
	videoOpened = true;
	frameReadBase = frameBase;
	currentFrame = 0;
}



void OCVVideoReader::CloseVideo()
{
	// Release the video and note that it is closed
	cvReleaseCapture( &inputVideo );
	videoOpened = false;
}


unsigned char* OCVVideoReader::ReadArbitraryFrame( int frameNum )
{
	/*
	if( !videoOpened )
		return NULL;

	lpbi = (LPBITMAPINFOHEADER)AVIStreamGetFrame(pgf, frameReadBase+frameNum);	// Grab Data From The AVI Stream
	pdata=(char *)lpbi+lpbi->biSize+lpbi->biClrUsed * sizeof(RGBQUAD);			// Pointer To Data Returned By AVIStreamGetFrame																				// (Skip The Header Info To Get To The Data)
	return (unsigned char*)pdata;
	*/

	return 0;
}


unsigned char* OCVVideoReader::ReadInitialFrame( int startFrame )
{
	/*
	printf("Opened: %d\n", aviOpened?1:0);
	if( !videoOpened )
		return NULL;

	currentFrame = startFrame;
	lpbi = (LPBITMAPINFOHEADER)AVIStreamGetFrame(pgf, frameReadBase+currentFrame);	// Grab Data From The AVI Stream
	printf("%d %d\n", lpbi->biWidth, lpbi->biHeight );
	pdata=(char *)lpbi+lpbi->biSize+lpbi->biClrUsed * sizeof(RGBQUAD);				// Pointer To Data Returned By AVIStreamGetFrame
	return (unsigned char*)pdata;
	*/

	return 0;
}

unsigned char* OCVVideoReader::ReadNextFrame( void )
{
	if( !videoOpened )
		return NULL;

	currentFrame = (currentFrame + 1) % (lastframe - frameReadBase);
	cvGrabFrame( inputVideo );
	curFrameImg = cvRetrieveFrame( inputVideo );

	// print some debugging data...
	printf("Image size: (%d x %d) = %d bytes\n", curFrameImg->width, curFrameImg->height, curFrameImg->imageSize );
	printf("Image depth: %d = IPL_DEPTH_8U ? %d\n", curFrameImg->depth, curFrameImg->depth == IPL_DEPTH_8U ? 1 : 0 );

	printf("%d %d %d, %d %d %d, %d %d %d\n", curFrameImg->imageData[0], curFrameImg->imageData[1], curFrameImg->imageData[2],
		curFrameImg->imageData[3], curFrameImg->imageData[4], curFrameImg->imageData[5],
		curFrameImg->imageData[6], curFrameImg->imageData[7], curFrameImg->imageData[8] );

	return (unsigned char*)curFrameImg->imageData;
}

