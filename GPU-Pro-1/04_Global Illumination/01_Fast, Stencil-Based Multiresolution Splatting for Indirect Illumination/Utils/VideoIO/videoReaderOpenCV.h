/**********************************************
** videoReaderOpenCV.h                       **
** -------------------                       **
**                                           **
** A simplistic video reader using OpenCV to **
**    open and grab RGB frames from a video  **
**    stream.                                **
**                                           **
** Chris Wyman (05/11/2009)                  **
**********************************************/


#ifndef __OPENCVVID_READER_H
#define __OPENCVVID_READER_H

#include "cv.h"
#include "highgui.h"

class OCVVideoReader
{
public:
	// Simplistic constructor and destructor.
	OCVVideoReader();
	~OCVVideoReader();

	// Open and close the video stream
	void OpenVideo( char* videoFile, int frameBase = 0 );
	void CloseVideo();

	// Can we read from this video?
	inline bool IsValid( void ) const			{ return videoOpened; }

	// Read a frame from the video sequence.  This can arbitrary (any 
	//    0 <= frameNum < GetFrameCount()) or starting from an initial, 
	//    arbitrary frame and incrementing via ReadNextFrame().  Note 
	//    that ReadNextFrame() performs wrapping automatically for 
	//    continuous video textures.  Later improvements may make the 
	//    wrapping behavior controllable.
	unsigned char* ReadArbitraryFrame( int frameNum );
	unsigned char* ReadInitialFrame( int startFrame );
	unsigned char* ReadNextFrame( void );

	// Get the video stream's width and height
	inline int GetWidth( void )  const			{ return width; }
	inline int GetHeight( void ) const			{ return height; }

	// Get the video stream's total frame count.
	inline int GetFrameCount( void ) const      { return lastframe; }

private:
	int					currentFrame;               // Current frame for ReadInitialFrame/ReadNextFrame

	int					frameReadBase;              // Offset into the file stream specified to OpenAVI
	bool				videoOpened;                // Has OpenVideo been called?

	CvCapture          *inputVideo;                 // OpenCV video stream
	long				lastframe;					// Total number of frames in the video
	int					width;						// Video Width
	int					height;						// Video Height
	IplImage           *curFrameImg;                // OpenCV frame structure
	char			   *frameData;					// Pointer To Texture Data
	int					mpf;						// Will Hold Rough Milliseconds Per Frame
};


#endif