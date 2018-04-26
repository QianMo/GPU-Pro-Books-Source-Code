/**********************************************
** videoReader.h                             **
** -------------                             **
**                                           **
** A simplistic video reader using VFW to    **
**    open and grab RGB frames from a video  **
**    stream.  Thanks to Greg Nichols for a  **
**    initial implementation of this class.  **
**                                           **
** Chris Wyman (05/11/2009)                  **
**********************************************/


#ifndef __VID_READER_H
#define __VID_READER_H

#include <windows.h>
#include <vfw.h>

class VideoReader
{
public:
	// Simplistic constructor and destructor.
	VideoReader();
	~VideoReader();

	// Open and close the video stream
	void OpenAVI( char* aviFile, int frameBase = 0 );
	void CloseAVI();

	// Can we read from this video?
	inline bool IsValid( void ) const			{ return aviOpened; }

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
	inline int GetWidth( void )  const			  { return width; }
	inline int GetHeight( void ) const			  { return height; }

	// Get the video stream's total frame count.
	inline int GetFrameCount( void ) const        { return lastframe; }

	inline float GetSecondsPerFrame( void ) const { return milliSec; }

private:
	int					currentFrame;               // Current frame for ReadInitialFrame/ReadNextFrame

	int					frameReadBase;              // Offset into the file stream specified to OpenAVI
	bool				aviOpened;                  // Has OpenAVI been called?

	AVISTREAMINFO		psi;						// Pointer To A Structure Containing Stream Info
	PAVISTREAM			pavi;						// Handle To An Open Stream
	PGETFRAME			pgf;						// Pointer To A GetFrame Object
	LPBITMAPINFOHEADER  lpbi;                       // Holds The Bitmap Header Information
	BITMAPINFOHEADER	bmih;						// Header Information For DrawDibDraw Decoding
	long				lastframe;					// Last Frame Of The Stream
	int					width;						// Video Width
	int					height;						// Video Height
	char				*pdata;						// Pointer To Texture Data
	int					mpf;						// Will Hold Rough Milliseconds Per Frame
	float               milliSec;					//   (same, only as float fractions of a second)
};


#endif