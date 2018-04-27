/******************************************************************************

 @File         CPVRTextureData.h

 @Title        Console Log

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI

 @Description  Class to hold the data part of a PVR texture.

******************************************************************************/
#ifndef CPVRTEXTUREDATA_H
#define CPVRTEXTUREDATA_H

#include <stdio.h>
#include "PVRTexLibGlobals.h"
#include "PVRTGlobal.h"
#include "Pixel.h"

namespace pvrtexlib
{
	
#ifdef __APPLE__
	/* The classes below are exported */
#pragma GCC visibility push(default)
#endif

class PVR_DLL CPVRTextureHeader;

class PVR_DLL CPVRTextureData
{
public:


	// channel enums
	enum{
		CHAN_A=0,
		CHAN_R,
		CHAN_G,
		CHAN_B,
		CHAN_RGB,
		CHAN_ARGB
	};


	/*******************************************************************************
	* Constructor
	* Description		: Blank constructor that allows the creation of a 
	*						valid but empty CPVRTextureData instance
	*******************************************************************************/
	CPVRTextureData();
	/*******************************************************************************
	* Constructor
	* Description		: Constructor that initialises an empty buffer to the size passed
	*******************************************************************************/
	CPVRTextureData(size_t stDataSize);
	/*******************************************************************************
	* Constructor
	* Description		: Constructor initialised to the data and size of data passed
	*******************************************************************************/
	CPVRTextureData(uint8 *pTextureData, size_t stDataSize);

	/*******************************************************************************
	* Constructor
	* Description		: Copy constructor
	*******************************************************************************/
	CPVRTextureData(const CPVRTextureData& original);
	/*******************************************************************************
	* Constructor
	* Description		: Assignment operator
	*******************************************************************************/
	CPVRTextureData& operator=(const CPVRTextureData& sData);

	/*******************************************************************************
	* Destructor
	*******************************************************************************/
	~CPVRTextureData();

	/*******************************************************************************
	* Function Name  : getData
	* In/Outputs	  : 
	* Description    : returns pointer to the data buffer in this instance
	*******************************************************************************/
	uint8* getData() const;

	/*******************************************************************************
	* Function Name  : getDataSize
	* In/Outputs	  : 
	* Description    : returns the recorded size of the data buffer in this instance
	*******************************************************************************/
	size_t getDataSize() const;

	/*******************************************************************************
	* Function Name  : clear
	* In/Outputs	  : 
	* Description    : releases any data stored and sets size to 0
	*******************************************************************************/
	void clear();

	/*******************************************************************************
	* Function Name  : ResizeBuffer
	* In/Outputs	  : stNewSize:		new size for the buffer
	* Description    : Resizes the buffer held by this texture. Copies across
	*					existing data to the new buffer, truncating it if the new
	*					buffer is smaller than the last.
	*******************************************************************************/
	void resizeBuffer(size_t stNewSize);
	/*******************************************************************************
	* Function Name  : convertToPrecMode
	* Description    : Converts the data held by the buffer to the new precision
	*					expanding the buffer.
	*					If the data is not a standard data type PVRTHROWS
	*******************************************************************************/
	void convertToPrecMode(const E_PRECMODE ePrecMode, CPVRTextureHeader& sHeader);

	/*******************************************************************************
	* Function Name  : WriteToFile
	* Description    : Writes the data in this instance to the passed file. Returns
	*					the number of bytes written.
	*******************************************************************************/
	size_t			writeToFile(FILE* const fFile)const;
	/*******************************************************************************
	* Function Name  : writeToIncludeFile
	* Description    : Writes the data in this instance to the passed file as a series
	*					of unsigned integers compatible with a C++ header. Returns
	*					the number of bytes written.
	*******************************************************************************/
	size_t			writeToIncludeFile(FILE* const fFile)const;
	/*******************************************************************************
	* Function Name  : append
	* Description    : will append the data from newData to the existing data in this
	*					CPVRTextureData instance. It may be wise to check if the 
	*					pixel types of these two instances match. PVRTHROWs if
	*					unsuccessful.
	*******************************************************************************/
	void			append(const CPVRTextureData& newData);

	/*******************************************************************************
	* Function Name  : RGBToAlpha
	* Description    : Dumps the maximum value of the RGB values in the current data
						into the alpha channel. Leaves the other channels untouched.
						Needs uint8 or PVRTHROWs
	*******************************************************************************/
	void			maxRGBToAlpha(	const unsigned int u32SurfaceNum,
									const CPVRTextureHeader& psInputHeader,
									const CPVRTextureData& sAlphaData);

	/*******************************************************************************
	* Function Name  : SwapChannels
	* Description    : Swaps the two specified channels.
	*					Data should be in a standard format
	*******************************************************************************/
	template<class tType>
	void SwapChannels(const E_COLOUR_CHANNEL eChannelA, const E_COLOUR_CHANNEL eChannelB)
	{
		Pixel<tType>* pixelData = (Pixel<tType>*)m_pData;
		unsigned int numPixels = m_stDataSize/(4*sizeof(tType));
		for(unsigned int i=0 ; i < numPixels ;++i)
		{
			PVRTswap<tType>((*pixelData)[eChannelA],(*pixelData)[eChannelB]);
			++pixelData;
		}
	}

	/*******************************************************************************
	* Function Name  : ClearChannel
	* Description    : Clears the specified channel to white
	*******************************************************************************/
	void ClearChannel(const CPVRTextureHeader& sHeader,const unsigned int u32Channel);

	private:
	uint8			*m_pData;			// pointer to image data
	size_t			m_stDataSize;		// size of data in bytes

};
	
#ifdef __APPLE__
#pragma GCC visibility pop
#endif

}

#endif // CPVRTEXTUREDATA_H

/*****************************************************************************
 End of file (CPVRTextureData.h)
*****************************************************************************/

