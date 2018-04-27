/******************************************************************************

 @File         PVRTTexture.cpp

 @Title        PVRTTexture

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Texture loading.

******************************************************************************/
#include <string.h>
#include <stdlib.h>

#include "PVRTTexture.h"

/*****************************************************************************
** Functions
*****************************************************************************/

/*!***************************************************************************
@Function		PVRTTextureLoadTiled
@Modified		pDst			Texture to place the tiled data
@Input			nWidthDst		Width of destination texture
@Input			nHeightDst		Height of destination texture
@Input			pSrc			Texture to tile
@Input			nWidthSrc		Width of source texture
@Input			nHeightSrc		Height of source texture
@Input 			nElementSize	Bytes per pixel
@Input			bTwiddled		True if the data is twiddled
@Description	Needed by PVRTTextureTile() in the various PVRTTextureAPIs
*****************************************************************************/
void PVRTTextureLoadTiled(
	PVRTuint8		* const pDst,
	const unsigned int	nWidthDst,
	const unsigned int	nHeightDst,
	const PVRTuint8	* const pSrc,
	const unsigned int	nWidthSrc,
	const unsigned int	nHeightSrc,
	const unsigned int	nElementSize,
	const bool			bTwiddled)
{
	unsigned int nXs, nYs;
	unsigned int nXd, nYd;
	unsigned int nIdxSrc, nIdxDst;

	for(nIdxDst = 0; nIdxDst < nWidthDst*nHeightDst; ++nIdxDst)
	{
		if(bTwiddled)
		{
			PVRTTextureDeTwiddle(nXd, nYd, nIdxDst);
		}
		else
		{
			nXd = nIdxDst % nWidthDst;
			nYd = nIdxDst / nWidthDst;
		}

		nXs = nXd % nWidthSrc;
		nYs = nYd % nHeightSrc;

		if(bTwiddled)
		{
			PVRTTextureTwiddle(nIdxSrc, nXs, nYs);
		}
		else
		{
			nIdxSrc = nYs * nWidthSrc + nXs;
		}

		memcpy(pDst + nIdxDst*nElementSize, pSrc + nIdxSrc*nElementSize, nElementSize);
	}
}

/*!***************************************************************************
@Function		PVRTTextureCreate
@Input			w			Size of the texture
@Input			h			Size of the texture
@Input			wMin		Minimum size of a texture level
@Input			hMin		Minimum size of a texture level
@Input			nBPP		Bits per pixel of the format
@Input			bMIPMap		Create memory for MIP-map levels also?
@Return			Allocated texture memory (must be free()d)
@Description	Creates a PVR_Texture_Header structure, including room for
				the specified texture, in memory.
*****************************************************************************/
PVR_Texture_Header *PVRTTextureCreate(
	unsigned int		w,
	unsigned int		h,
	const unsigned int	wMin,
	const unsigned int	hMin,
	const unsigned int	nBPP,
	const bool			bMIPMap)
{
	size_t			len;
	unsigned char	*p;

	len = 0;
	do
	{
		len += PVRT_MAX(w, wMin) * PVRT_MAX(h, hMin);
		w >>= 1;
		h >>= 1;
	}
	while(bMIPMap && (w || h));

	len = (len * nBPP) / 8;
	len += sizeof(PVR_Texture_Header);

	p = (unsigned char*)malloc(len);
	_ASSERT(p);
	return (PVR_Texture_Header*)p;
}


/*!***************************************************************************
 @Function		PVRTTextureTwiddle
 @Output		a	Twiddled value
 @Input			u	Coordinate axis 0
 @Input			v	Coordinate axis 1
 @Description	Combine a 2D coordinate into a twiddled value
*****************************************************************************/
void PVRTTextureTwiddle(unsigned int &a, const unsigned int u, const unsigned int v)
{
	_ASSERT(!((u|v) & 0xFFFF0000));
	a = 0;
	for(int i = 0; i < 16; ++i)
	{
		a |= ((u & (1 << i)) << (i+1));
		a |= ((v & (1 << i)) << (i+0));
	}
}

/*!***************************************************************************
 @Function		PVRTTextureDeTwiddle
 @Output		u	Coordinate axis 0
 @Output		v	Coordinate axis 1
 @Input			a	Twiddled value
 @Description	Extract 2D coordinates from a twiddled value.
*****************************************************************************/
void PVRTTextureDeTwiddle(unsigned int &u, unsigned int &v, const unsigned int a)
{
	u = 0;
	v = 0;
	for(int i = 0; i < 16; ++i)
	{
		u |= (a & (1 << ((2*i)+1))) >> (i+1);
		v |= (a & (1 << ((2*i)+0))) >> (i+0);
	}
}

/*****************************************************************************
 End of file (PVRTTexture.cpp)
*****************************************************************************/
