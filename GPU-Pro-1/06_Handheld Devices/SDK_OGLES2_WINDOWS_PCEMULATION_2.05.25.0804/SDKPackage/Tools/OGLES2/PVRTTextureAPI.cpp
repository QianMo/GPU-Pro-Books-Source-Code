/******************************************************************************

 @File         PVRTTextureAPI.cpp

 @Title        OGLES2\PVRTTextureAPI

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  OGLES2 texture loading.

******************************************************************************/
#include <string.h>
#include <stdlib.h>

#include "PVRTContext.h"
#include "PVRTgles2Ext.h"
#include "PVRTTexture.h"
#include "PVRTTextureAPI.h"
#include "PVRTDecompress.h"
#include "PVRTFixedPoint.h"
#include "PVRTMatrix.h"
#include "PVRTMisc.h"
#include "PVRTResourceFile.h"

/*****************************************************************************
** Functions
****************************************************************************/


/*!***************************************************************************
@Function		PVRTTextureTile
@Modified		pOut		The tiled texture in system memory
@Input			pIn			The source texture
@Input			nRepeatCnt	Number of times to repeat the source texture
@Description	Allocates and fills, in system memory, a texture large enough
				to repeat the source texture specified number of times.
*****************************************************************************/
void PVRTTextureTile(
	PVR_Texture_Header			**pOut,
	const PVR_Texture_Header	* const pIn,
	const int					nRepeatCnt)
{
	unsigned int		nFormat = 0, nType = 0, nBPP, nSize, nElW = 0, nElH = 0;
	unsigned char		*pMmSrc, *pMmDst;
	unsigned int		nLevel;
	PVR_Texture_Header	*psTexHeaderNew;

	_ASSERT(pIn->dwWidth);
	_ASSERT(pIn->dwWidth == pIn->dwHeight);
	_ASSERT(nRepeatCnt > 1);

	switch(pIn->dwpfFlags & PVRTEX_PIXELTYPE)
	{
	case OGL_RGBA_5551:
		nFormat		= GL_UNSIGNED_SHORT_5_5_5_1;
		nType		= GL_RGBA;
		nElW		= 1;
		nElH		= 1;
		break;
	case OGL_RGBA_8888:
		nFormat		= GL_UNSIGNED_BYTE;
		nType		= GL_RGBA;
		nElW		= 1;
		nElH		= 1;
		break;
	case OGL_PVRTC2:
		nFormat		= GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
		nType		= 0;
		nElW		= 8;
		nElH		= 4;
		break;
	case OGL_PVRTC4:
		nFormat		= GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
		nType		= 0;
		nElW		= 4;
		nElH		= 4;
		break;
	}

	nBPP = PVRTTextureFormatGetBPP(nFormat, nType);
	nSize = pIn->dwWidth * nRepeatCnt;

	psTexHeaderNew	= PVRTTextureCreate(nSize, nSize, nElW, nElH, nBPP, true);
	*psTexHeaderNew	= *pIn;
	pMmDst	= (unsigned char*)psTexHeaderNew + sizeof(*psTexHeaderNew);
	pMmSrc	= (unsigned char*)pIn + sizeof(*pIn);

	for(nLevel = 0; ((unsigned int)1 << nLevel) < nSize; ++nLevel)
	{
		int nBlocksDstW = PVRT_MAX((unsigned int)1, (nSize >> nLevel) / nElW);
		int nBlocksDstH = PVRT_MAX((unsigned int)1, (nSize >> nLevel) / nElH);
		int nBlocksSrcW = PVRT_MAX((unsigned int)1, (pIn->dwWidth >> nLevel) / nElW);
		int nBlocksSrcH = PVRT_MAX((unsigned int)1, (pIn->dwHeight >> nLevel) / nElH);
		int nBlocksS	= nBPP * nElW * nElH / 8;

		PVRTTextureLoadTiled(
			pMmDst,
			nBlocksDstW,
			nBlocksDstH,
			pMmSrc,
			nBlocksSrcW,
			nBlocksSrcH,
			nBlocksS,
			(pIn->dwpfFlags & PVRTEX_TWIDDLE) ? true : false);

		pMmDst += nBlocksDstW * nBlocksDstH * nBlocksS;
		pMmSrc += nBlocksSrcW * nBlocksSrcH * nBlocksS;
	}

	psTexHeaderNew->dwWidth = nSize;
	psTexHeaderNew->dwHeight = nSize;
	psTexHeaderNew->dwMipMapCount = nLevel;
	*pOut = psTexHeaderNew;
}


/*!***************************************************************************
 @Function		PVRTTextureLoadFromPointer
 @Input			pointer				Pointer to header-texture's structure
 @Modified		texName				the OpenGL ES texture name as returned by glBindTexture
 @Modified		psTextureHeader		Pointer to a PVR_Texture_Header struct. Modified to
									contain the header data of the returned texture Ignored if NULL.
 @Input			bAllowDecompress	Allow decompression if PVRTC is not supported in hardware.
 @Input			nLoadFromLevel		Which mipmap level to start loading from (0=all)
 @Input			texPtr				If null, texture follows header, else texture is here.
 @Return		PVR_SUCCESS on success
 @Description	Allows textures to be stored in C header files and loaded in. Can load parts of a
				mipmaped texture (ie skipping the highest detailed levels).In OpenGL Cube Map, each
				texture's up direction is defined as next (view direction, up direction),
				(+x,-y)(-x,-y)(+y,+z)(-y,-z)(+z,-y)(-z,-y).
				Sets the texture MIN/MAG filter to GL_LINEAR_MIPMAP_NEAREST/GL_LINEAR
				if mipmaps are present, GL_LINEAR/GL_LINEAR otherwise.
*****************************************************************************/
EPVRTError PVRTTextureLoadFromPointer(	const void* pointer,
										GLuint *const texName,
										const void *psTextureHeader,
										bool bAllowDecompress,
										const unsigned int nLoadFromLevel,
										const void * const texPtr)
{
	PVR_Texture_Header* psPVRHeader = (PVR_Texture_Header*)pointer;
	unsigned int u32NumSurfs;

	// perform checks for old PVR psPVRHeader
	if(psPVRHeader->dwHeaderSize!=sizeof(PVR_Texture_Header))
	{	// Header V1
		if(psPVRHeader->dwHeaderSize==PVRTEX_V1_HEADER_SIZE)
		{	// react to old psPVRHeader: i.e. fill in numsurfs as this is missing from old header
			PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer warning: this is an old pvr"
				" - you can use PVRTexTool to update its header.\n");
			if(psPVRHeader->dwpfFlags&PVRTEX_CUBEMAP)
				u32NumSurfs = 6;
			else
				u32NumSurfs = 1;
		}
		else
		{	// not a pvr at all
			PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer failed: not a valid pvr.\n");
			return PVR_FAIL;
		}
	}
	else
	{	// Header V2
		if(psPVRHeader->dwNumSurfs<1)
		{	// encoded with old version of PVRTexTool before zero numsurfs bug found.
			if(psPVRHeader->dwpfFlags & PVRTEX_CUBEMAP)
				u32NumSurfs = 6;
			else
				u32NumSurfs = 1;
		}
		else
		{
			u32NumSurfs = psPVRHeader->dwNumSurfs;
		}
	}

	GLuint textureName;
	GLenum textureFormat = 0;
	GLenum textureType = GL_RGB;
	GLenum eTarget;

	bool IsPVRTCSupported = CPVRTgles2Ext::IsGLExtensionSupported("GL_IMG_texture_compression_pvrtc");
#ifndef TARGET_OS_IPHONE
	bool IsETCSupported = CPVRTgles2Ext::IsGLExtensionSupported("GL_OES_compressed_ETC1_RGB8_texture");
#endif
	*texName = 0;	// install warning value
	bool IsCompressedFormatSupported = false, IsCompressedFormat = false;

	/* Only accept untwiddled data UNLESS texture format is PVRTC */
	if ( ((psPVRHeader->dwpfFlags & PVRTEX_TWIDDLE) == PVRTEX_TWIDDLE)
		&& ((psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)!=OGL_PVRTC2)
		&& ((psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)!=OGL_PVRTC4) )
	{
		// We need to load untwiddled textures -- hw will twiddle for us.
		PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer failed: texture should be untwiddled.\n");
		return PVR_FAIL;
	}

	switch(psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)
	{
	case OGL_RGBA_4444:
		textureFormat = GL_UNSIGNED_SHORT_4_4_4_4;
		textureType = GL_RGBA;
		break;

	case OGL_RGBA_5551:
		textureFormat = GL_UNSIGNED_SHORT_5_5_5_1;
		textureType = GL_RGBA;
		break;

	case OGL_RGBA_8888:
		textureFormat = GL_UNSIGNED_BYTE;
		textureType = GL_RGBA;
		break;

	/* New OGL Specific Formats Added */

	case OGL_RGB_565:
		textureFormat = GL_UNSIGNED_SHORT_5_6_5;
		textureType = GL_RGB;
		break;

	case OGL_RGB_555:
		PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer failed: pixel type OGL_RGB_555 not supported.\n");
		return PVR_FAIL; // Deal with exceptional case

	case OGL_RGB_888:
		textureFormat = GL_UNSIGNED_BYTE;
		textureType = GL_RGB;
		break;

	case OGL_I_8:
		textureFormat = GL_UNSIGNED_BYTE;
		textureType = GL_LUMINANCE;
		break;

	case OGL_AI_88:
		textureFormat = GL_UNSIGNED_BYTE;
		textureType = GL_LUMINANCE_ALPHA;
		break;

	case MGLPT_PVRTC2:
	case OGL_PVRTC2:
		if(IsPVRTCSupported)
		{
			IsCompressedFormatSupported = IsCompressedFormat = true;
			textureFormat = psPVRHeader->dwAlphaBitMask==0 ? GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG : GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG ;	// PVRTC2
		}
		else
		{
			if(bAllowDecompress)
			{
				IsCompressedFormatSupported = false;
				IsCompressedFormat = true;
				textureFormat = GL_UNSIGNED_BYTE;
				textureType = GL_RGBA;
				PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer warning: PVRTC2 not supported. Converting to RGBA8888 instead.\n");
			}
			else
			{
				PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer error: PVRTC2 not supported.\n");
				return PVR_FAIL;
			}
		}
		break;

	case MGLPT_PVRTC4:
	case OGL_PVRTC4:
		if(IsPVRTCSupported)
		{
			IsCompressedFormatSupported = IsCompressedFormat = true;
			textureFormat = psPVRHeader->dwAlphaBitMask==0 ? GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG : GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG ;	// PVRTC4
		}
		else
		{
			if(bAllowDecompress)
			{
				IsCompressedFormatSupported = false;
				IsCompressedFormat = true;
				textureFormat = GL_UNSIGNED_BYTE;
				textureType = GL_RGBA;
				PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer warning: PVRTC4 not supported. Converting to RGBA8888 instead.\n");
			}
			else
			{
				PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer error: PVRTC4 not supported.\n");
				return PVR_FAIL;
			}
		}
		break;

#ifndef TARGET_OS_IPHONE
	case ETC_RGB_4BPP:
		if(IsETCSupported)
		{
			IsCompressedFormatSupported = IsCompressedFormat = true;
			textureFormat = GL_ETC1_RGB8_OES;
		}
		else
		{
			if(bAllowDecompress)
			{
				IsCompressedFormatSupported = false;
				IsCompressedFormat = true;
				textureFormat = GL_UNSIGNED_BYTE;
				textureType = GL_RGBA;
				PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer warning: ETC not supported. Converting to RGBA8888 instead.\n");
			}
			else
			{
				PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer error: ETC not supported.\n");
				return PVR_FAIL;
			}
		}
		break;
#endif

	default:											// NOT SUPPORTED
		PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer failed: pixel type not supported.\n");
		return PVR_FAIL;
	}

	// load the texture up
	glPixelStorei(GL_UNPACK_ALIGNMENT,1);				// Never have row-aligned in psPVRHeaders

	glGenTextures(1, &textureName);

	//  check that this data is cube map data or not.
	if(psPVRHeader->dwpfFlags & PVRTEX_CUBEMAP)
		eTarget = GL_TEXTURE_CUBE_MAP;
	else
		eTarget = GL_TEXTURE_2D;

	glBindTexture(eTarget, textureName);

	if(glGetError())
	{
		PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer failed: glBindTexture() failed.\n");
		return PVR_FAIL;
	}

	for(unsigned int i=0; i<u32NumSurfs; i++)
	{
		char *theTexturePtr = (texPtr? (char*)texPtr :  (char*)psPVRHeader + psPVRHeader->dwHeaderSize) + psPVRHeader->dwTextureDataSize * i;
		char *theTextureToLoad = 0;
		int		nMIPMapLevel;
		int		nTextureLevelsNeeded = (psPVRHeader->dwpfFlags & PVRTEX_MIPMAP)? psPVRHeader->dwMipMapCount : 0;
		unsigned int		nSizeX= psPVRHeader->dwWidth, nSizeY = psPVRHeader->dwHeight;
		unsigned int		CompressedImageSize = 0;

		for(nMIPMapLevel = 0; nMIPMapLevel <= nTextureLevelsNeeded; nSizeX=PVRT_MAX(nSizeX/2, (unsigned int)1), nSizeY=PVRT_MAX(nSizeY/2, (unsigned int)1), nMIPMapLevel++)
		{
			theTextureToLoad = theTexturePtr;

			// Load the Texture

			// If the texture is PVRTC or ETC then use GLCompressedTexImage2D
			if(IsCompressedFormat)
			{
				/* Calculate how many bytes this MIP level occupies */
				if ((psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)==OGL_PVRTC2)
				{
					CompressedImageSize = ( PVRT_MAX(nSizeX, PVRTC2_MIN_TEXWIDTH) * PVRT_MAX(nSizeY, PVRTC2_MIN_TEXHEIGHT) * psPVRHeader->dwBitCount) / 8;
				}
				else if ((psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)==OGL_PVRTC4)
				{
					CompressedImageSize = ( PVRT_MAX(nSizeX, PVRTC4_MIN_TEXWIDTH) * PVRT_MAX(nSizeY, PVRTC4_MIN_TEXHEIGHT) * psPVRHeader->dwBitCount) / 8;
				}
				else
				{// ETC
					CompressedImageSize = ( PVRT_MAX(nSizeX, ETC_MIN_TEXWIDTH) * PVRT_MAX(nSizeY, ETC_MIN_TEXHEIGHT) * psPVRHeader->dwBitCount) / 8;
				}

				if(((signed int)nMIPMapLevel - (signed int)nLoadFromLevel) >= 0)
				{
					if(IsCompressedFormatSupported)
					{
						if(psPVRHeader->dwpfFlags&PVRTEX_CUBEMAP)
						{
							/* Load compressed texture data at selected MIP level */
							glCompressedTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i, nMIPMapLevel-nLoadFromLevel, textureFormat, nSizeX, nSizeY, 0,
											CompressedImageSize, theTextureToLoad);
						}
						else
						{
							/* Load compressed texture data at selected MIP level */
							glCompressedTexImage2D(GL_TEXTURE_2D, nMIPMapLevel-nLoadFromLevel, textureFormat, nSizeX, nSizeY, 0,
											CompressedImageSize, theTextureToLoad);

						}
					}
					else
					{
						// Convert PVRTC to 32-bit
						unsigned char *u8TempTexture = (unsigned char*)malloc(nSizeX*nSizeY*4);
						if ((psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)==OGL_PVRTC2)
						{
							PVRTDecompressPVRTC(theTextureToLoad, 1, nSizeX, nSizeY, u8TempTexture);
						}
						else if((psPVRHeader->dwpfFlags & PVRTEX_PIXELTYPE)==OGL_PVRTC4)
						{
							PVRTDecompressPVRTC(theTextureToLoad, 0, nSizeX, nSizeY, u8TempTexture);
						}
						else
						{	// ETC
							PVRTDecompressETC(theTextureToLoad, nSizeX, nSizeY, u8TempTexture, 0);
						}


						if(psPVRHeader->dwpfFlags&PVRTEX_CUBEMAP)
						{// Load compressed cubemap data at selected MIP level
							// Upload the texture as 32-bits
							glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i,nMIPMapLevel-nLoadFromLevel,GL_RGBA,
								nSizeX,nSizeY,0, GL_RGBA,GL_UNSIGNED_BYTE,u8TempTexture);
							FREE(u8TempTexture);
						}
						else
						{// Load compressed 2D data at selected MIP level
							// Upload the texture as 32-bits
							glTexImage2D(GL_TEXTURE_2D,nMIPMapLevel-nLoadFromLevel,GL_RGBA,
								nSizeX,nSizeY,0, GL_RGBA,GL_UNSIGNED_BYTE,u8TempTexture);
							FREE(u8TempTexture);
						}
					}
				}
			}
			else
			{
				if(((signed int)nMIPMapLevel - (signed int)nLoadFromLevel) >= 0)
				{
					if(psPVRHeader->dwpfFlags&PVRTEX_CUBEMAP)
					{
						/* Load uncompressed texture data at selected MIP level */
						glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+i,nMIPMapLevel-nLoadFromLevel,textureType,nSizeX,nSizeY,
							0, textureType,textureFormat,theTextureToLoad);
					}
					else
					{
						/* Load uncompressed texture data at selected MIP level */
						glTexImage2D(GL_TEXTURE_2D,nMIPMapLevel-nLoadFromLevel,textureType,nSizeX,nSizeY,0, textureType,textureFormat,theTextureToLoad);
					}
				}
			}



			if(glGetError())
			{
				PVRTErrorOutputDebug("PVRTTextureLoadPartialFromPointer failed: glBindTexture() failed.\n");
				return PVR_FAIL;
			}

			// offset the texture pointer by one mip-map level

			/* PVRTC case */
			if ( IsCompressedFormat )
			{
				theTexturePtr += CompressedImageSize;
			}
			else
			{
				/* New formula that takes into account bit counts inferior to 8 (e.g. 1 bpp) */
				theTexturePtr += (nSizeX * nSizeY * psPVRHeader->dwBitCount + 7) / 8;
			}
		}
	}

	*texName = textureName;

	if(psTextureHeader)
	{
		*(PVR_Texture_Header*)psTextureHeader = *psPVRHeader;
		((PVR_Texture_Header*)psTextureHeader)->dwPVR = PVRTEX_IDENTIFIER;
		((PVR_Texture_Header*)psTextureHeader)->dwNumSurfs = u32NumSurfs;
	}

	if(!psPVRHeader->dwMipMapCount)
	{
		glTexParameteri(eTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(eTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}
	else
	{
        glTexParameteri(eTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
		glTexParameteri(eTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	return PVR_SUCCESS;
}

/*!***************************************************************************
 @Function		PVRTTextureLoadFromPVR
 @Input			filename			Filename of the .PVR file to load the texture from
 @Modified		texName				the OpenGL ES texture name as returned by glBindTexture
 @Modified		psTextureHeader		Pointer to a PVR_Texture_Header struct. Modified to
									contain the header data of the returned texture Ignored if NULL.
 @Input			bAllowDecompress	Allow decompression if PVRTC is not supported in hardware.
 @Input			nLoadFromLevel		Which mipmap level to start loading from (0=all)
 @Return		PVR_SUCCESS on success
 @Description	Allows textures to be stored in binary PVR files and loaded in. Can load parts of a
				mipmaped texture (ie skipping the highest detailed levels).
				Sets the texture MIN/MAG filter to GL_LINEAR_MIPMAP_NEAREST/GL_LINEAR
				if mipmaps are present, GL_LINEAR/GL_LINEAR otherwise.
*****************************************************************************/
EPVRTError PVRTTextureLoadFromPVR(	const char * const filename,
									GLuint * const texName,
									const void *psTextureHeader,
									bool bAllowDecompress,
									const unsigned int nLoadFromLevel)
{
	CPVRTResourceFile TexFile(filename);
	if (!TexFile.IsOpen()) return PVR_FAIL;

	return PVRTTextureLoadFromPointer(	TexFile.DataPtr(),
										texName,
										psTextureHeader,
										bAllowDecompress,
										nLoadFromLevel);
}

/*!***************************************************************************
 @Function			PVRTTextureFormatGetBPP
 @Input				nFormat
 @Input				nType
 @Description		Returns the bits per pixel (BPP) of the format.
*****************************************************************************/
unsigned int PVRTTextureFormatGetBPP(const GLuint nFormat, const GLuint nType)
{
	switch(nFormat)
	{
	case GL_COMPRESSED_RGB_PVRTC_2BPPV1_IMG:
	case GL_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG:
		return 2;
	case GL_COMPRESSED_RGB_PVRTC_4BPPV1_IMG:
	case GL_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG:
		return 4;
	case GL_UNSIGNED_BYTE:
		switch(nType)
		{
		case GL_RGBA:
			return 32;
		}
	case GL_UNSIGNED_SHORT_5_5_5_1:
		switch(nType)
		{
		case GL_RGBA:
			return 16;
		}
	}

	return 0xFFFFFFFF;
}

/*****************************************************************************
 End of file (PVRTTextureAPI.cpp)
*****************************************************************************/
