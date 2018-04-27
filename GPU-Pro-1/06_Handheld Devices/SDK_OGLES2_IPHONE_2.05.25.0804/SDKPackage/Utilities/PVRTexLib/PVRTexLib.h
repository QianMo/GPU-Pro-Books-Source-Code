/******************************************************************************

 @File         PVRTexLib.h

 @Title        Console Log

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI

 @Description  Texture processing utility class. This is a singleton do not
               instantiate - use getPointer() instead.

******************************************************************************/
#ifndef PVRTEXLIB_H
#define PVRTEXLIB_H

/*****************************************************************************
* Includes
*****************************************************************************/
#include "PVRTexLibGlobals.h"
#include "PVRException.h"
#include "CPVRTextureData.h"
#include "CPVRTextureHeader.h"
#include "CPVRTexture.h"

namespace pvrtexlib
{
#ifdef __APPLE__
	/* The classes below are exported */
#pragma GCC visibility push(default)
#endif
	
	class PVR_DLL PVRTextureUtilities
	{

	public:
		/*******************************************************************************
		* Function Name  : getPointer
		* Returns		 : Pointer to the PVRTextureUtilities singleton
		* Description    : The main method of accessing the PVRTextureUtilities class.
		*******************************************************************************/
		static PVRTextureUtilities* getPointer();

		/*******************************************************************************
		* Function Name			: CompressPVR
		* In/Outputs		
		: sCompressedTexture	: Output CPVRTexture 
		: sDecompressedTexture	: Input CPVRTexture needs to be in a standard format
		: nMode					: Parameter value for specific image compressor - eg ETC
		* Description			: Takes a CPVRTexture in one of the standard formats
		*						: and compresses to the pixel type specified in the destination
		*						: PVRTexture. nMode specifies the quality mode.
		*******************************************************************************/
		void CompressPVR(	CPVRTexture& sDecompressedTexture,
			CPVRTexture& sCompressedTexture, const int nMode=0);
		/*******************************************************************************
		* Function Name			: CompressPVR
		* In/Outputs		
		: sDecompressedHeader	: Input CPVRTexture needs to be in a standard format
		: sDecompressedData		: Input CPVRTexture needs to be in a standard format
		: sCompressedHeader		: Output CPVRTextureHeader with output format set
		: sCompressedData		: Output CPVRTextureData
		: nMode					: Parameter value for specific image compressor - i.e. ETC
		* Description			: Takes a CPVRTextureHeader/CPVRTextureData pair in one of the
		*						: standard formats
		*						: and compresses to the pixel type specified in the destination
		*						: CPVRTextureHeader, the data goes in the destination CPVRTextureData.
		*						: nMode specifies the quality mode.
		*******************************************************************************/
		void CompressPVR(	CPVRTextureHeader			&sDecompressedHeader,
			CPVRTextureData				&sDecompressedData,
			CPVRTextureHeader			&sCompHeader,
			CPVRTextureData				&sCompData,
			const int					nMode=0);

		/*******************************************************************************
		* Function Name			: DecompressPVR
		* In/Outputs		
		: sCompressedTexture	: Input CPVRTexture 
		: sDecompressedTexture	: Output CPVRTexture will be in a standard format
		* Description			: Takes a CPVRTexture and decompresses it into a
		*						: standard format.
		*******************************************************************************/
		void DecompressPVR(CPVRTexture& sCompressedTexture,
			CPVRTexture&				sDecompressedTexture,
			const int					nMode=0);
		/*******************************************************************************
		* Function Name			: DecompressPVR
		* In/Outputs		
		: sCompressedHeader		: Input CPVRTextureHeader 
		: sCompressedData		: Input CPVRTextureData 
		: sDecompressedHeader	: Output CPVRTextureHeader will be in a standard format
		: sDecompressedData		: Output CPVRTextureData will be in a standard format
		* Description			: Takes a CPVRTextureHeader/Data pair and decompresses it into a
		*						: standard format.
		*******************************************************************************/
		void DecompressPVR(	CPVRTextureHeader		& sCompressedHeader,
			const CPVRTextureData		& sCompressedData,
			CPVRTextureHeader			& sDecompressedHeader,
			CPVRTextureData				& sDecompressedData,
			const int					nMode=0);

		/*******************************************************************************
		* Function Name			: ProcessRawPVR
		* In/Outputs		
		: sInputTexture			: Input CPVRTexture needs to be in a standard format
		: sOutputTexture		: Output CPVRTexture will be in a standard format (not necessarily the same)
		* Description			: Takes a CPVRTexture and processes it according to the differences in the passed
		*						:	output CPVRTexture and the passed parameters. Requires the input texture
		*						:	to be in a standard format.
		*******************************************************************************/
		bool ProcessRawPVR(	CPVRTexture&		sInputTexture,
			CPVRTextureHeader&		sProcessHeader,
			const bool				bDoBleeding=false,
			const float				fBleedRed=0.0f,
			const float				fBleedGreen=0.0f,
			const float				fBleedBlue=0.0f,
			E_RESIZE_MODE				eResizeMode=eRESIZE_BICUBIC );
		/*******************************************************************************
		* Function Name			: ProcessRawPVR
		* In/Outputs		
		: sInputTexture			: Input CPVRTexture needs to be in a standard format.
		: sOutputTexture		: Output CPVRTexture 
		* Description			: Takes a CPVRTexture and decompresses it into one of the standard
		*						: data formats.
		*******************************************************************************/
		bool ProcessRawPVR(	CPVRTextureHeader&		sInputHeader,
			CPVRTextureData&		sInputData,
			CPVRTextureHeader&		sProcessHeader,
			const bool				bDoBleeding=false,
			const float				fBleedRed=0.0f,
			const float				fBleedGreen=0.0f,
			const float				fBleedBlue=0.0f,
			E_RESIZE_MODE				eResizeMode=eRESIZE_BICUBIC );

	};

	
#ifdef __APPLE__
#pragma GCC visibility pop
#endif

}


#endif
/*****************************************************************************
End of file (pvr_utils.h)
*****************************************************************************/
