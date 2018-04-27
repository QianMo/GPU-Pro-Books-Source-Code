/******************************************************************************

 @File         PVRShell.cpp

 @Title        PVRShell

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Makes programming for 3D APIs easier by wrapping surface
               initialization, Texture allocation and other functions for use by a demo.

******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "PVRShell.h"
#include "PVRShellOS.h"
#include "PVRShellAPI.h"
#include "PVRShellImpl.h"

/*! This file simply defines a version string. It can be commented out. */
#include "../Builds/sdkver.h"
#ifndef PVRSDK_VERSION
#define PVRSDK_VERSION "n.nn.nn.nnnn"
#endif

/*! Define to automatically stop the app after x frames. If negative, run forever. */
#ifndef PVRSHELL_QUIT_AFTER_FRAME
#define PVRSHELL_QUIT_AFTER_FRAME -1
#endif

/*! Define to automatically stop the app after x amount of seconds. If negative, run forever. */
#ifndef PVRSHELL_QUIT_AFTER_TIME
#define PVRSHELL_QUIT_AFTER_TIME -1
#endif

/*! Define for the screen shot file name. */
#define PVRSHELL_SCREENSHOT_NAME	"PVRShell"

/*****************************************************************************
** Prototypes
*****************************************************************************/
static bool StringCopy(char *&pszStr, const char * const pszSrc);

/****************************************************************************
** Class: PVRShell
****************************************************************************/

/*!***********************************************************************
@Function			PVRShell
@Description		Constructor
*************************************************************************/
PVRShell::PVRShell()
{
	m_pShellInit = NULL;
	m_pShellData = new PVRShellData;

	m_pShellData->nShellPosX=0;
	m_pShellData->nShellPosY=0;

	m_pShellData->bFullScreen=false;	// note this may be overriden by some OS versions of PVRShell

	m_pShellData->nFSAAMode=0;
	m_pShellData->nColorBPP = 0;

	m_pShellData->nDieAfterFrames=PVRSHELL_QUIT_AFTER_FRAME;
	m_pShellData->fDieAfterTime=PVRSHELL_QUIT_AFTER_TIME;

	m_pShellData->bNeedPbuffer = false;
	m_pShellData->bNeedPixmap = false;
	m_pShellData->bNeedPixmapDisableCopy = false;
	m_pShellData->bNeedZbuffer = true;
	m_pShellData->bLockableBackBuffer = false;
	m_pShellData->bSoftwareRender = false;
	m_pShellData->bNeedStencilBuffer = false;
	m_pShellData->bNeedOpenVG = false;
	m_pShellData->bNeedAlphaFormatPre = false;
	m_pShellData->bUsingPowerSaving = true;
	m_pShellData->bOutputInfo = false;
	m_pShellData->bNoShellSwapBuffer = false;

	m_pShellData->pszAppName = 0;
	m_pShellData->pszExitMessage = 0;

	m_pShellData->nSwapInterval = 1;
	m_pShellData->nInitRepeats = 0;

	m_pShellData->nCaptureFrameStart = -1;
	m_pShellData->nCaptureFrameStop  = -1;

	m_pShellData->nPriority = 2;

	// Internal Data
	m_pShellData->bShellPosWasDefault = true;
	m_pShellData->nShellCurFrameNum = 0;
#ifdef PVRSHELL_FPS_OUTPUT
	m_pShellData->bOutputFPS = false;
#endif
}

/*!***********************************************************************
@Function			~PVRShell
@Description		Destructor
*************************************************************************/
PVRShell::~PVRShell()
{
	delete m_pShellData;
	m_pShellData = NULL;
}

// Allow user to set preferences from within InitApplication

/*!***********************************************************************
@Function		PVRShellSet
@Input			prefName				Name of preference to set to value
@Input			value					Value
@Return			true for success
@Description	This function is used to pass preferences to the PVRShell.
If used, it must be called from InitApplication().
*************************************************************************/
bool PVRShell::PVRShellSet(const prefNameBoolEnum prefName, const bool value)
{
	switch(prefName)
	{
	case prefFullScreen:
		m_pShellData->bFullScreen = value;
		return true;

	case prefPBufferContext:
		m_pShellData->bNeedPbuffer = value;
		return true;

	case prefPixmapContext:
		m_pShellData->bNeedPixmap = value;
		return true;

	case prefPixmapDisableCopy:
		m_pShellData->bNeedPixmapDisableCopy = value;
		return true;

	case prefZbufferContext:
		m_pShellData->bNeedZbuffer = value;
		return true;

	case prefLockableBackBuffer:
		m_pShellData->bLockableBackBuffer = value;
		return true;

	case prefSoftwareRendering:
		m_pShellData->bSoftwareRender = value;
		return true;

	case prefStencilBufferContext:
		m_pShellData->bNeedStencilBuffer = value;
		return true;

	case prefOpenVGContext:
		m_pShellData->bNeedOpenVG = value;
		return true;

	case prefAlphaFormatPre:
		m_pShellData->bNeedAlphaFormatPre = value;
		return true;

	case prefPowerSaving:
		m_pShellData->bUsingPowerSaving = value;
		return true;

	case prefOutputInfo:
		m_pShellData->bOutputInfo = value;
		return true;

	case prefNoShellSwapBuffer:
		m_pShellData->bNoShellSwapBuffer = value;
		return true;

#ifdef PVRSHELL_FPS_OUTPUT
	case prefOutputFPS:
		m_pShellData->bOutputFPS = value;
		return true;
#endif
	default:
		break;
	}
	return false;
}

/*!***********************************************************************
@Function		PVRShellGet
@Input			prefName				Name of preference to set to value
@Return			Value asked for.
@Description	This function is used to get parameters from the PVRShell
It can be called from any where in the program.
*************************************************************************/
bool PVRShell::PVRShellGet(const prefNameBoolEnum prefName) const
{
	switch(prefName)
	{
	case prefFullScreen:	return m_pShellData->bFullScreen;
	case prefIsRotated:	return (m_pShellData->nShellDimY > m_pShellData->nShellDimX);
	case prefPBufferContext:	return m_pShellData->bNeedPbuffer;
	case prefPixmapContext:	return m_pShellData->bNeedPixmap;
	case prefPixmapDisableCopy:	return m_pShellData->bNeedPixmapDisableCopy;
	case prefZbufferContext:	return m_pShellData->bNeedZbuffer;
	case prefLockableBackBuffer:	return m_pShellData->bLockableBackBuffer;
	case prefSoftwareRendering:	return m_pShellData->bSoftwareRender;
	case prefNoShellSwapBuffer: return m_pShellData->bNoShellSwapBuffer;
	case prefStencilBufferContext:	return m_pShellData->bNeedStencilBuffer;
	case prefOpenVGContext:	return m_pShellData->bNeedOpenVG;
	case prefAlphaFormatPre: return m_pShellData->bNeedAlphaFormatPre;
	case prefPowerSaving: return m_pShellData->bUsingPowerSaving;
	case prefOutputInfo:	return m_pShellData->bOutputInfo;
#ifdef PVRSHELL_FPS_OUTPUT
	case prefOutputFPS: return m_pShellData->bOutputFPS;
#endif
	default:	return false;
	}
}

/*!***********************************************************************
@Function		PVRShellSet
@Input			prefName				Name of preference to set to value
@Input			value					Value
@Return			true for success
@Description	This function is used to pass preferences to the PVRShell.
If used, it must be called from InitApplication().
*************************************************************************/
bool PVRShell::PVRShellSet(const prefNameFloatEnum prefName, const float value)
{
	switch(prefName)
	{
	case prefQuitAfterTime:
		m_pShellData->fDieAfterTime = value;
		return true;

	default:
		break;
	}
	return false;
}

/*!***********************************************************************
@Function		PVRShellGet
@Input			prefName				Name of preference to set to value
@Return			Value asked for.
@Description	This function is used to get parameters from the PVRShell
It can be called from any where in the program.
*************************************************************************/
float PVRShell::PVRShellGet(const prefNameFloatEnum prefName) const
{
	switch(prefName)
	{
	case prefQuitAfterTime:	return m_pShellData->fDieAfterTime;
	default:	return -1;
	}
}

/*!***********************************************************************
@Function		PVRShellSet
@Input			prefName				Name of preference to set to value
@Input			value					Value
@Return			true for success
@Description	This function is used to pass preferences to the PVRShell.
If used, it must be called from InitApplication().
*************************************************************************/
bool PVRShell::PVRShellSet(const prefNameIntEnum prefName, const int value)
{
	switch(prefName)
	{
	case prefWidth:
		if(value > 0)
		{
			m_pShellData->nShellDimX = value;
			return true;
		}
		return false;

	case prefHeight:
		if(value > 0)
		{
			m_pShellData->nShellDimY = value;
			return true;
		}
		return false;

	case prefPositionX:
		m_pShellData->bShellPosWasDefault = false;
		m_pShellData->nShellPosX = value;
		return true;

	case prefPositionY:
		m_pShellData->bShellPosWasDefault = false;
		m_pShellData->nShellPosY = value;
		return true;

	case prefQuitAfterFrame:
		m_pShellData->nDieAfterFrames = value;
		return true;

	case prefSwapInterval:
		return setSwapInterval(value);

	case prefInitRepeats:
		m_pShellData->nInitRepeats = value;
		return true;

	case prefFSAAMode:
		if(value >= 0 && value <= 2)
		{
			m_pShellData->nFSAAMode = value;
			return true;
		}
		return false;

	case prefColorBPP:
		if(value >= 0)
		{
			m_pShellData->nColorBPP = value;
			return true;
		}
		return false;

	case prefRotateKeys:
		{
			switch((PVRShellKeyRotate)value)
			{
			case PVRShellKeyRotateNone:
				m_pShellInit->m_eKeyMapUP = PVRShellKeyNameUP;
				m_pShellInit->m_eKeyMapLEFT = PVRShellKeyNameLEFT;
				m_pShellInit->m_eKeyMapDOWN = PVRShellKeyNameDOWN;
				m_pShellInit->m_eKeyMapRIGHT = PVRShellKeyNameRIGHT;
				break;
			case PVRShellKeyRotate90:
				m_pShellInit->m_eKeyMapUP = PVRShellKeyNameLEFT;
				m_pShellInit->m_eKeyMapLEFT = PVRShellKeyNameDOWN;
				m_pShellInit->m_eKeyMapDOWN = PVRShellKeyNameRIGHT;
				m_pShellInit->m_eKeyMapRIGHT = PVRShellKeyNameUP;
				break;
			case PVRShellKeyRotate180:
				m_pShellInit->m_eKeyMapUP = PVRShellKeyNameDOWN;
				m_pShellInit->m_eKeyMapLEFT = PVRShellKeyNameRIGHT;
				m_pShellInit->m_eKeyMapDOWN = PVRShellKeyNameUP;
				m_pShellInit->m_eKeyMapRIGHT = PVRShellKeyNameLEFT;
				break;
			case PVRShellKeyRotate270:
				m_pShellInit->m_eKeyMapUP = PVRShellKeyNameRIGHT;
				m_pShellInit->m_eKeyMapLEFT = PVRShellKeyNameUP;
				m_pShellInit->m_eKeyMapDOWN = PVRShellKeyNameLEFT;
				m_pShellInit->m_eKeyMapRIGHT = PVRShellKeyNameDOWN;
				break;
			default:
				return false;
			}
		}
			return true;
	case prefCaptureFrameStart:
		m_pShellData->nCaptureFrameStart = value;
		return true;
	case prefCaptureFrameStop:
		m_pShellData->nCaptureFrameStop  = value;
		return true;
	case prefPriority:
		return setPriority(value);
	default:
		break;
	}
	return false;
}

/*!***********************************************************************
@Function		PVRShellGet
@Input			prefName	Name of preference to return the value of
@Return			Value asked for.
@Description	This function is used to get parameters from the PVRShell
It can be called from any where in the program.
*************************************************************************/
int PVRShell::PVRShellGet(const prefNameIntEnum prefName) const
{
	switch(prefName)
	{
	case prefWidth:	return m_pShellData->nShellDimX;
	case prefHeight:	return m_pShellData->nShellDimY;
	case prefPositionX:	return m_pShellData->nShellPosX;
	case prefPositionY:	return m_pShellData->nShellPosY;
	case prefQuitAfterFrame:	return m_pShellData->nDieAfterFrames;
	case prefSwapInterval:	return m_pShellData->nSwapInterval;
	case prefInitRepeats:	return m_pShellData->nInitRepeats;
	case prefFSAAMode:	return m_pShellData->nFSAAMode;
	case prefCommandLineOptNum:	return m_pShellInit->m_CommandLine.m_nOptLen;
	case prefColorBPP: return m_pShellData->nColorBPP;
	case prefCaptureFrameStart: return m_pShellData->nCaptureFrameStart;
	case prefCaptureFrameStop: return m_pShellData->nCaptureFrameStop;
	case prefPriority: return m_pShellData->nPriority;
	default:
		{
			int n;

			if(m_pShellInit->ApiGet(prefName, &n))
				return n;
			if(m_pShellInit->OsGet(prefName, &n))
				return n;
			return -1;
		}
	}
}

/*!***********************************************************************
@Function		PVRShellSet
@Input			prefName				Name of preference to set to value
@Input			ptrValue				Value
@Return			true for success
@Description	This function is used to pass preferences to the PVRShell.
If used, it must be called from InitApplication().
*************************************************************************/
bool PVRShell::PVRShellSet(const prefNamePtrEnum prefName, const void * const ptrValue)
{
	return false;
}

/*!***********************************************************************
@Function		PVRShellGet
@Input			prefName				Name of preference to set to value
@Return			Value asked for.
@Description	This function is used to get parameters from the PVRShell
It can be called from any where in the program.
*************************************************************************/
void *PVRShell::PVRShellGet(const prefNamePtrEnum prefName) const
{
	switch(prefName)
	{
	case prefNativeWindowType:	return m_pShellInit->OsGetNativeWindowType();
	default:
		{
			void *p;

			if(m_pShellInit->ApiGet(prefName, &p))
				return p;
			if(m_pShellInit->OsGet(prefName, &p))
				return p;
			return NULL;
		}
	}
}

/*!***********************************************************************
@Function		PVRShellSet
@Input			prefName				Name of preference to set to value
@Input			ptrValue				Value
@Return			true for success
@Description	This function is used to pass preferences to the PVRShell.
If used, it must be called from InitApplication().
*************************************************************************/
bool PVRShell::PVRShellSet(const prefNameConstPtrEnum prefName, const void * const ptrValue)
{
	switch(prefName)
	{
	case prefAppName:
		StringCopy(m_pShellData->pszAppName, (char*)ptrValue);
		return true;
	case prefExitMessage:
		StringCopy(m_pShellData->pszExitMessage, (char*)ptrValue);
		PVRShellOutputDebug("Exit message has been set to: \"%s\".\n", ptrValue);
		return true;
	default:
		break;
	}
	return false;
}

/*!***********************************************************************
@Function		PVRShellGet
@Input			prefName				Name of preference to set to value
@Return			Value asked for.
@Description	This function is used to get parameters from the PVRShell
It can be called from any where in the program.
*************************************************************************/
const void *PVRShell::PVRShellGet(const prefNameConstPtrEnum prefName) const
{
	switch(prefName)
	{
	case prefAppName:
		return m_pShellData->pszAppName;
	case prefExitMessage:
		return m_pShellData->pszExitMessage;
	case prefReadPath:
		return m_pShellInit->GetReadPath();
	case prefWritePath:
		return m_pShellInit->GetWritePath();
	case prefCommandLine:
		return m_pShellInit->m_CommandLine.m_psOrig;
	case prefCommandLineOpts:
		return m_pShellInit->m_CommandLine.m_pOpt;
	case prefVersion:
		return PVRSDK_VERSION;
	default:
		return 0;
	}
}

/*!***********************************************************************
@Function		PVRShellScreenCaptureBuffer
@Input			Width			size of image to capture (relative to 0,0)
@Input			Height			size of image to capture (relative to 0,0)
@Modified		pLines			receives a pointer to an area of memory containing the screen buffer.
@Return			true for success
@Description	It will be stored as 24-bit per pixel, 8-bit per chanel RGB. The
memory should be freed with free() when no longer needed.
*************************************************************************/
bool PVRShell::PVRShellScreenCaptureBuffer(const int Width, const int Height, unsigned char **pLines)
{
	int Pitch;

	/* Compute line pitch */
	Pitch = 3*Width;
	if ( ((3*Width)%4)!=0 )
	{
		Pitch += 4-((3*Width)%4);
	}

	/* Allocate memory for line */
	*pLines=(unsigned char *)calloc(Pitch*Height, sizeof(unsigned char));
	if (!(*pLines)) return false;

	return m_pShellInit->ApiScreenCaptureBuffer(Width, Height, *pLines);
}

/*!***********************************************************************
@Function		PVRShellScreenSave
@Input			fname			base of file to save screen to
@Output			ofname			If non-NULL, receives the filename actually used
@Modified		pLines			image data to write out (24bpp, 8-bit per channel RGB)
@Return			true for success
@Description	Writes out the image data to a BMP file with basename
fname. The file written will be fname suffixed with a
number to make the file unique.
For example, if fname is "abc", this function will attempt
to save to "abc0000.bmp"; if that file already exists, it
will try "abc0001.bmp", repeating until a new filename is
found. The final filename used is returned in ofname.
*************************************************************************/
int PVRShell::PVRShellScreenSave(
								 const char			* const fname,
								 const unsigned char	* const pLines,
								 char				* const ofname)
{
	FILE		*file = 0;
	const char	*pszWritePath;
	char		*pszFileName;
	int			nScreenshotCount;
	int			dwWidth	= m_pShellData->nShellDimX;
	int			dwHeight= m_pShellData->nShellDimY;
	int 		error;

	pszWritePath = (const char*)PVRShellGet(prefWritePath);
	pszFileName = (char*)malloc(strlen(pszWritePath) + 200);

	/* Look for the first file name that doesn't already exist */
	for(nScreenshotCount = 0; nScreenshotCount < 10000; ++nScreenshotCount)
	{
		sprintf(pszFileName, "%s%s%04d.bmp", pszWritePath, fname, nScreenshotCount);
		file = fopen(pszFileName,"r");
		if(!file)
			break;
		fclose(file);
	}

	/* If all files already exist, replace the first one */
	if (nScreenshotCount==10000)
	{
		sprintf(pszFileName, "%s%s0000.bmp", pszWritePath, fname);
		PVRShellOutputDebug("PVRShell: *WARNING* : Overwriting %s\n", pszFileName);
	}

	if(ofname)	// requested the output file name
	{
		strcpy(ofname, pszFileName);
	}

	error = PVRShellWriteBMPFile(pszFileName, dwWidth, dwHeight, pLines);
	FREE(pszFileName);
	if (error)
	{
		return 10*error+1;
	}
	else
	{
		// No problem occured
		return 0;
	}
}

/*!***********************************************************************
@Function		PVRShellByteSwap
@Input			pBytes The bytes to swap
@Input			i32ByteNo The number of bytes to swap
@Description	Swaps the bytes in pBytes from little to big endian (or vice versa)
*************************************************************************/
inline void PVRShellByteSwap(unsigned char* pBytes, int i32ByteNo)
{
	int i = 0, j = i32ByteNo - 1;

	while(i < j)
	{
		unsigned char cTmp = pBytes[i];
		pBytes[i] = pBytes[j];
		pBytes[j] = cTmp;

		++i;
		--j;
	}
}

/*!***********************************************************************
@Function		PVRShellWriteBMPFile
@Input			pszFilename		file to save screen to
@Input			uWidth			the width of the data
@Input			uHeight			the height of the data
@Input			pImageData		image data to write out (24bpp, 8-bit per channel RGB)
@Return		0 on success
@Description	Writes out the image data to a BMP file with name fname.
*************************************************************************/
const int g_i32BMPHeaderSize = 14; /*!< The size of a BMP header */
const int g_i32BMPInfoSize   = 40; /*!< The size of a BMP info header */

int PVRShell::PVRShellWriteBMPFile(
								   const char			* const pszFilename,
								   const unsigned long	uWidth,
								   const unsigned long	uHeight,
								   const void			* const pImageData)
{
#define ByteSwap(x) PVRShellByteSwap((unsigned char*) &x, sizeof(x))

	int			Result = 1;
	unsigned int i,j;
	FILE*		fpDumpfile = 0;

	fpDumpfile = fopen(pszFilename, "wb");

	if (fpDumpfile != 0)
	{
		short int word = 0x0001;
		char *byte = (char*) &word;
		bool bLittleEndian = byte[0] ? true : false;

		int i32Line		  = uWidth * 3;
		int i32BytesPerLine = i32Line;
		unsigned int i32Align = 0;

		// round up to a dword boundary
		if(i32BytesPerLine & 3)
		{
			i32BytesPerLine |= 3;
			++i32BytesPerLine;
			i32Align = i32BytesPerLine - i32Line;
		}

		int i32RealSize = i32BytesPerLine *uHeight;

		// BMP Header
		unsigned short  bfType = 0x4D42;
		unsigned long   bfSize = g_i32BMPHeaderSize + g_i32BMPInfoSize + i32RealSize;
		unsigned short  bfReserved1 = 0;
		unsigned short  bfReserved2 = 0;
		unsigned long   bfOffBits = g_i32BMPHeaderSize + g_i32BMPInfoSize;

		// BMP Info Header
		unsigned long  biSize = g_i32BMPInfoSize;
		unsigned long  biWidth = uWidth;
		unsigned long  biHeight = uHeight;
		unsigned short biPlanes = 1;
		unsigned short biBitCount = 24;
		unsigned long  biCompression = 0L;
		unsigned long  biSizeImage = i32RealSize;
		unsigned long  biXPelsPerMeter = 0;
		unsigned long  biYPelsPerMeter = 0;
		unsigned long  biClrUsed = 0;
		unsigned long  biClrImportant = 0;

		unsigned char *pData = (unsigned char*) pImageData;

		if(!bLittleEndian)
		{
			for(i = 0; i < uWidth * uHeight; ++i)
				PVRShellByteSwap(pData + (3 * i), 3);

			ByteSwap(bfType);
			ByteSwap(bfSize);
			ByteSwap(bfOffBits);
			ByteSwap(biSize);
			ByteSwap(biWidth);
			ByteSwap(biHeight);
			ByteSwap(biPlanes);
			ByteSwap(biBitCount);
			ByteSwap(biCompression);
			ByteSwap(biSizeImage);
		}

		// Write Header.
		fwrite(&bfType		, 1, sizeof(bfType)		, fpDumpfile);
		fwrite(&bfSize		, 1, sizeof(bfSize)		, fpDumpfile);
		fwrite(&bfReserved1	, 1, sizeof(bfReserved1), fpDumpfile);
		fwrite(&bfReserved2	, 1, sizeof(bfReserved2), fpDumpfile);
		fwrite(&bfOffBits	, 1, sizeof(bfOffBits)	, fpDumpfile);

		// Write info header.
		fwrite(&biSize			, 1, sizeof(biSize)			, fpDumpfile);
		fwrite(&biWidth			, 1, sizeof(biWidth)		, fpDumpfile);
		fwrite(&biHeight		, 1, sizeof(biHeight)		, fpDumpfile);
		fwrite(&biPlanes		, 1, sizeof(biPlanes)		, fpDumpfile);
		fwrite(&biBitCount		, 1, sizeof(biBitCount)		, fpDumpfile);
		fwrite(&biCompression	, 1, sizeof(biCompression)	, fpDumpfile);
		fwrite(&biSizeImage		, 1, sizeof(biSizeImage)	, fpDumpfile);
		fwrite(&biXPelsPerMeter	, 1, sizeof(biXPelsPerMeter), fpDumpfile);
		fwrite(&biYPelsPerMeter	, 1, sizeof(biYPelsPerMeter), fpDumpfile);
		fwrite(&biClrUsed		, 1, sizeof(biClrUsed)		, fpDumpfile);
		fwrite(&biClrImportant	, 1, sizeof(biClrImportant)	, fpDumpfile);

		unsigned char align = 0x00;

		// Write image.
		if(i32Align == 0)
		{
			fwrite(pData, 1, i32RealSize, fpDumpfile);
		}
		else
		{
			for(i = 0; i < uHeight; i++)
			{
				pData += fwrite(pData, 1, i32Line, fpDumpfile);

				for(j = 0; j < i32Align; ++j)
					fwrite(&align, 1, 1, fpDumpfile);
			}
		}

		// Last but not least close the file.
		fclose(fpDumpfile);

		Result = 0;
	}
	else
	{
		PVRShellOutputDebug("PVRShell: Failed to open \"%s\" for writing screen dump.\n", pszFilename);
	}

	return Result;
}

/*!***********************************************************************
@Function		PVRShellGetTime
@Returns		A value which increments once per millisecond.
@Description	The number itself should be considered meaningless; an
application should use this function to determine how much
time has passed between two points (e.g. between each
frame).
*************************************************************************/
unsigned long PVRShell::PVRShellGetTime()
{
	// Read timer from a platform dependant function
	return m_pShellInit->OsGetTime();
}

/*!***********************************************************************
@Function		PVRShellIsKeyPressed
@Input			key		Code of the key to test
@Return			true if key was pressed
@Description	Check if a key was pressed. The keys on various devices
are mapped to the PVRShell-supported keys (listed in @a PVRShellKeyName) in
a platform-dependent manner, since most platforms have different input
devices. Check the SDK release notes for details on how the enum values
map to your device's input device.
*************************************************************************/
bool PVRShell::PVRShellIsKeyPressed(const PVRShellKeyName key)
{
	if(!m_pShellInit)
		return false;

	return m_pShellInit->DoIsKeyPressed(key);
}

/*!****************************************************************************
* @Class PVRShellCommandLine
*****************************************************************************/
/*!***********************************************************************
@Function			PVRShellCommandLine
@Description		Constructor
*************************************************************************/
PVRShellCommandLine::PVRShellCommandLine()
{
	memset(this, 0, sizeof(*this));
}

/*!***********************************************************************
@Function			PVRShellCommandLine
@Description		Destructor
*************************************************************************/
PVRShellCommandLine::~PVRShellCommandLine()
{
	delete [] m_psOrig;
	delete [] m_psSplit;
	FREE(m_pOpt);
}

/*!***********************************************************************
@Function		Parse
@Input			pStr Input string
@Description	Parse pStr for command-line options and store them in m_pOpt
*************************************************************************/
void PVRShellCommandLine::Parse(const char *pStr)
{
	size_t		len;
	int			nIn, nOut;
	bool		bInQuotes;
	SCmdLineOpt	opt;

	// Take a copy of the original
	len = strlen(pStr)+1;
	m_psOrig = new char[len];
	strcpy(m_psOrig, pStr);

	// Take a copy to be edited
	m_psSplit = new char[len];

	// Break the command line into options
	bInQuotes = false;
	opt.pArg = NULL;
	opt.pVal = NULL;
	nIn = -1;
	nOut = 0;
	do
	{
		++nIn;
		if(pStr[nIn] == '"')
		{
			bInQuotes = !bInQuotes;
		}
		else
		{
			if(bInQuotes && pStr[nIn] != 0)
			{
				if(!opt.pArg)
					opt.pArg = &m_psSplit[nOut];

				m_psSplit[nOut++] = pStr[nIn];
			}
			else
			{
				switch(pStr[nIn])
				{
				case '=':
					m_psSplit[nOut++] = 0;
					opt.pVal = &m_psSplit[nOut];
					break;

				case ' ':
				case '\t':
				case '\0':
					m_psSplit[nOut++] = 0;
					if(opt.pArg || opt.pVal)
					{
						// Increase list length if necessary
						if(m_nOptLen == m_nOptMax)
							m_nOptMax = m_nOptMax * 2 + 1;
						m_pOpt = (SCmdLineOpt*)realloc(m_pOpt, m_nOptMax * sizeof(*m_pOpt));
						if(!m_pOpt)
							return;

						// Add option to list
						m_pOpt[m_nOptLen++] = opt;
						opt.pArg = NULL;
						opt.pVal = NULL;
					}
					break;

				default:
					if(!opt.pArg)
						opt.pArg = &m_psSplit[nOut];

					m_psSplit[nOut++] = pStr[nIn];
					break;
				}
			}
		}
	} while(pStr[nIn]);
}

/*!***********************************************************************
@Function		Apply
@Input			shell
@Description	Apply the command-line options to shell
*************************************************************************/
void PVRShellCommandLine::Apply(PVRShell &shell)
{
	int i;
	const char *arg, *val;

	for(i = 0; i < m_nOptLen; ++i)
	{
		arg = m_pOpt[i].pArg;
		val = m_pOpt[i].pVal;

		if(!arg)
			continue;

		if(val)
		{
			if(_stricmp(arg, "-width") == 0)
			{
				shell.PVRShellSet(prefWidth, atoi(val));
			}
			else if(_stricmp(arg, "-height") == 0)
			{
				shell.PVRShellSet(prefHeight, atoi(val));
			}
			else if(_stricmp(arg, "-FSAAMode") == 0 || _stricmp(arg, "-aa") == 0)
			{
				shell.PVRShellSet(prefFSAAMode, atoi(val));
			}
			else if(_stricmp(arg, "-fullscreen") == 0)
			{
				shell.PVRShellSet(prefFullScreen, (atoi(val) != 0));
			}
			else if(_stricmp(arg, "-sw") == 0)
			{
				shell.PVRShellSet(prefSoftwareRendering, (atoi(val) != 0));
			}
			else if(_stricmp(arg, "-quitafterframe") == 0 || _stricmp(arg, "-qaf") == 0)
			{
				shell.PVRShellSet(prefQuitAfterFrame, atoi(val));
			}
			else if(_stricmp(arg, "-quitaftertime") == 0 || _stricmp(arg, "-qat") == 0)
			{
				shell.PVRShellSet(prefQuitAfterTime, (float)atof(val));
			}
			else if(_stricmp(arg, "-posx") == 0)
			{
				shell.PVRShellSet(prefPositionX, atoi(val));
			}
			else if(_stricmp(arg, "-posy") == 0)
			{
				shell.PVRShellSet(prefPositionY, atoi(val));
			}
			else if(_stricmp(arg, "-vsync") == 0)
			{
				shell.PVRShellSet(prefSwapInterval, atoi(val));
			}
			else if(_stricmp(arg, "-powersaving") == 0 || _stricmp(arg, "-ps") == 0)
			{
				shell.PVRShellSet(prefPowerSaving, (atoi(val) != 0));
			}
			else if(_stricmp(arg, "-colourbpp") == 0 || _stricmp(arg, "-colorbpp") == 0 ||_stricmp(arg, "-cbpp") == 0)
			{
				shell.PVRShellSet(prefColorBPP, atoi(val));
			}
			else if(_stricmp(arg, "-rotatekeys") == 0)
			{
				shell.PVRShellSet(prefRotateKeys, atoi(val));
			}
			else if(_stricmp(arg, "-c") == 0)
			{
				const char* pDash = strchr(val, '-');

				shell.PVRShellSet(prefCaptureFrameStart, atoi(val));

				if(!pDash)
					shell.PVRShellSet(prefCaptureFrameStop, atoi(val));
				else
					shell.PVRShellSet(prefCaptureFrameStop, atoi(pDash + 1));
			}
			else if(_stricmp(arg, "-priority") == 0)
			{
				shell.PVRShellSet(prefPriority, atoi(val));
			}
		}
		else
		{
			if(_stricmp(arg, "-version") == 0)
			{
				shell.PVRShellOutputDebug("Version: \"%s\"\n", shell.PVRShellGet(prefVersion));
			}
#ifdef PVRSHELL_FPS_OUTPUT
			else if(_stricmp(arg, "-fps") == 0)
			{
				shell.PVRShellSet(prefOutputFPS, true);
			}
#endif
			else if(_stricmp(arg, "-info") == 0)
			{
				shell.PVRShellSet(prefOutputInfo, true);
			}
		}
	}
}

/*!***************************************************************************
* @Class  PVRShellInit
****************************************************************************/
/*!***********************************************************************
@Function		PVRShellInit
@Description	Constructor
*************************************************************************/
PVRShellInit::PVRShellInit()
{
	memset(this, 0, sizeof(*this));
}

/*!***********************************************************************
@Function		~PVRShellInit
@Description	Destructor
*************************************************************************/
PVRShellInit::~PVRShellInit()
{
	delete [] m_pReadPath;
	m_pReadPath = NULL;

	delete [] m_pWritePath;
	m_pWritePath = NULL;
}

/*!***********************************************************************
@Function		Init
@Input			Shell
@Description	PVRShell Initialisation.
*************************************************************************/
void PVRShellInit::Init(PVRShell &Shell)
{
	m_pShell				= &Shell;
	m_pShell->m_pShellInit	= this;

	// set default direction key mappings
	m_eKeyMapDOWN = PVRShellKeyNameDOWN;
	m_eKeyMapLEFT = PVRShellKeyNameLEFT;
	m_eKeyMapUP = PVRShellKeyNameUP;
	m_eKeyMapRIGHT = PVRShellKeyNameRIGHT;

	OsInit();

	gShellDone = false;
	m_eState = ePVRShellInitApp;
}

/*!***********************************************************************
@Function		CommandLine
@Input			str A string containing the command-line
@Description	Receives the command-line from the application.
*************************************************************************/
void PVRShellInit::CommandLine(char *str)
{
	m_CommandLine.Parse(str);
#if defined(_DEBUG)
	m_pShell->PVRShellOutputDebug("PVRShell command line: %d/%d\n", m_CommandLine.m_nOptLen, m_CommandLine.m_nOptMax);
	for(int i = 0; i < m_CommandLine.m_nOptLen; ++i)
	{
		m_pShell->PVRShellOutputDebug("CL %d: \"%s\"\t= \"%s\".\n", i,
			m_CommandLine.m_pOpt[i].pArg ? m_CommandLine.m_pOpt[i].pArg : "",
			m_CommandLine.m_pOpt[i].pVal ? m_CommandLine.m_pOpt[i].pVal : "");
	}
#endif
}

/*!***********************************************************************
@Function		CommandLine
@Input			argc Number of strings in argv
@Input			argv An array of strings
@Description	Receives the command-line from the application.
*************************************************************************/
void PVRShellInit::CommandLine(int argc, char **argv)
{
	size_t	tot, len;
	char	*buf;
	int		i;

	tot = 0;
	for(i = 0; i < argc; ++i)
		tot += strlen(argv[i]);

	if(!tot)
	{
		CommandLine((char*) "");
		return;
	}

	// Add room for spaces and the \0
	tot += argc;

	buf = new char[tot];
	tot = 0;
	for(i = 0; i < argc; ++i)
	{
		len = strlen(argv[i]);
		strncpy(&buf[tot], argv[i], len);
		tot += len;
		buf[tot++] = ' ';
	}
	buf[tot-1] = 0;

	CommandLine(buf);

	delete [] buf;
}

/*!***********************************************************************
@Function		DoIsKeyPressed
@Input			key The key we're querying for
@description	Return 'true' if the specific key has been pressed.
*************************************************************************/
bool PVRShellInit::DoIsKeyPressed(const PVRShellKeyName key)
{
	if(key == nLastKeyPressed)
	{
		nLastKeyPressed = PVRShellKeyNameNull;
		return true;
	}
	else
	{
		return false;
	}
}

/*!***********************************************************************
@Function		KeyPressed
@Input			nkey The key that has been pressed
@description	Used by the OS-specific code to tell the Shell that a key has been pressed.
*************************************************************************/
void PVRShellInit::KeyPressed(PVRShellKeyName nKey)
{
	nLastKeyPressed = nKey;
}

/*!***********************************************************************
@Function		GetReadPath
@Returns		A path the application is capable of reading from
@description	Used by the OS-specific code to tell the Shell where to read external files from
*************************************************************************/
const char* PVRShellInit::GetReadPath() const
{
	return m_pReadPath;
}

/*!***********************************************************************
@Function		GetWritePath
@Returns		A path the applications is capable of writing to
@description	Used by the OS-specific code to tell the Shell where to write to
*************************************************************************/
const char* PVRShellInit::GetWritePath() const
{
	return m_pWritePath;
}

/*!****************************************************************************
@Function	  SetAppName
@Input		  str The application name
@Description  Sets the default app name (to be displayed by the OS)
*******************************************************************************/
void PVRShellInit::SetAppName(const char * const str)
{
	const char *pName = strrchr(str, PVRSHELL_DIR_SYM);

	if(pName)
	{
		++pName;
	}
	else
	{
		pName = str;
	}
	m_pShell->PVRShellSet(prefAppName, pName);
}

/*!***********************************************************************
@Function		SetReadPath
@Input			str The read path
@description	Set the path to where the application expects to read from.
*************************************************************************/
void PVRShellInit::SetReadPath(const char * const str)
{
	m_pReadPath = new char[strlen(str)+1];

	if(m_pReadPath)
	{
		strcpy(m_pReadPath, str);
		char* lastSlash = strrchr(m_pReadPath, PVRSHELL_DIR_SYM);
		lastSlash[1] = 0;
	}
}

/*!***********************************************************************
@Function		SetWritePath
@Input			str The write path
@description	Set the path to where the application expects to write to.
*************************************************************************/
void PVRShellInit::SetWritePath(const char * const str)
{
	m_pWritePath = new char[strlen(str)+1];

	if(m_pWritePath)
	{
		strcpy(m_pWritePath, str);
		char* lastSlash = strrchr(m_pWritePath, PVRSHELL_DIR_SYM);
		lastSlash[1] = 0;
	}
}

#ifdef PVRSHELL_FPS_OUTPUT
/*****************************************************************************
* Function Name  : FpsUpdate
* Description    : Calculates a value for frames-per-second (FPS).
*****************************************************************************/
void PVRShellInit::FpsUpdate()
{
	unsigned int ui32TimeDelta, ui32Time;

	ui32Time = m_pShell->PVRShellGetTime();
	++m_i32FpsFrameCnt;
	ui32TimeDelta = ui32Time - m_i32FpsTimePrev;

	if(ui32TimeDelta >= 1000)
	{
		float fFPS = 1000.0f * (float) m_i32FpsFrameCnt / (float) ui32TimeDelta;

		m_pShell->PVRShellOutputDebug("PVRShell: frame %d, FPS %.1f.\n",
			m_pShell->m_pShellData->nShellCurFrameNum, fFPS);

		m_i32FpsFrameCnt = 0;
		m_i32FpsTimePrev = ui32Time;
	}
}
#endif

/*****************************************************************************
* Function Name  : Run
* Returns        : false when the app should quit
* Description    : Main message loop / render loop
*****************************************************************************/
bool PVRShellInit::Run()
{
	static unsigned long StartTime = 0;

	switch(m_eState)
	{
	case ePVRShellInitApp:
		if(!m_pShell->InitApplication())
		{
			m_eState = ePVRShellExit;
			return true;
		}

	case ePVRShellInitInstance:
		{
			m_CommandLine.Apply(*m_pShell);

			// Output non-api specific data if required
			OutputInfo();

			// Perform OS initialisation
			if(!OsInitOS())
			{
				m_pShell->PVRShellOutputDebug("InitOS failed!\n");
				m_eState = ePVRShellQuitApp;
				return true;
			}

			// Initialize the 3D API
			if(!OsDoInitAPI())
			{
				m_pShell->PVRShellOutputDebug("InitAPI failed!\n");
				m_eState = ePVRShellReleaseOS;
				gShellDone = true;
				return true;
			}

			// Output api specific data if required
			OutputAPIInfo();

			// Initialise the app
			if(!m_pShell->InitView())
			{
				m_pShell->PVRShellOutputDebug("InitView failed!\n");
				m_eState = ePVRShellReleaseAPI;
				gShellDone = true;
				return true;
			}

			if(StartTime==0)
			{
				StartTime = OsGetTime();
			}

			m_eState = ePVRShellRender;
			return true;
		}
	case ePVRShellRender:
		{
			// Main message loop:
			if(!m_pShell->RenderScene())
				break;

			ApiRenderComplete();
			OsRenderComplete();

#ifdef PVRSHELL_FPS_OUTPUT
			if(m_pShell->m_pShellData->bOutputFPS)
				FpsUpdate();
#endif
			int nCurrentFrame = m_pShell->m_pShellData->nShellCurFrameNum;

			if(DoIsKeyPressed(PVRShellKeyNameScreenshot) || (nCurrentFrame >= m_pShell->m_pShellData->nCaptureFrameStart && nCurrentFrame <= m_pShell->m_pShellData->nCaptureFrameStop))
			{
				unsigned char *pBuf;
				if(m_pShell->PVRShellScreenCaptureBuffer(m_pShell->PVRShellGet(prefWidth), m_pShell->PVRShellGet(prefHeight), &pBuf))
				{
					if(m_pShell->PVRShellScreenSave(PVRSHELL_SCREENSHOT_NAME, pBuf) != 0)
					{
						m_pShell->PVRShellSet(prefExitMessage, "Screen-shot save failed.\n");
					}
				}
				else
				{
					m_pShell->PVRShellSet(prefExitMessage, "Screen capture failed.\n");
				}
				FREE(pBuf);
			}

			if(DoIsKeyPressed(PVRShellKeyNameQUIT))
				gShellDone = true;

			if(gShellDone)
				break;

			/* Quit if maximum number of allowed frames is reached */
			if((m_pShell->m_pShellData->nDieAfterFrames>=0) && (nCurrentFrame >= m_pShell->m_pShellData->nDieAfterFrames))
				break;

			/* Quit if maximum time is reached */
			if((m_pShell->m_pShellData->fDieAfterTime>=0.0f) && (((OsGetTime()-StartTime)*0.001f) >= m_pShell->m_pShellData->fDieAfterTime))
				break;

			m_pShell->m_pShellData->nShellCurFrameNum++;
			return true;
		}

	case ePVRShellReleaseView:
		m_pShell->ReleaseView();

	case ePVRShellReleaseAPI:
		OsDoReleaseAPI();

	case ePVRShellReleaseOS:
		OsReleaseOS();

		if(!gShellDone && m_pShell->m_pShellData->nInitRepeats)
		{
			--m_pShell->m_pShellData->nInitRepeats;
			m_eState = ePVRShellInitInstance;
			return true;
		}

	case ePVRShellQuitApp:
		// Final app tidy-up
		m_pShell->QuitApplication();

	case ePVRShellExit:
		OsExit();
		StringCopy(m_pShell->m_pShellData->pszAppName, 0);
		StringCopy(m_pShell->m_pShellData->pszExitMessage, 0);
		return false;
	}

	m_eState = (EPVRShellState)(m_eState + 1);
	return true;
}

/*!***********************************************************************
@Function		OutputInfo
@description	When prefOutputInfo is set to true this function outputs
various pieces of non-API dependent information via
PVRShellOutputDebug.
*************************************************************************/
void PVRShellInit::OutputInfo()
{
	if(m_pShell->PVRShellGet(prefOutputInfo))
	{
		m_pShell->PVRShellOutputDebug("\n");
		m_pShell->PVRShellOutputDebug("App name: %s\n"     , m_pShell->PVRShellGet(prefAppName));
		m_pShell->PVRShellOutputDebug("SDK version: %s\n"  , m_pShell->PVRShellGet(prefVersion));
		m_pShell->PVRShellOutputDebug("\n");
		m_pShell->PVRShellOutputDebug("Read path:  %s\n"    , m_pShell->PVRShellGet(prefReadPath));
		m_pShell->PVRShellOutputDebug("Write path: %s\n"   , m_pShell->PVRShellGet(prefWritePath));
		m_pShell->PVRShellOutputDebug("\n");
		m_pShell->PVRShellOutputDebug("Command-line: %s\n" , m_pShell->PVRShellGet(prefCommandLine));
		m_pShell->PVRShellOutputDebug("\n");
		m_pShell->PVRShellOutputDebug("Power saving: %s\n" , m_pShell->PVRShellGet(prefPowerSaving) ? "On" : "Off");
		m_pShell->PVRShellOutputDebug("FSAAMode requested: %i\n", m_pShell->PVRShellGet(prefFSAAMode));
		m_pShell->PVRShellOutputDebug("Fullscreen: %s\n", m_pShell->PVRShellGet(prefFullScreen) ? "Yes" : "No");
		m_pShell->PVRShellOutputDebug("PBuffer requested: %s\n", m_pShell->PVRShellGet(prefPBufferContext) ? "Yes" : "No");
		m_pShell->PVRShellOutputDebug("ZBuffer requested: %s\n", m_pShell->PVRShellGet(prefZbufferContext) ? "Yes" : "No");
		m_pShell->PVRShellOutputDebug("Stencil buffer requested: %s\n", m_pShell->PVRShellGet(prefStencilBufferContext) ? "Yes" : "No");

		if(m_pShell->PVRShellGet(prefColorBPP) > 0)
			m_pShell->PVRShellOutputDebug("Colour buffer size requested: %i\n", m_pShell->PVRShellGet(prefColorBPP));

		m_pShell->PVRShellOutputDebug("Software rendering requested: %s\n", m_pShell->PVRShellGet(prefSoftwareRendering) ? "Yes" : "No");
		m_pShell->PVRShellOutputDebug("OpenVG requested: %s\n", m_pShell->PVRShellGet(prefOpenVGContext) ? "Yes" : "No");
		m_pShell->PVRShellOutputDebug("Swap Interval requested: %i\n", m_pShell->PVRShellGet(prefSwapInterval));

		if(m_pShell->PVRShellGet(prefInitRepeats) > 0)
			m_pShell->PVRShellOutputDebug("No of Init repeats: %i\n", m_pShell->PVRShellGet(prefInitRepeats));

		if(m_pShell->PVRShellGet(prefQuitAfterFrame) != -1)
			m_pShell->PVRShellOutputDebug("Quit after frame:   %i\n", m_pShell->PVRShellGet(prefQuitAfterFrame));

		if(m_pShell->PVRShellGet(prefQuitAfterTime)  != -1.0f)
			m_pShell->PVRShellOutputDebug("Quit after time:    %f\n", m_pShell->PVRShellGet(prefQuitAfterTime));
	}
}

/****************************************************************************
** Local code
****************************************************************************/
/*!***********************************************************************
@Function		StringCopy
@Modified		pszStr The string to copy pszSrc into
@Input			pszSrc The source string to copy
@description	This function copies pszSrc into pszStr.
*************************************************************************/
static bool StringCopy(char *&pszStr, const char * const pszSrc)
{
	size_t len;

	FREE(pszStr);

	if(!pszSrc)
		return true;

	len = strlen(pszSrc)+1;
	pszStr = (char*)malloc(len);
	if(!pszStr)
		return false;

	strcpy(pszStr, pszSrc);
	return true;
}

/*****************************************************************************
End of file (PVRShell.cpp)
*****************************************************************************/
