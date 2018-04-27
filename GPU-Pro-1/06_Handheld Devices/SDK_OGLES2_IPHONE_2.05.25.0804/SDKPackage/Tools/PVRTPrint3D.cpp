/******************************************************************************

 @File         PVRTPrint3D.cpp

 @Title        PVRTPrint3D

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Displays a text string using 3D polygons. Can be done in two ways:
               using a window defined by the user or writing straight on the
               screen.

******************************************************************************/
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "PVRTGlobal.h"
#include "PVRTFixedPoint.h"
#include "PVRTMatrix.h"
#include "PVRTPrint3D.h"

/* Print3D texture data */
#include "PVRTPrint3DIMGLogo.h"
#include "PVRTPrint3DPVRLogo.h"
#include "PVRTPrint3Ddat.h"


/****************************************************************************
** Defines
****************************************************************************/
#define MAX_LETTERS				(5120)
#define MIN_CACHED_VTX			(0x1000)
#define MAX_CACHED_VTX			(0x00100000)
#define LINES_SPACING			(29.0f)

#define Print3D_WIN_EXIST	1
#define Print3D_WIN_ACTIVE	2
#define Print3D_WIN_TITLE	4
#define Print3D_WIN_STATIC	8
#define Print3D_FULL_OPAQUE	16
#define Print3D_FULL_TRANS	32
#define Print3D_ADJUST_SIZE	64
#define Print3D_NO_BORDER	128

/****************************************************************************
** Class: CPVRTPrint3D
****************************************************************************/

/*****************************************************************************
 @Function			CPVRTPrint3D
 @Description		Init some values.
*****************************************************************************/
CPVRTPrint3D::CPVRTPrint3D()
{
#if !defined(DISABLE_PRINT3D)

	// Initialise all variables
	memset(this, 0, sizeof(*this));

#endif
}

/*****************************************************************************
 @Function			~CPVRTPrint3D
 @Description		De-allocate the working memory
*****************************************************************************/
CPVRTPrint3D::~CPVRTPrint3D()
{
#if !defined (DISABLE_PRINT3D)
#endif
}

/*!***************************************************************************
 @Function			PVRTPrint3DSetTextures
 @Input				pContext		Context
 @Input				dwScreenX		Screen resolution along X
 @Input				dwScreenY		Screen resolution along Y
 @Input				bRotate			Rotate print3D by 90 degrees
 @Return			PVR_SUCCESS or PVR_FAIL
 @Description		Initialization and texture upload. Should be called only once
					for a given context.
*****************************************************************************/
EPVRTError CPVRTPrint3D::SetTextures(
	const SPVRTContext	* const pContext,
	const unsigned int	dwScreenX,
	const unsigned int	dwScreenY,
	const bool bRotate)
{
#if !defined (DISABLE_PRINT3D)

	int				i;
	bool			bStatus;

	/* Set the aspect ratio, so we can chage it without updating textures or anything else */
	float fX, fY;

	m_ui32ScreenDim[0] = bRotate ? dwScreenY : dwScreenX;
	m_ui32ScreenDim[1] = bRotate ? dwScreenX : dwScreenY;

	// Alter the X, Y resolutions if the screen isn't portrait.
	if(dwScreenX > dwScreenY)
	{
		fX = (float) dwScreenX;
		fY = (float) dwScreenY;
	}
	else
	{
		fX = (float) dwScreenY;
		fY = (float) dwScreenX;
	}

	m_fScreenScale[0] = (bRotate ? fY : fX) /640.0f;
	m_fScreenScale[1] = (bRotate ? fX : fY) /480.0f;

	/* Check whether textures are already set up just in case */
	if (m_bTexturesSet)
		return PVR_SUCCESS;

	if(!APIInit(pContext))
		return PVR_FAIL;

	/*
		This is the window background texture
		Type 0 because the data comes in TexTool rectangular format.
	*/
	bStatus = APIUpLoad4444(1, (unsigned char *)WindowBackground, 16, 0);
	if (!bStatus) return PVR_FAIL;

	bStatus = APIUpLoad4444(2, (unsigned char *)WindowPlainBackground, 16, 0);
	if (!bStatus) return PVR_FAIL;

	bStatus = APIUpLoad4444(3, (unsigned char *)WindowBackgroundOp, 16, 0);
	if (!bStatus) return PVR_FAIL;

	bStatus = APIUpLoad4444(4, (unsigned char *)WindowPlainBackgroundOp, 16, 0);
	if (!bStatus) return PVR_FAIL;

	/*
		This is the texture with the fonts.
		Type 1 because there is only alpha component (RGB are white).
	*/
	bStatus = APIUpLoad4444(0, (unsigned char *)PVRTPrint3DABC_Pixels, 256, 1);
	if (!bStatus) return PVR_FAIL;

	/* INDEX BUFFERS */
	m_pwFacesFont = (unsigned short*)malloc(PVRTPRINT3D_MAX_RENDERABLE_LETTERS*2*3*sizeof(unsigned short));

	if(!m_pwFacesFont)
		return PVR_FAIL;

	bStatus = APIUpLoadIcons((const PVRTuint32 *)PVRTPrint3DPVRLogo, (const PVRTuint32 *)PVRTPrint3DIMGLogo);
	if (!bStatus) return PVR_FAIL;

	/* Vertex indices for letters */
	for (i=0; i < PVRTPRINT3D_MAX_RENDERABLE_LETTERS; i++)
	{
		m_pwFacesFont[i*6+0] = 0+i*4;
		m_pwFacesFont[i*6+1] = 3+i*4;
		m_pwFacesFont[i*6+2] = 1+i*4;

		m_pwFacesFont[i*6+3] = 3+i*4;
		m_pwFacesFont[i*6+4] = 0+i*4;
		m_pwFacesFont[i*6+5] = 2+i*4;
	}

	m_nVtxCacheMax = MIN_CACHED_VTX;
	m_pVtxCache = (SPVRTPrint3DAPIVertex*)malloc(m_nVtxCacheMax * sizeof(*m_pVtxCache));
	m_nVtxCache = 0;

	if(!m_pVtxCache)
	{
		return PVR_FAIL;
	}

	/* Everything is OK */
	m_bTexturesSet = true;

	/* set all windows for an update */
	for (i=0; i<PVRTPRINT3D_MAX_WINDOWS; i++)
		m_pWin[i].bNeedUpdated = true;

	/* Return OK */
	return PVR_SUCCESS;

#else
	return PVR_SUCCESS;
#endif
}

/*!***************************************************************************
 @Function			PVRTPrint3D
 @Input				fPosX		Position of the text along X
 @Input				fPosY		Position of the text along Y
 @Input				fScale		Scale of the text
 @Input				Colour		Colour of the text
 @Input				pszFormat	Format string for the text
 @Return			PVR_SUCCESS or PVR_FAIL
 @Description		Display 3D text on screen.
					No window needs to be allocated to use this function.
					However, PVRTPrint3DSetTextures(...) must have been called
					beforehand.
					This function accepts formatting in the printf way.
*****************************************************************************/
EPVRTError CPVRTPrint3D::Print3D(float fPosX, float fPosY, const float fScale, unsigned int Colour, const char * const pszFormat, ...)
{
#if !defined (DISABLE_PRINT3D)

	va_list				args;
	static char			Text[MAX_LETTERS+1], sPreviousString[MAX_LETTERS+1];
	static float		XPosPrev, YPosPrev, fScalePrev;
	static unsigned int	ColourPrev;
	static unsigned int	nVertices;

	/* No textures! so... no window */
	if (!m_bTexturesSet)
	{
		PVRTErrorOutputDebug("PVRTPrint3DDisplayWindow : You must call PVRTPrint3DSetTextures()\nbefore using this function!!!\n");
		return PVR_FAIL;
	}

	/* Reading the arguments to create our Text string */
	va_start(args, pszFormat);
	vsprintf(Text, pszFormat, args);		// Could use _vsnprintf but e.g. LinuxVP does not support it
	va_end(args);

	/* nothing to be drawn */
	if(*Text == 0)
		return PVR_FAIL;

	/* Adjust input parameters */
	fPosX *= 640.0f/100.0f;
	fPosY *= 480.0f/100.0f;

	/* We check if the string has been changed since last time */
	if(
		strcmp (sPreviousString, Text) != 0 ||
		fPosX != XPosPrev ||
		fPosY != YPosPrev ||
		fScale != fScalePrev ||
		Colour != ColourPrev ||
		m_pPrint3dVtx == NULL)
	{
		/* copy strings */
		strcpy (sPreviousString, Text);
		XPosPrev = fPosX;
		YPosPrev = fPosY;
		fScalePrev = fScale;
		ColourPrev = Colour;

		/* Create Vertex Buffer (only if it doesn't exist) */
		if(m_pPrint3dVtx == 0)
		{
			m_pPrint3dVtx = (SPVRTPrint3DAPIVertex*)malloc(MAX_LETTERS*4*sizeof(SPVRTPrint3DAPIVertex));

			if(!m_pPrint3dVtx)
				return PVR_FAIL;
		}

		/* Fill up our buffer */
		nVertices = UpdateLine(0, 0.0f, fPosX, fPosY, fScale, Colour, Text, m_pPrint3dVtx);
	}

	// Draw the text
	DrawLineUP(m_pPrint3dVtx, nVertices);
#endif

	return PVR_SUCCESS;
}
/*!***************************************************************************
 @Function			DisplayDefaultTitle
 @Input				sTitle				Title to display
 @Input				sDescription		Description to display
 @Input				uDisplayLogo		1 = Display the logo
 @Return			PVR_SUCCESS or PVR_FAIL
 @Description		Creates a default title with predefined position and colours.
					It displays as well company logos when requested:
					0 = No logo
					1 = PowerVR logo
					2 = Img Tech logo
*****************************************************************************/
EPVRTError CPVRTPrint3D::DisplayDefaultTitle(const char * const pszTitle, const char * const pszDescription, const unsigned int uDisplayLogo)
{
	EPVRTError eRet = PVR_SUCCESS;

#if !defined (DISABLE_PRINT3D)

	/* Display Title
	 */
	if(pszTitle)
	{
		if(Print3D(0.0f, 1.0f, 1.2f,  PVRTRGBA(255, 255, 0, 255), pszTitle) != PVR_SUCCESS)
			eRet = PVR_FAIL;
	}

	/* Display Description
	 */
	if(pszDescription)
	{
		if(Print3D(0.0f, 8.0f, 0.9f,  PVRTRGBA(255, 255, 255, 255), pszDescription) != PVR_SUCCESS)
			eRet = PVR_FAIL;
	}

	m_uLogoToDisplay = uDisplayLogo;

#endif

	return eRet;
}
/*!***************************************************************************
 @Function			CreateDefaultWindow
 @Input				fPosX					Position X for the new window
 @Input				fPosY					Position Y for the new window
 @Input				nXSize_LettersPerLine
 @Input				sTitle					Title of the window
 @Input				sBody					Body text of the window
 @Return			Window handle
 @Description		Creates a default window.
					If Title is NULL the main body will have just one line
					(for InfoWin).
*****************************************************************************/
unsigned int CPVRTPrint3D::CreateDefaultWindow(float fPosX, float fPosY, int nXSize_LettersPerLine, char *sTitle, char *sBody)
{
#if !defined (DISABLE_PRINT3D)

	unsigned int dwActualWin;
	unsigned int dwFlags = ePVRTPrint3D_ADJUST_SIZE_ALWAYS;
	unsigned int dwBodyTextColor, dwBodyBackgroundColor;

	/* If no text is specified, return an error */
	if(!sBody && !sTitle) return 0xFFFFFFFF;

	/* If no title is specified, body text colours are different */
	if(!sTitle)
	{
		dwBodyTextColor			= PVRTRGBA(0xFF, 0xFF, 0x30, 0xFF);
		dwBodyBackgroundColor	= PVRTRGBA(0x20, 0x20, 0xB0, 0xE0);
	}
	else
	{
		dwBodyTextColor			= PVRTRGBA(0xFF, 0xFF, 0xFF, 0xFF);
		dwBodyBackgroundColor	= PVRTRGBA(0x20, 0x30, 0xFF, 0xE0);
	}

	/* Set window flags depending on title and body text were specified */
	if(!sBody)		dwFlags |= ePVRTPrint3D_DEACTIVATE_WIN;
	if(!sTitle)		dwFlags |= ePVRTPrint3D_DEACTIVATE_TITLE;

	/* Create window */
	dwActualWin = InitWindow(nXSize_LettersPerLine, (sTitle==NULL) ? 1:50);

	/* Set window properties */
	SetWindow(dwActualWin, dwBodyBackgroundColor, dwBodyTextColor, 0.5f, fPosX, fPosY, 20.0f, 20.0f);

	/* Set title */
	if (sTitle)
		SetTitle(dwActualWin, PVRTRGBA(0x20, 0x20, 0xB0, 0xE0), 0.6f, PVRTRGBA(0xFF, 0xFF, 0x30, 0xFF), sTitle, PVRTRGBA(0xFF, 0xFF, 0x30, 0xFF), (char*)"");

	/* Set window text */
	if (sBody)
		SetText(dwActualWin, sBody);

	/* Set window flags */
	SetWindowFlags(dwActualWin, dwFlags);

	m_pWin[dwActualWin].bNeedUpdated = true;

	/* Return window handle */
	return dwActualWin;

#else
	return 0;
#endif
}

/*!***************************************************************************
 @Function			InitWindow
 @Input				dwBufferSizeX		Buffer width
 @Input				dwBufferSizeY		Buffer height
 @Return			Window handle
 @Description		Allocate a buffer for a newly-created window and return its
					handle.
*****************************************************************************/
unsigned int CPVRTPrint3D::InitWindow(unsigned int dwBufferSizeX, unsigned int dwBufferSizeY)
{
#if !defined (DISABLE_PRINT3D)

	unsigned int		dwCurrentWin;

	/* Find the first available window */
	for (dwCurrentWin=1; dwCurrentWin<PVRTPRINT3D_MAX_WINDOWS; dwCurrentWin++)
	{
		/* If this window available? */
		if (!(m_pWin[dwCurrentWin].dwFlags & Print3D_WIN_EXIST))
		{
			/* Window available, exit loop */
			break;
		}
	}

	/* No more windows available? */
	if (dwCurrentWin == PVRTPRINT3D_MAX_WINDOWS)
	{
		_RPT0(_CRT_WARN,"\nPVRTPrint3DCreateWindow WARNING: PVRTPRINT3D_MAX_WINDOWS overflow.\n");
		return 0;
	}

	/* Set flags */
	m_pWin[dwCurrentWin].dwFlags = Print3D_WIN_TITLE  | Print3D_WIN_EXIST | Print3D_WIN_ACTIVE;

	/* Text Buffer */
	m_pWin[dwCurrentWin].dwBufferSizeX = dwBufferSizeX + 1;
	m_pWin[dwCurrentWin].dwBufferSizeY = dwBufferSizeY;
	m_pWin[dwCurrentWin].pTextBuffer  = (char *)calloc((dwBufferSizeX+2)*(dwBufferSizeY+2), sizeof(char));
	m_pWin[dwCurrentWin].bTitleTextL  = (char *)calloc(MAX_LETTERS, sizeof(char));
	m_pWin[dwCurrentWin].bTitleTextR  = (char *)calloc(MAX_LETTERS, sizeof(char));

	/* Memory allocation failed */
	if (!m_pWin[dwCurrentWin].pTextBuffer || !m_pWin[dwCurrentWin].bTitleTextL || !m_pWin[dwCurrentWin].bTitleTextR)
	{
		_RPT0(_CRT_WARN,"\nPVRTPrint3DCreateWindow : No memory enough for Text Buffer.\n");
		return 0;
	}

	/* Title */
	m_pWin[dwCurrentWin].fTitleFontSize	= 1.0f;
	m_pWin[dwCurrentWin].dwTitleFontColorL = PVRTRGBA(0xFF, 0xFF, 0x00, 0xFF);
	m_pWin[dwCurrentWin].dwTitleFontColorR = PVRTRGBA(0xFF, 0xFF, 0x00, 0xFF);
	m_pWin[dwCurrentWin].dwTitleBaseColor  = PVRTRGBA(0x30, 0x30, 0xFF, 0xFF); /* Dark Blue */

	/* Window */
	m_pWin[dwCurrentWin].fWinFontSize		= 0.5f;
	m_pWin[dwCurrentWin].dwWinFontColor	= PVRTRGBA(0xFF, 0xFF, 0xFF, 0xFF);
	m_pWin[dwCurrentWin].dwWinBaseColor	= PVRTRGBA(0x80, 0x80, 0xFF, 0xFF); /* Light Blue */
	m_pWin[dwCurrentWin].fWinPos[0]		= 0.0f;
	m_pWin[dwCurrentWin].fWinPos[1]		= 0.0f;
	m_pWin[dwCurrentWin].fWinSize[0]		= 20.0f;
	m_pWin[dwCurrentWin].fWinSize[1]		= 20.0f;
	m_pWin[dwCurrentWin].fZPos		        = 0.0f;
	m_pWin[dwCurrentWin].dwSort		    = 0;

	m_pWin[dwCurrentWin].bNeedUpdated = true;

	dwCurrentWin++;

	/* Returning the handle */
	return (dwCurrentWin-1);

#else
	return 0;
#endif
}

/*!***************************************************************************
 @Function			DeleteWindow
 @Input				dwWin		Window handle
 @Description		Delete the window referenced by dwWin.
*****************************************************************************/
void CPVRTPrint3D::DeleteWindow(unsigned int dwWin)
{
#if !defined (DISABLE_PRINT3D)

	int i;

	/* Release VertexBuffer */
	FREE(m_pWin[dwWin].pTitleVtxL);
	FREE(m_pWin[dwWin].pTitleVtxR);
	FREE(m_pWin[dwWin].pWindowVtxTitle);
	FREE(m_pWin[dwWin].pWindowVtxText);

	for(i=0; i<255; i++)
		FREE(m_pWin[dwWin].pLineVtx[i]);

	/* Only delete window if it exists */
	if(m_pWin[dwWin].dwFlags & Print3D_WIN_EXIST)
	{
		FREE(m_pWin[dwWin].pTextBuffer);
		FREE(m_pWin[dwWin].bTitleTextL);
		FREE(m_pWin[dwWin].bTitleTextR);
	}

	/* Reset flags */
	m_pWin[dwWin].dwFlags = 0;

#endif
}

/*!***************************************************************************
 @Function			DeleteAllWindows
 @Description		Delete all windows.
*****************************************************************************/
void CPVRTPrint3D::DeleteAllWindows()
{
#if !defined (DISABLE_PRINT3D)

	int unsigned i;

	for (i=0; i<PVRTPRINT3D_MAX_WINDOWS; i++)
		DeleteWindow (i);

#endif
}

/*!***************************************************************************
 @Function			DisplayWindow
 @Input				dwWin
 @Return			PVR_SUCCESS or PVR_FAIL
 @Description		Display window.
					This function MUST be called between a BeginScene/EndScene
					pair as it uses D3D render primitive calls.
					PVRTPrint3DSetTextures(...) must have been called beforehand.
*****************************************************************************/
EPVRTError CPVRTPrint3D::DisplayWindow(unsigned int dwWin)
{
#if !defined (DISABLE_PRINT3D)

	unsigned int	i;
	float			fTitleSize = 0.0f;

	/* No textures! so... no window */
	if (!m_bTexturesSet)
	{
		_RPT0(_CRT_WARN,"PVRTPrint3DDisplayWindow : You must call PVRTPrint3DSetTextures()\nbefore using this function!!!\n");
		return PVR_FAIL;
	}

	/* Update Vertex data only when needed */
	if(m_pWin[dwWin].bNeedUpdated)
	{
		/* TITLE */
		if(m_pWin[dwWin].dwFlags & Print3D_WIN_TITLE)
		{
			/* Set title size */
			if(m_pWin[dwWin].fTitleFontSize < 0.0f)
				fTitleSize = 8.0f + 16.0f;
			else
				fTitleSize = m_pWin[dwWin].fTitleFontSize * 23.5f + 16.0f;

			/* Title */
			UpdateTitleVertexBuffer(dwWin);

			/* Background */
			if (!(m_pWin[dwWin].dwFlags & Print3D_FULL_TRANS))
			{
				/* Draw title background */
				UpdateBackgroundWindow(
					dwWin, m_pWin[dwWin].dwTitleBaseColor,
					0.0f,
					m_pWin[dwWin].fWinPos[0],
					m_pWin[dwWin].fWinPos[1],
					m_pWin[dwWin].fWinSize[0],
					fTitleSize, &m_pWin[dwWin].pWindowVtxTitle);
			}
		}

		/* Main text */
		UpdateMainTextVertexBuffer(dwWin);

		UpdateBackgroundWindow(
			dwWin, m_pWin[dwWin].dwWinBaseColor,
			0.0f,
			m_pWin[dwWin].fWinPos[0],
			(m_pWin[dwWin].fWinPos[1] + fTitleSize),
			m_pWin[dwWin].fWinSize[0],
			m_pWin[dwWin].fWinSize[1], &m_pWin[dwWin].pWindowVtxText);

		/* Don't update until next change makes it needed */
		m_pWin[dwWin].bNeedUpdated = false;
	}

	// Ensure any previously drawn text has been submitted before drawing the window.
	Flush();

	/* Save current render states */
	APIRenderStates(0);

	/*
		DRAW TITLE
	*/
	if(m_pWin[dwWin].dwFlags & Print3D_WIN_TITLE)
	{
		if (!(m_pWin[dwWin].dwFlags & Print3D_FULL_TRANS))
		{
			DrawBackgroundWindowUP(dwWin, m_pWin[dwWin].pWindowVtxTitle, (m_pWin[dwWin].dwFlags & Print3D_FULL_OPAQUE) ? true : false, (m_pWin[dwWin].dwFlags & Print3D_NO_BORDER) ? false : true);
		}

		/* Left and Right text */
		DrawLineUP(m_pWin[dwWin].pTitleVtxL, m_pWin[dwWin].nTitleVerticesL);
		DrawLineUP(m_pWin[dwWin].pTitleVtxR, m_pWin[dwWin].nTitleVerticesR);
	}

	/*
		DRAW WINDOW
	*/
	if (m_pWin[dwWin].dwFlags & Print3D_WIN_ACTIVE)
	{
		/* Background */
		if (!(m_pWin[dwWin].dwFlags & Print3D_FULL_TRANS))
		{
			DrawBackgroundWindowUP(dwWin, m_pWin[dwWin].pWindowVtxText, (m_pWin[dwWin].dwFlags & Print3D_FULL_OPAQUE) ? true : false, (m_pWin[dwWin].dwFlags & Print3D_NO_BORDER) ? false : true);
		}

		/* Text, line by line */
		for (i=0; i<m_pWin[dwWin].dwBufferSizeY; i++)
		{
			DrawLineUP(m_pWin[dwWin].pLineVtx[i], m_pWin[dwWin].nLineVertices[i]);
		}
	}

	/* Restore render states */
	APIRenderStates(1);

#endif

	return PVR_SUCCESS;
}

/*!***************************************************************************
 @Function			SetText
 @Input				dwWin		Window handle
 @Input				Format		Format string
 @Return			PVR_SUCCESS or PVR_FAIL
 @Description		Feed the text buffer of window referenced by dwWin.
					This function accepts formatting in the printf way.
*****************************************************************************/
EPVRTError CPVRTPrint3D::SetText(unsigned int dwWin, const char *Format, ...)
{
#if !defined (DISABLE_PRINT3D)

	va_list			args;
	unsigned int			i;
	unsigned int			dwBufferSize, dwTotalLength = 0;
	unsigned int			dwPosBx, dwPosBy, dwSpcPos;
	char			bChar;
	unsigned int			dwCursorPos;
	static char	sText[MAX_LETTERS+1];

	/* If window doesn't exist then return from function straight away */
	if (!(m_pWin[dwWin].dwFlags & Print3D_WIN_EXIST))
		return PVR_FAIL;

	// Empty the window buffer
	memset(m_pWin[dwWin].pTextBuffer, 0, m_pWin[dwWin].dwBufferSizeX * m_pWin[dwWin].dwBufferSizeY * sizeof(char));

	/* Reading the arguments to create our Text string */
	va_start(args,Format);
	vsprintf(sText, Format, args);		// Could use _vsnprintf but e.g. LinuxVP does not support it
	va_end(args);

	dwCursorPos	= 0;

	m_pWin[dwWin].bNeedUpdated = true;

	/* Compute buffer size */
	dwBufferSize = (m_pWin[dwWin].dwBufferSizeX+1) * (m_pWin[dwWin].dwBufferSizeY+1);

	/* Compute length */
	while(dwTotalLength < dwBufferSize && sText[dwTotalLength] != 0)
		dwTotalLength++;

	/* X and Y pointer position */
	dwPosBx = 0;
	dwPosBy = 0;

	/* Process each character */
	for (i=0; i<dwTotalLength; i++)
	{
		/* Get current character in string */
		bChar = sText[i];

		/* Space (for word wrap only) */
		if (bChar == ' ')
		{
			/* Looking for the next space (or return or end) */
			dwSpcPos = 1;
			do
			{
				bChar = sText[i + dwSpcPos++];
			}
			while (bChar==' ' || bChar==0x0A || bChar==0);
			bChar = ' ';

			/*
				Humm, if this word is longer than the buffer don't move it.
				Otherwise check if it is at the end and create a return.
			*/
			if (dwSpcPos<m_pWin[dwWin].dwBufferSizeX && (dwPosBx+dwSpcPos)>m_pWin[dwWin].dwBufferSizeX)
			{
				/* Set NULL character */
				m_pWin[dwWin].pTextBuffer[dwCursorPos++] = 0;

				dwPosBx = 0;
				dwPosBy++;

				/* Set new cursor position */
				dwCursorPos = dwPosBy * m_pWin[dwWin].dwBufferSizeX;

				/* Don't go any further */
				continue;
			}
		}

		/* End of line */
		if (dwPosBx == (m_pWin[dwWin].dwBufferSizeX-1))
		{
			m_pWin[dwWin].pTextBuffer[dwCursorPos++] = 0;
			dwPosBx = 0;
			dwPosBy++;
		}

		/* Vertical Scroll */
		if (dwPosBy >= m_pWin[dwWin].dwBufferSizeY)
		{
			memcpy(m_pWin[dwWin].pTextBuffer,
				m_pWin[dwWin].pTextBuffer + m_pWin[dwWin].dwBufferSizeX,
				(m_pWin[dwWin].dwBufferSizeX-1) * m_pWin[dwWin].dwBufferSizeY);

			dwCursorPos -= m_pWin[dwWin].dwBufferSizeX;

			dwPosBx = 0;
			dwPosBy--;
		}

		/* Return */
		if (bChar == 0x0A)
		{
			/* Set NULL character */
			m_pWin[dwWin].pTextBuffer[dwCursorPos++] = 0;

			dwPosBx = 0;
			dwPosBy++;

			dwCursorPos = dwPosBy * m_pWin[dwWin].dwBufferSizeX;

			/* Don't go any further */
			continue;
		}

		/* Storing our character */
		if (dwCursorPos<dwBufferSize)
		{
			m_pWin[dwWin].pTextBuffer[dwCursorPos++] = bChar;
		}

		/* Increase position */
		dwPosBx++;
	}

	/* Automatic adjust of the window size */
	if (m_pWin[dwWin].dwFlags & Print3D_ADJUST_SIZE)
	{
		AdjustWindowSize(dwWin, 0);
	}

#endif

	return PVR_SUCCESS;
}

/*!***************************************************************************
 @Function			SetWindow
 @Input				dwWin			Window handle
 @Input				dwWinColor		Window colour
 @Input				dwFontColor		Font colour
 @Input				fFontSize		Font size
 @Input				fPosX			Window position X
 @Input				fPosY			Window position Y
 @Input				fSizeX			Window size X
 @Input				fSizeY			Window size Y
 @Description		Set attributes of window.
					Windows position and size are referred to a virtual screen
					of 100x100. (0,0) is the top-left corner and (100,100) the
					bottom-right corner.
					These values are the same for all resolutions.
*****************************************************************************/
void CPVRTPrint3D::SetWindow(unsigned int dwWin, unsigned int dwWinColor, unsigned int dwFontColor, float fFontSize,
						  float fPosX, float fPosY, float fSizeX, float fSizeY)
{
#if !defined (DISABLE_PRINT3D)

	/* Check if there is a real change */
	if(	m_pWin[dwWin].fWinFontSize		!= fFontSize ||
		m_pWin[dwWin].dwWinFontColor	!= dwFontColor ||
		m_pWin[dwWin].dwWinBaseColor	!= dwWinColor ||
		m_pWin[dwWin].fWinPos[0]		!= fPosX  * 640.0f/100.0f ||
		m_pWin[dwWin].fWinPos[1]		!= fPosY  * 480.0f/100.0f ||
		m_pWin[dwWin].fWinSize[0]		!= fSizeX * 640.0f/100.0f ||
		m_pWin[dwWin].fWinSize[1]		!= fSizeY * 480.0f/100.0f)
	{
		/* Set window properties */
		m_pWin[dwWin].fWinFontSize		= fFontSize;
		m_pWin[dwWin].dwWinFontColor	= dwFontColor;
		m_pWin[dwWin].dwWinBaseColor	= dwWinColor;
		m_pWin[dwWin].fWinPos[0]		= fPosX  * 640.0f/100.0f;
		m_pWin[dwWin].fWinPos[1]		= fPosY  * 480.0f/100.0f;
		m_pWin[dwWin].fWinSize[0]		= fSizeX * 640.0f/100.0f;
		m_pWin[dwWin].fWinSize[1]		= fSizeY * 480.0f/100.0f;

		m_pWin[dwWin].bNeedUpdated = true;
	}

#endif
}

/*!***************************************************************************
 @Function			SetTitle
 @Input				dwWin				Window handle
 @Input				dwBackgroundColor	Background color
 @Input				fFontSize			Font size
 @Input				dwFontColorLeft
 @Input				sTitleLeft
 @Input				dwFontColorRight
 @Input				sTitleRight
 @Description		Set window title.
*****************************************************************************/
void CPVRTPrint3D::SetTitle(unsigned int dwWin, unsigned int dwBackgroundColor, float fFontSize,
						 unsigned int dwFontColorLeft, char *sTitleLeft,
						 unsigned int dwFontColorRight, char *sTitleRight)
{
#if !defined (DISABLE_PRINT3D)

	FREE(m_pWin[dwWin].pTitleVtxL);
	FREE(m_pWin[dwWin].pTitleVtxR);

	if(sTitleLeft)  memcpy(m_pWin[dwWin].bTitleTextL, sTitleLeft , PVRT_MIN((size_t)(MAX_LETTERS-1), strlen(sTitleLeft )+1));
	if(sTitleRight) memcpy(m_pWin[dwWin].bTitleTextR, sTitleRight, PVRT_MIN((size_t)(MAX_LETTERS-1), strlen(sTitleRight)+1));

	/* Set title properties */
	m_pWin[dwWin].fTitleFontSize		= fFontSize;
	m_pWin[dwWin].dwTitleFontColorL	= dwFontColorLeft;
	m_pWin[dwWin].dwTitleFontColorR	= dwFontColorRight;
	m_pWin[dwWin].dwTitleBaseColor	= dwBackgroundColor;
	m_pWin[dwWin].fTextRMinPos		= GetLength(m_pWin[dwWin].fTitleFontSize, m_pWin[dwWin].bTitleTextL) + 10.0f;
	m_pWin[dwWin].bNeedUpdated		= true;

#endif
}

/*!***************************************************************************
 @Function			SetWindowFlags
 @Input				dwWin				Window handle
 @Input				dwFlags				Flags
 @Description		Set flags for window referenced by dwWin.
					A list of flag can be found at the top of this header.
*****************************************************************************/
void CPVRTPrint3D::SetWindowFlags(unsigned int dwWin, unsigned int dwFlags)
{
#if !defined (DISABLE_PRINT3D)

	/* Check if there is need of updating vertex buffers */
	if(	dwFlags & ePVRTPrint3D_ACTIVATE_TITLE ||
		dwFlags & ePVRTPrint3D_DEACTIVATE_TITLE ||
		dwFlags & ePVRTPrint3D_ADJUST_SIZE_ALWAYS)
		m_pWin[dwWin].bNeedUpdated = true;

	/* Set window flags */
	if (dwFlags & ePVRTPrint3D_ACTIVATE_WIN)		m_pWin[dwWin].dwFlags |= Print3D_WIN_ACTIVE;
	if (dwFlags & ePVRTPrint3D_DEACTIVATE_WIN)	m_pWin[dwWin].dwFlags &= ~Print3D_WIN_ACTIVE;
	if (dwFlags & ePVRTPrint3D_ACTIVATE_TITLE)	m_pWin[dwWin].dwFlags |= Print3D_WIN_TITLE;
	if (dwFlags & ePVRTPrint3D_DEACTIVATE_TITLE) m_pWin[dwWin].dwFlags &= ~Print3D_WIN_TITLE;
	if (dwFlags & ePVRTPrint3D_FULL_OPAQUE)		m_pWin[dwWin].dwFlags |= Print3D_FULL_OPAQUE;
	if (dwFlags & ePVRTPrint3D_FULL_TRANS)		m_pWin[dwWin].dwFlags |= Print3D_FULL_TRANS;

	if (dwFlags & ePVRTPrint3D_ADJUST_SIZE_ALWAYS)
	{
		m_pWin[dwWin].dwFlags |= Print3D_ADJUST_SIZE;
		AdjustWindowSize(dwWin, 0);
	}

	if (dwFlags & ePVRTPrint3D_NO_BORDER)	m_pWin[dwWin].dwFlags |= Print3D_NO_BORDER;

#endif
}

/*!***************************************************************************
 @Function			AdjustWindowSize
 @Input				dwWin				Window handle
 @Input				dwMode				dwMode 0 = Both, dwMode 1 = X only,  dwMode 2 = Y only
 @Description		Calculates window size so that all text fits in the window.
*****************************************************************************/
void CPVRTPrint3D::AdjustWindowSize(unsigned int dwWin, unsigned int dwMode)
{
#if !defined (DISABLE_PRINT3D)

	int unsigned i;
	unsigned int dwPointer = 0;
	float fMax = 0.0f, fLength;

	if (dwMode==1 || dwMode==0)
	{
		/* Title horizontal Size */
		if(m_pWin[dwWin].dwFlags & Print3D_WIN_TITLE)
		{
			fMax = GetLength(m_pWin[dwWin].fTitleFontSize, m_pWin[dwWin].bTitleTextL);

			if (m_pWin[dwWin].bTitleTextR)
			{
				fMax += GetLength(m_pWin[dwWin].fTitleFontSize, m_pWin[dwWin].bTitleTextR) + 12.0f;
			}
		}

		/* Body horizontal size (line by line) */
		for (i=0; i<m_pWin[dwWin].dwBufferSizeY; i++)
		{
			fLength = GetLength(m_pWin[dwWin].fWinFontSize, (m_pWin[dwWin].pTextBuffer + dwPointer));

			if (fLength > fMax) fMax = fLength;

			dwPointer += m_pWin[dwWin].dwBufferSizeX;
		}

		m_pWin[dwWin].fWinSize[0] = fMax - 2.0f + 16.0f;
	}

	/* Vertical Size */
	if(dwMode==0 || dwMode==2)
	{
		if(m_pWin[dwWin].dwBufferSizeY < 2)
		{
			i = 0;
		}
		else
		{
			/* Looking for the last line */
			i=m_pWin[dwWin].dwBufferSizeY;
			while(i)
			{
				--i;
				if (m_pWin[dwWin].pTextBuffer[m_pWin[dwWin].dwBufferSizeX * i])
					break;
			}
		}

		if (m_pWin[dwWin].fWinFontSize>0)
			m_pWin[dwWin].fWinSize[1] = (float)(i+1) * LINES_SPACING * m_pWin[dwWin].fWinFontSize + 16.0f;
		else
			m_pWin[dwWin].fWinSize[1] = ((float)(i+1) * 12.0f) + 16.0f;
	}

	m_pWin[dwWin].bNeedUpdated = true;

#endif
}

/*!***************************************************************************
 @Function			GetSize
 @Output			pfWidth				Width of the string in pixels
 @Output			pfHeight			Height of the string in pixels
 @Input				fFontSize			Font size
 @Input				sString				String to take the size of
 @Description		Returns the size of a string in pixels.
*****************************************************************************/
void CPVRTPrint3D::GetSize(
	float		* const pfWidth,
	float		* const pfHeight,
	const float	fFontSize,
	const char	* sString)
{
#if !defined (DISABLE_PRINT3D)

	unsigned char Val;
	float fScale, fSize;

	if(sString == NULL) {
		if(pfWidth)
			*pfWidth = 0;
		if(pfHeight)
			*pfHeight = 0;
		return;
	}

	if(fFontSize > 0.0f) /* Arial font */
	{
		fScale = fFontSize;
		fSize  = 0.0f;

		Val = *sString++;
		while(Val)
		{
			if(Val==' ')
				Val = '0';

			if(Val>='0' && Val <= '9')
				Val = '0'; /* That's for fixing the number width */

			fSize += PVRTPrint3DSize_Bold[Val] * 40.0f * fScale ;

			/* these letters are too narrow due to a bug in the table */
			if(Val=='i' || Val == 'l' || Val == 'j')
				fSize += 0.4f* fScale;
			Val = *sString++;
		}

		if(pfHeight)
			*pfHeight = m_fScreenScale[1] * fScale * 27.0f * (100.0f / 640.0f);
	}
	else /* System font */
	{
		fScale = 255.0f;
		fSize  = 0.0f;

		Val = *sString++;
		while (Val)
		{
			if(Val == ' ') {
				fSize += 5.0f;
				continue;
			}

			if(Val>='0' && Val <= '9')
				Val = '0'; /* That's for fixing the number width */

			fSize += PVRTPrint3DSize_System[Val]  * fScale * (100.0f / 640.0f);
			Val = *sString++;
		}

		if(pfHeight)
			*pfHeight = m_fScreenScale[1] * 12.0f;
	}

	if(pfWidth)
		*pfWidth = fSize;

#endif
}

/*!***************************************************************************
 @Function			GetAspectRatio
 @Output			dwScreenX		Screen resolution X
 @Output			dwScreenY		Screen resolution Y
 @Description		Returns the current resolution used by Print3D
*****************************************************************************/
void CPVRTPrint3D::GetAspectRatio(unsigned int *dwScreenX, unsigned int *dwScreenY)
{
#if !defined (DISABLE_PRINT3D)

	*dwScreenX = (int)(640.0f * m_fScreenScale[0]);
	*dwScreenY = (int)(480.0f * m_fScreenScale[1]);

#endif
}

/*************************************************************
*					 PRIVATE FUNCTIONS						 *
**************************************************************/

/*!***************************************************************************
 @Function			UpdateBackgroundWindow
 @Return			true if succesful, false otherwise.
 @Description		Draw a generic rectangle (with or without border).
*****************************************************************************/
bool CPVRTPrint3D::UpdateBackgroundWindow(unsigned int /*dwWin*/, unsigned int Color, float fZPos, float fPosX, float fPosY, float fSizeX, float fSizeY, SPVRTPrint3DAPIVertex **ppVtx)
{
	int				i;
	SPVRTPrint3DAPIVertex	*vBox;
	float			fU[] = { 0.0f, 0.0f, 6.0f, 6.0f, 10.0f,10.0f, 16.0f,16.0f,10.0f,16.0f,10.0f,16.0f,6.0f,6.0f,0.0f,0.0f};
	float			fV[] = { 0.0f, 6.0f, 0.0f, 6.0f, 0.0f, 6.0f, 0.0f, 6.0f, 10.0f, 10.0f, 16.0f,16.0f, 16.0f, 10.0f, 16.0f, 10.0f};

	/* Create our vertex buffers */
	if(*ppVtx==0)
	{
		*ppVtx = (SPVRTPrint3DAPIVertex*)malloc(16*sizeof(SPVRTPrint3DAPIVertex));

		if(!*ppVtx)
			return false;
	}
	vBox = *ppVtx;


	/* Removing the border */
	fSizeX -= 16.0f ;
	fSizeY -= 16.0f ;

	/* Set Z position, color and texture coordinates in array */
	for (i=0; i<16; i++)
	{
		vBox[i].sz		= f2vt(fZPos);
		vBox[i].color	= Color;
		vBox[i].tu		= f2vt(fU[i]/16.0f);
		vBox[i].tv		= f2vt(1.0f - fV[i]/16.0f);
	}

	/* Set coordinates in array */
	vBox[0].sx = f2vt((fPosX + fU[0]) * m_fScreenScale[0]);
	vBox[0].sy = f2vt((fPosY + fV[0]) * m_fScreenScale[1]);

	vBox[1].sx = f2vt((fPosX + fU[1]) * m_fScreenScale[0]);
	vBox[1].sy = f2vt((fPosY + fV[1]) * m_fScreenScale[1]);

	vBox[2].sx = f2vt((fPosX + fU[2]) * m_fScreenScale[0]);
	vBox[2].sy = f2vt((fPosY + fV[2]) * m_fScreenScale[1]);

	vBox[3].sx = f2vt((fPosX + fU[3]) * m_fScreenScale[0]);
	vBox[3].sy = f2vt((fPosY + fV[3]) * m_fScreenScale[1]);

	vBox[4].sx = f2vt((fPosX + fU[4] + fSizeX) * m_fScreenScale[0]);
	vBox[4].sy = f2vt((fPosY + fV[4]) * m_fScreenScale[1]);

	vBox[5].sx = f2vt((fPosX + fU[5] + fSizeX) * m_fScreenScale[0]);
	vBox[5].sy = f2vt((fPosY + fV[5]) * m_fScreenScale[1]);

	vBox[6].sx = f2vt((fPosX + fU[6] + fSizeX) * m_fScreenScale[0]);
	vBox[6].sy = f2vt((fPosY + fV[6]) * m_fScreenScale[1]);

	vBox[7].sx = f2vt((fPosX + fU[7] + fSizeX) * m_fScreenScale[0]);
	vBox[7].sy = f2vt((fPosY + fV[7]) * m_fScreenScale[1]);

	vBox[8].sx = f2vt((fPosX + fU[8] + fSizeX) * m_fScreenScale[0]);
	vBox[8].sy = f2vt((fPosY + fV[8] + fSizeY) * m_fScreenScale[1]);

	vBox[9].sx = f2vt((fPosX + fU[9] + fSizeX) * m_fScreenScale[0]);
	vBox[9].sy = f2vt((fPosY + fV[9] + fSizeY) * m_fScreenScale[1]);

	vBox[10].sx = f2vt((fPosX + fU[10] + fSizeX) * m_fScreenScale[0]);
	vBox[10].sy = f2vt((fPosY + fV[10] + fSizeY) * m_fScreenScale[1]);

	vBox[11].sx = f2vt((fPosX + fU[11] + fSizeX) * m_fScreenScale[0]);
	vBox[11].sy = f2vt((fPosY + fV[11] + fSizeY) * m_fScreenScale[1]);

	vBox[12].sx = f2vt((fPosX + fU[12]) * m_fScreenScale[0]);
	vBox[12].sy = f2vt((fPosY + fV[12] + fSizeY) * m_fScreenScale[1]);

	vBox[13].sx = f2vt((fPosX + fU[13]) * m_fScreenScale[0]);
	vBox[13].sy = f2vt((fPosY + fV[13] + fSizeY) * m_fScreenScale[1]);

	vBox[14].sx = f2vt((fPosX + fU[14]) * m_fScreenScale[0]);
	vBox[14].sy = f2vt((fPosY + fV[14] + fSizeY) * m_fScreenScale[1]);

	vBox[15].sx = f2vt((fPosX + fU[15]) * m_fScreenScale[0]);
	vBox[15].sy = f2vt((fPosY + fV[15] + fSizeY) * m_fScreenScale[1]);

	if(m_fScreenScale[0]*640.0f<m_fScreenScale[1]*480.0f)
	{
		Rotate(vBox, 16);
	}

	/* No problem occured */
	return true;
}

/*!***************************************************************************
 @Function			UpdateLine
 @Description
*****************************************************************************/
unsigned int CPVRTPrint3D::UpdateLine(const unsigned int dwWin, const float fZPos, float XPos, float YPos, const float fScale, const unsigned int Colour, const char * const Text, SPVRTPrint3DAPIVertex * const pVertices)
{
	unsigned	i=0, VertexCount=0;
	unsigned	Val;
	float		XSize = 0.0f, XFixBug,	YSize = 0, TempSize;
	float		UPos,	VPos;
	float		USize,	VSize;
	float		fWinClipX[2],fWinClipY[2];
	float		fScaleX, fScaleY, fPreXPos;

	/* Nothing to update */
	if (Text==NULL) return 0;

	_ASSERT(m_pWin[dwWin].dwFlags & Print3D_WIN_EXIST || !dwWin);

	if (fScale>0)
	{
		fScaleX = m_fScreenScale[0] * fScale * 255.0f;
		fScaleY = m_fScreenScale[1] * fScale * 27.0f;
	}
	else
	{
		fScaleX = m_fScreenScale[0] * 255.0f;
		fScaleY = m_fScreenScale[1] * 12.0f;
	}

	XPos *= m_fScreenScale[0];
	YPos *= m_fScreenScale[1];

	fPreXPos = XPos;

	/*
		Calculating our margins
	*/
	if (dwWin)
	{
		fWinClipX[0] = (m_pWin[dwWin].fWinPos[0] + 6.0f) * m_fScreenScale[0];
		fWinClipX[1] = (m_pWin[dwWin].fWinPos[0] + m_pWin[dwWin].fWinSize[0] - 6.0f) * m_fScreenScale[0];

		fWinClipY[0] = (m_pWin[dwWin].fWinPos[1] + 6.0f) * m_fScreenScale[1];
		fWinClipY[1] = (m_pWin[dwWin].fWinPos[1] + m_pWin[dwWin].fWinSize[1]  + 9.0f) * m_fScreenScale[1];

		if(m_pWin[dwWin].dwFlags & Print3D_WIN_TITLE)
		{
			if (m_pWin[dwWin].fTitleFontSize>0)
			{
				fWinClipY[0] +=  m_pWin[dwWin].fTitleFontSize * 25.0f  * m_fScreenScale[1];
				fWinClipY[1] +=  m_pWin[dwWin].fTitleFontSize * 25.0f *  m_fScreenScale[1];
			}
			else
			{
				fWinClipY[0] +=  10.0f * m_fScreenScale[1];
				fWinClipY[1] +=  8.0f  * m_fScreenScale[1];
			}
		}
	}

	while (true)
	{
		Val = (int)Text[i++];

		/* End of the string */
		if (Val==0 || i>MAX_LETTERS) break;

		/* It is SPACE so don't draw and carry on... */
		if (Val==' ')
		{
			if (fScale>0)	XPos += 10.0f/255.0f * fScaleX;
			else			XPos += 5.0f * m_fScreenScale[0];
			continue;
		}

		/* It is SPACE so don't draw and carry on... */
		if (Val=='#')
		{
			if (fScale>0)	XPos += 1.0f/255.0f * fScaleX;
			else			XPos += 5.0f * m_fScreenScale[0];
			continue;
		}

		/* It is RETURN so jump a line */
		if (Val==0x0A)
		{
			XPos = fPreXPos - XSize;
			YPos += YSize;
			continue;
		}

		/* If fScale is negative then select the small set of letters (System) */
		if (fScale < 0.0f)
		{
			XPos    += XSize;
			UPos    =  PVRTPrint3DU_System[Val];
			VPos    =  PVRTPrint3DV_System[Val] - 0.0001f; /* Some cards need this little bit */
			YSize   =  fScaleY;
			XSize   =  PVRTPrint3DSize_System[Val] * fScaleX;
			USize	=  PVRTPrint3DSize_System[Val];
			VSize	=  12.0f/255.0f;
		}
		else /* Big set of letters (Bold) */
		{
			XPos    += XSize;
			UPos    =  PVRTPrint3DU_Bold[Val];
			VPos    =  PVRTPrint3DV_Bold[Val] - 1.0f/230.0f;
			YSize   =  fScaleY;
			XSize   =  PVRTPrint3DSize_Bold[Val] * fScaleX;
			USize	=  PVRTPrint3DSize_Bold[Val];
			VSize	=  29.0f/255.0f;
		}

		/*
			CLIPPING
		*/
		XFixBug = XSize;

		if (0)//dwWin) /* for dwWin==0 (screen) no clipping */
		{
			/* Outside */
			if (XPos>fWinClipX[1]  ||  (YPos)>fWinClipY[1])
			{
				continue;
			}

			/* Clip X */
			if (XPos<fWinClipX[1] && XPos+XSize > fWinClipX[1])
			{
				TempSize = XSize;

				XSize = fWinClipX[1] - XPos;

				if (fScale < 0.0f)
					USize	=  PVRTPrint3DSize_System[Val] * (XSize/TempSize);
				else
					USize	=  PVRTPrint3DSize_Bold[Val] * (XSize/TempSize);
			}

			/*
				Clip Y
			*/
			if (YPos<fWinClipY[1] && YPos+YSize > fWinClipY[1])
			{
				TempSize = YSize;
				YSize = fWinClipY[1] - YPos;

				if(fScale < 0.0f)
				 	VSize	=  (YSize/TempSize)*12.0f/255.0f;
				else
					VSize	=  (YSize/TempSize)*28.0f/255.0f;
			}
		}


		/* Filling vertex data */
		pVertices[VertexCount+0].sx		= f2vt(XPos);
		pVertices[VertexCount+0].sy		= f2vt(YPos);
		pVertices[VertexCount+0].sz		= f2vt(fZPos);
		pVertices[VertexCount+0].tu		= f2vt(UPos);
		pVertices[VertexCount+0].tv		= f2vt(VPos);

		pVertices[VertexCount+1].sx		= f2vt(XPos+XSize);
		pVertices[VertexCount+1].sy		= f2vt(YPos);
		pVertices[VertexCount+1].sz		= f2vt(fZPos);
		pVertices[VertexCount+1].tu		= f2vt(UPos+USize);
		pVertices[VertexCount+1].tv		= f2vt(VPos);

		pVertices[VertexCount+2].sx		= f2vt(XPos);
		pVertices[VertexCount+2].sy		= f2vt(YPos+YSize);
		pVertices[VertexCount+2].sz		= f2vt(fZPos);
		pVertices[VertexCount+2].tu		= f2vt(UPos);
		pVertices[VertexCount+2].tv		= f2vt(VPos-VSize);

		pVertices[VertexCount+3].sx		= f2vt(XPos+XSize);
		pVertices[VertexCount+3].sy		= f2vt(YPos+YSize);
		pVertices[VertexCount+3].sz		= f2vt(fZPos);
		pVertices[VertexCount+3].tu		= f2vt(UPos+USize);
		pVertices[VertexCount+3].tv		= f2vt(VPos-VSize);

		pVertices[VertexCount+0].color	= Colour;
		pVertices[VertexCount+1].color	= Colour;
		pVertices[VertexCount+2].color	= Colour;
		pVertices[VertexCount+3].color	= Colour;

		VertexCount += 4;

		XSize = XFixBug;

		/* Fix number width */
		if (Val >='0' && Val <='9')
		{
			if (fScale < 0.0f)
				XSize = PVRTPrint3DSize_System[(int)'0'] * fScaleX;
			else
				XSize = PVRTPrint3DSize_Bold[(int)'0'] * fScaleX;
		}
	}

	if(m_fScreenScale[0]*640.0f<m_fScreenScale[1]*480.0f)
	{
		Rotate(pVertices, VertexCount);
	}

	return VertexCount;
}

/*!***************************************************************************
 @Function			DrawLineUP
 @Return			true or false
 @Description		Draw a single line of text.
*****************************************************************************/
bool CPVRTPrint3D::DrawLineUP(SPVRTPrint3DAPIVertex *pVtx, unsigned int nVertices)
{
	if(!nVertices)
		return true;

	_ASSERT((nVertices % 4) == 0);
	_ASSERT((nVertices/4) < MAX_LETTERS);

	while(m_nVtxCache + (int)nVertices > m_nVtxCacheMax) {
		if(m_nVtxCache + nVertices > MAX_CACHED_VTX) {
			_RPT1(_CRT_WARN, "Print3D: Out of space to cache text! (More than %d vertices!)\n", MAX_CACHED_VTX);
			return false;
		}

		m_nVtxCacheMax	= PVRT_MIN(m_nVtxCacheMax * 2, MAX_CACHED_VTX);
		m_pVtxCache		= (SPVRTPrint3DAPIVertex*)realloc(m_pVtxCache, m_nVtxCacheMax * sizeof(*m_pVtxCache));
		_ASSERT(m_pVtxCache);
		_RPT1(_CRT_WARN, "Print3D: TextCache increased to %d vertices.\n", m_nVtxCacheMax);
	}

	memcpy(&m_pVtxCache[m_nVtxCache], pVtx, nVertices * sizeof(*pVtx));
	m_nVtxCache += nVertices;
	return true;
}

/*!***************************************************************************
 @Function			UpdateTitleVertexBuffer
 @Return			true or false
 @Description
*****************************************************************************/
bool CPVRTPrint3D::UpdateTitleVertexBuffer(unsigned int dwWin)
{
	float fRPos;
	unsigned int dwLenL = 0, dwLenR = 0;

	/* Doesn't exist */
	if (!(m_pWin[dwWin].dwFlags & Print3D_WIN_EXIST) && dwWin)
		return false;

	/* Allocate our buffers if needed */
	if(m_pWin[dwWin].pTitleVtxL==0 || m_pWin[dwWin].pTitleVtxR==0)
	{
		dwLenL = (unsigned int)strlen(m_pWin[dwWin].bTitleTextL);
		FREE(m_pWin[dwWin].pTitleVtxL);
		if(dwLenL)
			m_pWin[dwWin].pTitleVtxL = (SPVRTPrint3DAPIVertex*)malloc(dwLenL*4*sizeof(SPVRTPrint3DAPIVertex));

		dwLenR = m_pWin[dwWin].bTitleTextR ? (unsigned int)strlen(m_pWin[dwWin].bTitleTextR) : 0;
		FREE(m_pWin[dwWin].pTitleVtxR);
		if(dwLenR)
			m_pWin[dwWin].pTitleVtxR = (SPVRTPrint3DAPIVertex*)malloc(dwLenR*4*sizeof(SPVRTPrint3DAPIVertex));
	}

	/* Left title */
	if (dwLenL)
	{
		m_pWin[dwWin].nTitleVerticesL = UpdateLine(dwWin, 0.0f,
			(m_pWin[dwWin].fWinPos[0] + 6.0f),
			(m_pWin[dwWin].fWinPos[1] + 7.0f),
			m_pWin[dwWin].fTitleFontSize,
			m_pWin[dwWin].dwTitleFontColorL,
			m_pWin[dwWin].bTitleTextL,
			m_pWin[dwWin].pTitleVtxL);
	}
	else
	{
		m_pWin[dwWin].nTitleVerticesL = 0;
		m_pWin[dwWin].pTitleVtxL = NULL;
	}

	/* Right title */
	if (dwLenR)
	{
		/* Compute position */
		fRPos = GetLength(m_pWin[dwWin].fTitleFontSize,m_pWin[dwWin].bTitleTextR);

		fRPos = m_pWin[dwWin].fWinSize[0]  - fRPos - 6.0f;

		/* Check that we're not under minimum position */
		if(fRPos<m_pWin[dwWin].fTextRMinPos)
			fRPos = m_pWin[dwWin].fTextRMinPos;

		/* Add window position */
		fRPos += m_pWin[dwWin].fWinPos[0];

		/* Print text */
		m_pWin[dwWin].nTitleVerticesR = UpdateLine(dwWin, 0.0f,
			fRPos,
			m_pWin[dwWin].fWinPos[1] + 7.0f,
			m_pWin[dwWin].fTitleFontSize,
			m_pWin[dwWin].dwTitleFontColorR,
			m_pWin[dwWin].bTitleTextR,
			m_pWin[dwWin].pTitleVtxR);
	}
	else
	{
		m_pWin[dwWin].nTitleVerticesR = 0;
		m_pWin[dwWin].pTitleVtxR = NULL;
	}

	return true;
}

/*!***************************************************************************
 @Function			UpdateMainTextVertexBuffer
 @Return			true or false
 @Description
*****************************************************************************/
bool CPVRTPrint3D::UpdateMainTextVertexBuffer(unsigned int dwWin)
{
	int i;
	float		fNewPos, fTitleSize;
	unsigned int		dwPointer = 0, dwLen;

	/* Doesn't exist */
	if (!(m_pWin[dwWin].dwFlags & Print3D_WIN_EXIST) && dwWin) return false;

	/* No text to update vertices */
	if(m_pWin[dwWin].pTextBuffer==NULL) return true;

	/* Well, once we've got our text, allocate it to draw it later */
	/* Text, line by line */
	for (i = 0; i < (int) m_pWin[dwWin].dwBufferSizeY; i++)
	{
		/* line length */
		dwLen = (unsigned int)strlen(&m_pWin[dwWin].pTextBuffer[dwPointer]);
		if(dwLen==0)
		{
			m_pWin[dwWin].nLineVertices[i] = 0;
			m_pWin[dwWin].pLineVtx[i] = NULL;
		}
		else
		{
			/* Create Vertex Buffer (one per line) */
			if (m_pWin[dwWin].pLineVtx[i]==0)
			{
				m_pWin[dwWin].pLineVtx[i] = (SPVRTPrint3DAPIVertex*)malloc(m_pWin[dwWin].dwBufferSizeX *4*sizeof(SPVRTPrint3DAPIVertex));

				if(!m_pWin[dwWin].pLineVtx[i])
					return false;
			}

			/* Compute new text position */
			fTitleSize = 0.0f;
			if(m_pWin[dwWin].fTitleFontSize < 0.0f)
			{
				/* New position for alternate font */
				if(m_pWin[dwWin].dwFlags & Print3D_WIN_TITLE)
					fTitleSize = 8.0f +16;
				fNewPos = fTitleSize + (float)(i * 12.0f);
			}
			else
			{
				/* New position for normal font */
				if(m_pWin[dwWin].dwFlags & Print3D_WIN_TITLE)
					fTitleSize = m_pWin[dwWin].fTitleFontSize * 23.5f + 16.0f;
				fNewPos = fTitleSize + (float)(i * m_pWin[dwWin].fWinFontSize) * LINES_SPACING;
			}

			/* Print window text */
			m_pWin[dwWin].nLineVertices[i] = UpdateLine(dwWin, 0.0f,
				(m_pWin[dwWin].fWinPos[0] + 6.0f),
				(m_pWin[dwWin].fWinPos[1] + 6.0f + fNewPos),
				m_pWin[dwWin].fWinFontSize, m_pWin[dwWin].dwWinFontColor,
				&m_pWin[dwWin].pTextBuffer[dwPointer],
				m_pWin[dwWin].pLineVtx[i]);
		}

		/* Increase pointer */
		dwPointer += m_pWin[dwWin].dwBufferSizeX;
	}

	return true;
}

/*!***************************************************************************
 @Function			GetLength
 @Description		calculates the size in pixels.
*****************************************************************************/
float CPVRTPrint3D::GetLength(float fFontSize, char *sString)
{
	unsigned char Val;
	float fScale, fSize;

	if(sString == NULL)
		return 0.0f;

	if (fFontSize>=0) /* Arial font */
	{
		fScale = fFontSize * 255.0f;
		fSize  = 0.0f;

		Val = *sString++;
		while (Val)
		{
			if(Val==' ')
			{
				fSize += 10.0f * fFontSize;
			}
			else
			{
				if(Val>='0' && Val <= '9') Val = '0'; /* That's for fixing the number width */
				fSize += PVRTPrint3DSize_Bold[Val] * fScale ;
			}
			Val = *sString++;
		}
	}
	else /* System font */
	{
		fScale = 255.0f;
		fSize  = 0.0f;

		Val = *sString++;
		while (Val)
		{
			if (Val==' ')
			{
				fSize += 5.0f;
			}
			else
			{
				if(Val>='0' && Val <= '9') Val = '0'; /* That's for fixing the number width */
				fSize += PVRTPrint3DSize_System[Val]  * fScale;
			}
			Val = *sString++;
		}
	}

	return (fSize);
}

void CPVRTPrint3D::Rotate(SPVRTPrint3DAPIVertex * const pv, const unsigned int nCnt)
{
	unsigned int	i;
	VERTTYPE		x, y;

	for(i = 0; i < nCnt; ++i)
	{
		x = VERTTYPEDIV((VERTTYPE&)pv[i].sx, f2vt(640.0f * m_fScreenScale[0]));
		y = VERTTYPEDIV((VERTTYPE&)pv[i].sy, f2vt(480.0f * m_fScreenScale[1]));
		(VERTTYPE&)pv[i].sx = VERTTYPEMUL(y, f2vt(640.0f * m_fScreenScale[0]));
		(VERTTYPE&)pv[i].sy = VERTTYPEMUL(f2vt(1.0f) - x, f2vt(480.0f * m_fScreenScale[1]));
	}
}

/****************************************************************************
** Local code
****************************************************************************/

/*****************************************************************************
 End of file (PVRTPrint3D.cpp)
*****************************************************************************/
