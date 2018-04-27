/******************************************************************************

 @File         PVRTPrint3D.h

 @Title        PVRTPrint3D

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Code to print text through the 3D interface.

******************************************************************************/
#ifndef _PVRTPRINT3D_H_
#define _PVRTPRINT3D_H_

#include "PVRTGlobal.h"
#include "PVRTContext.h"
#include "PVRTError.h"

/****************************************************************************
** Defines
****************************************************************************/
#define PVRTPRINT3D_MAX_WINDOWS				(512)
#define PVRTPRINT3D_MAX_RENDERABLE_LETTERS	(0xFFFF >> 2)

/****************************************************************************
** Enums
****************************************************************************/

/*!***************************************************************************
 dwFlags for PVRTPrint3DSetWindowFlags
*****************************************************************************/
typedef enum {
	ePVRTPrint3D_ACTIVATE_WIN		=	0x01,
	ePVRTPrint3D_DEACTIVATE_WIN		=	0x02,
	ePVRTPrint3D_ACTIVATE_TITLE		=	0x04,
	ePVRTPrint3D_DEACTIVATE_TITLE	=	0x08,
	ePVRTPrint3D_FULL_OPAQUE		=	0x10,
	ePVRTPrint3D_FULL_TRANS			=	0x20,
	ePVRTPrint3D_ADJUST_SIZE_ALWAYS	=	0x40,
	ePVRTPrint3D_NO_BORDER			=	0x80
} EPVRTPrint3DFlags;

/*!***************************************************************************
 Logo flags for DisplayDefaultTitle
*****************************************************************************/
typedef enum {
	ePVRTPrint3DLogoNone  = 0x00,
	ePVRTPrint3DLogoPVR = 0x02,
	ePVRTPrint3DLogoIMG = 0x04,
	ePVRTPrint3DSDKLogo = ePVRTPrint3DLogoIMG
} EPVRTPrint3DLogo;

/****************************************************************************
** Structures
****************************************************************************/
/*!**************************************************************************
@Struct SPVRTPrint3DAPIVertex
@Brief A structure for Print3Ds vertex type
****************************************************************************/
struct SPVRTPrint3DAPIVertex
{
	VERTTYPE		sx, sy, sz, rhw;
	unsigned int	color;
	VERTTYPE		tu, tv;
};

/*!**************************************************************************
@Struct SPVRTPrint3DWIN
@Brief A structure for Print3Ds data
****************************************************************************/
struct SPVRTPrint3DWIN
{
	unsigned int			dwFlags;

	bool					bNeedUpdated;

	// Text Buffer
	char					*pTextBuffer;
	unsigned int			dwBufferSizeX;
	unsigned int			dwBufferSizeY;

	// Title
	float					fTitleFontSize;
	float					fTextRMinPos;
	unsigned int			dwTitleFontColorL;
	unsigned int			dwTitleFontColorR;
	unsigned int			dwTitleBaseColor;
	char					*bTitleTextL;
	char					*bTitleTextR;
	unsigned int			nTitleVerticesL;
	unsigned int			nTitleVerticesR;
	SPVRTPrint3DAPIVertex	*pTitleVtxL;
	SPVRTPrint3DAPIVertex	*pTitleVtxR;

	// Window
	float					fWinFontSize;
	unsigned int			dwWinFontColor;
	unsigned int			dwWinBaseColor;
	float					fWinPos[2];
	float					fWinSize[2];
	float					fZPos;
	unsigned int			dwSort;
	unsigned int			nLineVertices[255]; // every line of text is allocated and drawn apart.
	SPVRTPrint3DAPIVertex	*pLineVtx[256];
	SPVRTPrint3DAPIVertex	*pWindowVtxTitle;
	SPVRTPrint3DAPIVertex	*pWindowVtxText;
};

struct SPVRTPrint3DAPI;

/*!***************************************************************************
 @Class CPVRTPrint3D
 @Brief Display text/logos on the screen
*****************************************************************************/
class CPVRTPrint3D
{
public:
	/*!***************************************************************************
	 @Function			CPVRTPrint3D
	 @Description		Init some values.
	*****************************************************************************/
	CPVRTPrint3D();
	/*!***************************************************************************
	 @Function			~CPVRTPrint3D
	 @Description		De-allocate the working memory
	*****************************************************************************/
	~CPVRTPrint3D();

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
	EPVRTError SetTextures(
		const SPVRTContext	* const pContext,
		const unsigned int	dwScreenX,
		const unsigned int	dwScreenY,
		const bool bRotate = false);

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
	EPVRTError Print3D(float fPosX, float fPosY, const float fScale, unsigned int Colour, const char * const pszFormat, ...);

	/*!***************************************************************************
	 @Function			DisplayDefaultTitle
	 @Input				pszTitle			Title to display
	 @Input				pszDescription		Description to display
	 @Input				uDisplayLogo		1 = Display the logo
	 @Return			PVR_SUCCESS or PVR_FAIL
	 @Description		Creates a default title with predefined position and colours.
						It displays as well company logos when requested:
						0 = No logo
						1 = PowerVR logo
						2 = Img Tech logo
	*****************************************************************************/
	 EPVRTError DisplayDefaultTitle(const char * const pszTitle, const char * const pszDescription, const unsigned int uDisplayLogo);

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
	unsigned int CreateDefaultWindow(float fPosX, float fPosY, int nXSize_LettersPerLine, char *sTitle, char *sBody);

	/*!***************************************************************************
	 @Function			InitWindow
	 @Input				dwBufferSizeX		Buffer width
	 @Input				dwBufferSizeY		Buffer height
	 @Return			Window handle
	 @Description		Allocate a buffer for a newly-created window and return its
						handle.
	*****************************************************************************/
	unsigned int InitWindow(unsigned int dwBufferSizeX, unsigned int dwBufferSizeY);

	/*!***************************************************************************
	 @Function			DeleteWindow
	 @Input				dwWin		Window handle
	 @Description		Delete the window referenced by dwWin.
	*****************************************************************************/
	void DeleteWindow(unsigned int dwWin);

	/*!***************************************************************************
	 @Function			DeleteAllWindows
	 @Description		Delete all windows.
	*****************************************************************************/
	void DeleteAllWindows();

	/*!***************************************************************************
	 @Function			DisplayWindow
	 @Input				dwWin
	 @Return			PVR_SUCCESS or PVR_FAIL
	 @Description		Display window.
						This function MUST be called between a BeginScene/EndScene
						pair as it uses D3D render primitive calls.
						PVRTPrint3DSetTextures(...) must have been called beforehand.
	*****************************************************************************/
	EPVRTError DisplayWindow(unsigned int dwWin);

	/*!***************************************************************************
	 @Function			SetText
	 @Input				dwWin		Window handle
	 @Input				Format		Format string
	 @Return			PVR_SUCCESS or PVR_FAIL
	 @Description		Feed the text buffer of window referenced by dwWin.
						This function accepts formatting in the printf way.
	*****************************************************************************/
	EPVRTError SetText(unsigned int dwWin, const char *Format, ...);

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
	void SetWindow(unsigned int dwWin, unsigned int dwWinColor, unsigned int dwFontColor, float fFontSize,
							  float fPosX, float fPosY, float fSizeX, float fSizeY);

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
	void SetTitle(unsigned int dwWin, unsigned int dwBackgroundColor, float fFontSize,
							 unsigned int dwFontColorLeft, char *sTitleLeft,
							 unsigned int dwFontColorRight, char *sTitleRight);

	/*!***************************************************************************
	 @Function			SetWindowFlags
	 @Input				dwWin				Window handle
	 @Input				dwFlags				Flags
	 @Description		Set flags for window referenced by dwWin.
						A list of flag can be found at the top of this header.
	*****************************************************************************/
	void SetWindowFlags(unsigned int dwWin, unsigned int dwFlags);

	/*!***************************************************************************
	 @Function			AdjustWindowSize
	 @Input				dwWin				Window handle
	 @Input				dwMode				dwMode 0 = Both, dwMode 1 = X only,  dwMode 2 = Y only
	 @Description		Calculates window size so that all text fits in the window.
	*****************************************************************************/
	void AdjustWindowSize(unsigned int dwWin, unsigned int dwMode);

	/*!***************************************************************************
	 @Function			GetSize
	 @Output			pfWidth				Width of the string in pixels
	 @Output			pfHeight			Height of the string in pixels
	 @Input				fFontSize			Font size
	 @Input				sString				String to take the size of
	 @Description		Returns the size of a string in pixels.
	*****************************************************************************/
	void GetSize(
		float		* const pfWidth,
		float		* const pfHeight,
		const float	fFontSize,
		const char	* sString);

	/*!***************************************************************************
	 @Function			GetAspectRatio
	 @Output			dwScreenX		Screen resolution X
	 @Output			dwScreenY		Screen resolution Y
	 @Description		Returns the current resolution used by Print3D
	*****************************************************************************/
	void GetAspectRatio(unsigned int *dwScreenX, unsigned int *dwScreenY);

private:
	/*!***************************************************************************
	 @Function			UpdateBackgroundWindow
	 @Return			true if succesful, false otherwise.
	 @Description		Draw a generic rectangle (with or without border).
	*****************************************************************************/
	bool UpdateBackgroundWindow(unsigned int dwWin, unsigned int Color, float fZPos, float fPosX, float fPosY, float fSizeX, float fSizeY, SPVRTPrint3DAPIVertex **ppVtx);

	/*!***************************************************************************
	 @Function			UpdateLine
	 @Description
	*****************************************************************************/
	unsigned int UpdateLine(const unsigned int dwWin, const float fZPos, float XPos, float YPos, const float fScale, const unsigned int Colour, const char * const Text, SPVRTPrint3DAPIVertex * const pVertices);

	/*!***************************************************************************
	 @Function			DrawLineUP
	 @Return			true or false
	 @Description		Draw a single line of text.
	*****************************************************************************/
	bool DrawLineUP(SPVRTPrint3DAPIVertex *pVtx, unsigned int nVertices);

	/*!***************************************************************************
	 @Function			UpdateTitleVertexBuffer
	 @Return			true or false
	 @Description
	*****************************************************************************/
	bool UpdateTitleVertexBuffer(unsigned int dwWin);

	/*!***************************************************************************
	 @Function			UpdateMainTextVertexBuffer
	 @Return			true or false
	 @Description
	*****************************************************************************/
	bool UpdateMainTextVertexBuffer(unsigned int dwWin);

	/*!***************************************************************************
	 @Function			GetLength
	 @Description		calculates the size in pixels.
	*****************************************************************************/
	float GetLength(float fFontSize, char *sString);

	/*!***************************************************************************
	 @Function			Rotate
	 @Description		Rotates the vertices to fit on screen.
	*****************************************************************************/
	void Rotate(SPVRTPrint3DAPIVertex * const pv, const unsigned int nCnt);

private:
	SPVRTPrint3DAPI			*m_pAPI;
	unsigned int			m_uLogoToDisplay;
	unsigned short			*m_pwFacesFont;
	SPVRTPrint3DAPIVertex	*m_pPrint3dVtx;
	float					m_fScreenScale[2];
	unsigned int			m_ui32ScreenDim[2];
	bool					m_bTexturesSet;
	SPVRTPrint3DAPIVertex	*m_pVtxCache;
	int						m_nVtxCache;
	int						m_nVtxCacheMax;
	SPVRTPrint3DWIN			m_pWin[PVRTPRINT3D_MAX_WINDOWS];

public:
	/*!***************************************************************************
	 @Function			PVRTPrint3DReleaseTextures
	 @Description		Deallocate the memory allocated in PVRTPrint3DSetTextures(...)
	*****************************************************************************/
	void ReleaseTextures();

	/*!***************************************************************************
	 @Function			Flush
	 @Description		Flushes all the print text commands
	*****************************************************************************/
	int Flush();

private:
	/*!***************************************************************************
	 @Function			APIInit
	 @Return			true or false
	 @Description		Initialization and texture upload. Should be called only once
						for a given context.
	*****************************************************************************/
	bool APIInit(const SPVRTContext	* const pContext);

	/*!***************************************************************************
	 @Function			APIRelease
	 @Description		Deinitialization.
	*****************************************************************************/
	void APIRelease();

	/*!***************************************************************************
	 @Function			APIUpLoadIcons
	 @Return			true or false
	 @Description		Initialization and texture upload. Should be called only once
						for a given context.
	*****************************************************************************/
	bool APIUpLoadIcons(
		const PVRTuint32 * const pPVR,
		const PVRTuint32 * const pIMG);

	/*!***************************************************************************
	 @Function			APIUpLoad4444
	 @Return			true if succesful, false otherwise.
	 @Description		Reads texture data from *.dat and loads it in
						video memory.
	*****************************************************************************/
	bool APIUpLoad4444(unsigned int TexID, unsigned char *pSource, unsigned int nSize, unsigned int nMode);

	/*!***************************************************************************
	 @Function			DrawBackgroundWindowUP
	 @Description
	*****************************************************************************/
	void DrawBackgroundWindowUP(unsigned int dwWin, SPVRTPrint3DAPIVertex *pVtx, const bool bIsOp, const bool bBorder);

	/*!***************************************************************************
	 @Function			APIRenderStates
	 @Description		Stores, writes and restores Render States
	*****************************************************************************/
	void APIRenderStates(int nAction);

	/*!***************************************************************************
	 @Function			APIDrawLogo
	 @Description		nPos = -1 to the left
						nPos = +1 to the right
	*****************************************************************************/
	void APIDrawLogo(unsigned int uLogoToDisplay, int nPos);
};


#endif /* _PVRTPRINT3D_H_ */

/*****************************************************************************
 End of file (PVRTPrint3D.h)
*****************************************************************************/
