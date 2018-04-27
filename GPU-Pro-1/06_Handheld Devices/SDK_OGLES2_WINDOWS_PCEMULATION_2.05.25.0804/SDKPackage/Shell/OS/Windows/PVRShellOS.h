/******************************************************************************

 @File         PVRShellOS.h

 @Title        Windows/PVRShellOS

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     WinCE/Windows

 @Description  Makes programming for 3D APIs easier by wrapping surface
               initialization, Texture allocation and other functions for use by a demo.

******************************************************************************/
#ifndef _PVRSHELLOS_
#define _PVRSHELLOS_

#include <windows.h>

/* The following defines are for Windows PC platforms only */
#if defined(WIN32) && !defined(UNDER_CE)
/* Enable the following 2 lines for memory leak checking - also see WinMain() */
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif

#ifdef UNDER_CE
/* Pocket PC and WinCE 5.0 only */
#ifdef DEBUG
#define _ASSERT(X) { (X) ? 0 : DebugBreak(); }
#define _ASSERTE _ASSERT
#else
#define _ASSERT(X) /* */
#define _ASSERTE(X) /* */
#endif
#endif

#define PVRSHELL_DIR_SYM	'\\'
#define vsnprintf _vsnprintf

//#define NO_GDI 1 /* Remove the GDI functions */

/*!***************************************************************************
 PVRShellInitOS
 @Brief Class. Interface with specific Operative System.
*****************************************************************************/
class PVRShellInitOS
{
public:
	HDC			m_hDC;
	HWND		m_hWnd;

	// Pixmap support: variables for the pixmap
	HBITMAP		m_hBmPixmap, m_hBmPixmapOld;
	HDC			m_hDcPixmap;

	HACCEL		m_hAccelTable;
	HINSTANCE	m_hInstance;
	int			m_nCmdShow;

	bool		m_bHaveFocus;

	float m_vec2PointerLocation[2];
	bool m_bPointer;	// is there a valid pointer position stored?
	unsigned int	m_u32ButtonState;

#ifdef WIN32_PLATFORM_WFSP
	// Time at which SHIdleTimerReset() was last called
	unsigned long	m_nTimeIdleTimeReset;
#endif

public:
#ifndef NO_GDI
	ATOM MyRegisterClass();
#endif
};

#endif /* _PVRSHELLOS_ */
/*****************************************************************************
 End of file (PVRShellOS.h)
*****************************************************************************/
