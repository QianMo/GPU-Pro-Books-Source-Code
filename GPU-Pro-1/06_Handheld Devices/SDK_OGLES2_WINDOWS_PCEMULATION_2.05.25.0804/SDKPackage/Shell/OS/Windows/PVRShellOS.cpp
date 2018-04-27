/******************************************************************************

 @File         PVRShellOS.cpp

 @Title        Windows/PVRShellOS

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     WinCE/Windows

 @Description  Makes programming for 3D APIs easier by wrapping window creation
               and other functions for use by a demo.

******************************************************************************/
#include <windows.h>
#include <TCHAR.H>
#include <stdio.h>

#include "PVRShell.h"
#include "PVRShellAPI.h"
#include "PVRShellOS.h"
#include "PVRShellImpl.h"


#if !(WINVER >= 0x0500)
	#define COMPILE_MULTIMON_STUBS
	#include <multimon.h>
#endif

/* PocketPC platforms require this additional header */
#if UNDER_CE == 420 || defined(WIN32_PLATFORM_PSPC) || defined(WIN32_PLATFORM_WFSP)
#include "Aygshell.h"	// Allow proper full-screen mode
#pragma comment( lib, "aygshell" )
#endif

#ifdef UNDER_CE

#ifdef WIN32_PLATFORM_WFSP
#include "tpcshell.h"	// Prevent the OS from timing-out the app
#endif

/* The required variables for the power notification */
bool				g_bClosing;			// used to shut down the power notifications thread
HANDLE				g_hPower = NULL;	// an event that will be signaled by power notifications
DWORD PowerNotificationThread(PVOID pContext);

#define WM_APP_DONTRENDER	WM_APP + 1
#define WM_APP_RENDER		WM_APP + 2
bool g_bRendering = true;

#include <pm.h>			// Allow the backlight to be controlled
#include <Msgqueue.h>	// For CloseMsgQueue etc
#endif


/****************************************************************************
	Defines
*****************************************************************************/
/*! The class name for the window */
#define WINDOW_CLASS _T("PVRShellClass")

/*! Maximum size to create string array for determining the read/write paths */
#define DIR_BUFFER_LEN	(10240)

#ifdef BUILD_OVG
/*! X dimension of the window that is created */
#define SHELL_DISPLAY_DIM_X	240
/*! Y dimension of the window that is created */
#define SHELL_DISPLAY_DIM_Y	320
#else
/*! X dimension of the window that is created */
#define SHELL_DISPLAY_DIM_X	640
/*! Y dimension of the window that is created */
#define SHELL_DISPLAY_DIM_Y	480
#endif


/*****************************************************************************
	Declarations
*****************************************************************************/
LRESULT CALLBACK	WndProc(HWND, UINT, WPARAM, LPARAM);

/*!***************************************************************************
	Class: PVRShellInit
*****************************************************************************/

/*!***********************************************************************
@Function		PVRShellOutputDebug
@Input			format			printf style format followed by arguments it requires
@Description	Writes the resultant string to the debug output (e.g. using
				printf(), OutputDebugString(), ...). Check the SDK release notes for
				details on how the string is output.
*************************************************************************/
void PVRShell::PVRShellOutputDebug(char const * const format, ...) const
{
	va_list arg;
	char	buf[1024];

	va_start(arg, format);
	vsnprintf(buf, 1024, format, arg);
	va_end(arg);

	// Passes the data to a platform dependant function
	m_pShellInit->OsDisplayDebugString(buf);
}

/*!***********************************************************************
 @Function		OsInit
 @description	Initialisation for OS-specific code.
*************************************************************************/
void PVRShellInit::OsInit()
{

#if defined UNDER_CE
	m_pShell->m_pShellData->bFullScreen = true; // WinCE overrides default to use fullscreen
#endif

	m_hAccelTable = 0;

	m_pShell->m_pShellData->nShellDimX = SHELL_DISPLAY_DIM_X;
	m_pShell->m_pShellData->nShellDimY = SHELL_DISPLAY_DIM_Y;

	m_hDC = 0;
	m_hWnd = 0;

	// Pixmap support: init variables to 0
	m_hBmPixmap = 0;
	m_hBmPixmapOld = 0;
	m_hDcPixmap = 0;

	/*
		Construct the binary path for GetReadPath() and GetWritePath()
	*/
	{
		/* Allocate memory for strings and return 0 if allocation failed */
		TCHAR* exeNameTCHAR = new TCHAR[DIR_BUFFER_LEN];
		char* exeName = new char[DIR_BUFFER_LEN];
		if(exeNameTCHAR && exeName)
		{
			DWORD retSize;

			/*
				Get the data path and a default application name
			*/

			// Get full path of executable
			retSize = GetModuleFileName(NULL, exeNameTCHAR, DIR_BUFFER_LEN);

			if (DIR_BUFFER_LEN > (int)retSize)
			{
				/* Get string length in char */
				retSize = (DWORD)_tcslen(exeNameTCHAR);

				/* Convert TChar to char */
				for (DWORD i = 0; i <= retSize; i++)
				{
					exeName[i] = (char)exeNameTCHAR[i];
				}

				SetAppName(exeName);
				SetReadPath(exeName);
				SetWritePath(exeName);
			}
		}

		delete [] exeName;
		delete [] exeNameTCHAR;
	}

	m_bPointer = false;		// have no valid pointer location at startup
	m_u32ButtonState = 0;	// clear mouse button state at startup
}

/*!***********************************************************************
 @Function		OsInitOS
 @description	Saves instance handle and creates main window
				In this function, we save the instance handle in a global variable and
				create and display the main program window.
*************************************************************************/
bool PVRShellInit::OsInitOS()
#ifndef NO_GDI
{
	MONITORINFO sMInfo;
	TCHAR		*appName;
	RECT		winRect;
	POINT		p;

	MyRegisterClass();

#ifdef UNDER_CE
	if(!m_pShell->m_pShellData->bUsingPowerSaving)
	{
		/*
			See "Program Applications to Turn the Smartphone Backlight Off and On":
				<http://msdn.microsoft.com/library/default.asp?url=/library/en-us/mobilesdk5/html/mob5tskHowToProgramApplicationsToTurnSmartphoneBacklightOffOn.asp>
		*/
		SetPowerRequirement(_T("BKL1:"), D0, POWER_NAME, NULL, 0);

		/*
			It is also necessary to prevent the display driver entering
			power-saving mode.

			The string used here is hard-coded. The following function can
			probably be used to enumerate available strings:
				RequestDeviceNotifications()
		*/
		SetPowerRequirement(_T("{EB91C7C9-8BF6-4a2d-9AB8-69724EED97D1}\\\\Windows\\ddi.dll"), D0, POWER_NAME, NULL, 0);
    }
#endif

	/*
		Build the window title
	*/
	{
		const char		*pszName, *pszSeparator, *pszVersion;
		size_t			len;
		unsigned int	out, in;

		pszName			= (const char*)m_pShell->PVRShellGet(prefAppName);
		pszSeparator	= STR_WNDTITLE;
		pszVersion		= (const char*)m_pShell->PVRShellGet(prefVersion);

		len = strlen(pszName)+strlen(pszSeparator)+strlen(pszVersion)+1;
		appName = new TCHAR[len];

		for(out = 0; appName[out] = pszName[out]; ++out);
		for(in = 0; appName[out] = pszSeparator[in]; ++in, ++out);
		for(in = 0; appName[out] = pszVersion[in]; ++in, ++out);
		_ASSERT(out == len-1);
	}

	/*
		Retrieve the monitor information.

		MonitorFromWindow() doesn't work, because the window hasn't been
		created yet.
	*/
	{
		HMONITOR	hMonitor;
		BOOL		bRet;

		p.x			= m_pShell->m_pShellData->nShellPosX;
		p.y			= m_pShell->m_pShellData->nShellPosY;
		hMonitor	= MonitorFromPoint(p, MONITOR_DEFAULTTOPRIMARY);
		sMInfo.cbSize = sizeof(sMInfo);
		bRet = GetMonitorInfo(hMonitor, &sMInfo);
		_ASSERT(bRet);
	}

	/*
		Reduce the window size until it fits on screen
	*/
	while(
		(m_pShell->m_pShellData->nShellDimX > (sMInfo.rcMonitor.right - sMInfo.rcMonitor.left)) ||
		(m_pShell->m_pShellData->nShellDimY > (sMInfo.rcMonitor.bottom - sMInfo.rcMonitor.top)))
	{
		m_pShell->m_pShellData->nShellDimX >>= 1;
		m_pShell->m_pShellData->nShellDimY >>= 1;
	}


	/*
		Create the window
	*/

	if(m_pShell->m_pShellData->bFullScreen)
	{
		m_hWnd = CreateWindow(WINDOW_CLASS, appName, WS_VISIBLE | WS_SYSMENU,CW_USEDEFAULT, CW_USEDEFAULT, m_pShell->m_pShellData->nShellDimX, m_pShell->m_pShellData->nShellDimY,
				NULL, NULL, m_hInstance, this);

		SetWindowLong(m_hWnd,GWL_STYLE,GetWindowLong(m_hWnd,GWL_STYLE) &~ WS_CAPTION);
		SetWindowPos(m_hWnd,HWND_NOTOPMOST,0,0,0,0,SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED);
	}
	else
	{
		int x, y;

		SetRect(&winRect,
			m_pShell->m_pShellData->nShellPosX,
			m_pShell->m_pShellData->nShellPosY,
			m_pShell->m_pShellData->nShellPosX+m_pShell->m_pShellData->nShellDimX,
			m_pShell->m_pShellData->nShellPosY+m_pShell->m_pShellData->nShellDimY);
		AdjustWindowRectEx(&winRect, WS_CAPTION|WS_SYSMENU, false, 0);

		x = m_pShell->m_pShellData->nShellPosX - winRect.left;
		winRect.left += x;
		winRect.right += x;

		y = m_pShell->m_pShellData->nShellPosY - winRect.top;
		winRect.top += y;
		winRect.bottom += y;

		if(m_pShell->m_pShellData->bShellPosWasDefault)
		{
			x = CW_USEDEFAULT;
			y = CW_USEDEFAULT;
		}
		else
		{
			x = winRect.left;
			y = winRect.top;
		}

		m_hWnd = CreateWindow(WINDOW_CLASS, appName, WS_VISIBLE|WS_CAPTION|WS_SYSMENU,
			x, y, winRect.right-winRect.left, winRect.bottom-winRect.top, NULL, NULL, m_hInstance, this);
	}

	if(!m_hWnd)
		return false;

#ifdef UNDER_CE
	// For the Axim (X50v, X51v)
	RegisterHotKey(m_hWnd, 0xC1, MOD_WIN, 0xC1);
	RegisterHotKey(m_hWnd, 0xC2, MOD_WIN, 0xC2);
	RegisterHotKey(m_hWnd, 0xC3, MOD_WIN, 0xC3);
	RegisterHotKey(m_hWnd, 0xC4, MOD_WIN, 0xC4);
#endif

	if(m_pShell->m_pShellData->bFullScreen)
	{
		m_pShell->m_pShellData->nShellDimX = sMInfo.rcMonitor.right;
		m_pShell->m_pShellData->nShellDimY = sMInfo.rcMonitor.bottom;
		SetWindowPos(m_hWnd,HWND_TOPMOST,0,0,m_pShell->m_pShellData->nShellDimX,m_pShell->m_pShellData->nShellDimY,0);

#if UNDER_CE == 420 || defined(WIN32_PLATFORM_PSPC) || defined(WIN32_PLATFORM_WFSP)
		/*
			Fix for 'Top menu-bar' corruption (occurring on exiting egltest in
			fullscreen mode on PocketPC).

			See "How To Create Full-Screen Applications for the PocketPC":
				<http://support.microsoft.com/default.aspx?scid=kb;%5Bln%5D;266244>
		*/
		SetForegroundWindow(m_hWnd);
		SHFullScreen(m_hWnd, SHFS_HIDETASKBAR | SHFS_HIDESIPBUTTON | SHFS_HIDESTARTICON);
		MoveWindow(m_hWnd,
					sMInfo.rcMonitor.left,
					sMInfo.rcMonitor.top,
					sMInfo.rcMonitor.right,
					sMInfo.rcMonitor.bottom,
					TRUE);
#endif
	}

	m_hDC = GetDC(m_hWnd);
	ShowWindow(m_hWnd, m_nCmdShow);
	UpdateWindow(m_hWnd);

	p.x = 0;
	p.y = 0;
	ClientToScreen(m_hWnd, &p);
	m_pShell->m_pShellData->nShellPosX = p.x;
	m_pShell->m_pShellData->nShellPosY = p.y;

	delete [] appName;
	return true;
}
#else
{
    return true;
}
#endif
/*!***********************************************************************
 @Function		OsReleaseOS
 @description	Destroys main window
*************************************************************************/
void PVRShellInit::OsReleaseOS()
{
#ifndef NO_GDI
	ReleaseDC(m_hWnd, m_hDC);
	DestroyWindow(m_hWnd);
#endif
}

/*!***********************************************************************
 @Function		OsExit
 @description	Destroys main window
*************************************************************************/
void PVRShellInit::OsExit()
{
	const char	*szText;

	/*
		Show the exit message to the user
	*/
	szText		= (const char*)m_pShell->PVRShellGet(prefExitMessage);

#ifndef NO_GDI
	int			i, nT, nC;
	const char	*szCaption;
	TCHAR		*tzText, *tzCaption;

	szCaption	= (const char*)m_pShell->PVRShellGet(prefAppName);

	if(!szText || !szCaption)
		return;

	nT = (int)strlen(szText) + 1;
	nC = (int)strlen(szCaption) + 1;

	tzText = (TCHAR*)malloc(nT * sizeof(*tzText));
	tzCaption = (TCHAR*)malloc(nC * sizeof(*tzCaption));

	for(i = 0; tzText[i] = szText[i]; ++i);
	for(i = 0; tzCaption[i] = szCaption[i]; ++i);

	MessageBox(NULL, tzText, tzCaption, MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND);

	FREE(tzText);
	FREE(tzCaption);
#else
	OsDisplayDebugString(szText);
#endif

}

/*!***********************************************************************
 @Function		OsDoInitAPI
 @Return		true on success
 @description	Perform API initialisation and bring up window / fullscreen
*************************************************************************/
bool PVRShellInit::OsDoInitAPI()
{
#ifndef NO_GDI
	// Pixmap support: create the pixmap
	if(m_pShell->m_pShellData->bNeedPixmap)
	{
		m_hDcPixmap = CreateCompatibleDC(m_hDC);
		m_hBmPixmap = CreateCompatibleBitmap(m_hDC, 640, 480);
	}
#endif
	if(!ApiInitAPI())
	{
		return false;
	}
#ifndef NO_GDI
	// Pixmap support: select the pixmap into a device context (DC) ready for blitting
	if(m_pShell->m_pShellData->bNeedPixmap)
	{
		m_hBmPixmapOld = (HBITMAP)SelectObject(m_hDcPixmap, m_hBmPixmap);
	}

	SetForegroundWindow(m_hWnd);
#endif
	/* No problem occured */
	return true;
}

/*!***********************************************************************
 @Function		OsDoReleaseAPI
 @description	Clean up after we're done
*************************************************************************/
void PVRShellInit::OsDoReleaseAPI()
{
#ifndef NO_GDI
#endif
	ApiReleaseAPI();
#ifndef NO_GDI
	if(m_pShell->m_pShellData->bNeedPixmap)
	{
		// Pixmap support: free the pixmap
		SelectObject(m_hDcPixmap, m_hBmPixmapOld);
		DeleteDC(m_hDcPixmap);
		DeleteObject(m_hBmPixmap);
	}

#endif
}

/*!***********************************************************************
 @Function		OsRenderComplete
 @Returns		false when the app should quit
 @description	Main message loop / render loop
*************************************************************************/
void PVRShellInit::OsRenderComplete()
#ifndef NO_GDI
{
	MSG		msg;

#ifdef WIN32_PLATFORM_WFSP
	{
		unsigned long nTime = m_pShell->PVRShellGetTime();

		if((nTime - m_nTimeIdleTimeReset) > 8000)
		{
			/*
				Prevent Smartphone timing-out the app and returning to the
				menu.

				Performance is reduced if this is called every frame, so here
				it is called every 8 seconds; 10 seconds is the minimum value
				that the user can set in "Settings/Home Screen/Time out" on
				Windows Smartphone.
			*/
			SHIdleTimerReset();
			m_nTimeIdleTimeReset = nTime;
		}

		if(!m_bHaveFocus)
		{
			m_pShell->PVRShellOutputDebug("Window lost focus, quitting.\n");
			gShellDone = true;
		}
	}
#endif

	/*
		Process the message queue
	*/
#ifdef UNDER_CE
	/*
		If the screen is off and we aren't rendering then keep going round the message loop
		waiting for the WM_APP_RENDER message.
	*/
	while((g_bRendering || m_pShell->m_pShellInit->gShellDone) ?
		PeekMessage(&msg, m_hWnd, 0, 0, PM_REMOVE): GetMessage(&msg, m_hWnd, WM_APP_RENDER, WM_APP_RENDER) > 0)
#else
	while(PeekMessage(&msg, m_hWnd, 0, 0, PM_REMOVE))
#endif
	{
		if (!TranslateAccelerator(msg.hwnd, m_hAccelTable, &msg))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

}
#else
{
}
#endif

/*!***********************************************************************
 @Function		OsPixmapCopy
 @Return		true if the copy succeeded
 @description	When using pixmaps, copy the render to the display
*************************************************************************/
bool PVRShellInit::OsPixmapCopy()
{
#ifndef NO_GDI
	return (BitBlt(m_hDC, 0, 0, 640, 480, m_hDcPixmap, 0, 0, SRCCOPY) == TRUE);
#else
	return 0;
#endif
}

/*!***********************************************************************
 @Function		OsGetNativeDisplayType
 @Return		The 'NativeDisplayType' for EGL
 @description	Called from InitAPI() to get the NativeDisplayType
*************************************************************************/
void *PVRShellInit::OsGetNativeDisplayType()
{
	return m_hDC; // Null when NO_GDI defined
}

/*!***********************************************************************
 @Function		OsGetNativePixmapType
 @Return		The 'NativePixmapType' for EGL
 @description	Called from InitAPI() to get the NativePixmapType
*************************************************************************/
void *PVRShellInit::OsGetNativePixmapType()
{
	// Pixmap support: return the pixmap
	return m_hBmPixmap; // Null when NO_GDI defined
}

/*!***********************************************************************
 @Function		OsGetNativeWindowType
 @Return		The 'NativeWindowType' for EGL
 @description	Called from InitAPI() to get the NativeWindowType
*************************************************************************/
void *PVRShellInit::OsGetNativeWindowType()
{
	return m_hWnd;  // Null when NO_GDI defined
}

/*!***********************************************************************
 @Function		OsGet
 @Input			prefName	Name of value to get
 @Modified		pn A pointer set to the value asked for
 @Returns		true on success
 @Description	Retrieves OS-specific data
*************************************************************************/
bool PVRShellInit::OsGet(const prefNameIntEnum prefName, int *pn)
{
	switch(prefName)
	{
	case prefButtonState:
		*pn = m_u32ButtonState;
		return true;
	};
	return false;
}

/*!***********************************************************************
 @Function		OsGet
 @Input			prefName	Name of value to get
 @Modified		pp A pointer set to the value asked for
 @Returns		true on success
 @Description	Retrieves OS-specific data
*************************************************************************/
bool PVRShellInit::OsGet(const prefNamePtrEnum prefName, void **pp)
{
	switch(prefName)
	{
	case prefHINSTANCE:
		*pp = m_hInstance;
		return true;
	case prefPointerLocation:
		if(m_bPointer)
		{
			*pp = m_vec2PointerLocation;
		}
		else
			return false;
		return true;

	default:
		return false;
	}
}

/*!***********************************************************************
 @Function		OsDisplayDebugString
 @Input			str		string to output
 @Description	Prints a debug string
*************************************************************************/
void PVRShellInit::OsDisplayDebugString(char const * const str)
{
	if(str)
	{
#ifdef UNICODE
		wchar_t	strc[1024];
		int		i;

		for(i = 0; (str[i] != '\0') && (i < (sizeof(strc) / sizeof(*strc))); ++i)
		{
			strc[i] = (wchar_t)str[i];
		}

		OutputDebugString(strc);
#else
		OutputDebugString(str);
#endif
	}
}

/*!***********************************************************************
 @Function		OsGetTime
 @Return		Time in milliseconds since the beginning of the application
 @Description	Gets the time in milliseconds since the beginning of the application
*************************************************************************/
unsigned long PVRShellInit::OsGetTime()
{
	return (unsigned long)GetTickCount();
}

/*****************************************************************************
 Class: PVRShellInitOS
*****************************************************************************/

/*!******************************************************************************************
@function		MyRegisterClass()
@description	Registers the window class.
				This function and its usage is only necessary if you want this code
				to be compatible with Win32 systems prior to the 'RegisterClassEx'
				function that was added to Windows 95. It is important to call this function
				so that the application will get 'well formed' small icons associated
				with it.
**********************************************************************************************/
#ifndef NO_GDI
ATOM PVRShellInitOS::MyRegisterClass()
{
	WNDCLASS wc;

    wc.style			= CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc		= (WNDPROC)WndProc;
    wc.cbClsExtra		= 0;
    wc.cbWndExtra		= 0;
    wc.hInstance		= m_hInstance;
    wc.hIcon			= LoadIcon(m_hInstance, _T("ICON"));
    wc.hCursor			= 0;
    wc.lpszMenuName		= 0;
	wc.hbrBackground	= (HBRUSH) GetStockObject(WHITE_BRUSH);
    wc.lpszClassName	= WINDOW_CLASS;

	return RegisterClass(&wc);
}
#endif
/*****************************************************************************
 Global code
*****************************************************************************/

void doButtonDown(HWND hWnd, PVRShellInit *pData, EPVRShellButtonState eButton, LPARAM lParam)
{
	RECT rcWinDimensions;
	GetClientRect(hWnd,&rcWinDimensions);
	pData->m_vec2PointerLocation[0] = (float)LOWORD(lParam)/(float)(rcWinDimensions.right);
	pData->m_vec2PointerLocation[1] = (float)HIWORD(lParam)/(float)(rcWinDimensions.bottom);
	SetCapture(hWnd);	// must be within window so capture
	pData->m_bPointer = true;
	pData->m_u32ButtonState |= eButton;
}
bool doButtonUp(HWND hWnd, PVRShellInit *pData, EPVRShellButtonState eButton)
{
	pData->m_u32ButtonState &= (~eButton);
	if(pData->m_vec2PointerLocation[0]<0.f ||
		pData->m_vec2PointerLocation[0]>1.f ||
		pData->m_vec2PointerLocation[1]<0.f ||
		pData->m_vec2PointerLocation[1]>1.f)
	{	// pointer has left window
		if(pData->m_u32ButtonState==0)
		{	// only release capture if mouse buttons have been released
			ReleaseCapture();
		}
		pData->m_bPointer = false;
		return false;
	}
	return true;
}

/*!***************************************************************************
@function		WndProc
@input			hWnd		Handle to the window
@input			message		Specifies the message
@input			wParam		Additional message information
@input			lParam		Additional message information
@returns		result code to OS
@description	Processes messages for the main window.
*****************************************************************************/
#ifndef NO_GDI
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	PVRShellInit	*pData = (PVRShellInit*)(__int64)GetWindowLong(hWnd, GWL_USERDATA);

	switch (message)
	{
	case WM_CREATE:
		{
			CREATESTRUCT	*pCreate = (CREATESTRUCT*)lParam;
			SetWindowLong(hWnd, GWL_USERDATA, (LONG)(LONG_PTR)pCreate->lpCreateParams); // Not ideal, but WinCE doesn't have SetWindowLongPtr

#ifdef UNDER_CE
			g_bClosing = false;
			// To receive power notifications, you need a second thread.
			// Dealing with multiple threads is a large part of the
			// complexity of this approach to stopping animations.
			HANDLE hThread;
			hThread = CreateThread(NULL, 0, PowerNotificationThread, hWnd, 0, NULL);

			// Close the thread handle since we don’t need it to monitor the thread’s exit.
			// If we don’t close it here the OS will not be able to release internal resources
			// until the process terminates.
			if (hThread != NULL)
				CloseHandle(hThread);
#endif

			break;
		}
	case WM_PAINT:
		break;
	case WM_DESTROY:
		return 0;
#ifdef UNDER_CE
	// The two messages that the power notification thread may post
	case WM_APP_DONTRENDER:
		g_bRendering = false;
		break;
	case WM_APP_RENDER:
		g_bRendering = true;
		break;
#endif
	case WM_CLOSE:
		pData->gShellDone = true;
		return 0;
	case WM_QUIT:
		return 0;
	case WM_LBUTTONDOWN:
		{
			doButtonDown(hWnd,pData,ePVRShellButtonLeft,lParam);
			break;
		}
	case WM_LBUTTONUP:
		{
			if(!doButtonUp(hWnd,pData,ePVRShellButtonLeft))
				return false;
		break;
		}
	case WM_RBUTTONDOWN:
		{
			doButtonDown(hWnd,pData,ePVRShellButtonRight,lParam);
			break;
		}
	case WM_RBUTTONUP:
		{
			if(!doButtonUp(hWnd,pData,ePVRShellButtonRight))
				return false;
			break;
		}
	case WM_MBUTTONDOWN:
		{
			doButtonDown(hWnd,pData,ePVRShellButtonMiddle,lParam);
			break;
		}
	case WM_MBUTTONUP:
		{
			if(!doButtonUp(hWnd,pData,ePVRShellButtonMiddle))
				return false;
			break;
		}
	case WM_MOUSEMOVE:
		{
			RECT rcWinDimensions;
			GetClientRect(hWnd,&rcWinDimensions);
			pData->m_vec2PointerLocation[0] = (float)LOWORD(lParam)/(float)(rcWinDimensions.right);
			pData->m_vec2PointerLocation[1] = (float)HIWORD(lParam)/(float)(rcWinDimensions.bottom);
			if(pData->m_vec2PointerLocation[0]<0.f ||
				pData->m_vec2PointerLocation[0]>1.f ||
				pData->m_vec2PointerLocation[1]<0.f ||
				pData->m_vec2PointerLocation[1]>1.f)
			{	// pointer has left window
				if(pData->m_u32ButtonState==0)
				{	// only release capture if mouse buttons have been released
					ReleaseCapture();
				}
				pData->m_bPointer = false;
				return false;
			}
			else
			{	// pointer is inside window
				pData->m_bPointer = true;
			}
			break;
		}
#ifdef UNDER_CE
	// For the Axim (X50v, X51v)
	case WM_HOTKEY:
		{
			switch(HIWORD(lParam))
			{
			case 0xC1:
				pData->KeyPressed(PVRShellKeyNameScreenshot);
				break;
			case 0xC2:
				pData->KeyPressed(PVRShellKeyNameACTION1);
				break;
			case 0xC3:
				pData->KeyPressed(PVRShellKeyNameACTION2);
				break;
			case 0xC4:
				pData->KeyPressed(PVRShellKeyNameQUIT);
				break;
			}
			break;
		}
#endif
	case WM_SETFOCUS:
		pData->m_bHaveFocus = true;
		return 0;
	case WM_KILLFOCUS:
		pData->m_bHaveFocus = false;
		return 0;
	case WM_KEYDOWN:
	{
		switch(wParam)
		{
		case VK_ESCAPE:
		case 0xC1:
			pData->KeyPressed(PVRShellKeyNameQUIT);
			break;
		case VK_UP:
		case 0x35:
			pData->KeyPressed(pData->m_eKeyMapUP);
			break;
		case VK_DOWN:
		case 0x30:
			pData->KeyPressed(pData->m_eKeyMapDOWN);
			break;
		case VK_LEFT:
		case 0x37:
			pData->KeyPressed(pData->m_eKeyMapLEFT);
			break;
		case VK_RIGHT:
		case 0x39:
			pData->KeyPressed(pData->m_eKeyMapRIGHT);
			break;
		case VK_F23:	// X50v select button
		case VK_SPACE:
		case 0x38:
			pData->KeyPressed(PVRShellKeyNameSELECT);
			break;
		case '1':
		case 0x34:
			pData->KeyPressed(PVRShellKeyNameACTION1);
			break;
		case '2':
		case 0x36:
			pData->KeyPressed(PVRShellKeyNameACTION2);
			break;
		case VK_F11:
		case 0xC2:
			pData->KeyPressed(PVRShellKeyNameScreenshot);
			break;
		}
	}
	default:
		break;
	}

	return DefWindowProc(hWnd, message, wParam, lParam);
}
#endif
/*!***************************************************************************
@function		WinMain
@input			hInstance		Application instance from OS
@input			hPrevInstance	Always NULL
@input			lpCmdLine		command line from OS
@input			nCmdShow		Specifies how the window is to be shown
@returns		result code to OS
@description	Main function of the program
*****************************************************************************/
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, TCHAR *lpCmdLine, int nCmdShow)
{
	size_t			i;
	char			*pszCmdLine;
	PVRShellInit	init;

#if defined(WIN32) && !defined(UNDER_CE)
	// Enable memory-leak reports
	_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG));
#endif

	// Get a char-array command line as the input may be UNICODE
	i = _tcslen(lpCmdLine) + 1;
	pszCmdLine = new char[i];

	while(i)
	{
		--i;
		pszCmdLine[i] = (char)lpCmdLine[i];
	}

	//	Create the demo, process the command line, create the OS initialiser.
	PVRShell *pDemo = NewDemo();

	if(!pDemo)
		return EXIT_ERR_CODE;

	init.Init(*pDemo);
	init.CommandLine(pszCmdLine);
	init.m_hInstance = hInstance;
	init.m_nCmdShow = nCmdShow;

	//	Initialise/run/shutdown
	while(init.Run());

#if defined(UNDER_CE) && !defined(NO_GDI)
	g_bClosing = true;

	if(g_hPower != NULL)
	{
		CloseMsgQueue(g_hPower);
		g_hPower = NULL;
	}
#endif

	delete pDemo;
	delete [] pszCmdLine;

	return EXIT_NOERR_CODE;
}

#if defined(UNDER_CE) && !defined(NO_GDI)
/*!***************************************************************************
@function		PowerNotificationThread
@input			pContext		A void pointer that is the hWnd
@returns		result code to OS
@description	The Power notification thread. This thread monitors the power
				notifications to decide what state the device is in. This code
				is based on the code described in the msdn blog entry "Power
				to the people" by Mike Calligaro
				(http://blogs.msdn.com/windowsmobile/archive/2005/08/01/446240.aspx).
*****************************************************************************/
DWORD PowerNotificationThread(PVOID pContext)
{
	HWND hWnd = (HWND)pContext;

	BYTE pbMsgBuf[sizeof(POWER_BROADCAST) + sizeof(POWER_BROADCAST_POWER_INFO)];

	MSGQUEUEOPTIONS msgopts;
	HANDLE hReq = NULL;

	BOOL fContinue = false;

	// Create our message queue
	memset(&msgopts, 0, sizeof(msgopts));
    msgopts.dwSize = sizeof(msgopts);
    msgopts.dwFlags = MSGQUEUE_NOPRECOMMIT;
    msgopts.dwMaxMessages = 0;
    msgopts.cbMaxMessage = sizeof(pbMsgBuf);
    msgopts.bReadAccess = TRUE;

	if(!g_hPower)
		g_hPower = CreateMsgQueue(NULL, &msgopts);

	if(g_hPower != NULL)
	{
		// Request power notifications
		hReq = RequestPowerNotifications(g_hPower, PBT_TRANSITION);

		if(hReq != NULL)
		{
			PVRShellInit	*pData;
			pData = (PVRShellInit*)(__int64) GetWindowLong(hWnd, GWL_USERDATA);

			while(!g_bClosing)
			{
				if(!pData->m_pShell->PVRShellGet(prefPowerSaving))
					g_bClosing = true;
				else
				{
					DWORD dwSize, dwFlags;

					/*
						Note the importance of having an INFINITE timeout here.  If you have a shorter timeout,
						you're polling, which burns power the same way the animation does.  You'd be creating
						the problem you're trying to solve.
					*/
					WaitForSingleObject(g_hPower, INFINITE);

					if(ReadMsgQueue(g_hPower, pbMsgBuf, sizeof(pbMsgBuf), &dwSize, 0, &dwFlags))
					{
						POWER_BROADCAST *pPower = (POWER_BROADCAST*)pbMsgBuf;

						/*
							If the screen is on then post a message to say we want to render. However,
							if it is off, or the device has suspended or the state says the user has
							gone idle then cease rendering.
						*/
						if((pPower->Flags & POWER_STATE_ON))
							PostMessage(hWnd, WM_APP_RENDER, 0, 0);
						else if((pPower->Flags & (POWER_STATE_OFF | POWER_STATE_SUSPEND | POWER_STATE_IDLE)))
							PostMessage(hWnd, WM_APP_DONTRENDER, 0, 0);
	#ifdef POWER_STATE_USERIDLE
						else if((pPower->Flags & POWER_STATE_USERIDLE))
							PostMessage(hWnd, WM_APP_DONTRENDER, 0, 0);
	#endif
					}
				}
			}

			StopPowerNotifications(hReq);
		}

		CloseMsgQueue(g_hPower);
		g_hPower = NULL;
	}

	return (0);
}


#endif

/*****************************************************************************
 End of file (PVRShellOS.cpp)
*****************************************************************************/
