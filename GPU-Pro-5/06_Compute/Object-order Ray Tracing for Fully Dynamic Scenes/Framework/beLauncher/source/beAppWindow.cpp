/*****************************************************/
/* breeze Framework Launch Lib  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beLauncherInternal/stdafx.h"
#include "beLauncher/beAppWindow.h"

#include "../resource/beLauncherResource.h"
#include <lean/logging/win_errors.h>


namespace beLauncher
{

namespace
{

#ifdef _M_X64
	typedef INT_PTR DLG_PROC_RESULT;
#else
	typedef BOOL DLG_PROC_RESULT;
#endif

DLG_PROC_RESULT CALLBACK AppWindowDlgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	ApplicationWindow *pAppWindow = reinterpret_cast<ApplicationWindow*>(::GetWindowLongPtrW(hWnd, GWLP_USERDATA));

	switch (uMsg)
	{
	case WM_INITDIALOG:
		{
			pAppWindow = reinterpret_cast<ApplicationWindow*>(lParam);
			::SetWindowLongPtrW(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pAppWindow));

			LEAN_ASSERT(pAppWindow);

			pAppWindow->WindowCreated(hWnd);
		}
		return TRUE;

	case WM_MOVE:
	case WM_SIZE:
		{
			if (pAppWindow)
			{
				RECT screenRect;

				::GetClientRect(hWnd, &screenRect);
				::ClientToScreen(hWnd, &screenRect);
				
				if (uMsg == WM_MOVE)
					pAppWindow->WindowMoved(hWnd, screenRect);
				else
					pAppWindow->WindowResized(hWnd, screenRect);
			}
		}
		break;

	case WM_CLOSE:
		{
			LEAN_ASSERT(pAppWindow);

			 if (pAppWindow->WindowClosed(hWnd))
				 ::DestroyWindow(hWnd);
		}
		return TRUE;

	case WM_DESTROY:
		{
			LEAN_ASSERT(pAppWindow);

			pAppWindow->WindowDestroyed(hWnd);

			PostQuitMessage(0);
		}
		return TRUE;
	}

	LRESULT result = 0;

	if (pAppWindow && pAppWindow->HandleMessage(hWnd, uMsg, wParam, lParam, result))
	{
		SetWindowLongPtr(hWnd, DWLP_MSGRESULT, result);
		return TRUE;
	}

	return FALSE;
}

DLG_PROC_RESULT CALLBACK AuxWindowDlgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
	case WM_SETFOCUS:
		::SetFocus( ::GetParent(hWnd) );
		return TRUE;
		break;
	}

	return FALSE;
}

} // namespace
} // namespace

// Creates a default application window.
HWND beLauncher::CreateAppWindow(const lean::utf16_ntri &title, HINSTANCE hInstance, DLGPROC proc, void *param, HICON hIcon, bool fullScreen)
{
	HWND hWnd = ::CreateDialogParamW(hInstance, MAKEINTRESOURCE(IDD_BE_MAINWINDOW), NULL, proc, reinterpret_cast<LPARAM>(param));

	::SetWindowTextW(hWnd, title.c_str());

	if (hIcon == NULL)
		hIcon = ::LoadIconW(NULL, IDI_APPLICATION);

	::SendMessageW(hWnd, WM_SETICON, static_cast<WPARAM>(ICON_SMALL), reinterpret_cast<LPARAM>(hIcon));
	::SendMessageW(hWnd, WM_SETICON, static_cast<WPARAM>(ICON_BIG), reinterpret_cast<LPARAM>(hIcon));

	// Windows Vista+ mess up window rendering in full-screen modes
	if (fullScreen)
		::SetWindowLongPtrW(hWnd, GWL_STYLE, WS_POPUP | WS_SYSMENU);

	return hWnd;
}

/// Creates a default application window.
HWND beLauncher::CreateAppWindow(const lean::utf16_ntri &title, HINSTANCE hInstance, ApplicationWindow *pAppWindow,
	HICON hIcon, bool fullScreen)
{
	return CreateAppWindow(title, hInstance, &AppWindowDlgProc, pAppWindow, hIcon, fullScreen);
}

// Creates an auxiliary window.
HWND beLauncher::CreateAuxWindow(HINSTANCE hInstance, HWND hParentWnd, DLGPROC proc, void *param)
{
	if (!proc)
		proc = &AuxWindowDlgProc;

	return ::CreateDialogParamW(hInstance, MAKEINTRESOURCE(IDD_BE_AUXWINDOW),
		hParentWnd, proc,
		reinterpret_cast<LPARAM>(param));
}

// Runs the application window.
void beLauncher::RunAppWindow(ApplicationWindow *pAppWindow, HWND hFocusWnd)
{
	MSG msg = { 0 };

	if (!::IsWindowVisible(hFocusWnd))
		::ShowWindow(hFocusWnd, SW_SHOW);

	while (true)
	{
		while (::PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				return;

			if (!::IsDialogMessageW(NULL, &msg))
			{
				::TranslateMessage(&msg);
				::DispatchMessageW(&msg);
			}
		}

		if (pAppWindow)
			pAppWindow->WindowRun(hFocusWnd);
	}
}

// Scales the application window to fit the given resolution.
void beLauncher::ScaleWindow(HWND hWnd, lean::uint4 width, lean::uint4 height)
{
	RECT windowRect = { 0, 0, width, height };
	::AdjustWindowRect(&windowRect, ::GetWindowLongW(hWnd, GWL_STYLE), false);

	::SetWindowPos(hWnd, 
		NULL,
		0, 0,
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		SWP_NOMOVE | SWP_NOOWNERZORDER );
}

// Centers the application window on the given primary screen.
void beLauncher::CenterWindow(HWND hWnd)
{
	RECT windowRect;
	::GetWindowRect(hWnd, &windowRect);

	::SetWindowPos(hWnd, 
		NULL,
		(::GetSystemMetrics(SM_CXSCREEN) - (windowRect.right - windowRect.left))  / 2,
		(::GetSystemMetrics(SM_CYSCREEN) - (windowRect.bottom - windowRect.top))  / 2,
		0, 0,
		SWP_NOSIZE | SWP_NOOWNERZORDER );
}

// Transforms a client rectangle to screen.
void ClientToScreen(HWND hWnd, RECT *pRect)
{
	union POINT_RECT
	{
		RECT r;
		POINT p[2];
	} &rect = reinterpret_cast<POINT_RECT&>(*pRect);

	::ClientToScreen(hWnd, &rect.p[0]);
	::ClientToScreen(hWnd, &rect.p[1]);
}

// Transforms a screen rectangle to client.
void ScreenToClient(HWND hWnd, RECT *pRect)
{
	union POINT_RECT
	{
		RECT r;
		POINT p[2];
	} &rect = reinterpret_cast<POINT_RECT&>(*pRect);

	::ScreenToClient(hWnd, &rect.p[0]);
	::ScreenToClient(hWnd, &rect.p[1]);
}
