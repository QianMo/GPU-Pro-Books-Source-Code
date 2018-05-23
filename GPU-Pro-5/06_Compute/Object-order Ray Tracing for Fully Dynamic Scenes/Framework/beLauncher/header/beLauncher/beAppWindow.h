/*****************************************************/
/* breeze Framework Launch Lib  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_LAUNCHER_APPWINDOW
#define BE_LAUNCHER_APPWINDOW

#include "beLauncher.h"
#include <Windows.h>

namespace beLauncher
{

/// Application window interface.
class LEAN_INTERFACE ApplicationWindow
{
	LEAN_INTERFACE_BEHAVIOR(ApplicationWindow)

public:
	/// Called when the window has been created.
	virtual void WindowCreated(HWND hWnd) { }
	/// Called when the window has been moved.
	virtual void WindowMoved(HWND hWnd, const RECT &rect) { }
	/// Called when the window has been resized.
	virtual void WindowResized(HWND hWnd, const RECT &rect) { }
	/// Called when the window is ready for frame updates.
	virtual void WindowRun(HWND hWnd) = 0;
	/// Called when the window is about to be closed. Return false to keep the window open.
	virtual bool WindowClosed(HWND hWnd) { return true; }
	/// Called when the window has been destroyed.
	virtual void WindowDestroyed(HWND hWnd) { }

	/// Called for every other message in the queue.
	virtual bool HandleMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam, LRESULT &result) { return false; }
};

/// Creates a default application window.
BE_LAUNCHER_API HWND CreateAppWindow(const lean::utf16_ntri &title, HINSTANCE hInstance, DLGPROC proc, void *param = nullptr,
	HICON hIcon = NULL, bool fullScreen = false);
/// Creates a default application window.
BE_LAUNCHER_API HWND CreateAppWindow(const lean::utf16_ntri &title, HINSTANCE hInstance, ApplicationWindow *pAppWindow,
	HICON hIcon = NULL, bool fullScreen = false);

/// Creates an auxiliary window.
BE_LAUNCHER_API HWND CreateAuxWindow(HINSTANCE hInstance, HWND hParentWnd, DLGPROC proc = nullptr, void *param = nullptr);

/// Scales the application window to fit the given resolution.
BE_LAUNCHER_API void ScaleWindow(HWND hWnd, lean::uint4 width, lean::uint4 height);
/// Centers the application window on the given primary screen.
BE_LAUNCHER_API void CenterWindow(HWND hWnd);

/// Runs the application window.
BE_LAUNCHER_API void RunAppWindow(ApplicationWindow *pAppWindow, HWND hFocusWnd);

}

// WinAPI Enhancements

/// Transforms a client rectangle to screen.
BE_LAUNCHER_API void ClientToScreen(HWND hWnd, RECT *pRect);
/// Transforms a screen rectangle to client.
BE_LAUNCHER_API void ScreenToClient(HWND hWnd, RECT *pRect);

#endif