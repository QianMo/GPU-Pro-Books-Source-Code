
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "Resource.h"

#include "../CPU.h"
#include "../BaseApp.h"
#include <direct.h>

#include <mmsystem.h>
#pragma comment (lib, "winmm.lib")


extern BaseApp *app;

#define GETX(l) (int(l & 0xFFFF))
#define GETY(l) (int(l) >> 16)

BOOL active = TRUE;

LRESULT CALLBACK WinProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam){
	switch (message){
	case WM_PAINT:
		PAINTSTRUCT paint;
		BeginPaint(hwnd, &paint);
		active = !IsRectEmpty(&paint.rcPaint);
		EndPaint(hwnd, &paint);
		break;
	case WM_MOUSEMOVE:
		static int lastX, lastY;
		int x, y;
		x = GETX(lParam);
		y = GETY(lParam);
		app->onMouseMove(x, y, x - lastX, y - lastY);
		lastX = x;
		lastY = y;
		break;
	case WM_KEYDOWN:
		app->onKey((unsigned int) wParam, true);
		break;
	case WM_KEYUP:
		app->onKey((unsigned int) wParam, false);
		break;
	case WM_LBUTTONDOWN:
		app->onMouseButton(GETX(lParam), GETY(lParam), MOUSE_LEFT, true);
 		break;
	case WM_LBUTTONUP:
		app->onMouseButton(GETX(lParam), GETY(lParam), MOUSE_LEFT, false);
		break;
	case WM_RBUTTONDOWN:
		app->onMouseButton(GETX(lParam), GETY(lParam), MOUSE_RIGHT, true);
 		break;
	case WM_RBUTTONUP:
		app->onMouseButton(GETX(lParam), GETY(lParam), MOUSE_RIGHT, false);
		break;
	case WM_MOUSEWHEEL:
		static int scroll;
		int s;

		scroll += GET_WHEEL_DELTA_WPARAM(wParam);
		s = scroll / WHEEL_DELTA;
		scroll %= WHEEL_DELTA;

		POINT point;
		point.x = GETX(lParam);
		point.y = GETY(lParam);
		ScreenToClient(hwnd, &point);

		if (s != 0) app->onMouseWheel(point.x, point.y, s);
		break;
	case WM_SIZE:
		app->onSize(LOWORD(lParam), HIWORD(lParam));
		break;
	case WM_WINDOWPOSCHANGED:
		if ((((LPWINDOWPOS) lParam)->flags & SWP_NOSIZE) == 0){
			RECT rect;
			GetClientRect(hwnd, &rect);
			int w = rect.right - rect.left;
			int h = rect.bottom - rect.top;
			if (w > 0 && h > 0) app->onSize(w, h);
		}
		break;
	case WM_SYSKEYDOWN:
		if ((lParam & (1 << 29)) && wParam == KEY_ENTER){
			app->toggleFullscreen();
		} else {
			app->onKey((unsigned int) wParam, true);
		}
		break;
	case WM_SYSKEYUP:
		app->onKey((unsigned int) wParam, false);
		break;
	case WM_CREATE:
		ShowWindow(hwnd, SW_SHOW);
		break;
	case WM_CLOSE:
		app->closeWindow(true, true);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hwnd, message, wParam, lParam);
	}
	return 0;
}

#ifdef _DEBUG
#include <crtdbg.h>
#endif

#include <stdio.h>

int WINAPI WinMain(HINSTANCE hThisInst, HINSTANCE hLastInst, LPSTR lpszCmdLine, int nCmdShow){
#ifdef _DEBUG
	int flag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG); // Get current flag
	flag |= _CRTDBG_LEAK_CHECK_DF; // Turn on leak-checking bit
	flag |= _CRTDBG_CHECK_ALWAYS_DF; // Turn on CrtCheckMemory
//	flag |= _CRTDBG_DELAY_FREE_MEM_DF;
	_CrtSetDbgFlag(flag); // Set flag to the new value
#endif
	initCPU();

	// Make sure we're running in the exe's path
	char path[MAX_PATH];
	if (GetModuleFileName(NULL, path, sizeof(path))){
		char *slash = strrchr(path, '\\');
		if (slash) *slash = '\0';
        chdir(path);
	}

	MSG msg;
	WNDCLASS wincl;
	wincl.hInstance = hThisInst;
	wincl.lpszClassName = "Humus";
	wincl.lpfnWndProc = WinProc;
	wincl.style = 0;
	wincl.hIcon = LoadIcon(hThisInst, MAKEINTRESOURCE(IDI_MAINICON));
	wincl.hCursor = LoadCursor(NULL, IDI_APPLICATION);
	wincl.lpszMenuName = NULL;
	wincl.cbClsExtra = 0;
	wincl.cbWndExtra = 0;
	wincl.hbrBackground = NULL;
	if (!RegisterClass(&wincl)) return 0;

	app->setInstance(hThisInst);

	//JOYINFO joyInfo;
	//bool useJoystick = (joyGetPos(JOYSTICKID1, &joyInfo) == JOYERR_NOERROR);
	JOYCAPS joyCaps;
	DWORD joyFlags = 0;
	float xScale = 0, xBias = 0;
	float yScale = 0, yBias = 0;
	float zScale = 0, zBias = 0;
	float rScale = 0, rBias = 0;
	float uScale = 0, uBias = 0;
	float vScale = 0, vBias = 0;
	if (joyGetDevCaps(JOYSTICKID1, &joyCaps, sizeof(joyCaps)) == JOYERR_NOERROR){
		joyFlags = JOY_RETURNX | JOY_RETURNY | JOY_RETURNBUTTONS;
		xScale = 2.0f / float(int(joyCaps.wXmax) - int(joyCaps.wXmin));
		xBias  = 1.0f - joyCaps.wXmax * xScale;
		yScale = 2.0f / float(int(joyCaps.wYmax) - int(joyCaps.wYmin));
		yBias  = 1.0f - joyCaps.wYmax * yScale;

		if (joyCaps.wCaps & JOYCAPS_HASZ){
			joyFlags |= JOY_RETURNZ;
			zScale = 2.0f / float(int(joyCaps.wZmax) - int(joyCaps.wZmin));
			zBias  = 1.0f - joyCaps.wZmax * zScale;
		}
		if (joyCaps.wCaps & JOYCAPS_HASR){
			joyFlags |= JOY_RETURNR;
			rScale = 2.0f / float(int(joyCaps.wRmax) - int(joyCaps.wRmin));
			rBias  = 1.0f - joyCaps.wRmax * rScale;
		}
		if (joyCaps.wCaps & JOYCAPS_HASU){
			joyFlags |= JOY_RETURNU;
			uScale = 2.0f / float(int(joyCaps.wUmax) - int(joyCaps.wUmin));
			uBias  = 1.0f - joyCaps.wUmax * uScale;
		}
		if (joyCaps.wCaps & JOYCAPS_HASV){
			joyFlags |= JOY_RETURNV;
			vScale = 2.0f / float(int(joyCaps.wVmax) - int(joyCaps.wVmin));
			vBias  = 1.0f - joyCaps.wVmax * vScale;
		}
	}

	// Initialize timer
	app->initTime();

	app->loadConfig();
	app->initGUI();

	/*
		Force the main thread to always run on CPU 0.
		This is done because on some systems QueryPerformanceCounter returns a bit different counter values
		on the different CPUs (contrary to what it's supposed to do), which can cause negative frame times
		if the thread is scheduled on the other CPU in the next frame. This can cause very jerky behavior and
		appear as if frames return out of order.
	*/
	SetThreadAffinityMask(GetCurrentThread(), 1);

	if (app->init()){
		app->resetCamera();

		do {
			app->loadConfig();

			if (!app->initCaps()) break;
			if (!app->initAPI()) break;

			if (!app->load()){
				app->closeWindow(true, false);
			}

			while (true){

				InvalidateRect(app->getWindow(), NULL, FALSE);

				while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)){
					//TranslateMessage(&msg);
					DispatchMessage(&msg);
				}
				if (msg.message == WM_QUIT) break;

				if (active){
					/*
						Joystick support
					*/
					if (joyFlags){
						static DWORD lastButtons = 0;
						static DWORD lastXpos = 0, lastYpos = 0, lastZpos = 0;
						static DWORD lastRpos = 0, lastUpos = 0, lastVpos = 0;

						JOYINFOEX joyInfo;
						joyInfo.dwSize = sizeof(joyInfo);
						joyInfo.dwFlags = joyFlags;

						if (joyGetPosEx(JOYSTICKID1, &joyInfo) == JOYERR_NOERROR){
							DWORD changed = lastButtons ^ joyInfo.dwButtons;
							if (changed){
								for (uint i = 0; i < joyCaps.wNumButtons; i++){
									// Only call App for buttons that changed
									if (changed & 1){
										app->onJoystickButton(i, ((joyInfo.dwButtons >> i) & 1) != 0);
									}
									changed >>= 1;
								}

								lastButtons = joyInfo.dwButtons;
							}
							if ((joyInfo.dwFlags & JOY_RETURNX) && joyInfo.dwXpos != lastXpos){
								app->onJoystickAxis(0, joyInfo.dwXpos * xScale + xBias);
								lastXpos = joyInfo.dwXpos;
							}
							if ((joyInfo.dwFlags & JOY_RETURNY) && joyInfo.dwYpos != lastYpos){
								app->onJoystickAxis(1, joyInfo.dwYpos * yScale + yBias);
								lastYpos = joyInfo.dwYpos;
							}
							if ((joyInfo.dwFlags & JOY_RETURNZ) && joyInfo.dwZpos != lastZpos){
								app->onJoystickAxis(2, joyInfo.dwZpos * zScale + zBias);
								lastZpos = joyInfo.dwZpos;
							}
							if ((joyInfo.dwFlags & JOY_RETURNR) && joyInfo.dwRpos != lastRpos){
								app->onJoystickAxis(3, joyInfo.dwRpos * rScale + rBias);
								lastRpos = joyInfo.dwRpos;
							}
							if ((joyInfo.dwFlags & JOY_RETURNU) && joyInfo.dwUpos != lastUpos){
								app->onJoystickAxis(4, joyInfo.dwUpos * uScale + uBias);
								lastUpos = joyInfo.dwUpos;
							}
							if ((joyInfo.dwFlags & JOY_RETURNV) && joyInfo.dwVpos != lastVpos){
								app->onJoystickAxis(5, joyInfo.dwVpos * vScale + vBias);
								lastVpos = joyInfo.dwVpos;
							}
						}
					}

					app->updateTime();
					app->makeFrame();
				} else {
					Sleep(100);
				}
			}
		} while (!app->isDone());

		app->exit();
	}

	delete app;

	return (int) msg.wParam;
}
