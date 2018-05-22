#include "Main.h"

#include <windows.h>
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <fstream>
#include <strsafe.h>
#include "MemoryLeakTracker.h"

HWND g_hWnd;
Main *g_Main;
bool g_EndApp;
bool g_Maximized;
int  g_Fullscreen;

using namespace std;

#define SHOW_FPS 

//--------------------------------------------------------------------------------------
// Called every time the application receives a message
//--------------------------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	PAINTSTRUCT ps;
	HDC hdc;

	if(g_Main && g_Main->MsgProc(hWnd, message, wParam, lParam))
		return 0;

	switch (message) 
	{
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		EndPaint(hWnd, &ps);
		break;

	case WM_SIZE:
		if(wParam == SIZE_MAXIMIZED)
			g_Maximized = true;
		else
			if(g_Maximized && wParam == SIZE_RESTORED)
				g_Maximized = false;
			else
				break;
		

	case WM_EXITSIZEMOVE:
		{
			RECT rc;
			GetClientRect(g_hWnd, &rc);
			int width = (int)rc.right - rc.left;
			int height = (int)rc.bottom - rc.top;

			if(g_Main->WindowSizeChanged(width, height) != CORE_OK)
			{
				g_EndApp = true;
				MessageBox(g_hWnd, L"Backbuffer resize failed!", L"Error", MB_ICONERROR);
			}

			break;
		}

	case WM_LBUTTONDOWN:
		g_Main->SetMouseButton(false, true);
		break;
	case WM_LBUTTONUP:
		g_Main->SetMouseButton(false, false);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}

	return 0;
}

// Register class and create window
HRESULT InitWindow( HINSTANCE hInstance, int nCmdShow )
{
	// Register class
	WNDCLASSEX wcex;
	wcex.cbSize		    = sizeof(WNDCLASSEX); 
	wcex.style          = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc    = WndProc;
	wcex.cbClsExtra     = 0;
	wcex.cbWndExtra     = 0;
	wcex.hInstance      = hInstance;
	wcex.hIcon          = NULL;
	wcex.hCursor        = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName   = NULL;
	wcex.lpszClassName  = L"DeepShadowMapsClass";
	wcex.hIconSm        = NULL;//LoadIcon(wcex.hInstance, (LPCTSTR)IDI_APPICON);
	if(!RegisterClassEx(&wcex))
		return E_FAIL;

	std::ifstream fp ("config.txt", std::ios::in); 
	int resV = 720;
	int resH = 1280;
	g_Fullscreen = 0;
	if (fp.is_open())
	{
		fp >> resH >> resV >> g_Fullscreen;
		fp.close();
	}

	// Create window
	// TODO Create window based on screen resolution
	RECT rc = {0, 0, resH, resV};
	AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
	g_hWnd = CreateWindow(L"DeepShadowMapsClass", L"DeepShadowMaps", WS_OVERLAPPEDWINDOW,
		(GetSystemMetrics(SM_CXFULLSCREEN) - rc.right + rc.left) / 2, 
		(GetSystemMetrics(SM_CYFULLSCREEN) - rc.bottom + rc.top) / 2, rc.right - rc.left, rc.bottom - rc.top, NULL, NULL, hInstance, NULL);
	if(!g_hWnd)
		return E_FAIL;

	ShowWindow(g_hWnd, nCmdShow);

	return S_OK;
}

float CalcFPS(float timeRunning)
{
	static UINT frames = 0;
	static float last = -1.0f;
	static float lastFPS = 0.0f;

	if (last < 0.0f) 
		last = timeRunning;

	frames += 1;

	if(timeRunning - last >= 1.0f) 
	{
		lastFPS = (float)frames * 1 / (timeRunning - last);
		last = timeRunning;
		frames = 0;
	}
	return lastFPS;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
{
	if(FAILED(InitWindow(hInstance, nCmdShow)))
		return 0;

	double time = 0.0;
	double lastTime = 0.0;
	LONGLONG	startTime;	
	LONGLONG	endTime;
	LONGLONG	frequency;
	g_EndApp = false;
	g_Maximized = false;
	bool bContinue = true;

	#ifdef _DEBUG
//	InitMemoryTracker();
//	_CrtSetBreakAlloc(1616);
	#endif

	if(CreateMain(g_hWnd, &g_Main, g_Fullscreen != 0) != CORE_OK)
	{
		MessageBox(g_hWnd, L"Engine initialization failed!", L"Error", MB_ICONERROR);
		return -1;
	}
	
		
	QueryPerformanceFrequency((LARGE_INTEGER*)&frequency);

	// Main message loop
	MSG msg = {0};
	while(msg.message != WM_QUIT && bContinue)
	{
		if(g_EndApp) break;
		QueryPerformanceCounter((LARGE_INTEGER*)&startTime);
		

		if(PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			bContinue = g_Main->NewFrame((float)(time - lastTime), (float)time, GetActiveWindow() == g_hWnd);
			
			WCHAR title[200];
			StringCbPrintf(title, 200, L"DeepShadowMaps - %.2f", CalcFPS((float)time));
			#ifdef SHOW_FPS
			SetWindowText(g_hWnd, title);
			#endif
			QueryPerformanceCounter((LARGE_INTEGER*)&endTime);
			lastTime = time;
			time += (endTime - startTime) / (double)frequency;
		}
	}

	SAFE_RELEASE(g_Main);

	return (int)msg.wParam;
}


