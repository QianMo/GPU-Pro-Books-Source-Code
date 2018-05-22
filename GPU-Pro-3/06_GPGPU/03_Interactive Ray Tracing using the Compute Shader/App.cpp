// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

// ------------------------------------------------
// App.cpp
// ------------------------------------------------
// Defines the entry point for the application.

#include <ArgumentsParser.h>
extern ArgumentsParser m_Parser = ArgumentsParser(); 
#include "Common.h"
#include "RayTracerCS.h"

typedef RayTracerCS RayTracer;

// Global Variables
HWND						g_hWnd = NULL;						// Main window
RayTracer*					g_pRayTracer = NULL;				// RT application
Scene*						g_pScene = NULL;					// Scene data

// Functions
int							InitApp();							// Initialize raytracer application.
int							InitWindow();						// Initialize window properties.
void						UpdateWindowTitle();				// Update Window Title.
template<class T> int		Run( T* pApp );						// Message loop
LRESULT CALLBACK			WndProc(HWND, UINT, WPARAM, LPARAM);// Control user input

//--------------------------------------------------------------------------------------
// Entry point of the application.
//--------------------------------------------------------------------------------------
int main( int argc, char*argv[] )
{
	g_hWnd = GetConsoleWindow();
	MoveWindow(g_hWnd,1050,1,850,1070,1);

	// Initialize the application (CPU or GPU version)
	if( InitApp() ) { return 0; }
	
	// Initialize the window
	if( InitWindow() ) { return 0;	}

	g_pRayTracer = new RayTracer(g_pScene, g_hWnd);
	
	// Main message loop:
	Run(g_pRayTracer);

	return 0;
}

//--------------------------------------------------------------------------------------
// Run message loop
//--------------------------------------------------------------------------------------
template<class T>
int Run( T* pApp )
{
	printf("Application started.\n");

	MSG msg;
	msg.message = WM_NULL;
	PeekMessage(&msg, NULL, 0U, 0U, PM_NOREMOVE);
	while (msg.message != WM_QUIT)
	{
		if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
        {
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        }
		else
		{
			pApp->Render();
			UpdateWindowTitle();
		}
	}
	return (int) msg.wParam;
}

//--------------------------------------------------------------------------------------
// Initialize application
//--------------------------------------------------------------------------------------
int InitApp()
{
	printf("Initializing application...\n");
	
	m_Parser.ParseData();

	std::map<std::string,std::vector<Point>> vObjects;
	const char* sModel = m_Parser.GetModelName();
	std::vector<Point> vPositions;
	vPositions.push_back(Point(0.0f,0.0f,0.0f));

	vObjects.insert( std::pair<const char*,std::vector<Point>>(sModel, vPositions) );
	
	printf("Loading %s...\n", sModel);
	// Create scene with an object-position list
	g_pScene = new Scene(vObjects);

	return 0;
}

//--------------------------------------------------------------------------------------
// Register class and create window
//--------------------------------------------------------------------------------------
int InitWindow( )
{
	printf("Initializing window...\n");

	HINSTANCE hInstance = (HINSTANCE)GetWindowLong(g_hWnd, GWL_HINSTANCE);

    // Register class
	WNDCLASSEX wcex;
    wcex.cbSize = sizeof( WNDCLASSEX );
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon( hInstance, ( LPCTSTR )"RayTracer" );
    wcex.hCursor = LoadCursor( NULL, IDC_ARROW );
    wcex.hbrBackground = ( HBRUSH )( COLOR_WINDOW + 1 );
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = L"RayTracer";
    wcex.hIconSm = LoadIcon( wcex.hInstance, ( LPCTSTR )"RayTracer" );
    if( !RegisterClassEx( &wcex ) )
        return E_FAIL;

    // Create window
	RECT rc = { 0, 0, WIDTH, HEIGHT };
    AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
    g_hWnd = CreateWindow( L"RayTracer", L"Direct3D - Ray Tracer",
                           WS_OVERLAPPEDWINDOW,
                           0,0, rc.right - rc.left, rc.bottom - rc.top, NULL, NULL, hInstance,
                           NULL );
    if( !g_hWnd )
        return E_FAIL;

	// Display window on screen
    ShowWindow( g_hWnd, SW_SHOW );

    return 0;
}

//--------------------------------------------------------------------------------------
// Processes messages for the main window.
//--------------------------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	PAINTSTRUCT ps;
	HDC hdc;
	
	switch (message)
	{
	case WM_PAINT:
		hdc = BeginPaint( hWnd, &ps );
		EndPaint( hWnd, &ps );
		break;
	case WM_DESTROY:
		printf("Quit application...\n");
		PostQuitMessage(0);
		break;
	}
	return g_pRayTracer->WndProc(hWnd, message, wParam, lParam);
}

//--------------------------------------------------------------------------------------
// Updates Frames per second (FPS) and print this info on Window title.
//--------------------------------------------------------------------------------------
void UpdateWindowTitle()
{
	char aux[1024];
	char* str = g_pScene->GetObjects()[0]->GetModel()->GetAccelStructure()->GetName();
	if (g_pRayTracer->UpdateWindowTitle(aux, str))
	{
		SetWindowTextA(g_hWnd,aux);
	}
}