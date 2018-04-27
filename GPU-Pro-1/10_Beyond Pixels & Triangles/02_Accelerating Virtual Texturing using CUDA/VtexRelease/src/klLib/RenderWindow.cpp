/*
Copyright (c) 2005-2009 Charles-Frederik Hollemeersch

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
*/

/* Based on NEHE opengl init code */

#define _WIN32_WINDOWS 0x501
#include <windows.h>
#include "RenderWindow.h"

#ifdef USES_GLEW
#include <gl\glew.h>
#include <gl\wglew.h>
#endif

#include <gl\gl.h>

// This automatically includes the opengl libs and saves on the visual studio setup
#pragma comment( lib, "opengl32.lib" )

struct RenderWindowInfo {
    HDC			hDC;		// Private GDI Device Context
    HGLRC		hRC;		// Permanent Rendering Context
    HWND		hWnd;		// Holds Our Window Handle
    HINSTANCE	hInstance;		// Holds The Instance Of The Application
    bool	    active;		// Window Active Flag Set To TRUE By Default
    bool	    fullscreen;	// Fullscreen Flag Set To Fullscreen Mode By Default
    int         oldButtons;
    RenderWindow::Listener *listener;
};

static RenderWindowInfo rwi;
LRESULT	CALLBACK RenderWindowProc(HWND, UINT, WPARAM, LPARAM);

GLvoid KillRenderWindow(GLvoid)
{
	if (rwi.fullscreen)
	{
		ChangeDisplaySettings(NULL,0);
		ShowCursor(TRUE);
	}

	if (rwi.hRC) {
		if (!wglMakeCurrent(NULL,NULL)) {
			MessageBox(NULL,"Release Of DC And RC Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		}

		if (!wglDeleteContext(rwi.hRC)) {
			MessageBox(NULL,"Release Rendering Context Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		}

		rwi.hRC=NULL;
	}

	if (rwi.hDC && !ReleaseDC(rwi.hWnd,rwi.hDC))
	{
		MessageBox(NULL,"Release Device Context Failed.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		rwi.hDC=NULL;
	}

	if (rwi.hWnd && !DestroyWindow(rwi.hWnd))
	{
		MessageBox(NULL,"Could Not Release hWnd.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		rwi.hWnd=NULL;	
	}

	if (!UnregisterClass("RenderWindow",rwi.hInstance))
	{
		MessageBox(NULL,"Could Not Unregister Class.","SHUTDOWN ERROR",MB_OK | MB_ICONINFORMATION);
		rwi.hInstance=NULL;
	}
}
 
bool CreateRenderWindow(const char* title, int width, int height, int bits, bool fullscreenflag) 
{
	GLuint		PixelFormat;			// Holds The Results After Searching For A Match
	WNDCLASS	wc;						// Windows Class Structure
	DWORD		dwExStyle;				// Window Extended Style
	DWORD		dwStyle;				// Window Style
	RECT		WindowRect;				// Grabs Rectangle Upper Left / Lower Right Values
	WindowRect.left=(long)0;			// Set Left Value To 0
	WindowRect.right=(long)width;		// Set Right Value To Requested Width
	WindowRect.top=(long)0;				// Set Top Value To 0
	WindowRect.bottom=(long)height;		// Set Bottom Value To Requested Height

	rwi.fullscreen=fullscreenflag;			// Set The Global Fullscreen Flag

	rwi.hInstance		= GetModuleHandle(NULL);
	wc.style			= CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wc.lpfnWndProc		= (WNDPROC) RenderWindowProc;
	wc.cbClsExtra		= 0;
	wc.cbWndExtra		= 0;
	wc.hInstance		= rwi.hInstance;
	wc.hIcon			= LoadIcon(NULL, IDI_WINLOGO);
	wc.hCursor			= LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground	= NULL;
	wc.lpszMenuName		= NULL;
	wc.lpszClassName	= "RenderWindow";

	if (!RegisterClass(&wc)) {
		MessageBox(NULL,"Failed To Register The Window Class.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;
	}
	
	if (rwi.fullscreen) {
	    DEVMODE dmScreenSettings;
		memset(&dmScreenSettings,0,sizeof(dmScreenSettings));
		dmScreenSettings.dmSize=sizeof(dmScreenSettings);
		dmScreenSettings.dmPelsWidth	= width;
		dmScreenSettings.dmPelsHeight	= height;
		dmScreenSettings.dmBitsPerPel	= bits;
		dmScreenSettings.dmFields=DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT;

		// Try to go fullscreen
		if (ChangeDisplaySettings(&dmScreenSettings,CDS_FULLSCREEN)!=DISP_CHANGE_SUCCESSFUL)
		{
			// If The Mode Fails, Offer Two Options.  Quit Or Use Windowed Mode.
			if (MessageBox(NULL,"The Requested Fullscreen Mode Is Not Supported By\nYour Video Card. Use Windowed Mode Instead?","NeHe GL",MB_YESNO|MB_ICONEXCLAMATION)==IDYES)
			{
				rwi.fullscreen=false;
            } else {
				return false;
			}
		}
	}

	if (rwi.fullscreen)	{
		dwExStyle=WS_EX_APPWINDOW;
		dwStyle=WS_POPUP;
		//ShowCursor(FALSE);
	} else {
		dwExStyle=WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
		dwStyle=WS_OVERLAPPEDWINDOW;
	}

    // Fix the rectangle so we have the correct requested client size
	AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle);

    int initWindowPosX = 0;
    int initWindowPosY = 0;

	if (!(rwi.hWnd=CreateWindowEx(dwExStyle,"RenderWindow",	title,
								dwStyle | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
								initWindowPosX, initWindowPosY,
								WindowRect.right-WindowRect.left,
								WindowRect.bottom-WindowRect.top,
								NULL, NULL, rwi.hInstance, NULL)))
	{
		KillRenderWindow();	
		MessageBox(NULL,"Window Creation Error.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;
	}

	static	PIXELFORMATDESCRIPTOR pfd=				// pfd Tells Windows How We Want Things To Be
	{
		sizeof(PIXELFORMATDESCRIPTOR),				// Size Of This Pixel Format Descriptor
		1,											// Version Number
		PFD_DRAW_TO_WINDOW |						// Format Must Support Window
		PFD_SUPPORT_OPENGL |						// Format Must Support OpenGL
		PFD_DOUBLEBUFFER,							// Must Support Double Buffering
		PFD_TYPE_RGBA,								// Request An RGBA Format
		bits,										// Select Our Color Depth
		0, 0, 0, 0, 0, 0,							// Color Bits Ignored
		0,											// No Alpha Buffer
		0,											// Shift Bit Ignored
		0,											// No Accumulation Buffer
		0, 0, 0, 0,									// Accumulation Bits Ignored
		16,											// 16Bit Z-Buffer (Depth Buffer)  
		0,											// No Stencil Buffer
		0,											// No Auxiliary Buffer
		PFD_MAIN_PLANE,								// Main Drawing Layer
		0,											// Reserved
		0, 0, 0										// Layer Masks Ignored
	};
	
	if (!(rwi.hDC=GetDC(rwi.hWnd)))	{
		KillRenderWindow();
		MessageBox(NULL,"Can't Create A GL Device Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;
	}

	if (!(PixelFormat=ChoosePixelFormat(rwi.hDC,&pfd))) {
		KillRenderWindow();
		MessageBox(NULL,"Can't Find A Suitable PixelFormat.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;
	}

	if(!SetPixelFormat(rwi.hDC,PixelFormat,&pfd)) {
		KillRenderWindow();
		MessageBox(NULL,"Can't Set The PixelFormat.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;
	}

	if (!(rwi.hRC=wglCreateContext(rwi.hDC))) {
		KillRenderWindow();
		MessageBox(NULL,"Can't Create A GL Rendering Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;
	}

	if(!wglMakeCurrent(rwi.hDC,rwi.hRC)) {
		KillRenderWindow();
		MessageBox(NULL,"Can't Activate The GL Rendering Context.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;
	}

	ShowWindow(rwi.hWnd,SW_SHOW);
	SetForegroundWindow(rwi.hWnd);
	SetFocus(rwi.hWnd);

#ifdef USES_GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        return false;
    }
#endif

	if (!rwi.listener->glInit()) {
		KillRenderWindow();
		MessageBox(NULL,"Initialization Failed.","ERROR",MB_OK|MB_ICONEXCLAMATION);
		return false;
	}

    rwi.listener->resize(width,height,(float)width/(float)height);

	return true;
}

LRESULT CALLBACK RenderWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {

	switch (uMsg)
	{
		case WM_ACTIVATE:
		{
			if (!HIWORD(wParam))
			{
				rwi.active=true;
			}
			else
			{
				rwi.active=false;
			}

			return 0;
		}

		case WM_SYSCOMMAND:
		{
			switch (wParam)
			{
				case SC_SCREENSAVE:
				case SC_MONITORPOWER:
				return 0;
			}
			break;
		}

		case WM_CLOSE:
		{
            if ( rwi.listener->canClose() ) {
                PostQuitMessage(0);
            }
			return 0;
		}

		case WM_KEYDOWN:
		{
			rwi.listener->keyDown(wParam);
			return 0;
		}

		case WM_KEYUP:
		{
			rwi.listener->keyUp(wParam);
			return 0;
		}

        case WM_CHAR:
        {        
            rwi.listener->onChar((int)wParam);
            return 0;
        }

        case WM_MOUSEMOVE:
        {
            int x = LOWORD(lParam);
            int y = HIWORD(lParam);
            rwi.listener->mouseMove(x,y);
            //break; This specially doesn't break in case move changes the keystate also
            return 0;
        }
        case WM_LBUTTONDOWN:
	    case WM_LBUTTONUP:
	    case WM_RBUTTONDOWN:
	    case WM_RBUTTONUP:
	    case WM_MBUTTONDOWN:
	    case WM_MBUTTONUP:
		{
			int	temp;
			temp = 0;

			if (wParam & MK_LBUTTON)
				temp |= 1;

			if (wParam & MK_RBUTTON)
				temp |= 2;

			if (wParam & MK_MBUTTON)
				temp |= 4;


            int changed = rwi.oldButtons ^ temp;

            if ( changed ) {
                if ( temp & 1 ) {
                    rwi.listener->mouseDown(RenderWindow::Listener::MOUSE_LEFT);
                } else {
                    rwi.listener->mouseUp(RenderWindow::Listener::MOUSE_LEFT);
                }
                if ( temp & 2 ) {
                    rwi.listener->mouseDown(RenderWindow::Listener::MOUSE_RIGHT);
                } else {
                    rwi.listener->mouseUp(RenderWindow::Listener::MOUSE_RIGHT);
                }
                if ( temp & 4 ) {
                    rwi.listener->mouseDown(RenderWindow::Listener::MOUSE_MIDDLE);
                } else {
                    rwi.listener->mouseUp(RenderWindow::Listener::MOUSE_MIDDLE);
                }
            }

            rwi.oldButtons = temp;
			return 0;
		}

	    case WM_MOUSEWHEEL:
		{
			int zDelta = (short)HIWORD( wParam ) / 120;
            rwi.listener->mouseWheel(zDelta);
			return 0;
        }

		case WM_SIZE:
		{
            int width = LOWORD(lParam);
            int height = HIWORD(lParam);
            rwi.listener->resize(width,height,(float)width/(float)height);
			return 0;
		}
	}

	// Pass All Unhandled Messages To DefWindowProc
	return DefWindowProc(hWnd,uMsg,wParam,lParam);
}

bool RenderWindow::Open(RenderWindow::Listener *listener,
                        const char *title,
                        int width, int height,
                        bool fullscreen, int bpp)
{
    rwi.active = true;
    rwi.fullscreen = false;
    rwi.hDC = NULL;
    rwi.hInstance = NULL;
    rwi.hRC = NULL;
    rwi.hWnd = NULL;
    rwi.oldButtons = 0;

    rwi.listener = listener;
    if (!CreateRenderWindow(title,width,height,bpp,fullscreen)) {
		return false;
	}

    return true;
}

bool RenderWindow::Update(void) {
	MSG		msg;

	// pump the message loop
	while (PeekMessage (&msg, NULL, 0, 0, PM_NOREMOVE)) {
		if ( !GetMessage (&msg, NULL, 0, 0) ) {
			return false;
		}

		TranslateMessage (&msg);
      	DispatchMessage (&msg);
	}

    return true;
}

bool RenderWindow::SwapBuffers(void) {
    SwapBuffers(rwi.hDC);
    return true;
}

void RenderWindow::Close(void) {
    rwi.listener->glFree();
	KillRenderWindow();
}
