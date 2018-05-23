//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------
#include "CPUTWindowWin.h"

#ifdef _DEBUG

TCHAR lpMsgBuf[100]; // declare global in case error is about lack of resources
_inline void HandleWin32Error()
{
	
	DWORD err = GetLastError();
	FormatMessage(
		FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		err,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		lpMsgBuf,
		100,
		NULL );
	ASSERT(false, lpMsgBuf);
	
}
#else
_inline void HandleWin32Error() {}
#endif

// static initializers
CPUT* CPUTWindowWin::mCPUT=NULL;
bool CPUTWindowWin::mbMaxMinFullScreen=false;

// Constructor
//-----------------------------------------------------------------------------
CPUTWindowWin::CPUTWindowWin():mhInst(0),
    mhWnd(0),
    mAppClosedReturnCode(0)
{
    mAppTitle.clear();
}

// Destructor
//-----------------------------------------------------------------------------
CPUTWindowWin::~CPUTWindowWin()
{
    mAppTitle.clear();
}

// Create window
//-----------------------------------------------------------------------------
CPUTResult CPUTWindowWin::Create(CPUT* cput, const cString WindowTitle, const int windowWidth, const int windowHeight, int windowX, int windowY)
{
    if(mhWnd)
    {
        return CPUT_ERROR_WINDOW_ALREADY_EXISTS;
    }

    ASSERT( (windowX < GetSystemMetrics(SM_CXFULLSCREEN) && (windowX>=-1)), _L("You are attempting to create a window outside the desktop coordinates.  Check your CPUTWindowCreationParams"));
    ASSERT( (windowY < GetSystemMetrics(SM_CYFULLSCREEN) && (windowY>=-1)), _L("You are attempting to create a window outside the desktop coordinates.  Check your CPUTWindowCreationParams"));

    // get the hInstance of this executable
    mhInst = GetModuleHandle(NULL);
    if(NULL==mhInst)
    {
        return CPUT_ERROR_CANNOT_GET_WINDOW_INSTANCE;
    }

    // set up app title (if not specified)
    mAppTitle = WindowTitle;

    if(0==mAppTitle.compare(_L("")))
    {
        mAppTitle = _L("CPUT Sample");
    }

    // Register the Win32 class for this app
    WNDCLASS wc;
    if(TRUE == GetClassInfo(mhInst, mAppTitle.c_str(), &wc))
    {
        // point to the existing one
        mhInst = wc.hInstance;
    }
    else
    {
        // register a new windows class
        ATOM classID;
        classID = MyRegisterClass(mhInst);
        if(0==classID)
        {
			HandleWin32Error();
            return CPUT_ERROR_WINDOW_CANNOT_REGISTER_APP;
        }
    }


	// Perform Win32 instance initialization
    const int nCmdShow = SW_SHOWNORMAL;
	if (false == InitInstance(nCmdShow, windowWidth, windowHeight, windowX, windowY))
	{
		return CPUT_ERROR_CANNOT_GET_WINDOW_INSTANCE;
	}

    // store the CPUT pointer
    mCPUT = (CPUT*) cput;

    return CPUT_SUCCESS;
}



// Destroy window
//-----------------------------------------------------------------------------
int CPUTWindowWin::Destroy()
{
	DestroyWindow(mhWnd);
	mCPUT = NULL;
    return true;
}


// Window return code on close
//-----------------------------------------------------------------------------
int CPUTWindowWin::ReturnCode()
{
    return mAppClosedReturnCode;
}

//  Register window class
//-----------------------------------------------------------------------------
ATOM CPUTWindowWin::MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEX wcex;

	wcex.cbSize = sizeof(WNDCLASSEX);

    // load icon from resource file
    LPCTSTR iconPathName= L"CPUT.ico";
    UINT icon_flags = LR_LOADFROMFILE | LR_DEFAULTSIZE;
    HANDLE hIcon = LoadImage(hInstance, iconPathName, IMAGE_ICON, 0, 0, icon_flags);

    // set up RegisterClass struct
    wcex.style			= CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc	= WndProc;
	wcex.cbClsExtra		= 0;
	wcex.cbWndExtra		= 0;
	wcex.hInstance		= hInstance;
	wcex.hIcon			= (HICON) hIcon; //LoadIcon(hInstance, iconSTR);
	wcex.hCursor		= LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground	= (HBRUSH)(COLOR_WINDOW+1);
	wcex.lpszMenuName	= NULL;
    wcex.lpszClassName	= mAppTitle.c_str();
	wcex.hIconSm		= NULL; // LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL)); // no small icon for now

    // register the window class
	return RegisterClassEx(&wcex);
}



// InitInstance
// Saves the windows instance handle, creates, and displays the main program
// window
//-----------------------------------------------------------------------------
BOOL CPUTWindowWin::InitInstance(int nCmdShow, int windowWidth, int windowHeight, int windowX, int windowY)
{
    // assure we have a valid hInstance
    ASSERT(NULL!=mhInst, _L(""));

   // zero sized windows means - you choose the size. :)
   if( (0==windowWidth) || (0==windowHeight) )
   {
       CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
       pServices->GetDesktopDimensions(&windowWidth, &windowHeight);

       // default window size 1280x720
       // but if screen is smaller than 1280x720, then pick 1/3 the screen size 
       // so that it doesn't appear off the edges
       if(1280>windowWidth)
       {
           windowWidth = (2*windowWidth)/3;
           windowHeight = (2*windowHeight)/3;
       }
       else
       {
        
        windowWidth=1280;
        windowHeight=720;
       }
   }

   // set up size structure
   RECT rc = { 0, 0, windowWidth, windowHeight };
   AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );

   // if x = -1, then let windows decide where to put it
   if(-1==windowX)
   {
       windowX = CW_USEDEFAULT;
   }

   // create the window
   mhWnd = CreateWindow(mAppTitle.c_str(), mAppTitle.c_str(),
       WS_OVERLAPPEDWINDOW,
       windowX, //CW_USEDEFAULT,
       windowY, //CW_USEDEFAULT,
       rc.right - rc.left,
       rc.bottom - rc.top,
       NULL,
       NULL,
       mhInst,
       NULL);

   if (!mhWnd)
   {
      return FALSE;
   }

   ShowWindow(mhWnd, nCmdShow);
   UpdateWindow(mhWnd);

   // initialize the OS services with the hWND so you can make
   // reference to this object
   CPUTOSServices *pServices = CPUTOSServices::GetOSServices();
   pServices->SethWnd(mhWnd);

   return TRUE;
}


//
// WndProc
// Handles the main message loop's events/messages
//-----------------------------------------------------------------------------
LRESULT CALLBACK CPUTWindowWin::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    CPUTEventHandledCode handledCode = CPUT_EVENT_UNHANDLED;
    LRESULT res;

    switch (message)
	{
	case WM_COMMAND:
        int     wmId, wmEvent;
		wmId    = LOWORD(wParam);
		wmEvent = HIWORD(wParam);

        // handle any menu item events here
        // see reference code in file history for examples
		break;

    case WM_KEYDOWN:
        if(mCPUT)
        {
            CPUTKey key = ConvertSpecialKeyCode(wParam, lParam);
            if(KEY_NONE!=key)
            {
                handledCode = mCPUT->CPUTHandleKeyboardEvent( key );
            }
        }
        break;

    case WM_CHAR: // WM_KEYDOWN: gives you EVERY key - including shifts/etc
        if(mCPUT)
        {
            CPUTKey key = ConvertKeyCode(wParam, lParam);
            if(KEY_NONE!=key)
            {
                handledCode = mCPUT->CPUTHandleKeyboardEvent( key );
            }
        }
        break;

    case WM_LBUTTONDBLCLK:
    case WM_MBUTTONDBLCLK:
    case WM_RBUTTONDBLCLK:
        // handle double-click events
        break;

    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_MBUTTONDOWN:
    case WM_MBUTTONUP:
    case WM_RBUTTONDOWN:
    case WM_RBUTTONUP:
    case WM_MOUSEMOVE:
        if(mCPUT)
        {
            CPUTMouseState state = ConvertMouseState(wParam);

            short xPos = LOWORD(lParam);
            short yPos = HIWORD(lParam);

            handledCode = mCPUT->CPUTHandleMouseEvent(xPos, yPos, 0, state);
        }
        break;

    case WM_MOUSEWHEEL:
        if(mCPUT)
        {
            // get mouse position
            short xPos = LOWORD(lParam);
            short yPos = HIWORD(lParam);

            // get wheel delta
            int wheel = GET_WHEEL_DELTA_WPARAM(wParam);  // one 'click'

            handledCode = mCPUT->CPUTHandleMouseEvent(xPos, yPos, wheel, CPUT_MOUSE_WHEEL);
        }
        return 0;
        break;

	case WM_PAINT:
	    PAINTSTRUCT ps;
	    HDC hdc;
		hdc = BeginPaint(hWnd, &ps);
        EndPaint(hWnd, &ps);
        break;

    case WM_SIZING:
    case WM_MOVING:
    case WM_ERASEBKGND:
        // overriding this to do nothing avoids flicker and
        // the expense of re-creating tons of gfx contexts as it resizes
        break;

    //case WM_ACTIVATE:
        // check for maximize/minimize
      //  break;

    case WM_SIZE:
        int width, height;
        height = HIWORD(lParam);
        width  = LOWORD(lParam);
            
        RECT windowRect;
        if(0==GetClientRect(hWnd, &windowRect)) // this gets the client area inside the window frame *excluding* frames/menu bar/etc
            break;
        width = windowRect.right - windowRect.left;        
        height = windowRect.bottom - windowRect.top;

        // if we have shrunk to 0 width/height - do not pass on this kind of resize - leads to 
        // various render target resizing warnings
        if(0==width || 0==height)
            break;

        if(mCPUT)
        {
            // maximize/minimize effect
            if( (SIZE_MAXIMIZED == wParam) ) 
            {
                // resize for new max/min size
                mCPUT->ResizeWindow(width,height);
                mbMaxMinFullScreen = true;
            }
            else if(SIZE_RESTORED == wParam)
            {
                if(true == mbMaxMinFullScreen)
                {
                    // resize for new max/min size
                    mCPUT->ResizeWindow(width,height);
                    mbMaxMinFullScreen = false;
                }
                else
                {
                    // do a stretch-blit while actively sizing by just rendering to un-resized back buffer
                    mCPUT->ResizeWindowSoft(width, height);
                }
            }
        }
        break;
    
    case WM_EXITSIZEMOVE:
        // update the system's size and make callback
        if(mCPUT)
        {
            RECT windowRect;
            if(0==GetClientRect(hWnd, &windowRect)) // this gets the client area inside the window frame *excluding* frames/menu bar/etc
                break;

            width = windowRect.right - windowRect.left;
            height = windowRect.bottom - windowRect.top;

            // if we have shrunk to 0 width/height - do not pass on this kind of resize - leads to 
            // various render target resizing warnings
            if(0==width || 0==height)
                break;

            mCPUT->ResizeWindow(width,height);
        }
        break;


    case WM_DESTROY:
        // time to shut down the system
        PostQuitMessage(0);
        break;

	default:
        // we don't handle it - pass it on thru to parent
        res = DefWindowProc(hWnd, message, wParam, lParam);
        return res;
	}

    // translate handled code
    if(CPUT_EVENT_HANDLED == handledCode)
    {
        return 1;
    }

	return 0;
}


// Convert OS specific key events to CPUT events
//-----------------------------------------------------------------------------da
CPUTKey CPUTWindowWin::ConvertKeyCode(WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(wParam);
    UNREFERENCED_PARAMETER(lParam);
    switch(wParam)
    {
    case 'a':
    case 'A':
        return KEY_A;
    case 'b':
    case 'B':
        return KEY_B;
    case 'c':
    case 'C':
        return KEY_C;
    case 'd':
    case 'D':
        return KEY_D;
    case 'e':
    case 'E':
        return KEY_E;
    case 'f':
    case 'F':
        return KEY_F;
    case 'g':
    case 'G':
        return KEY_G;
    case 'h':
    case 'H':
        return KEY_H;
    case 'i':
    case 'I':
        return KEY_I;
    case 'j':
    case 'J':
        return KEY_J;
    case 'k':
    case 'K':
        return KEY_K;
    case 'l':
    case 'L':
        return KEY_L;
    case 'm':
    case 'M':
        return KEY_M;
    case 'n':
    case 'N':
        return KEY_N;
    case 'o':
    case 'O':
        return KEY_O;
    case 'p':
    case 'P':
        return KEY_P;
    case 'Q':
    case 'q':
        return KEY_Q;
    case 'r':
    case 'R':
        return KEY_R;
    case 's':
    case 'S':
        return KEY_S;
    case 't':
    case 'T':
        return KEY_T;
    case 'u':
    case 'U':
        return KEY_U;
    case 'v':
    case 'V':
        return KEY_V;
    case 'w':
    case 'W':
        return KEY_W;
    case 'x':
    case 'X':
        return KEY_X;
    case 'y':
    case 'Y':
        return KEY_Y;
    case 'z':
    case 'Z':
        return KEY_Z;


        // number keys
    case '1':
        return KEY_1;
    case '2':
        return KEY_2;
    case '3':
        return KEY_3;
    case '4':
        return KEY_4;
    case '5':
        return KEY_5;
    case '6':
        return KEY_6;
    case '7':
        return KEY_7;
    case '8':
        return KEY_8;
    case '9':
        return KEY_9;
    case '0':
        return KEY_0;


    // symbols
    case ' ':
        return KEY_SPACE;
    case '`':
        return KEY_BACKQUOTE;
    case '~':
        return KEY_TILDE;
    case '!':
        return KEY_EXCLAMATION;
    case '@':
        return KEY_AT;
    case '#':
        return KEY_HASH;
    case '$':
        return KEY_$;
    case '%':
        return KEY_PERCENT;
    case '^':
        return KEY_CARROT;
    case '&':
        return KEY_ANDSIGN;
    case '*':
        return KEY_STAR;
    case '(':
        return KEY_OPENPAREN;
    case ')':
        return KEY_CLOSEPARN;
    case '_':
        return KEY__;
    case '-':
        return KEY_MINUS;
    case '+':
        return KEY_PLUS;
    case '[':
        return KEY_OPENBRACKET;
    case ']':
        return KEY_CLOSEBRACKET;
    case '{':
        return KEY_OPENBRACE;
    case '}':
        return KEY_CLOSEBRACE;
    case '\\':
        return KEY_BACKSLASH;
    case '|':
        return KEY_PIPE;
    case ';':
        return KEY_SEMICOLON;
    case ':':
        return KEY_COLON;
    case '\'':
        return KEY_SINGLEQUOTE;
    case '\"':
        return KEY_QUOTE;
    case ',':
        return KEY_COMMA;
    case '.':
        return KEY_PERIOD;
    case '/':
        return KEY_SLASH;
    case '<':
        return KEY_LESS;
    case '>':
        return KEY_GREATER;
    case '?':
        return KEY_QUESTION;
    }

    return KEY_NONE;
}

// Convert extended key events to CPUT events
//-----------------------------------------------------------------------------
CPUTKey CPUTWindowWin::ConvertSpecialKeyCode(WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch(wParam)
    {
    // function keys
    case VK_F1:
        return KEY_F1;
    case VK_F2:
        return KEY_F2;
    case VK_F3:
        return KEY_F3;
    case VK_F4:
        return KEY_F4;
    case VK_F5:
        return KEY_F5;
    case VK_F6:
        return KEY_F6;
    case VK_F7:
        return KEY_F7;
    case VK_F8:
        return KEY_F8;
    case VK_F9:
        return KEY_F9;
    case VK_F10:
        return KEY_F10;
    case VK_F11:
        return KEY_F11;
    case VK_F12:
        return KEY_F12;


    // special keys
    case VK_HOME:
        return KEY_HOME;
    case VK_END:
        return KEY_END;
    case VK_PRIOR:
        return KEY_PAGEUP;
    case VK_NEXT:
        return KEY_PAGEDOWN;
    case VK_INSERT:
        return KEY_INSERT;
    case VK_DELETE:
        return KEY_DELETE;

    case VK_BACK:
        return KEY_BACKSPACE;
    case VK_TAB:
        return KEY_TAB;
    case VK_RETURN:
        return KEY_ENTER;

    case VK_PAUSE:
        return KEY_PAUSE;
    case VK_CAPITAL:
        return KEY_CAPSLOCK;
    case VK_ESCAPE:
        return KEY_ESCAPE;

    case VK_UP:
        return KEY_UP;
    case VK_DOWN:
        return KEY_DOWN;
    case VK_LEFT:
        return KEY_LEFT;
    case VK_RIGHT:
        return KEY_RIGHT;
    }

    return KEY_NONE;
}

// Convert mouse state to CPUT state
//-----------------------------------------------------------------------------
CPUTMouseState CPUTWindowWin::ConvertMouseState(WPARAM wParam)
{
    CPUTMouseState eState=CPUT_MOUSE_NONE;

    if( wParam & MK_CONTROL)
        eState = (CPUTMouseState) (eState | static_cast<int>(CPUT_MOUSE_CTRL_DOWN));

    if( wParam & MK_SHIFT)
        eState = (CPUTMouseState) (eState | static_cast<int>(CPUT_MOUSE_SHIFT_DOWN));

    if( wParam & MK_LBUTTON)
        eState = (CPUTMouseState) (eState | static_cast<int>(CPUT_MOUSE_LEFT_DOWN));

    if( wParam & MK_MBUTTON)
        eState = (CPUTMouseState) (eState | static_cast<int>(CPUT_MOUSE_MIDDLE_DOWN));

    if( wParam & MK_RBUTTON)
        eState = (CPUTMouseState) (eState | static_cast<int>(CPUT_MOUSE_RIGHT_DOWN));


    return eState;
}

// Main message pump
//-----------------------------------------------------------------------------
int CPUTWindowWin::StartMessageLoop()
{
	//
	// Message pump
	//
    MSG msg = { 0 };
	bool fRunning = true;
    while(fRunning)
    {
        // PeekMessage() is a passthru on no events
        // so it allows us to render while no events are present
        if( PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) )
        {
			if (msg.message == WM_QUIT)
			{
				PostQuitMessage(0);
				fRunning = false;
			}
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        } else
		{
            // trigger render and other calls
            mCPUT->InnerExecutionLoop();
        }
    }
	
	//
	// Drain out the rest of the message queue.
	//
	while( PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) )
	{
		TranslateMessage( &msg );
		DispatchMessage( &msg );
	}

	if (UnregisterClass(mAppTitle.c_str(), mhInst) == 0) {
		HandleWin32Error();
	}

	//
	// Set the window handle to NULL to indicate window shutdown is complete
	//
	mhWnd = NULL;

    // return code
    mAppClosedReturnCode =  (int) msg.wParam;
	return mAppClosedReturnCode;
}
