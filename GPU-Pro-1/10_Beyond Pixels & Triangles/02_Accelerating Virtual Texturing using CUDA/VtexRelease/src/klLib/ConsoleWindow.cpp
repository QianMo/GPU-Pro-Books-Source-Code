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

#define _WIN32_WINDOWS 0x501
#include <windows.h>
#include "ConsoleWindow.h"

#ifndef NO_STL
#include <vector>
#include <string>
#endif NO_STL

#define IDC_INPUT 101
#define IDC_LOG   102

#define CON_WIDTH   640
#define CON_HEIGHT  480
#define CON_SPACING 2
#define CON_FONTSIZE 10

struct ConsoleWindowInfo {
    HWND		hWnd;        // Main window
    HWND        hwndLog; // Edit control with console text
    HWND        hwndInput;   // Edit control where the user can enter commands
    WNDPROC     defaultInputProc; //Default windows procedure for edit commands
    WNDPROC     defaultLogProc; //Default windows procedure for edit commands
    HFONT       hfFont;
    HINSTANCE	hInstance;
	HBRUSH      hbrNormalBackground;
	HBRUSH      hbrErrorBackground;
	HBRUSH      currentBackgroundBrush; //points to hbrNormalBackground or hbrErrorBackground
    bool	    active;
    ConsoleWindow::Listener *listener;
    bool        visible;
	bool		quitOnClose;
	bool		inError;
#ifndef NO_STL
    std::vector<std::string> commandHistory;
    int                      historyIndex;
#endif
};

static ConsoleWindowInfo cwi;
LRESULT	CALLBACK ConsoleWindowProc(HWND, UINT, WPARAM, LPARAM);
LRESULT	CALLBACK InputWindowProc(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK LogWindowProc(HWND, UINT, WPARAM, LPARAM);

bool KillConsoleWindow(void)
{
    if ( !cwi.hWnd ) return true;

    ShowWindow(cwi.hWnd, SW_HIDE);
	CloseWindow(cwi.hWnd);
	if (!DestroyWindow(cwi.hWnd)) {
		cwi.hWnd=NULL;	
        return false;
	}

    DeleteObject (cwi.hfFont);
    DeleteObject (cwi.hbrNormalBackground);
    DeleteObject (cwi.hbrErrorBackground);

	if (!UnregisterClass("ConsoleWindow",cwi.hInstance)) {
		cwi.hInstance=NULL;
        return false;
	}

    cwi.hWnd = NULL;
    return true;
}
 
bool CreateConsoleWindow(const char *titleText) {
    cwi.hInstance = GetModuleHandle(NULL);

	WNDCLASS wc;
	memset( &wc, 0, sizeof( wc ) );
	wc.style         = 0;
	wc.lpfnWndProc   = (WNDPROC)ConsoleWindowProc;
	wc.cbClsExtra    = 0;
	wc.cbWndExtra    = 0;
	wc.hInstance     = cwi.hInstance;
	wc.hIcon		 = LoadIcon(NULL, IDI_WINLOGO);
	wc.hCursor		 = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = NULL;//(HBRUSH)(COLOR_BTNFACE+1);
	wc.lpszMenuName  = NULL;
	wc.lpszClassName = "ConsoleWindow";

    if ( !RegisterClass (&wc) ) {
		return false;
    }

	int dwStyle = WS_POPUPWINDOW | WS_CAPTION | WS_SIZEBOX | WS_MAXIMIZEBOX | WS_MINIMIZEBOX;

	RECT windowRect;
	windowRect.left=0;			
	windowRect.right=CON_WIDTH;
	windowRect.top=0;
	windowRect.bottom=CON_HEIGHT;

	AdjustWindowRect(&windowRect, dwStyle, false);

    int initWindowPosX = 100;
    int initWindowPosY = 100;

	cwi.hWnd = CreateWindowEx(0, "ConsoleWindow", titleText, dwStyle,
							  initWindowPosX, initWindowPosX,
                              windowRect.right - windowRect.left,
							  windowRect.bottom- windowRect.top,
                              NULL, NULL, wc.hInstance, NULL);

	if ( cwi.hWnd == NULL ) {
		return false;
	}

	// The edit box that contains the console log
	cwi.hwndLog = CreateWindow("edit", NULL, WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_BORDER | ES_LEFT | 
                               ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
							   CON_SPACING, CON_SPACING, CON_WIDTH - 2 * CON_SPACING, CON_HEIGHT - 3 * CON_SPACING - 20,
                               cwi.hWnd, (HMENU)IDC_LOG, wc.hInstance, NULL );

	if ( cwi.hwndLog == NULL ) {
		return false;
	}

    // Setup a fixed width font
	HDC hDC = GetDC(cwi.hWnd);

    //Calculate the font size from pixel size
	int fontSize = -MulDiv( CON_FONTSIZE, GetDeviceCaps( hDC, LOGPIXELSY), 72);

	cwi.hfFont = CreateFont(fontSize, 0, 0, 0, FW_LIGHT, 0, 0, 0,
							DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
							DEFAULT_QUALITY, FF_MODERN | FIXED_PITCH, "Consolas" );

    // On winxp without office 2k9 consolas is not available
    if ( cwi.hfFont == NULL ) {
	    cwi.hfFont = CreateFont(fontSize, 0, 0, 0, FW_LIGHT, 0, 0, 0,
							    DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
							    DEFAULT_QUALITY, FF_MODERN | FIXED_PITCH, "Courier New" );
    }

	if ( cwi.hfFont == NULL ) {
		return false;
	}

	ReleaseDC( cwi.hWnd, hDC );

    // Set the font on the console edit box
	SendMessage( cwi.hwndLog, WM_SETFONT, ( WPARAM )cwi.hfFont, 0 );

    // Override the default windows editbox function
	cwi.defaultLogProc = (WNDPROC)SetWindowLong(cwi.hwndLog, GWL_WNDPROC, (long)InputWindowProc);

	// The create editbox where the user can enter text
	cwi.hwndInput = CreateWindow("edit", NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | ES_LEFT | ES_AUTOHSCROLL,
								 CON_SPACING, CON_HEIGHT - CON_SPACING - 20, CON_WIDTH - 2 * CON_SPACING, 20,
                                 cwi.hWnd, (HMENU)IDC_INPUT, wc.hInstance, NULL);

	if ( cwi.hwndInput == NULL ) {
		return false;
	}

    // Override the default windows editbox function
	cwi.defaultInputProc = (WNDPROC)SetWindowLong(cwi.hwndInput, GWL_WNDPROC, (long)InputWindowProc);

    // Set the font on the input box too
	SendMessage(cwi.hwndInput, WM_SETFONT, ( WPARAM ) cwi.hfFont, 0 );

    return true;
}

void CloseOrHide(void) {
	if ( cwi.inError || cwi.listener->canQuit() ) {
		PostQuitMessage(0);
	} else {
		ConsoleWindow::Hide();
	}
}

LRESULT CALLBACK ConsoleWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {

	switch (uMsg)
	{
		case WM_CREATE:
		{
			cwi.hbrNormalBackground = CreateSolidBrush(GetSysColor(COLOR_BTNFACE));
			cwi.currentBackgroundBrush = cwi.hbrNormalBackground;
			cwi.hbrErrorBackground = CreateSolidBrush( RGB( 255, 64, 64));

			SetTimer( hWnd, 1, 500, NULL );
			return 0;
		}
	    case WM_ACTIVATE:
        {
		    if ( LOWORD( wParam ) != WA_INACTIVE ) {
			    SetFocus(cwi.hwndInput);
		    }
		    return 0;
        }

	    case WM_CLOSE:
        {
			CloseOrHide();
		    return 0;
        }

		case WM_SIZE:
		{
            int width = LOWORD(lParam);
            int height = HIWORD(lParam);

            MoveWindow(cwi.hwndLog, CON_SPACING, CON_SPACING, width - 2 * CON_SPACING, height - 3 * CON_SPACING - 20, true);
            MoveWindow(cwi.hwndInput, CON_SPACING, height - CON_SPACING - 20, width - 2 * CON_SPACING, 20, true);

			return 0;
		}

		case WM_ERASEBKGND:
		{
			RECT r;
			GetWindowRect( hWnd, &r );

			r.bottom = r.bottom - r.top + 1;
			r.right = r.right - r.left + 1;
			r.top = 0;
			r.left = 0;

			FillRect((HDC)wParam, &r, cwi.currentBackgroundBrush);
			
			return 1;
		}

		case WM_TIMER:
		{
			if ( wParam == 1 ) {
				if ( cwi.inError ) {
					if ( cwi.currentBackgroundBrush == cwi.hbrNormalBackground ) {
						cwi.currentBackgroundBrush = cwi.hbrErrorBackground;
					} else {
						cwi.currentBackgroundBrush = cwi.hbrNormalBackground;
					}

					InvalidateRect(hWnd, NULL, true);
				}
			}
			return 0;
		}
    }

    return DefWindowProc( hWnd, uMsg, wParam, lParam );
}

LRESULT CALLBACK LogWindowProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam ) {
	switch ( uMsg )
	{
	    case WM_CHAR:
        {
			if ( wParam == ConsoleWindow::CLOSE_CHAR ) {
				CloseOrHide();
                return 0;
            }
            break;
		}
    }

    return CallWindowProc( cwi.defaultLogProc, hWnd, uMsg, wParam, lParam );
}

LRESULT CALLBACK InputWindowProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam ) {

	switch ( uMsg )
	{
        // We are about to loose keyboard focus...
	    case WM_KILLFOCUS:
        {
            // If it is our parent window don't let it happen give it to us
            // do not capture focus from the log window in case people want
            // to copy paste from it
		    if ( (HWND) wParam == cwi.hWnd ) {
			    SetFocus( hWnd );
			    return 0;
		    }
		    break;
        }

	    case WM_CHAR:
        {
            // If we have a return...
		    if ( wParam == 13 ) {                
                char buffer[2048];
			    GetWindowText( cwi.hwndInput, buffer, sizeof(buffer));
			    SetWindowText( cwi.hwndInput, "" );
                
				// Make sure it is null terminated...
                buffer[sizeof(buffer)-1] = 0;

                // Send to listener, in error mode we don't allow any commands to be send
				if ( !cwi.inError ) {
					cwi.listener->command(buffer);
#ifndef NO_STL
					cwi.commandHistory.push_back(buffer);
					cwi.historyIndex = -1;
#endif
				}
			    return 0;
            } else if ( wParam == ConsoleWindow::CLOSE_CHAR ) {
				CloseOrHide();
                return 0;
            }
            int a = 0;
            break;
		}

#ifndef NO_STL
        // Capture VK_UP / VK_DOWN to scroll trough the command history
        case WM_KEYUP:
        {
            if ( wParam == VK_UP ) {

                if ( cwi.historyIndex == -1 ) {
                    cwi.historyIndex = cwi.commandHistory.size()-1;
                } else if ( cwi.historyIndex > 0 ) {
                    cwi.historyIndex--;
                }

                if ( cwi.historyIndex >= 0 ) {
                    SetWindowText(cwi.hwndInput, cwi.commandHistory[cwi.historyIndex].c_str() );
                    // Put cursor at end of input
                    int len = cwi.commandHistory[cwi.historyIndex].size();
                    SendMessage(cwi.hwndInput, EM_SETSEL, len, len);
                }

                return 0;
            } else if ( wParam == VK_DOWN ) {

                if ( cwi.historyIndex >= 0 ) {
                    cwi.historyIndex++;
                
                    if ( cwi.historyIndex >= cwi.commandHistory.size() ) {
                        cwi.historyIndex = -1;
                        SetWindowText( cwi.hwndInput, "");
                    } else {
                        SetWindowText(cwi.hwndInput, cwi.commandHistory[cwi.historyIndex].c_str() );
                        // Put cursor at end of input
                        int len = cwi.commandHistory[cwi.historyIndex].size();
                        SendMessage(cwi.hwndInput, EM_SETSEL, len, len);
                    }

                }

                return 0;
            } 
            break;
        }
        case WM_KEYDOWN:
        {
            if ( wParam == VK_UP ) {
                return 0;
            } else if ( wParam == VK_DOWN ) {
                return 0;
            } 
            break;
        }
#endif

	}

    // Let the rest be handled by the default edit box...
    return CallWindowProc( cwi.defaultInputProc, hWnd, uMsg, wParam, lParam );
}

bool ConsoleWindow::Create(ConsoleWindow::Listener *listener, const char *titleText) {
    cwi.hWnd = NULL;
    cwi.hwndLog = NULL;
    cwi.hwndInput = NULL;
    cwi.defaultInputProc = NULL;
    cwi.hfFont = NULL;
    cwi.hInstance = NULL;
    cwi.listener = listener;
    cwi.visible = false;
    cwi.historyIndex = -1;
	cwi.quitOnClose = false;
	cwi.currentBackgroundBrush = NULL;
	cwi.hbrNormalBackground = NULL;
	cwi.hbrErrorBackground = NULL;
	cwi.inError = false;
    return CreateConsoleWindow(titleText);
}

bool ConsoleWindow::Destroy(void) {
   return KillConsoleWindow(); 
}

bool ConsoleWindow::Update(void) {
	MSG		msg;

	//Pump the message loop
	while (PeekMessage (&msg, NULL, 0, 0, PM_NOREMOVE) > 0 ) {
		if ( GetMessage (&msg, NULL, 0, 0) == 0 ) {
			return false;
		}

		if ( cwi.inError ) {
			// We are in a fatal error, it is probably not a good idea to go and let other 
			// message handlers than the ones of the console window run...
            if ( msg.message == WM_TIMER || msg.hwnd == cwi.hWnd || IsChild(cwi.hWnd,msg.hwnd) ) {
				TranslateMessage (&msg);
      			DispatchMessage (&msg);
            }
		} else {
			TranslateMessage (&msg);
      		DispatchMessage (&msg);
		}
	}

    return true;
}

bool ConsoleWindow::Show(void) {
    if (!cwi.hWnd) return false;

    if ( !cwi.visible ) {
        cwi.historyIndex = -1;

        // Show the window
	    ShowWindow( cwi.hWnd, SW_SHOWDEFAULT);
    
        // Activate console input field
	    SetFocus( cwi.hwndInput );

        cwi.visible = true;
    }

    // Scroll the text to the end
    SendMessage( cwi.hwndLog, EM_LINESCROLL, 0, 0xffff);
	SetForegroundWindow( cwi.hWnd );
	UpdateWindow( cwi.hWnd );

    return true;
}

bool ConsoleWindow::Hide(void) {
    if (!cwi.hWnd) return false;

    if ( cwi.visible ) {
        cwi.visible = false;
        ShowWindow( cwi.hWnd, SW_HIDE );
    }

    return true;
}

bool ConsoleWindow::Toggle(void) {
    if (!cwi.hWnd) return false;

    if ( cwi.visible ) {
        return ConsoleWindow::Hide();
    } else {
        return ConsoleWindow::Show();
    }
}

#define APPEND_BUFFER_SIZE 2048

bool ConsoleWindow::AppendText(const char *text) {
    if (!cwi.hWnd) return false;
    char buffer[APPEND_BUFFER_SIZE];
    int numOut = 0;


	while ( *text && numOut < (APPEND_BUFFER_SIZE-2) ) {
		if ( text[0] == '\n' && text[1] == '\r' ) {
			buffer[numOut++] = '\r';
			buffer[numOut++] = '\n';
            text++;
		} else if ( *text == '\r' ) {
			buffer[numOut++] = '\r';
			buffer[numOut++] = '\n';
		} else if ( *text == '\n' ) {
			buffer[numOut++] = '\r';
			buffer[numOut++] = '\n';
		} else {
			buffer[numOut++] = *text;
		}
        text++;
	}

    buffer[numOut++] = 0;

    // This is more like magic incantations... seems to work all right

    // Save the old selection
    DWORD start;
    DWORD end;
    SendMessage(cwi.hwndLog, EM_GETSEL, (WPARAM)&start, (LPARAM)&end);  
    
    // If we have no selection scroll to the last line
    if ( start == 0 && end == 0 ) {
        SendMessage(cwi.hwndLog, EM_LINESCROLL , 0, 0xffff);
    }
   
    // Set input position at end of edit box
    int len = SendMessage(cwi.hwndLog, WM_GETTEXTLENGTH, 0, 0);
    SendMessage(cwi.hwndLog, EM_SETSEL, len, len);

    // Append
	SendMessage(cwi.hwndLog, EM_REPLACESEL, 0, (LPARAM)buffer);

    // Restore old selection
    SendMessage(cwi.hwndLog, EM_SETSEL, (WPARAM)start, (LPARAM)end);  

    return true;
}

void ConsoleWindow::FatalError(void) {
	cwi.quitOnClose = true;
	cwi.inError = true;
	cwi.currentBackgroundBrush = cwi.hbrErrorBackground;
	ConsoleWindow::Show();

	while ( ConsoleWindow::Update() ) {
		Sleep(10);
	}

	KillConsoleWindow();
	exit(0);
}