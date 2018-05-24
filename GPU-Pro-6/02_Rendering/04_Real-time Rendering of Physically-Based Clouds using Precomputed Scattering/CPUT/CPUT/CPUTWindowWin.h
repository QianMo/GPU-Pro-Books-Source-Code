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
#ifndef __WINDOWWIN_H__
#define __WINDOWWIN_H__

#include "CPUT.h"
#include "CPUTOSServicesWin.h"
#include "CPUTResource.h" // win resource.h customized for CPUT

#include <windows.h>
#include <winuser.h> // for character codes
#include <cstringt.h> // for CString class
#include <atlstr.h> // CString class

// Forward declarations
class CPUT;

// OS-specific window class
//-----------------------------------------------------------------------------
class CPUTWindowWin
{
public:
    // construction
    CPUTWindowWin();
    ~CPUTWindowWin();

    // Creates a graphics-context friendly window
    CPUTResult Create(CPUT* cput, const cString WindowTitle, const int windowWidth, const int windowHeight, int windowX, int windowY);

    // Main windows message loop that handles and dispatches messages
    int StartMessageLoop();
    int Destroy();
    int ReturnCode();

    // return the HWND/Window handle for the created window
    HWND GetHWnd() { return mhWnd;};

protected:
    HINSTANCE           mhInst;					// current instance
    HWND                mhWnd;                     // window handle
    int                 mAppClosedReturnCode;      // windows OS return code
    cString             mAppTitle;                 // title put at top of window
    static CPUT*        mCPUT;                     // CPUT reference for callbacks

    static bool         mbMaxMinFullScreen;

    // Window creation helper functions
    ATOM MyRegisterClass(HINSTANCE hInstance);
    BOOL InitInstance(int nCmdShow, int windowWidth, int windowHeight, int windowX, int windowY);
    static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

    // CPUT conversion helper functions
    static CPUTMouseState ConvertMouseState(WPARAM wParam);
    static CPUTKey ConvertKeyCode(WPARAM wParam, LPARAM lParam);
    static CPUTKey ConvertSpecialKeyCode(WPARAM wParam, LPARAM lParam);
};


#endif //#ifndef __WINDOWWIN_H__
