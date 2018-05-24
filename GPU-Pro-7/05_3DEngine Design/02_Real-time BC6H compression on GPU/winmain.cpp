#include "stdafx.h"
#include "app.h"

bool    gDestroy = false;
bool    gActive = true;
HWND    gWndHandle;

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
        case WM_DESTROY:
            gDestroy = true;
            break;

        case WM_ACTIVATE:
            gActive = wParam != WA_INACTIVE;
            break;

        case WM_KEYDOWN:
            gApp.OnKeyDown( wParam );
            break;

        case WM_LBUTTONDOWN:
            gApp.OnLButtonDown( LOWORD( lParam ), HIWORD( lParam ) );
            break;

        case WM_LBUTTONUP:
            gApp.OnLButtonUp( LOWORD( lParam ), HIWORD( lParam ) );
            break;

        case WM_MOUSEMOVE:
            gApp.OnMouseMove( LOWORD( lParam ), HIWORD( lParam ) );
            break;

        case WM_MOUSEWHEEL:
            gApp.OnMouseWheel( GET_WHEEL_DELTA_WPARAM( wParam ) );
            break;

         case WM_SIZE:
            gApp.OnResize();
            break;
    }

    return DefWindowProc(hWnd, msg, wParam, lParam);
}

void MainLoop(HINSTANCE hInst)
{
    MSG Msg;
    while (!gDestroy)
    {
        if (PeekMessage(&Msg, NULL, 0, 0, PM_NOREMOVE))
        {
            if (!GetMessage(&Msg, NULL, 0, 0))
                return;

            TranslateMessage(&Msg);
            DispatchMessage(&Msg);
        }
        else
        {
            if (gActive)
            {
                gApp.Render();
                Sleep(1);
            }
        }
    }
}

INT WINAPI WinMain(HINSTANCE hInst, HINSTANCE, LPSTR lpCmdLine, INT)
{
    wchar_t const* appName = L"rt_bc6h_encoder_gpu";
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), 0, MsgProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, appName, NULL };

    RegisterClassEx(&wc);

    DWORD const dwStyle = WS_SYSMENU | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SIZEBOX;
    RECT rcWindowSize;
    SetRect( &rcWindowSize, 0, 0, 1280, 720 );
    AdjustWindowRect( &rcWindowSize, dwStyle, FALSE );

    RECT rcDesktop;
    GetClientRect( GetDesktopWindow(), &rcDesktop );

    if ( rcWindowSize.bottom < rcDesktop.bottom )
    {
        rcWindowSize.bottom -= rcWindowSize.top;
        rcWindowSize.top = 0;
    }

    if ( rcWindowSize.right < rcDesktop.right )
    {
        int iTranslate = ( rcDesktop.right - ( rcWindowSize.right - rcWindowSize.left) ) / 2;
        rcWindowSize.left += iTranslate;
        rcWindowSize.right += iTranslate;
    }

    gWndHandle = CreateWindow( appName, appName, dwStyle, rcWindowSize.left, rcWindowSize.top, 
        rcWindowSize.right - rcWindowSize.left, rcWindowSize.bottom - rcWindowSize.top, 
        GetDesktopWindow(), nullptr, wc.hInstance, nullptr );

    gApp.Init( gWndHandle );
    ShowWindow( gWndHandle, SW_SHOWDEFAULT );
    UpdateWindow( gWndHandle );

    MainLoop( hInst );

    UnregisterClass(appName, wc.hInstance);
    gApp.Release();
    return 0;
}