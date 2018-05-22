//-------------------------------------------------------------------------------------------------
// File: Main.cpp
// Author: Ben Mistal
// Copyright 2010-2012 Mistal Research, Inc.
//-------------------------------------------------------------------------------------------------
#include "Main.h"

int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
	CMain& main = CMain::GetMain( hInstance, hPrevInstance, lpCmdLine, nCmdShow );

	return main.Main();

} // end int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )

CMain::CMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
	:	m_hInstance( hInstance ),
		m_hPrevInstance( hPrevInstance ),
		m_lpCmdLine( lpCmdLine ),
		m_nCmdShow( nCmdShow )
{
	m_hWnd	= NULL;

} // end CMain::CMain()

CMain::~CMain()
{
} // end CMain::~CMain()

CMain& CMain::GetMain(	HINSTANCE hInstance /* = NULL */,
						HINSTANCE hPrevInstance /* = NULL */,
						LPWSTR lpCmdLine /* = NULL */,
						int nCmdShow /* = NULL */ )
{
	static CMain main( hInstance, hPrevInstance, lpCmdLine, nCmdShow );

	return main;

} // end CMain& CMain::GetMain( ... );

int CMain::Main()
{
    if( FAILED( InitWindow() ) )
	{
        return 0;

	} // end if( FAILED( InitWindow() ) )

    if( FAILED( m_engine.InitDevice( m_hWnd ) ) )
    {
		::MessageBoxA( m_hWnd, "Sorry, unable to create a compatible DirectX 11 rendering device.", "TerrainTessellation", MB_OK );
        return 0;

    } // end if( FAILED( m_engine.InitDevice() ) )

    // Main message loop
    MSG msg = { 0 };
    while( msg.message != WM_QUIT )
    {
        if( ::PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
        {
			::TranslateMessage( &msg );
            ::DispatchMessage( &msg );

        } // end if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
        else
        {
			m_engine.Render();

        } // end else

    } // end while( msg.message != WM_QUIT )

    return ( int )msg.wParam;

} // end int CMain::Main()

HRESULT CMain::InitWindow()
{
    // Register class
    WNDCLASSEX wcex;
	memset( &wcex, 0, sizeof( WNDCLASSEX ) );
    wcex.cbSize			= sizeof( WNDCLASSEX );
    wcex.style			= CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc	= CMain::WndProc;
    wcex.cbClsExtra		= 0;
    wcex.cbWndExtra		= 0;
    wcex.hInstance		= m_hInstance;
    wcex.hIcon			= 0;
    wcex.hCursor		= LoadCursor( NULL, IDC_ARROW );
    wcex.hbrBackground	= ( HBRUSH )( COLOR_WINDOW + 1 );
    wcex.lpszMenuName	= NULL;
    wcex.lpszClassName	= L"TerrainTessellationWindowClass";
    wcex.hIconSm		= 0;

    if( !::RegisterClassEx( &wcex ) )
	{
        return E_FAIL;

	} // end if( !::RegisterClassEx( &wcex ) )

    // Create window
    RECT rc = { 0, 0, nHSize, nVSize };
    ::AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
    m_hWnd = ::CreateWindow(	L"TerrainTessellationWindowClass",
								L"TerrainTessellation",
								WS_OVERLAPPEDWINDOW,
								CW_USEDEFAULT, 
								CW_USEDEFAULT, 
								rc.right - rc.left, 
								rc.bottom - rc.top,
								NULL, 
								NULL, 
								m_hInstance,
								NULL );
    if( !m_hWnd )
	{
        return E_FAIL;

	} // end  if( !m_hWnd )

    ::ShowWindow( m_hWnd, m_nCmdShow );

    return S_OK;

} // end HRESULT CMain::InitWindow()

LRESULT CALLBACK CMain::WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )
{
    PAINTSTRUCT ps;
    HDC hdc;

    switch( message )
    {
        case WM_PAINT:
            hdc = ::BeginPaint( hWnd, &ps );
            ::EndPaint( hWnd, &ps );
            break;

        case WM_DESTROY:
            ::PostQuitMessage( 0 );
            break;

        default:
			CMain& main = CMain::GetMain();
			main.m_engine.MessageHandler( message, wParam, lParam );			
            return ::DefWindowProc( hWnd, message, wParam, lParam );

    } // end switch( message )

    return 0;

} // end LRESULT CALLBACK CMain::WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )