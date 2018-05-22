//-------------------------------------------------------------------------------------------------
// File: Main.h
// Author: Ben Mistal
// Copyright 2010-2012 Mistal Research, Inc.
//-------------------------------------------------------------------------------------------------
#include <windows.h>
#include "Engine.h"

class CMain
{
public: // Construction / Destruction

	CMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow );
	virtual ~CMain();

public: // Public Member Functions

	static CMain& GetMain(	HINSTANCE hInstance = NULL,
							HINSTANCE hPrevInstance = NULL,
							LPWSTR lpCmdLine = NULL,
							int nCmdShow = NULL );
	int Main();

protected: // Protected Member Functions

	HRESULT InitWindow();
	static LRESULT CALLBACK WndProc( HWND, UINT, WPARAM, LPARAM );

protected: // Protected Member Variables

	HINSTANCE	m_hInstance;
	HINSTANCE	m_hPrevInstance;
	LPWSTR		m_lpCmdLine;
	int			m_nCmdShow;
	HWND		m_hWnd;

	CEngine		m_engine;

}; // end class CMain