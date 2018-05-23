// beGraphics.cpp : Defines the exported functions for the DLL application.
//

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/beGraphics.h"

// Opens a message box containing version information.
void beGraphics::InfoBox()
{
	::MessageBoxW(NULL, L"beGraphics build " _T(__TIMESTAMP__) L".", L"Version info", MB_OK);
}
