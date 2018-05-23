// beMath.cpp : Defines the exported functions for the DLL application.
//

#include "beMathInternal/stdafx.h"
#include "beMath/beMath.h"

// Opens a message box containing version information.
void beMath::InfoBox()
{
	::MessageBoxW(NULL, L"beMath build " _T(__TIMESTAMP__) L".", L"Version info", MB_OK);
}
