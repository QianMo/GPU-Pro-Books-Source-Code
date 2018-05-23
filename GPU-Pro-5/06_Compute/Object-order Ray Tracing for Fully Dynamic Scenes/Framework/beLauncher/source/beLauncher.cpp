// beLauncher.cpp : Defines the exported functions for the DLL application.
//

#include "beLauncherInternal/stdafx.h"
#include "beLauncher/beLauncher.h"

// Opens a message box containing version information.
void beLauncher::InfoBox()
{
	::MessageBoxW(NULL, L"beLauncher build " _T(__TIMESTAMP__) L".", L"Version info", MB_OK);
}
