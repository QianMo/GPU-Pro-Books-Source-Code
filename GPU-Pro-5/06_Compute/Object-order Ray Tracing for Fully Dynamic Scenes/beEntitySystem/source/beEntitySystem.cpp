// beEntitySystem.cpp : Defines the exported functions for the DLL application.
//

#include "beEntitySystemInternal/stdafx.h"
#include "beEntitySystem/beEntitySystem.h"

// Opens a message box containing version information.
void beEntitySystem::InfoBox()
{
	::MessageBoxW(NULL, L"beEntitySystem build " _T(__TIMESTAMP__) L".", L"Version info", MB_OK);
}
