// beScene.cpp : Defines the exported functions for the DLL application.
//

#include "beSceneInternal/stdafx.h"
#include "beScene/beScene.h"

// Opens a message box containing version information.
void beScene::InfoBox()
{
	::MessageBoxW(NULL, L"beScene build " _T(__TIMESTAMP__) L".", L"Version info", MB_OK);
}
