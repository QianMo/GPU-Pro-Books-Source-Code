// beCore.cpp : Defines the exported functions for the DLL application.
//

#include "beCoreInternal/stdafx.h"
#include "beCore/beCore.h"

// Opens a message box containing version information.
void beCore::InfoBox()
{
	::MessageBoxW(NULL, L"beCore build " _T(__TIMESTAMP__) L".", L"Version info", MB_OK);
}

/*
#include "beFileSystem.cpp"
#include "beFileWatch.cpp"
#include "beGenericSerialization.cpp"
#include "beIdentifiers.cpp"
#include "beJob.cpp"
#include "beParameters.cpp"
#include "bePersistentIDs.cpp"
#include "bePropertySerialization.cpp"
#include "beReflectionProperties.cpp"
#include "beTextSerialization.cpp"
#include "beThreadPool.cpp"
#include "dllmain.cpp"
*/