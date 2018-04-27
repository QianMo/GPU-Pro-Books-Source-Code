#ifndef ENGINE_PLATFORMWINDOWCONFIG_H_INCLUDED
#define ENGINE_PLATFORMWINDOWCONFIG_H_INCLUDED

#if MD_WIN_PLATFORM
#include "SysWinDrv/Src/WinWindowConfig.h"
namespace Mod
{
	typedef WinWindowConfig PlatformWindowConfig;
}
#else
#error Unsupported platform
#endif

#endif