#ifndef DEBUGLIB_PRECOMPILED_H_INCLUDED
#define DEBUGLIB_PRECOMPILED_H_INCLUDED

#include "PrecompiledCommon.h"

#ifdef MD_WIN_PLATFORM
#include <Windows.h>
#else
#error Unsupported platform
#endif

#include "Win/winundef.h"

#endif