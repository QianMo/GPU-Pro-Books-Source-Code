#ifndef ENGINE_PRECOMPILED_H_INCLUDED
#define ENGINE_PRECOMPILED_H_INCLUDED

#include "PrecompiledCommon.h"

#ifdef MD_WIN_PLATFORM
#include "windows.h"
#endif

#ifdef CreateWindow
#undef CreateWindow
#endif

#endif