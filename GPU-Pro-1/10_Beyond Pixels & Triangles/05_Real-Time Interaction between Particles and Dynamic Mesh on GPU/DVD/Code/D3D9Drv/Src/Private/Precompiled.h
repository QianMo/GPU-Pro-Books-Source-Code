#ifndef D3D9DRV_PRECOMPILED_H_INCLUDED
#define D3D9DRV_PRECOMPILED_H_INCLUDED

#include <Windows.h>

#include <d3d9.h>
#include <d3dx9.h>

#include "PrecompiledCommon.h"

#include <exception>
#include <numeric>

#ifdef CreateWindow
#undef CreateWindow
#endif

#include "win/ComPtr.h"
#include "win/winundef.h"

#include "Math/Src/Types.h"

#if defined(MD_WIN32_PLATFORM) || defined(MD_WIN64_PLATFORM)
#define MD_D3D_CALLING_CONV __stdcall
#endif

#ifdef _DEBUG
#define MD_D3DV(expr) MD_FERROR_ON_FALSE((expr)==D3D_OK)
#else
#define MD_D3DV(expr) expr
#endif

#endif