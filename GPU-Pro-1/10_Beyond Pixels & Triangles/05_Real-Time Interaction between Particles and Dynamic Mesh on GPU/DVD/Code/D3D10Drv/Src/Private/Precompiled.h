#ifndef D3D10DRV_PRECOMPILED_H_INCLUDED
#define D3D10DRV_PRECOMPILED_H_INCLUDED

#define MD_D3D10_STATIC_LINK

#if 0
#define MD_D3D10_1
#endif

#define MD_USE_PERF_FUNCTIONS 1

#include <Windows.h>

#include <DXGI.h>

#ifdef MD_D3D10_1
#include <D3D10_1.h>
#else
#include <D3D10.h>
#endif

#if MD_USE_PERF_FUNCTIONS
#include <d3d9.h>
#pragma comment(lib,"d3d9")
#endif

#include <D3DX10.h>

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
#ifdef MD_D3D10_1
#pragma comment(lib,"d3d10_1")
#else
#pragma comment(lib,"d3d10")
#endif
#else
#define MD_D3D_CALLING_CONV
#endif

#endif