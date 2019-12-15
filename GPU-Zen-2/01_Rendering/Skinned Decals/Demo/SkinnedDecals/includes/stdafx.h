#ifndef AFX_STDAFX_H__CBEDBAE5_D5EC_45D2_9505_F12B4E77C708
#define AFX_STDAFX_H__CBEDBAE5_D5EC_45D2_9505_F12B4E77C708

#include <Windows.h>
#include <direct.h>
#include <mmsystem.h>
#include <stdio.h>
#include <assert.h>
#include <sys/stat.h>
#include <memory.h>
#include <float.h>
#include <math.h>

#include <dxgi1_4.h>
#include <d3d12.h>
#include <D3Dcompiler.h>
#include <d3dx12.h>
#include <pix.h>

#include <wrl.h>
#include <fstream>
#include <string>
#include <sstream>

#include <imgui.h>

#include <maths.h>

#pragma warning(disable : 4800) // disable int to bool warnings
#pragma warning(disable : 4996) // disable string security warnings

using namespace Microsoft::WRL;

// overload memory operators to get new with no-throwing behavior
void* __cdecl operator new(size_t size);

void* __cdecl operator new[](size_t size);

void __cdecl operator delete(void *p);

void __cdecl operator delete[](void *p);

#define SAFE_DELETE(x) { if(x) delete (x); (x) = nullptr; }
#define SAFE_DELETE_ARRAY(x) { if(x) delete [] (x); (x) = nullptr; }

// safe delete LIST of pointers
#define SAFE_DELETE_PLIST(x) { for(UINT i=0; i<(x).GetSize(); i++) { SAFE_DELETE((x)[i]) } (x).Erase(); }

#define BITFLAGS_ENUM(T, X) \
 enum X: T; \
 inline X operator | (X a, X b) { return X(T(a) | (T)b); } \
 inline X &operator |= (X &a, X b) { return (X&)((T&)a |= (T)b) ; } \
 inline X operator & (X a, X b) { return X((T)a & (T)b); } \
 inline X &operator &= (X &a, X b) { return (X&)((T&)a &= (T)b); } \
 inline X operator ^ (X a, X b) { return X((T)a ^ (T)b); } \
 inline X &operator ^= (X &a, X b) { return (X&)((T&)a ^= (T)b); } \
 inline X operator ~ (X a) { return X(~((T)a)); } \
 enum X: T

#define DEMO_MAX_STRING 256 // max length of a string
#define DEMO_MAX_FILENAME 512 // max length of file name
#define DEMO_MAX_FILEPATH 4096 // max length of file path   

#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080
#define VSYNC_ENABLED 0
// #define USE_WARP_DEVICE

#endif
