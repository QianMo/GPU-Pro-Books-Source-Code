/*
This file is part of TileBasedShadows demo to make it run on Unix-like systems.
It is tailored specifically for this demo to keep the original source code as much
untouched as possible and as such it might or might not work somewhere else.

Nikita Kindt
*/

#ifndef UNIX_PORT_H
#define UNIX_PORT_H

#include <GL/glew.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <SDL.h>
#include <cstdint>
#include <stdio.h>

typedef SDL_Window* HWND;
typedef SDL_Renderer* HDC;
typedef SDL_GLContext HGLRC;
typedef int BOOL;
typedef void* HINSTANCE;
typedef Uint32 UINT;
typedef int WPARAM;
typedef Sint32 LONG; /* LP64 vs LLP64 ! */
typedef Uint64 LONGLONG;
typedef Uint64 LARGE_INTEGER;

struct POINT
{
  LONG x, y;
};
struct RECT
{
  LONG left, top, right, bottom;
};

#define E_FAIL -1
#define VK_LBUTTON 1
#define VK_RBUTTON 2
#define VK_F1 (SDLK_F1 & ~SDLK_SCANCODE_MASK)
#define VK_F2 (SDLK_F2 & ~SDLK_SCANCODE_MASK)
#define VK_ESCAPE (SDLK_ESCAPE & ~SDLK_SCANCODE_MASK)
#define WM_LBUTTONDOWN 1
#define WM_RBUTTONDOWN 2
#define WM_LBUTTONUP 3
#define WM_RBUTTONUP 4
#define WM_KEYDOWN 5
#define WM_KEYUP 6
#define MB_OK 1
#define MB_ICONEXCLAMATION 2
#define MB_ICONERROR 4
#define BI_RGB 0
#define MessageBox(_h, _t, _c, _f) { Uint32 f = SDL_MESSAGEBOX_INFORMATION; \
    if((_f & MB_ICONEXCLAMATION) != 0) f = SDL_MESSAGEBOX_WARNING; \
    else if ((_f & MB_ICONERROR) != 0) f =  SDL_MESSAGEBOX_ERROR; \
    SDL_ShowSimpleMessageBox(f, _c, _t, nullptr); }
#define OutputDebugString(_t) { printf("%s", _t); } /* Write to stdout */
#define SwapBuffers(_h) SDL_GL_SwapWindow(Demo::window->GetHWnd())
#define QueryPerformanceFrequency(_p) false
#define QueryPerformanceCounter(_p) false
#define timeGetTime() SDL_GetTicks()
#define fopen_s(_f, _n, _m) { *_f = fopen(_n, _m); }
#define vsprintf_s(_b, _f, _a) vsnprintf(_b, sizeof(_b), _f, _a)
#define _snprintf snprintf
#define _vscprintf(_f, _a) 512
#define _stat stat
#define _chdir chdir

inline int ShowCursor(BOOL show)
{
  auto r = SDL_ShowCursor(show != 0 ? 1 : 0);
  return r == 0 ? -1 : r; /* Take care of infinite loop */
}

#pragma pack(push, packing)
#pragma pack(1)
struct BITMAPINFOHEADER
{
  std::uint32_t biSize;
  std::int32_t biWidth;
  std::int32_t biHeight;
  std::uint16_t biPlanes;
  std::uint16_t biBitCount;
  std::uint32_t biCompression;
  std::uint32_t biSizeImage;
  std::int32_t biXPelsPerMeter;
  std::int32_t biYPelsPerMeter;
  std::uint32_t biClrUsed;
  std::uint32_t biClrImportant;
};
struct BITMAPFILEHEADER
{
  std::uint16_t bfType;
  std::uint32_t bfSize;
  std::uint16_t bfReserved1;
  std::uint16_t bfReserved2;
  std::uint32_t bfOffBits;
};
#pragma pack(pop, packing)

#endif /* UNIX_PORT_H */
