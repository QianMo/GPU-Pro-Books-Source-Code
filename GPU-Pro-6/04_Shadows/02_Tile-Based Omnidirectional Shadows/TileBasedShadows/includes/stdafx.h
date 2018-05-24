#ifndef AFX_STDAFX_H__83FFB9D0_9394_47D5_A90D_5BE5684EC708
#define AFX_STDAFX_H__83FFB9D0_9394_47D5_A90D_5BE5684EC708

#ifndef UNIX_PORT
  #include <Windows.h>
  #include <direct.h>
  #include <mmsystem.h>
  #include <glew.h>
  #include <wglext.h>
  #include <wglExtensions.h>
#endif /* UNIX_PORT */

#include <stdio.h>
#include <assert.h>
#include <sys/stat.h>
#include <memory.h>
#include <float.h>
#include <math.h>

// stl
#include <fstream>
#include <string>
#include <sstream>

// GUI
#include <AntTweakBar.h>

#include <maths.h>

#pragma warning(disable : 4800) // disable int to bool warnings
#pragma warning(disable : 4996) // disable string security warnings

// overload memory operators to get new with no-throwing behavior
inline void* __cdecl operator new(size_t size)
{
  return malloc(size); 
}

inline void* __cdecl operator new[](size_t size)
{ 
  return malloc(size); 
}

inline void __cdecl operator delete(void *p)
{ 
  free(p); 
}

inline void __cdecl operator delete[](void *p)
{ 
  free(p); 
}

#define SAFE_DELETE(x) { if(x) delete (x); (x) = NULL; }
#define SAFE_DELETE_ARRAY(x) { if(x) delete [] (x); (x) = NULL; }

// safe delete LIST of pointers
#define SAFE_DELETE_PLIST(x) { for(unsigned int i=0; i<(x).GetSize(); i++) { SAFE_DELETE((x)[i]) } (x).Erase(); }

#define DEMO_MAX_STRING 256 // max length of a string
#define DEMO_MAX_FILENAME 512 // max length of file name
#define DEMO_MAX_FILEPATH 4096 // max length of file path   

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720
#define VSYNC_ENABLED 0

// Maximum number of point lights used in this demo. 
#define MAX_NUM_POINT_LIGHTS 256

#endif