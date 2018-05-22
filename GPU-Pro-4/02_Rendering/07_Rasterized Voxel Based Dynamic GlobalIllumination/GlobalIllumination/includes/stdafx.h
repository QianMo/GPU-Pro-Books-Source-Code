#ifndef AFX_STDAFX_H__25664952_6DB8_47a3_9C6F_7BD5583D8B19
#define AFX_STDAFX_H__25664952_6DB8_47a3_9C6F_7BD5583D8B19

#include <windows.h>
#include <stdio.h>
#include <assert.h>
#include <direct.h>
#include <sys/stat.h>
#include <mmsystem.h>
#include <memory.h>
#include <float.h>
#include <math.h>

// stl
#include <fstream>
#include <string>
#include <sstream>

// DirectX 11
#include <d3d11.h>
#include <d3dx11.h>
#include <d3dcompiler.h>
#include <DXGI.h>

#include <maths.h>

#pragma warning(disable : 4800) // disable int to bool warnings
#pragma warning(disable : 4996) // disable security warnings

#define SAFE_DELETE(x) { if(x) delete (x);(x)=NULL; }
#define SAFE_DELETE_ARRAY(x) { if(x) delete [] (x);(x)=NULL;  }

// safe delete LIST of pointers
#define SAFE_DELETE_PLIST(x) { for(int i=0;i<(x).GetSize();i++) {SAFE_DELETE((x)[i])} (x).Erase(); }

#define SAFE_RELEASE(x) { if(x) (x)->Release();(x)=NULL; }

#define DEMO_MAX_STRING 256 // max length of a string
#define DEMO_MAX_FILENAME 512 // max length of file name
#define DEMO_MAX_FILEPATH 4096 // max length of file path   

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720
#define VSYNC_ENABLED 0

// To avoid long loading times, per default pre-compiled shaders are used. However, if
// you want to modify the shader-source, you have to un-define the below define, so that
// the shaders will be re-generated.
#define USE_SHADER_BIN 

#endif