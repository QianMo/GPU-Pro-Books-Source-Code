//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------
#ifndef __CPUTBASE_H__
#define __CPUTBASE_H__

// Master #defines for which target
#define CPUT_FOR_DX11

#include <stdlib.h>
#include <crtdbg.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <assert.h>
#include "CPUTMath.h"
#include "CPUTEventHandler.h"
#include "CPUTCallbackHandler.h"
#include "CPUTTimer.h"

#ifdef CPUT_GPA_INSTRUMENTATION
// For D3DPERF_* calls, you also need d3d9.lib included
#include <d3d9.h>               // required for all the pix D3DPERF_BeginEvent()/etc calls
#include <ittnotify.h>
#include "CPUTITTTaskMarker.h"  // markup helper for GPA Platform Analyzer tags
#include "CPUTPerfTaskMarker.h" // markup helper for GPA Frame Analyzer tags

// GLOBAL instrumentation junk
enum CPUT_GPA_INSTRUMENTATION_STRINGS{
    GPA_HANDLE_CPUT_CREATE = 0,
    GPA_HANDLE_CONTEXT_CREATION,
    GPA_HANDLE_SYSTEM_INITIALIZATION,
    GPA_HANDLE_MAIN_MESSAGE_LOOP,
    GPA_HANDLE_EVENT_DISPATCH_AND_HANDLE,
    GPA_HANDLE_LOAD_SET,
    GPA_HANDLE_LOAD_MODEL,
    GPA_HANDLE_LOAD_MATERIAL,
    GPA_HANDLE_LOAD_TEXTURE,
    GPA_HANDLE_LOAD_CAMERAS,
    GPA_HANDLE_LOAD_LIGHTS,
    GPA_HANDLE_LOAD_VERTEX_SHADER,
    GPA_HANDLE_LOAD_GEOMETRY_SHADER,
    GPA_HANDLE_LOAD_PIXEL_SHADER,
    GPA_HANDLE_DRAW_GUI,
    GPA_HANDLE_STRING_ENUMS_SIZE,
};
#endif // CPUT_GPA_INSTRUMENTATION


// Heap corruption, ASSERT, and TRACE defines
//-----------------------------------------------------------------------------
#ifdef _DEBUG
    #include <tchar.h>
    #define TRACE(String)  {OutputDebugString(String);}
    #define DEBUGMESSAGEBOX(Title, Text) { CPUTOSServices::GetOSServices()->OpenMessageBox(Title, Text);}
    #define ASSERT(Condition, Message) { if( !(Condition) ) { cString msg = cString(_T(__FUNCTION__)) + _L(":  ") + Message; OutputDebugString(msg.c_str()); DEBUGMESSAGEBOX(_L("Assert"), msg ); } assert(Condition);}
    #define HEAPCHECK     { int  heapstatus = _heapchk(); ASSERT(_HEAPOK == heapstatus, _L("Heap corruption") ); }
    // #define HEAPCHECK     {}
#else
    #define ASSERT(Condition, Message)
    #define TRACE(String)
    #define DEBUGMESSAGEBOX(Title, Text)
    #define HEAPCHECK
#endif // _DEBUG


// Error codes
//-----------------------------------------------------------------------------
typedef enum CPUTResult
{
    // success
    CPUT_SUCCESS = 0x00000000,

    // warnings
//    CPUT_WARNING_OUT_OF_RANGE,
    CPUT_WARNING_NOT_FOUND,
//    CPUT_WARNING_ALREADY_EXISTS,
//    CPUT_WARNING_FILE_IN_SEARCH_PATH_BUT_NOT_WHERE_SPECIFIED,
//    CPUT_WARNING_PHONG_SHADER_MISSING_TEXTURE,
    CPUT_WARNING_CANCELED,
//    CPUT_WARNING_NO_SUITABLE_FORMAT_FOUND,
//
    CPUT_WARNING_SHADER_INPUT_SLOT_NOT_MATCHED,
//
//    // file errors
    CPUT_ERROR_FILE_NOT_FOUND = 0xF0000001,
    CPUT_ERROR_FILE_READ_ERROR = CPUT_ERROR_FILE_NOT_FOUND+1,
    CPUT_ERROR_FILE_CLOSE_ERROR = CPUT_ERROR_FILE_NOT_FOUND+2,
    CPUT_ERROR_FILE_IO_ERROR = CPUT_ERROR_FILE_NOT_FOUND+3,
    CPUT_ERROR_FILE_NO_SUCH_DEVICE_OR_ADDRESS = CPUT_ERROR_FILE_NOT_FOUND+4,
    CPUT_ERROR_FILE_BAD_FILE_NUMBER = CPUT_ERROR_FILE_NOT_FOUND+5,
    CPUT_ERROR_FILE_NOT_ENOUGH_MEMORY = CPUT_ERROR_FILE_NOT_FOUND+6,
    CPUT_ERROR_FILE_PERMISSION_DENIED = CPUT_ERROR_FILE_NOT_FOUND+7,
    CPUT_ERROR_FILE_DEVICE_OR_RESOURCE_BUSY = CPUT_ERROR_FILE_NOT_FOUND+8,
    CPUT_ERROR_FILE_EXISTS = CPUT_ERROR_FILE_NOT_FOUND+9,
    CPUT_ERROR_FILE_IS_A_DIRECTORY = CPUT_ERROR_FILE_NOT_FOUND+10,
    CPUT_ERROR_FILE_TOO_MANY_OPEN_FILES = CPUT_ERROR_FILE_NOT_FOUND+11,
    CPUT_ERROR_FILE_TOO_LARGE = CPUT_ERROR_FILE_NOT_FOUND+12,
    CPUT_ERROR_FILE_DEVICE_FULL = CPUT_ERROR_FILE_NOT_FOUND+13,
    CPUT_ERROR_FILE_FILENAME_TOO_LONG = CPUT_ERROR_FILE_NOT_FOUND+14,
    CPUT_ERROR_FILE_PATH_ERROR = CPUT_ERROR_FILE_NOT_FOUND+15,
    CPUT_ERROR_FILE_ERROR = CPUT_ERROR_FILE_NOT_FOUND+16,
//
//    CPUT_ERROR_DIRECTORY_NOT_FOUND = CPUT_ERROR_FILE_NOT_FOUND+21,
//
//    // subsystem errors
    CPUT_ERROR_INVALID_PARAMETER = 0xF0000100,
    CPUT_ERROR_NOT_FOUND = CPUT_ERROR_INVALID_PARAMETER+1,
//    CPUT_ERROR_COMPONENT_NOT_INITIALIZED = CPUT_ERROR_INVALID_PARAMETER+2,
//    CPUT_ERROR_SUBSYSTEM_OUT_OF_MEMORY = CPUT_ERROR_INVALID_PARAMETER+3,
//    CPUT_ERROR_OUT_OF_BOUNDS = CPUT_ERROR_INVALID_PARAMETER+4,
//    CPUT_ERROR_HEAP_CORRUPTION = CPUT_ERROR_INVALID_PARAMETER+5,
//
//    // image format errors
    CPUT_ERROR_UNSUPPORTED_IMAGE_FORMAT = 0xF0000200,
//    CPUT_ERROR_ERROR_LOADING_IMAGE = CPUT_ERROR_UNSUPPORTED_IMAGE_FORMAT+1,
    CPUT_ERROR_UNSUPPORTED_SRGB_IMAGE_FORMAT,
//
//    // shader loading errors
//    CPUT_SHADER_LOAD_ERROR = 0xF0000300,
//    CPUT_SHADER_COMPILE_ERROR = CPUT_SHADER_LOAD_ERROR+1,
//    CPUT_SHADER_LINK_ERROR = CPUT_SHADER_LOAD_ERROR+2,
//    CPUT_SHADER_REGISTRATION_ERROR = CPUT_SHADER_LOAD_ERROR+3,
//    CPUT_SHADER_CONSTANT_BUFFER_ERROR = CPUT_SHADER_LOAD_ERROR+4,
//    CPUT_SHADER_REFLECTION_ERROR = CPUT_SHADER_LOAD_ERROR+5,
//
//    // texture loading errors
    CPUT_TEXTURE_LOAD_ERROR = 0xF0000400,
    CPUT_ERROR_TEXTURE_FILE_NOT_FOUND = CPUT_TEXTURE_LOAD_ERROR+1,
//
//    // GUI errors
    CPUT_GUI_GEOMETRY_CREATION_ERROR = 0xF0000500,
//    CPUT_GUI_SAMPLER_CREATION_ERROR = CPUT_GUI_GEOMETRY_CREATION_ERROR+1,
//    CPUT_GUI_TEXTURE_CREATION_ERROR = CPUT_GUI_GEOMETRY_CREATION_ERROR+2,
//    CPUT_GUI_CANNOT_CREATE_CONTROL = CPUT_GUI_GEOMETRY_CREATION_ERROR+3,
    CPUT_GUI_INVALID_CONTROL_ID = CPUT_GUI_GEOMETRY_CREATION_ERROR+4,
//
//    // Texture loading errors
//    CPUT_FONT_TEXTURE_TYPE_ERROR = 0xF0000600,
//    CPUT_FONT_TEXTURE_LOAD_ERROR = CPUT_FONT_TEXTURE_TYPE_ERROR+1,
//
//    // Model loading errors
//    CPUT_ERROR_MODEL_LOAD_ERROR = 0xF0000650,
//    CPUT_ERROR_MODEL_FILE_NOT_FOUND = CPUT_ERROR_MODEL_LOAD_ERROR+1,
//
//    // Shader errors
    CPUT_ERROR_VERTEX_LAYOUT_PROBLEM = 0xF0000700,
//    CPUT_ERROR_VERTEX_BUFFER_CREATION_PROBLEM = CPUT_ERROR_VERTEX_LAYOUT_PROBLEM+1,
//    CPUT_ERROR_INDEX_BUFFER_CREATION_PROBLEM = CPUT_ERROR_VERTEX_LAYOUT_PROBLEM+2,
//    CPUT_ERROR_UNSUPPORTED_VERTEX_ELEMENT_TYPE = CPUT_ERROR_VERTEX_LAYOUT_PROBLEM+3,
//    CPUT_ERROR_INDEX_BUFFER_LAYOUT_PROBLEM = CPUT_ERROR_VERTEX_LAYOUT_PROBLEM+4,
    CPUT_ERROR_SHADER_INPUT_SLOT_NOT_MATCHED = CPUT_ERROR_VERTEX_LAYOUT_PROBLEM+5,
//
//
//    // Context creation errors
//    CPUT_ERROR_CONTEXT_CREATION_FAILURE = 0xF0000C00,
//    CPUT_ERROR_SWAP_CHAIN_CREATION_FAILURE = CPUT_ERROR_CONTEXT_CREATION_FAILURE+1,
//    CPUT_ERROR_RENDER_TARGET_VIEW_CREATION_FAILURE = CPUT_ERROR_CONTEXT_CREATION_FAILURE+2,
//
//    // Depth buffer errors
//    CPUT_ERROR_DEPTH_BUFFER_CREATION_ERROR = 0xF0000800,
//    CPUT_ERROR_DEPTH_STENCIL_BUFFER_CREATION_ERROR = CPUT_ERROR_DEPTH_BUFFER_CREATION_ERROR+1,
//    CPUT_ERROR_RASTER_STATE_CREATION_ERROR = CPUT_ERROR_DEPTH_BUFFER_CREATION_ERROR+2,
//
//    // GUI shaders
    CPUT_ERROR_INITIALIZATION_GUI_VERTEX_SHADER_NOT_FOUND = 0xF0000130,
    CPUT_ERROR_INITIALIZATION_GUI_PIXEL_SHADER_NOT_FOUND = CPUT_ERROR_INITIALIZATION_GUI_VERTEX_SHADER_NOT_FOUND+1,
    CPUT_ERROR_INITIALIZATION_GUI_CONTROL_TEXTURES_NOT_FOUND = CPUT_ERROR_INITIALIZATION_GUI_VERTEX_SHADER_NOT_FOUND+2,
//
//    // gfx system errors
//    CPUT_ERROR_GFX_SUBSYSTEM_BUSY = 0xF0000B00,
//    CPUT_ERROR_GFX_SUBSYSTEM_TO_MANY_OBJECTS = CPUT_ERROR_GFX_SUBSYSTEM_BUSY+1,
//
//    // window layer errors
    CPUT_ERROR_WINDOW_CANNOT_REGISTER_APP = 0xF0000D00,
    CPUT_ERROR_WINDOW_ALREADY_EXISTS = CPUT_ERROR_WINDOW_CANNOT_REGISTER_APP+1,
//    CPUT_ERROR_CANNOT_GET_WINDOW_CLASS = CPUT_ERROR_WINDOW_CANNOT_REGISTER_APP+3,
    CPUT_ERROR_CANNOT_GET_WINDOW_INSTANCE = CPUT_ERROR_WINDOW_CANNOT_REGISTER_APP+4,
//    CPUT_ERROR_WINDOW_OS_PROPERTY_GET_ERROR = CPUT_ERROR_WINDOW_CANNOT_REGISTER_APP+5,
//
//    // AssetLibrary/AssetSet errors
    CPUT_ERROR_ASSET_LIBRARY_INVALID_LIBRARY = 0xF0000E00,
//    CPUT_ERROR_ASSET_SET_INVALID_TYPE = CPUT_ERROR_ASSET_LIBRARY_INVALID_LIBRARY+1,
//    CPUT_ERROR_ASSET_LIBRARY_OBJECT_NOT_FOUND,
//    CPUT_ERROR_ASSET_ALREADY_EXISTS = CPUT_ERROR_ASSET_LIBRARY_INVALID_LIBRARY+3,
//
//    // Paramter block errors.
    CPUT_ERROR_PARAMETER_BLOCK_NOT_FOUND = 0xF0000F00,
//
//    // misc errors
//    CPUT_ERROR_FULLSCREEN_SWITCH_ERROR = 0xF0000F00,
} CPUTResult;

static int gRefCount = 0;
//handy defines
//-----------------------------------------------------------------------------
#define SAFE_RELEASE(p)     {if((p)){HEAPCHECK; gRefCount = (p)->Release(); (p)=NULL; HEAPCHECK;} }
#define SAFE_DELETE(p)      {if((p)){HEAPCHECK; delete (p);     (p)=NULL;HEAPCHECK; }}
#define SAFE_DELETE_ARRAY(p){if((p)){HEAPCHECK; delete[](p);    (p)=NULL;HEAPCHECK; }}
#define UNREFERENCED_PARAMETER(P) (P)

// CPUT data types
//-----------------------------------------------------------------------------
#define CPUTSUCCESS(returnCode) ((returnCode) < 0xF0000000)
#define CPUTFAILED(returnCode) ((returnCode) >= 0xF0000000)


//typedef UINT CPUTResult;
typedef unsigned int UINT;
typedef unsigned long DWORD;

// color
struct CPUTColor4
{
    float r;
    float g;
    float b;
    float a;

    bool operator == (const CPUTColor4& rhs) const
    {
        return((rhs.r == r) && 
               (rhs.g == g) &&
               (rhs.b == b) &&
               (rhs.a == a));
    }
    bool operator != (const CPUTColor4& rhs) const
    {
        return((rhs.r != r) || 
               (rhs.g != g) ||
               (rhs.b != b) ||
               (rhs.a != a));
    }
};

// where the loader should start looking from to locate files
enum CPUT_PATH_SEARCH_MODE
{
    CPUT_PATH_SEARCH_RESOURCE_DIRECTORY,
    CPUT_PATH_SEARCH_NONE,
};

// string size limitations
const UINT CPUT_MAX_PATH = 2048;
const UINT CPUT_MAX_STRING_LENGTH = 1024;
const UINT CPUT_MAX_SHADER_ERROR_STRING_LENGTH = 8192;
const UINT CPUT_MAX_DIGIT_STRING_LENGTH = 5;



// Data format types used in interpreting mesh data
enum CPUT_DATA_FORMAT_TYPE
{
    CPUT_UNKNOWN=0,

    CPUT_DOUBLE=1,
    CPUT_F32=2,

    CPUT_U64=3,
    CPUT_I64=4,

    CPUT_U32=5,
    CPUT_I32=6,

    CPUT_U16=7,
    CPUT_I16=8,

    CPUT_U8=9,
    CPUT_I8=10,

    CPUT_CHAR=11,
    CPUT_BOOL=12,
};

// Corresponding sizes (in bytes) that match CPUT_DATA_FORMAT_TYPE
const int CPUT_DATA_FORMAT_SIZE[] =
{
        0, //CPUT_UNKNOWN=0,
        
        8, //CPUT_DOUBLE,
        4, //CPUT_F32,
        
        8, //CPUT_U64,
        8, //CPUT_I64,
        
        4, //CPUT_U32,
        4, //CPUT_I32,
        
        2, //CPUT_U16,
        2, //CPUT_I16,
        
        1, //CPUT_U8,
        1, //CPUT_I8,

        1, //CPUT_CHAR
        1, //CPUT_BOOL
};

//-----------------------------------------------------------------------------
enum eCPUTMapType
{
    CPUT_MAP_UNDEFINED = 0,
    CPUT_MAP_READ = 1,
    CPUT_MAP_WRITE = 2,
    CPUT_MAP_READ_WRITE = 3,
    CPUT_MAP_WRITE_DISCARD = 4,
    CPUT_MAP_NO_OVERWRITE = 5
};

// routines to support unicode + multibyte
// TODO: Move to string file
//-----------------------------------------------------------------------------
#if defined (UNICODE) || defined(_UNICODE)

    // define string and literal types
    #define cString std::wstring
	#define cStringStream std::wstringstream
	#define cFile std::wfstream
    #define _L(x)      L##x

    // convert integer to wide/unicode ascii
    //-----------------------------------------------------------------------------
    inline std::wstring itoc(const int integer)
    {
        wchar_t wcstring[CPUT_MAX_STRING_LENGTH];
        swprintf_s(&wcstring[0], CPUT_MAX_STRING_LENGTH, _L("%d"),integer);
        std::wstring ws(wcstring);

        return ws;
    }

    // convert pointer to wide/unicode ascii
    //-----------------------------------------------------------------------------
    inline std::wstring ptoc(const void *pPointer)
    {
        std::wstringstream wstream;
        //std::ostringstream os;
        wstream << pPointer;

        std::wstring address;
        address = wstream.str();

        return address;
    }

    // convert char* to wide/unicode string
    //-----------------------------------------------------------------------------
    inline std::wstring s2ws(const char* stringArg)
    {
        // compute the size of the buffer I need to allocate
        size_t numConvertedChars;
        mbstowcs_s(&numConvertedChars, NULL, 0, stringArg, _TRUNCATE);
        numConvertedChars++;  // +1 for null termination
        if(numConvertedChars>CPUT_MAX_STRING_LENGTH)
        {
            numConvertedChars = CPUT_MAX_STRING_LENGTH;
        }

        // allocate the converted string and copy
        wchar_t *pWString = new wchar_t[numConvertedChars];
        mbstowcs_s(&numConvertedChars, pWString, numConvertedChars, stringArg, _TRUNCATE);
        std::wstring ws(pWString);
        delete [] pWString;
        return ws;
        /*
        // alternate method - less 'safe', but possibly useful for unix
        std::string s = stringArg;

        std::wstring ws(s.begin(), s.end());
        ws.assign(s.begin(), s.end());

        return ws;
        */
    }

    // convert wide/unicode string to char
    //-----------------------------------------------------------------------------
    inline char* ws2s(std::wstring string)
    {
        size_t numConverted, finalCount;

        // what size of buffer (in bytes) do we need to allocate for conversion?
        wcstombs_s(&numConverted, NULL, 0, string.c_str(), CPUT_MAX_STRING_LENGTH);
        numConverted+=2; // for null termination
        char *pBuffer = new char[numConverted];

        // do the actual conversion
        wcstombs_s(&finalCount, pBuffer, numConverted, string.c_str(), CPUT_MAX_STRING_LENGTH);

        return pBuffer;
    }

#else

    // define string and literal types
    #define cString std::string
	#define cStringStream std::stringstream
	#define cFile std::fstream
    #define _L(x)      x

    // conversion routine
    //-----------------------------------------------------------------------------
    inline std::string s2ws(const char* stringArg) { return std::string(stringArg); }

    // convert integer to char string
    //-----------------------------------------------------------------------------
    inline std::string itoc(const int integer)
    {
        char string[CPUT_MAX_STRING_LENGTH];
        sprintf_s(string, CPUT_MAX_STRING_LENGTH, "%d",integer);
        std::string s(string);

        return s;
    }

    // convert pointer to wide/unicode ascii
    //-----------------------------------------------------------------------------
    inline std::string ptoc(const void *pPointer)
    {
        std::ostringstream stream;

        stream << pPointer;

        std::string address;
        address = stream.str();

        return address;
    }

    // conversion from ws2s
    // Doesn't do anything in multibyte version since string is already a char*
    //-----------------------------------------------------------------------------
    inline char* ws2s(const char* string)
    {
        return const_cast<char*>(string);
    }
#endif

#ifdef CPUT_FOR_DX11
#include "CPUTRenderTarget.h"
#else    
    #error You must supply a target graphics API (ex: #define CPUT_FOR_DX11), or implement the target API for this file.
#endif

class CPUTCamera;
class CPUTRenderStateBlock;

// CPUT class
//-----------------------------------------------------------------------------
class CPUT:public CPUTEventHandler, public CPUTCallbackHandler
{
protected:
    CPUTCamera  *mpCamera;
    CPUTCamera  *mpShadowCamera;
	CPUTTimer   *mpTimer;
    float3       mLightColor; // TODO: Get from light(s)
    float3       mAmbientColor;
    CPUTBuffer  *mpBackBuffer;
    CPUTBuffer  *mpDepthBuffer;
    CPUTTexture *mpBackBufferTexture;
    CPUTTexture *mpDepthBufferTexture;

public:
    CPUT() :
        mpCamera(NULL),
        mpShadowCamera(NULL),
        mAmbientColor(0.2f, 0.2f, 0.2f),
        mLightColor(1.0f, 1.0f, 1.0f),
        mpBackBuffer(NULL),
        mpDepthBuffer(NULL),
        mpBackBufferTexture(NULL),
        mpDepthBufferTexture(NULL)
    {}
    virtual ~CPUT() {}

    CPUTCamera  *GetCamera() { return mpCamera; }
    CPUTCamera  *GetShadowCamera() { return mpShadowCamera; } // TODO: Support more than one.
    virtual void InnerExecutionLoop() {;}
    virtual void ResizeWindowSoft(UINT width, UINT height) {UNREFERENCED_PARAMETER(width);UNREFERENCED_PARAMETER(height);}
    virtual void ResizeWindow(UINT width, UINT height) {
        CPUTRenderTargetColor::SetActiveWidthHeight( width, height );
        CPUTRenderTargetDepth::SetActiveWidthHeight( width, height );
    }
    virtual void DeviceShutdown(){}

    virtual CPUTEventHandledCode CPUTHandleKeyboardEvent(CPUTKey key) {UNREFERENCED_PARAMETER(key);return CPUT_EVENT_UNHANDLED;}
    virtual CPUTEventHandledCode CPUTHandleMouseEvent(int x, int y, int wheel, CPUTMouseState state) {UNREFERENCED_PARAMETER(x);UNREFERENCED_PARAMETER(y);UNREFERENCED_PARAMETER(wheel);UNREFERENCED_PARAMETER(state);return CPUT_EVENT_UNHANDLED;}

    float3 &GetAmbientColor() { return mAmbientColor; }
    void    SetAmbientColor( float3 &ambientColor ) {  mAmbientColor = ambientColor; }
    float3 &GetLightColor() { return mLightColor; }
    void    SetLightColor( float3 &lightColor ) {  mLightColor = lightColor; }
};

// Include this here to make sure ASSERT resolves correctly
#include "CPUTOSServicesWin.h"

void CPUTSetDebugName( void *pResource, cString name );
#endif // #ifndef __CPUTBASE_H__
