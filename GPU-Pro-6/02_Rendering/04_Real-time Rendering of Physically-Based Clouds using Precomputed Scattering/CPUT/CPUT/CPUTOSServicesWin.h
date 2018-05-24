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
#ifndef __CPUTOSServicesWin_H__
#define __CPUTOSServicesWin_H__



#include "CPUT.h"

// OS includes
#include <windows.h>
#include <errno.h>  // file open error codes
#include <string>   // wstring



class CPUTOSServices
{
public:
    static CPUTOSServices* GetOSServices();
    static CPUTResult DeleteOSServices();

    // screen/window dimensions
    void GetClientDimensions( int *pWidth, int *pHeight);
    void GetClientDimensions( int *pX, int *pY, int *pWidth, int *pHeight);
    void GetDesktopDimensions(int *pWidth, int *pHeight);
    bool IsWindowMaximized();
    bool IsWindowMinimized();
    bool DoesWindowHaveFocus();

    // Mouse capture - 'binds'/releases all mouse input to this window
    void CaptureMouse();
    void ReleaseMouse();
    

    //Working directory manipulation
    CPUTResult GetWorkingDirectory(cString *pPath);
    CPUTResult SetWorkingDirectory(const cString &path);
    CPUTResult GetExecutableDirectory(cString *pExecutableDir);

    // Path helpers
    CPUTResult ResolveAbsolutePathAndFilename(const cString &fileName, cString *pResolvedPathAndFilename);
    CPUTResult SplitPathAndFilename(const cString &sourceFilename, cString *pDrive, cString *pDir, cString *pfileName, cString *pExtension);

    // file handling
    CPUTResult DoesFileExist(const cString &pathAndFilename);
    CPUTResult DoesDirectoryExist(const cString &path);
    CPUTResult OpenFile(const cString &fileName, FILE **pFilePointer);
    CPUTResult ReadFileContents(const cString &fileName, UINT *psizeInBytes, void **ppData);

    // File dialog box
    CPUTResult OpenFileDialog(const cString &filter, cString *pfileName);

    // Informational Message box
    CPUTResult OpenMessageBox(cString title, cString text);

    // error handling
    inline void Assert(bool bCondition) {assert(bCondition);}
    void OutputConsoleString(cString &outputString);
    CPUTResult TranslateFileError(errno_t err);

    // hwnd setup
    inline void SethWnd(const HWND hWnd) { mhWnd = hWnd; };
    inline void GetWindowHandle(HWND *phWnd) { *phWnd = mhWnd; };

    // special keys
    bool ControlKeyPressed(CPUTKey &key);



private:
    CPUTOSServices();
     ~CPUTOSServices();
    static CPUTOSServices *mpOSServices;   // singleton object
    HWND                   mhWnd;
    cString                mCPUTResourceDirectory;
    bool                   FileFoundButWithError(CPUTResult result);

#ifdef CPUT_GPA_INSTRUMENTATION
public:
    // GPA instrumentation (only available in Profile build)
    void GetInstrumentationPointers(__itt_domain **ppGPADomain, CPUT_GPA_INSTRUMENTATION_STRINGS eString, __itt_string_handle **ppGPAStringHandle);
    void SetInstrumentationPointers(__itt_domain *pGPADomain, CPUT_GPA_INSTRUMENTATION_STRINGS eString, __itt_string_handle *pGPAStringHandle);

private:
    // GPA instrumentation member variables
    __itt_domain *mpGPADomain;
    __itt_string_handle *mppGPAStringHandles[GPA_HANDLE_STRING_ENUMS_SIZE];
#endif

};
#endif // __CPUTOSServicesWin_H__
