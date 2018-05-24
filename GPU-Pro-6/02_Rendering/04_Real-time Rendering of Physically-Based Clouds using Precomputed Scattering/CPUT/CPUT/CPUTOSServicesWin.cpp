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
#include "CPUTOSServicesWin.h"
#include "tchar.h"
CPUTOSServices* CPUTOSServices::mpOSServices = NULL;

// Constructor
//-----------------------------------------------------------------------------
CPUTOSServices::CPUTOSServices():mhWnd(NULL)
{
    //mCPUTMediaDirectory.clear();
    mCPUTResourceDirectory.clear();
}

// Destructor
//-----------------------------------------------------------------------------
CPUTOSServices::~CPUTOSServices()
{
    //mCPUTMediaDirectory.clear();
    mCPUTResourceDirectory.clear();
}

// Singleton GetOSServices()
//-----------------------------------------------------------------------------
CPUTOSServices* CPUTOSServices::GetOSServices()
{
    if(NULL==mpOSServices)
        mpOSServices = new CPUTOSServices();
    return mpOSServices;
}

// Singleton destroyer
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::DeleteOSServices()
{
    if(mpOSServices)
    {
        delete mpOSServices;
        mpOSServices = NULL;
    }

    return CPUT_SUCCESS;
}

// Get the OS window dimensions
//-----------------------------------------------------------------------------
void CPUTOSServices::GetClientDimensions(int *pX, int *pY, int *pWidth, int *pHeight)
{
    RECT windowRect;
    if(0==GetClientRect(mhWnd, &windowRect))
    {
        return;
    }
    *pX      = windowRect.left;
    *pY      = windowRect.top;
    *pWidth  = windowRect.right - windowRect.left;
    *pHeight = windowRect.bottom - windowRect.top;
}

// Get the OS window client area dimensions
//-----------------------------------------------------------------------------
void CPUTOSServices::GetClientDimensions(int *pWidth, int *pHeight)
{
    RECT windowRect;
    if(0==GetClientRect(mhWnd, &windowRect))
    {
        return;
    }
    *pWidth  = windowRect.right - windowRect.left;
    *pHeight = windowRect.bottom - windowRect.top;
}

// Get the desktop dimensions
//-----------------------------------------------------------------------------
void CPUTOSServices::GetDesktopDimensions(int *pWidth, int *pHeight)
{
    *pWidth  = GetSystemMetrics(SM_CXFULLSCREEN);  // alternate method: GetSystemMetrics(SM_CXSCREEN);
    *pHeight = GetSystemMetrics(SM_CYFULLSCREEN);  // alternate method: GetSystemMetrics(SM_CYSCREEN);
}


// Returns true if the window is currently maximized
//-----------------------------------------------------------------------------
bool CPUTOSServices::IsWindowMaximized()
{
    WINDOWPLACEMENT WindowPlacement;
    WindowPlacement.length = sizeof(WindowPlacement);
    GetWindowPlacement(mhWnd,  &WindowPlacement);

    if(SW_SHOWMAXIMIZED == WindowPlacement.showCmd)
    {
        return true;
    }
    
    return false;
}

// Returns true if the window is currently minimized
//-----------------------------------------------------------------------------
bool CPUTOSServices::IsWindowMinimized()
{
    WINDOWPLACEMENT WindowPlacement;
    WindowPlacement.length = sizeof(WindowPlacement);
    GetWindowPlacement(mhWnd,  &WindowPlacement);

    if(SW_SHOWMAXIMIZED == WindowPlacement.showCmd)
    {
        return true;
    }
    
    return false;
}

// Returns true if the CPUT window is currently the 'focused' window on the 
// desktop
//-----------------------------------------------------------------------------
bool CPUTOSServices::DoesWindowHaveFocus()
{
    HWND hFocusedWindow = GetActiveWindow();
    if(mhWnd == hFocusedWindow)
    {
        return true;
    }
    return false;
}

// Retrieves the current working directory
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::GetWorkingDirectory(cString *pPath)
{
    TCHAR pPathAsTchar[CPUT_MAX_PATH];
    DWORD result = GetCurrentDirectory(CPUT_MAX_PATH, pPathAsTchar);
    ASSERT( result, _L("GetCurrentDirectory returned 0.") );
    *pPath = pPathAsTchar;
    return CPUT_SUCCESS;
}

// Sets the current working directory
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::SetWorkingDirectory(const cString &path)
{
    BOOL result = SetCurrentDirectory(path.c_str());
    ASSERT( 0 != result, _L("Error setting current directory.") );
    return CPUT_SUCCESS;
}

// Gets the location of the executable's directory
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::GetExecutableDirectory(cString *pExecutableDir)
{
    TCHAR   pFilename[CPUT_MAX_PATH];
    DWORD result = GetModuleFileName(NULL, pFilename, CPUT_MAX_PATH);
    ASSERT( 0 != result, _L("Unable to get executable's working directory."));

    // strip off the executable name+ext
    cString ResolvedPathAndFilename;
    ResolveAbsolutePathAndFilename(pFilename, &ResolvedPathAndFilename);
    cString Drive, Dir, Filename, Ext;
    SplitPathAndFilename(ResolvedPathAndFilename, &Drive, &Dir, &Filename, &Ext);

    // store and return
    *pExecutableDir = Drive + Dir;

    return CPUT_SUCCESS;
}

// Split up the supplied path+fileName into its constituent parts
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::SplitPathAndFilename(const cString &sourceFilename, cString *pDrive, cString *pDir, cString *pFileName, cString *pExtension)
{
    TCHAR pSplitDrive[CPUT_MAX_PATH];
    TCHAR pSplitDirs[CPUT_MAX_PATH];
    TCHAR pSplitFile[CPUT_MAX_PATH];
    TCHAR pSplitExt[CPUT_MAX_PATH];
#if defined (UNICODE) || defined(_UNICODE)
    #define SPLITPATH _wsplitpath_s
#else
    #define SPLITPATH _splitpath_s
#endif
    errno_t result = SPLITPATH(sourceFilename.c_str(), pSplitDrive, CPUT_MAX_PATH, pSplitDirs, CPUT_MAX_PATH, pSplitFile, CPUT_MAX_PATH, pSplitExt, CPUT_MAX_PATH);
    ASSERT( 0 == result, _L("Error splitting path") );

    // return the items the user wants
    *pDrive     = pSplitDrive;
    *pDir       = pSplitDirs;
    *pFileName  = pSplitFile;
    *pExtension = pSplitExt;

    return CPUT_SUCCESS;
}

// Takes a relative/full path+fileName and returns the absolute path with drive
// letter, absolute path, fileName and extension of this file.
// Truncates total path/file length to CPUT_MAX_PATH
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::ResolveAbsolutePathAndFilename(const cString &fileName, cString *pResolvedPathAndFilename)
{
    TCHAR pFullPathAndFilename[CPUT_MAX_PATH];
    DWORD result = GetFullPathName(fileName.c_str(), CPUT_MAX_PATH, pFullPathAndFilename, NULL);
    ASSERT( 0 != result, _L("Error getting full path name") );
    *pResolvedPathAndFilename = pFullPathAndFilename;

    return CPUT_SUCCESS;
}

// Verifies that file exists at specified path
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::DoesFileExist(const cString &pathAndFilename)
{
    // check for file existence
    // attempt to open it where they said it was
    FILE *pFile = NULL;
#if defined (UNICODE) || defined(_UNICODE)
    errno_t err = _wfopen_s(&pFile, pathAndFilename.c_str(), _L("r"));
#else
    errno_t err = fopen_s(&pFile, pathAndFilename.c_str(), _L("r"));
#endif
    if(0 == err)
    {
        // yep - file exists
        fclose(pFile);
        return CPUT_SUCCESS;
    }

    // not found, translate the file error and return it
    return TranslateFileError(err);
}

// Verifies that directory exists.
// Returns success if the directory exists and is readable (failure may mean
// it's busy or permissions denied on win32)
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::DoesDirectoryExist(const cString &path)
{
    DWORD fileAttribs;
#if defined (UNICODE) || defined(_UNICODE)
    fileAttribs = GetFileAttributesW(path.c_str());
#else
    fileAttribs = GetFileAttributesA(path.c_str());
#endif
    ASSERT( INVALID_FILE_ATTRIBUTES != fileAttribs, _L("Failed getting file attributes") );
    return CPUT_SUCCESS;
}

// Open a file and return file pointer to it
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::OpenFile(const cString &fileName, FILE **ppFilePointer)
{
#if defined (UNICODE) || defined(_UNICODE)
    errno_t err = _wfopen_s(ppFilePointer, fileName.c_str(), _L("r"));
#else
    errno_t err = fopen_s(ppFilePointer, fileName.c_str(), "r");
#endif

    return TranslateFileError(err);
}

// Read the entire contents of a file and return a pointer/size to it
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::ReadFileContents(const cString &fileName, UINT *pSizeInBytes, void **ppData)
{
    FILE *pFile = NULL;
#if defined (UNICODE) || defined(_UNICODE)
    errno_t err = _wfopen_s(&pFile, fileName.c_str(), _L("r"));
#else
    errno_t err = fopen_s(&pFile, fileName.c_str(), "r");
#endif
    if(0 == err)
    {
        // get file size
        fseek(pFile, 0, SEEK_END);
	    *pSizeInBytes = ftell(pFile);
        fseek (pFile, 0, SEEK_SET);

        // allocate buffer
        *ppData = (void*) new char[*pSizeInBytes];
        ASSERT( ppData, _L("Out of memory") );

        // read it all in
        UINT numBytesRead = (UINT) fread(*ppData, sizeof(char), *pSizeInBytes, pFile);
        ASSERT( numBytesRead == *pSizeInBytes, _L("File read byte count mismatch.") );

        // close and return
        fclose(pFile);
        return CPUT_SUCCESS;
    }

    // some kind of file error, translate the error code and return it
    return TranslateFileError(err);
}

// Open the OS's 'open a file' dialog box
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::OpenFileDialog(const cString &filter, cString *pfileName)
{
    OPENFILENAME ofn;       // common dialog box structure
    TCHAR szFile[260];       // buffer for file name

    // Initialize OPENFILENAME
    UNREFERENCED_PARAMETER(filter);
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = mhWnd;
    ofn.lpstrFilter = _L("All\0*.*\0Text\0*.TXT\0\0");//Filter.c_str(); //"All\0*.*\0Text\0*.TXT\0";
    ofn.lpstrFile = szFile;
    // Set lpstrFile[0] to '\0' so that GetOpenFileName does not
    // use the contents of szFile to initialize itself.
    ofn.lpstrFile[0] = '\0';
    ofn.nMaxFile = sizeof(szFile);
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    if(0==GetOpenFileName(&ofn))
    {
        return CPUT_WARNING_CANCELED;
    }

    *pfileName = szFile;
    return CPUT_SUCCESS;
}

// Open a system dialog box
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::OpenMessageBox(cString title, cString text)
{
	::MessageBox(NULL, text.c_str(), title.c_str(), MB_OK);

    return CPUT_SUCCESS;
}

// Returns whether the specified control key is pressed or not
//-----------------------------------------------------------------------------
bool CPUTOSServices::ControlKeyPressed(CPUTKey &key)
{
    int nVirtKey;

    switch(key)
    {
    case KEY_LEFT_SHIFT:
        nVirtKey = VK_LSHIFT;
        break;
    case KEY_RIGHT_SHIFT:
        nVirtKey = VK_RSHIFT;
        break;
    case KEY_LEFT_CTRL:
        nVirtKey = VK_LCONTROL;
        break;
    case KEY_RIGHT_CTRL:
        nVirtKey = VK_RCONTROL;
        break;
    case KEY_LEFT_ALT:
        nVirtKey = VK_LMENU;
        break;
    case KEY_RIGHT_ALT:
        nVirtKey = VK_RMENU;
        break;
    default:
        return false;
    };

    SHORT result = GetKeyState(nVirtKey);

    // return true/false if handled
    return (result & 0x8000)!=0;
}

// mouse
// this function 'captures' the mouse and makes it ONLY available to this app
// User cannot click on any other app until you call ReleaseMouse, so use this
// carefully
//-----------------------------------------------------------------------------
void CPUTOSServices::CaptureMouse()
{
    SetCapture(mhWnd);
}

// Releases a captured mouse
//-----------------------------------------------------------------------------
void CPUTOSServices::ReleaseMouse()
{
    ReleaseCapture();
}

// outputs a string to the debug out.
// In Visual Studio - this is the Output pane
//-----------------------------------------------------------------------------
void CPUTOSServices::OutputConsoleString(cString &OutputString)
{
    OutputDebugString(OutputString.c_str());
}

// Did file open error indicate a returnable file/system problem? (or just not found)
//-----------------------------------------------------------------------------
bool CPUTOSServices::FileFoundButWithError(CPUTResult result)
{
    bool IsError = false;

    // Is the result a file error code?
    switch(result)
    {
    case CPUT_ERROR_FILE_IO_ERROR:
    case CPUT_ERROR_FILE_NOT_ENOUGH_MEMORY:
    case CPUT_ERROR_FILE_PERMISSION_DENIED:
    case CPUT_ERROR_FILE_DEVICE_OR_RESOURCE_BUSY:
    case CPUT_ERROR_FILE_IS_A_DIRECTORY:
    case CPUT_ERROR_FILE_TOO_MANY_OPEN_FILES:
    case CPUT_ERROR_FILE_TOO_LARGE:
    case CPUT_ERROR_FILE_FILENAME_TOO_LONG:
        IsError = true;
        break;

    default:
        // nope - good to go
        IsError = false;
    }

    return IsError;
}

// Translate a file operation error code
//-----------------------------------------------------------------------------
CPUTResult CPUTOSServices::TranslateFileError(errno_t err)
{
    if(0==err)
    {
        return CPUT_SUCCESS;
    }

    // see: http://msdn.microsoft.com/en-us/library/t3ayayh1.aspx
    // for list of all error codes
    CPUTResult result = CPUT_ERROR_FILE_ERROR;

    switch(err)
    {
    case ENOENT: result = CPUT_ERROR_FILE_NOT_FOUND;                 break; // file/dir not found
    case EIO:    result = CPUT_ERROR_FILE_IO_ERROR;                  break;
    case ENXIO:  result = CPUT_ERROR_FILE_NO_SUCH_DEVICE_OR_ADDRESS; break;
    case EBADF:  result = CPUT_ERROR_FILE_BAD_FILE_NUMBER;           break;
    case ENOMEM: result = CPUT_ERROR_FILE_NOT_ENOUGH_MEMORY;         break;
    case EACCES: result = CPUT_ERROR_FILE_PERMISSION_DENIED;         break;
    case EBUSY:  result = CPUT_ERROR_FILE_DEVICE_OR_RESOURCE_BUSY;   break;
    case EEXIST: result = CPUT_ERROR_FILE_EXISTS;                    break;
    case EISDIR: result = CPUT_ERROR_FILE_IS_A_DIRECTORY;            break;
    case ENFILE: result = CPUT_ERROR_FILE_TOO_MANY_OPEN_FILES;       break;
    case EFBIG:  result = CPUT_ERROR_FILE_TOO_LARGE;                 break;
    case ENOSPC: result = CPUT_ERROR_FILE_DEVICE_FULL;               break;
    case ENAMETOOLONG: result = CPUT_ERROR_FILE_FILENAME_TOO_LONG;   break;
    default:
        // unknown file error type - assert so you can add it to the list
        ASSERT(0,_L("Unkown error code"));
    }
    return result;
}

#ifdef CPUT_GPA_INSTRUMENTATION
// Allows you to get the global/domain-wide instrumentation markers needed
// to mark events in GPA
//-----------------------------------------------------------------------------
void CPUTOSServices::GetInstrumentationPointers(__itt_domain **ppGPADomain, CPUT_GPA_INSTRUMENTATION_STRINGS eString, __itt_string_handle **ppGPAStringHandle)
{
    *ppGPADomain = mpGPADomain;
    *ppGPAStringHandle = mppGPAStringHandles[eString];
}

// Set the global/domain-wide instrumtation markers needed to mark events
// in GPA
//-----------------------------------------------------------------------------
void CPUTOSServices::SetInstrumentationPointers(__itt_domain *pGPADomain, CPUT_GPA_INSTRUMENTATION_STRINGS eString, __itt_string_handle *pGPAStringHandle)
{
    mpGPADomain = pGPADomain;
    mppGPAStringHandles[eString] = pGPAStringHandle;
}
#endif
