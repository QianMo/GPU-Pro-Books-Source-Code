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

#pragma once


#define LOG_ERROR(ErrorMsg, ...)\
{                                       \
    TCHAR FormattedErrorMsg[256];       \
    _stprintf_s(FormattedErrorMsg, sizeof(FormattedErrorMsg)/sizeof(FormattedErrorMsg[0]), ErrorMsg, __VA_ARGS__ ); \
    TCHAR FullErrorMsg[1024];           \
    _stprintf_s(FullErrorMsg, sizeof(FullErrorMsg)/sizeof(FullErrorMsg[0]), _T("The following error occured in the %s function() (%s, line %d):\n%s"), _T(__FUNCTION__), _T(__FILE__), __LINE__, FormattedErrorMsg); \
    MessageBox(NULL, FullErrorMsg, _T("Error"), MB_ICONERROR|MB_OK ); \
}

#define CHECK_HR(Result, ErrorMsg, ...)\
    if( FAILED(Result) )                \
        LOG_ERROR(ErrorMsg, __VA_ARGS__);

#define CHECK_HR_RET(Result, ErrorMsg, ...)\
    if( FAILED(Result) )                    \
    {                                       \
        LOG_ERROR(ErrorMsg, __VA_ARGS__);   \
        return Result;                      \
    }
