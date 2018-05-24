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
#ifndef __CPUTPERFTASKMARKER_H__
#define __CPUTPERFTASKMARKER_H__



#include "CPUT.h"


#ifdef CPUT_GPA_INSTRUMENTATION

// GPA instrumentation helper class - only available in profile build
// Allows you to easily add 'task markers' to certain events
//-----------------------------------------------------------------------------
class CPUTPerfTaskMarker
{
public:
    CPUTPerfTaskMarker(DWORD color, wchar_t *pString);
    ~CPUTPerfTaskMarker();
private:

};
#endif

#endif // #ifndef __CPUTPERFTASKMARKER_H__