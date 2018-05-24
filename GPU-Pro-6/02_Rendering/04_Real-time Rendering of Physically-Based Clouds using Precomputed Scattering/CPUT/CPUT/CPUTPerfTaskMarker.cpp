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
#include "CPUTPerfTaskMarker.h"

#ifdef CPUT_GPA_INSTRUMENTATION
// constructor
// automatically creates an ITT begin marker at the start of this task
//-----------------------------------------------------------------------------
CPUTPerfTaskMarker::CPUTPerfTaskMarker(DWORD color, wchar_t *pString)
{
    D3DPERF_BeginEvent(color, pString);
}

// destructor
// when class goes out of scope, this marker will automatically be called
//-----------------------------------------------------------------------------
CPUTPerfTaskMarker::~CPUTPerfTaskMarker()
{
    D3DPERF_EndEvent();
}
#else
    // This is a bit of a hack to get the compiler not to complain about this being an empty file
    // during the compilation in any mode that doesn't have CPUT_GPA_INSTRUMENTATION defined
    #define CPUTPerfTaskMarkerNotEmpty()   namespace { char CPUTPerfTaskMarkerDummy##__LINE__; }
    CPUTPerfTaskMarkerNotEmpty();
#endif
