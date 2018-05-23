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
#include "CPUTITTTaskMarker.h"


#ifdef CPUT_GPA_INSTRUMENTATION
// Constructor
// automatically creates an ITT begin marker at the start of this task
//-----------------------------------------------------------------------------
CPUTITTTaskMarker::CPUTITTTaskMarker(__itt_domain *pITTDomain, __itt_string_handle *pITTStringHandle)
{
    mpITTDomain = pITTDomain;
    mpITTStringHandle = pITTStringHandle;

    __itt_task_begin(pITTDomain, __itt_null, __itt_null, pITTStringHandle);
}


// Destructor
// When the class goes out of scope, this marker will automatically be called
// and the domain 'closed'
//-----------------------------------------------------------------------------
CPUTITTTaskMarker::~CPUTITTTaskMarker()
{
    __itt_task_end(mpITTDomain);
}
#else
    // This is a bit of a hack to get the compiler not to complain about this being an empty file
    // during the compilation in any mode that doesn't have CPUT_GPA_INSTRUMENTATION defined
    #define CPUTITTTaskMarkerNotEmpty()   namespace { char CPUTITTTaskMarkerDummy##__LINE__; }
    CPUTITTTaskMarkerNotEmpty();
#endif
