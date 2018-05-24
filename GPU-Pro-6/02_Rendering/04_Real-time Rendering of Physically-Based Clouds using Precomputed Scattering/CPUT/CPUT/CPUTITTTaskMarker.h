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
#ifndef __CPUTITTTASKMARKER_H__
#define __CPUTITTTASKMARKER_H__



#include "CPUT.h"

#ifdef CPUT_GPA_INSTRUMENTATION

// GPA ITT instrumentation helper class - only available in profile build
// Automatically open/close marks an ITT marker event for GPA
//-----------------------------------------------------------------------------
class CPUTITTTaskMarker
{
public:
    CPUTITTTaskMarker(__itt_domain *pITTDomain, __itt_string_handle *pITTStringHandle);
    ~CPUTITTTaskMarker();
private:
    __itt_domain *mpITTDomain;
    __itt_string_handle *mpITTStringHandle;
};

#endif
#endif // #ifndef __CPUTITTTASKMARKER_H__