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
#ifndef _CPUTBUFFER_H
#define _CPUTBUFFER_H

#include "CPUT.h"
#include "CPUTRefCount.h"

// TODO: Move to dedicated file
class CPUTBuffer : public CPUTRefCount
{
protected:
    cString      mName;
    eCPUTMapType mMappedType;

    ~CPUTBuffer(){
        mName.clear();
    } // Destructor is not public.  Must release instead of delete.
public:
    CPUTBuffer(){mMappedType = CPUT_MAP_UNDEFINED;}
    CPUTBuffer(cString &name) {mName = name; mMappedType = CPUT_MAP_UNDEFINED;}
};

#endif //_CPUTBUFFER_H
