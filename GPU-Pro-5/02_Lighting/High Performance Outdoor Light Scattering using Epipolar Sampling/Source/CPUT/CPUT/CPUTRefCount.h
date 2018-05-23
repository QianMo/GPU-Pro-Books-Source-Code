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
#ifndef __CPUTREFCOUNT_H__
#define __CPUTREFCOUNT_H__

#include "CPUT.h"

// Reference counting class
//-----------------------------------------------------------------------------
class CPUTRefCount
{
private:
    mutable UINT mRefCount;

protected:
    virtual ~CPUTRefCount(){} // Destructor is not public.  Must release instead of delete.

public:
    CPUTRefCount():mRefCount(1){}
    int AddRef() const { return ++mRefCount; }
    int GetRefCount() const { return mRefCount; }
    int Release() const
    {
        UINT u = --mRefCount;
        if(0==mRefCount)
        {
            delete this;
        }
        return u;
    }
};

#endif // __CPUTREFCOUNT_H__


