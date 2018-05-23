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
#ifndef _CPUTRENDERSTATEBLOCK_H
#define _CPUTRENDERSTATEBLOCK_H

#include "CPUT.h"
#include "CPUTRefCount.h"

class CPUTRenderParameters;

class CPUTRenderStateBlock : public CPUTRefCount
{
protected:
    static CPUTRenderStateBlock *mpDefaultRenderStateBlock;

    cString mMaterialName;
    
    ~CPUTRenderStateBlock(){} // Destructor is not public.  Must release instead of delete.

public:
    static CPUTRenderStateBlock *CreateRenderStateBlock( const cString &name, const cString &absolutePathAndFilename );
    static CPUTRenderStateBlock *GetDefaultRenderStateBlock() { return mpDefaultRenderStateBlock; }
    static void SetDefaultRenderStateBlock( CPUTRenderStateBlock *pBlock ) { SAFE_RELEASE( mpDefaultRenderStateBlock ); mpDefaultRenderStateBlock = pBlock; }


    CPUTRenderStateBlock(){}
    virtual CPUTResult LoadRenderStateBlock(const cString &fileName) = 0;
    virtual void SetRenderStates(CPUTRenderParameters &renderParams) = 0;
    virtual void CreateNativeResources() = 0;
};

#endif // _CPUTRENDERSTATEBLOCK_H
