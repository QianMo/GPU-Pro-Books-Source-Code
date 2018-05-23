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
#ifndef _CPUTSHADERDX11_H
#define _CPUTSHADERDX11_H

#include "CPUT.h"
#include "CPUTRefCount.h"

class CPUTConfigBlock;

class CPUTShaderDX11 : public CPUTRefCount
{
protected:
    ID3DBlob          *mpBlob;

     // Destructor is not public.  Must release instead of delete.
    ~CPUTShaderDX11(){ SAFE_RELEASE(mpBlob); }

public:
    CPUTShaderDX11() : mpBlob(NULL) {}
    CPUTShaderDX11(ID3DBlob *pBlob) : mpBlob(pBlob) {}
    ID3DBlob *GetBlob() { return mpBlob; }

    bool ShaderRequiresPerModelPayload( CPUTConfigBlock &properties );
};

#endif //_CPUTPIXELSHADER_H
