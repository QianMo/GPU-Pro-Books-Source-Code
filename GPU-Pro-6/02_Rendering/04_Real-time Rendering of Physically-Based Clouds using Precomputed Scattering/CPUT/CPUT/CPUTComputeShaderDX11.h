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
#ifndef _CPUTCOMPUTESHADERDX11_H
#define _CPUTCOMPUTESHADERDX11_H

#include "CPUT.h"
#include "CPUTShaderDX11.h"

class CPUTComputeShaderDX11 : public CPUTShaderDX11
{
protected:
    ID3D11ComputeShader *mpComputeShader;

     // Destructor is not public.  Must release instead of delete.
    ~CPUTComputeShaderDX11(){ SAFE_RELEASE(mpComputeShader) }

public:
    static CPUTComputeShaderDX11 *CreateComputeShader(
        const cString  &name,
        ID3D11Device   *pD3dDevice,
        const cString  &shaderMain,
        const cString  &shaderProfile
    );

    static CPUTComputeShaderDX11 *CreateComputeShaderFromMemory(
        const cString        &name,
        ID3D11Device         *pD3dDevice,
        const cString        &shaderMain,
        const cString        &shaderProfile,
        const char           *pShaderSource
    );
    CPUTComputeShaderDX11() : mpComputeShader(NULL), CPUTShaderDX11(NULL) {}
    CPUTComputeShaderDX11(ID3D11ComputeShader *pD3D11ComputeShader, ID3DBlob *pBlob) : mpComputeShader(pD3D11ComputeShader), CPUTShaderDX11(pBlob) {}
    ID3DBlob *GetBlob() { return mpBlob; }
    ID3D11ComputeShader *GetNativeComputeShader() { return mpComputeShader; }
};

#endif //_CPUTCOMPUTESHADER_H
