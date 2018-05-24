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
#ifndef _CPUTVERTEXSHADERDX11_H
#define _CPUTVERTEXSHADERDX11_H

#include "CPUT.h"
#include "CPUTShaderDX11.h"

class CPUTVertexShaderDX11 : public CPUTShaderDX11
{
protected:
    ID3D11VertexShader *mpVertexShader;

    // Destructor is not public.  Must release instead of delete.
    ~CPUTVertexShaderDX11(){ SAFE_RELEASE(mpVertexShader) }

public:
    static CPUTVertexShaderDX11 *CreateVertexShader(
        const cString        &name,
        ID3D11Device         *pD3dDevice,
        const cString        &shaderMain,
        const cString        &shaderProfile
    );
static CPUTVertexShaderDX11 *CreateVertexShaderFromMemory(
        const cString        &name,
        ID3D11Device         *pD3dDevice,
        const cString        &shaderMain,
        const cString        &shaderProfile,
        const char           *pShaderSource
    );

    CPUTVertexShaderDX11() : mpVertexShader(NULL), CPUTShaderDX11(NULL) {}
    CPUTVertexShaderDX11(ID3D11VertexShader *pD3D11VertexShader, ID3DBlob *pBlob) : mpVertexShader(pD3D11VertexShader), CPUTShaderDX11(pBlob) {}
    ID3D11VertexShader *GetNativeVertexShader() { return mpVertexShader; }
};

#endif //_CPUTVERTEXSHADER_H
