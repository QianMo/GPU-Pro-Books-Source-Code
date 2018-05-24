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
#ifndef _CPUTDOMAINSHADERDX11_H
#define _CPUTDOMAINSHADERDX11_H

#include "CPUT.h"
#include "CPUTShaderDX11.h"

class CPUTDomainShaderDX11 : public CPUTShaderDX11
{
protected:
    ID3D11DomainShader *mpDomainShader;

    // Destructor is not public.  Must release instead of delete.
    ~CPUTDomainShaderDX11(){ SAFE_RELEASE(mpDomainShader) }

public:
    static CPUTDomainShaderDX11 *CreateDomainShader(
        const cString        &name,
        ID3D11Device         *pD3dDevice,
        const cString        &shaderMain,
        const cString        &shaderProfile
    );
    static CPUTDomainShaderDX11 *CreateDomainShaderFromMemory(
        const cString        &name,
        ID3D11Device         *pD3dDevice,
        const cString        &shaderMain,
        const cString        &shaderProfile,
        const char           *pShaderSource
    );


    CPUTDomainShaderDX11() : mpDomainShader(NULL), CPUTShaderDX11(NULL) {}
    CPUTDomainShaderDX11(ID3D11DomainShader *pD3D11DomainShader, ID3DBlob *pBlob) : mpDomainShader(pD3D11DomainShader), CPUTShaderDX11(pBlob) {}
    ID3DBlob *GetBlob() { return mpBlob; }
    ID3D11DomainShader *GetNativeDomainShader() { return mpDomainShader; }
};

#endif //_CPUTDOMAINSHADER_H
