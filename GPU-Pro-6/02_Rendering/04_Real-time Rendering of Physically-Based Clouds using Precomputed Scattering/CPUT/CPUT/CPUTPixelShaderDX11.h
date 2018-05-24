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
#ifndef _CPUTPIXELSHADERDX11_H
#define _CPUTPIXELSHADERDX11_H

#include "CPUT.h"
#include "CPUTShaderDX11.h"

class CPUTPixelShaderDX11 : public CPUTShaderDX11
{
protected:
    ID3D11PixelShader *mpPixelShader;

     // Destructor is not public.  Must release instead of delete.
    ~CPUTPixelShaderDX11(){ SAFE_RELEASE(mpPixelShader); }

public:
    static CPUTPixelShaderDX11 *CreatePixelShader(
        const cString        &name,
        ID3D11Device         *pD3dDevice,
        const cString        &shaderMain,
        const cString        &shaderProfile
    );

    static CPUTPixelShaderDX11 *CreatePixelShaderFromMemory(
        const cString        &name,
        ID3D11Device         *pD3dDevice,
        const cString        &shaderMain,
        const cString        &shaderProfile,
        const char           *pShaderSource
    );

    CPUTPixelShaderDX11() : mpPixelShader(NULL), CPUTShaderDX11(NULL) {}
    CPUTPixelShaderDX11(ID3D11PixelShader *pD3D11PixelShader, ID3DBlob *pBlob) : mpPixelShader(pD3D11PixelShader), CPUTShaderDX11(pBlob) {}
    ID3D11PixelShader *GetNativePixelShader() { return mpPixelShader; }
};

#endif //_CPUTPIXELSHADER_H
