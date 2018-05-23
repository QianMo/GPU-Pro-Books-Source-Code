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
#ifndef _CPUTHULLSHADERDX11_H
#define _CPUTHULLSHADERDX11_H

#include "CPUT.h"
#include "CPUTShaderDX11.h"

class CPUTHullShaderDX11 : public CPUTShaderDX11
{
protected:
    ID3D11HullShader *mpHullShader;

    // Destructor is not public.  Must release instead of delete.
    ~CPUTHullShaderDX11(){ SAFE_RELEASE(mpHullShader); }

public:
    static CPUTHullShaderDX11 *CreateHullShader(
        const cString        &name,
        ID3D11Device         *pD3dDevice,
        const cString        &shaderMain,
        const cString        &shaderProfile
    );
    static CPUTHullShaderDX11 *CreateHullShaderFromMemory(
        const cString        &name,
        ID3D11Device         *pD3dDevice,
        const cString        &shaderMain,
        const cString        &shaderProfile,
        const char           *pShaderSource
    );


    CPUTHullShaderDX11() : mpHullShader(NULL), CPUTShaderDX11(NULL) {}
    CPUTHullShaderDX11(ID3D11HullShader *pD3D11HullShader, ID3DBlob *pBlob) : mpHullShader(pD3D11HullShader), CPUTShaderDX11(pBlob) {}
    ID3DBlob *GetBlob() { return mpBlob; }
    ID3D11HullShader *GetNativeHullShader() { return mpHullShader; }
};

#endif //_CPUTHULLSHADER_H
