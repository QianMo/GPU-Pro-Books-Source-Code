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
#ifndef _CPUTBUFFERDX11_H
#define _CPUTBUFFERDX11_H

#include "CPUTBuffer.h"
#include "CPUT_DX11.h"

//--------------------------------------------------------------------------------------
// TODO: Move to dedicated file
class CPUTBufferDX11 : public CPUTBuffer
{
private:
    // resource view pointer
    ID3D11ShaderResourceView  *mpShaderResourceView;
    ID3D11UnorderedAccessView *mpUnorderedAccessView;
    ID3D11Buffer              *mpBuffer;
    ID3D11Buffer              *mpBufferStaging;

    // Destructor is not public.  Must release instead of delete.
    ~CPUTBufferDX11() {
        SAFE_RELEASE( mpShaderResourceView );
        SAFE_RELEASE( mpUnorderedAccessView );
        SAFE_RELEASE( mpBuffer );
        SAFE_RELEASE( mpBufferStaging );
    }

public:
    CPUTBufferDX11() :
        mpShaderResourceView(NULL),
        mpUnorderedAccessView(NULL),
        mpBuffer(NULL),
        mpBufferStaging(NULL)
    {
    }
    CPUTBufferDX11(cString &name, ID3D11Buffer *pBuffer) :
        mpBuffer(pBuffer),
        mpBufferStaging(NULL),
        mpShaderResourceView(NULL),
        mpUnorderedAccessView(NULL),
        CPUTBuffer(name)
    {
        if(pBuffer) pBuffer->AddRef();
    }

    CPUTBufferDX11(cString &name, ID3D11Buffer *pBuffer, ID3D11ShaderResourceView *pView) :
        mpBuffer(pBuffer),
        mpBufferStaging(NULL),
        mpShaderResourceView(pView),
        mpUnorderedAccessView(NULL),
        CPUTBuffer(name)
    {
        if(pBuffer) pBuffer->AddRef();
        if(pView) pView->AddRef();
    }

    CPUTBufferDX11(cString &name, ID3D11Buffer *pBuffer, ID3D11UnorderedAccessView *pView) :
        mpBuffer(pBuffer),
        mpBufferStaging(NULL),
        mpShaderResourceView(NULL),
        mpUnorderedAccessView(pView),
        CPUTBuffer(name)
    {
        if(pBuffer) pBuffer->AddRef();
        if(pView) pView->AddRef();
    }

    ID3D11ShaderResourceView *GetShaderResourceView()
    {
        return mpShaderResourceView;
    }

    ID3D11UnorderedAccessView *GetUnorderedAccessView()
    {
        return mpUnorderedAccessView;
    }

    void SetShaderResourceView(ID3D11ShaderResourceView *pShaderResourceView)
    {
        // release any resource view we might already be pointing too
        SAFE_RELEASE( mpShaderResourceView );
        mpShaderResourceView = pShaderResourceView;
        mpShaderResourceView->AddRef();
    }
    void SetUnorderedAccessView(ID3D11UnorderedAccessView *pUnorderedAccessView)
    {
        // release any resource view we might already be pointing too
        SAFE_RELEASE( mpUnorderedAccessView );
        mpUnorderedAccessView = pUnorderedAccessView;
        mpUnorderedAccessView->AddRef();
    }
    void SetBufferAndViews(ID3D11Buffer *pBuffer, ID3D11ShaderResourceView *pShaderResourceView, ID3D11UnorderedAccessView *pUnorderedAccessView )
    {
        SAFE_RELEASE(mpBuffer);
        mpBuffer = pBuffer;
        if(mpBuffer) mpBuffer->AddRef();

        // release any resource view we might already be pointing too
        SAFE_RELEASE( mpShaderResourceView );
        mpShaderResourceView = pShaderResourceView;
        if(mpShaderResourceView) mpShaderResourceView->AddRef();

        // release any resource view we might already be pointing too
        SAFE_RELEASE( mpUnorderedAccessView );
        mpUnorderedAccessView = pUnorderedAccessView;
        if(mpUnorderedAccessView) mpUnorderedAccessView->AddRef();
    }
    ID3D11Buffer *GetNativeBuffer() { return mpBuffer; }
    D3D11_MAPPED_SUBRESOURCE  MapBuffer(   CPUTRenderParameters &params, eCPUTMapType type, bool wait=true );
    void                      UnmapBuffer( CPUTRenderParameters &params );
    void ReleaseBuffer()
    {
        SAFE_RELEASE(mpShaderResourceView);
        SAFE_RELEASE(mpUnorderedAccessView);
        SAFE_RELEASE(mpBuffer);
        SAFE_RELEASE(mpBufferStaging);
    }
};
#endif //_CPUTBUFFERDX11_H

