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
#ifndef _CPUTTEXTUREDX11_H
#define _CPUTTEXTUREDX11_H

#include "CPUTTexture.h"
#include "CPUT_DX11.h"
#include <d3d11.h>
#include <d3DX11.h>

class CPUTTextureDX11 : public CPUTTexture
{
private:
    // resource view pointer
    CD3D11_TEXTURE2D_DESC     mDesc;
    ID3D11ShaderResourceView *mpShaderResourceView;
    ID3D11Resource           *mpTexture;
    ID3D11Resource           *mpTextureStaging;

    // Destructor is not public.  Must release instead of delete.
    ~CPUTTextureDX11() {
        SAFE_RELEASE( mpShaderResourceView );
        SAFE_RELEASE( mpTexture );
        SAFE_RELEASE( mpTextureStaging );
    }

public:
    static const cString &GetDXGIFormatString(DXGI_FORMAT Format);
    static CPUTResult     GetSRGBEquivalent(DXGI_FORMAT inFormat, DXGI_FORMAT& sRGBFormat);
    static bool           DoesExistEquivalentSRGBFormat(DXGI_FORMAT inFormat);
    static CPUTTexture   *CreateTexture( const cString &name, const cString &absolutePathAndFilename, bool loadAsSRGB );
    static CPUTResult     CreateNativeTexture(
                              ID3D11Device *pD3dDevice,
                              const cString &fileName,
                              ID3D11ShaderResourceView **ppShaderResourceView,
                              ID3D11Resource **ppTexture,
                              bool forceLoadAsSRGB
                          );

    CPUTTextureDX11() :
        mpShaderResourceView(NULL),
        mpTexture(NULL),
        mpTextureStaging(NULL)
    {}
    CPUTTextureDX11(cString &name) :
        mpShaderResourceView(NULL),
        mpTexture(NULL),
        mpTextureStaging(NULL),
        CPUTTexture(name)
    {}
    CPUTTextureDX11(cString &name, ID3D11Resource *pTextureResource, ID3D11ShaderResourceView *pSrv ) :
        mpTextureStaging(NULL),
        CPUTTexture(name)
    {
        mpShaderResourceView = pSrv;
        if(mpShaderResourceView) pSrv->AddRef();
        mpTexture = pTextureResource;
        if(mpTexture) mpTexture->AddRef();
    }

    void ReleaseTexture()
    {
        SAFE_RELEASE(mpShaderResourceView);
        SAFE_RELEASE(mpTexture);
    }
    void SetTexture(ID3D11Resource *pTextureResource, ID3D11ShaderResourceView *pSrv )
    {
        mpShaderResourceView = pSrv;
        if(mpShaderResourceView) pSrv->AddRef();

        mpTexture = pTextureResource;
        if(mpTexture) mpTexture->AddRef();
    }

    ID3D11ShaderResourceView* GetShaderResourceView()
    {
        return mpShaderResourceView;
    }

    void SetTextureAndShaderResourceView(ID3D11Resource *pTexture, ID3D11ShaderResourceView *pShaderResourceView)
    {
        // release any resources we might already be pointing too
        SAFE_RELEASE( mpTexture );
        SAFE_RELEASE( mpTextureStaging ); // Now out-of sync.  Will be recreated on next Map().
        SAFE_RELEASE( mpShaderResourceView );
        mpTexture = pTexture;
        if( mpTexture ) mpTexture->AddRef();
        mpShaderResourceView = pShaderResourceView;
        mpShaderResourceView->AddRef();
    }
    D3D11_MAPPED_SUBRESOURCE  MapTexture(   CPUTRenderParameters &params, eCPUTMapType type, bool wait=true );
    void                      UnmapTexture( CPUTRenderParameters &params );
};

#endif //_CPUTTEXTUREDX11_H

