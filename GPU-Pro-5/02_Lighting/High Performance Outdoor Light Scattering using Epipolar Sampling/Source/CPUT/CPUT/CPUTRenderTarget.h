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
#ifndef _CPUTRENDERTARGET_H
#define _CPUTRENDERTARGET_H

#include "CPUT.h"
#include "d3d11.h"

class CPUTMaterial;
class CPUTRenderParameters;
class CPUTTexture;
class CPUTBuffer;
class CPUTRenderTargetDepth;

class CPUTRenderTargetColor
{
public:
    static ID3D11RenderTargetView *GetActiveRenderTargetView() { return spActiveRenderTargetView; }
    static void                    SetActiveRenderTargetView(ID3D11RenderTargetView *pView) { spActiveRenderTargetView = pView; }
    static void                    SetActiveWidthHeight( UINT width, UINT height ) {sCurrentWidth = width; sCurrentHeight=height; }
    static UINT                    GetActiveWidth()  {return sCurrentWidth; }
    static UINT                    GetActiveHeight() {return sCurrentHeight; }

protected:
    static UINT                    sCurrentWidth;
    static UINT                    sCurrentHeight;
    static ID3D11RenderTargetView *spActiveRenderTargetView;

    cString                        mName;
    UINT                           mWidth;
    UINT                           mHeight;
    bool                           mRenderTargetSet;
    bool                           mHasUav;
                                  
    DXGI_FORMAT                    mColorFormat;
    UINT                           mMultiSampleCount;
                                  
    CD3D11_TEXTURE2D_DESC          mColorDesc;
    CPUTTexture                   *mpColorTexture;
    CPUTBuffer                    *mpColorBuffer;
    ID3D11Texture2D               *mpColorTextureDX;
    ID3D11ShaderResourceView      *mpColorSRV;
    ID3D11UnorderedAccessView     *mpColorUAV;
    ID3D11RenderTargetView        *mpColorRenderTargetView;

    CPUTTexture                   *mpColorTextureMSAA;
    ID3D11Texture2D               *mpColorTextureDXMSAA;
    ID3D11ShaderResourceView      *mpColorSRVMSAA;

    ID3D11Texture2D               *mpColorTextureDXStaging;
    eCPUTMapType                   mMappedType;

    UINT                           mSavedWidth;
    UINT                           mSavedHeight;
    ID3D11RenderTargetView        *mpSavedColorRenderTargetView;
    ID3D11DepthStencilView        *mpSavedDepthStencilView;

public:
    CPUTRenderTargetColor() :
        mWidth(0),
        mHeight(0),
        mRenderTargetSet(false),
        mHasUav(false),
        mColorFormat(DXGI_FORMAT_UNKNOWN),
        mMultiSampleCount(1),
        mpColorTexture(NULL),
        mpColorBuffer(NULL),
        mpColorTextureDX(NULL),
        mpColorTextureDXMSAA(NULL),
        mpColorTextureMSAA(NULL),
        mpColorSRV(NULL),
        mpColorUAV(NULL),
        mpColorSRVMSAA(NULL),
        mpColorTextureDXStaging(NULL),
        mpColorRenderTargetView(NULL),
        mSavedWidth(0),
        mSavedHeight(0),
        mpSavedColorRenderTargetView(NULL),
        mpSavedDepthStencilView(NULL)
    {
    }

    ~CPUTRenderTargetColor();

    HRESULT CreateRenderTarget(
        cString     textureName,
        UINT        width,
        UINT        height,
        DXGI_FORMAT colorFormat,
        UINT        multiSampleCount = 1,
        bool        createUAV = false,
        bool        recreate = false
    );

    HRESULT RecreateRenderTarget(
        UINT        width,
        UINT        height,
        DXGI_FORMAT colorFormat = DXGI_FORMAT_UNKNOWN,
        UINT        multiSampleCount = 1
    );

    void SetRenderTarget(
        CPUTRenderParameters &renderParams,
        CPUTRenderTargetDepth *pDepthBuffer = NULL,
        DWORD renderTargetIndex=0,
        const float *pClearColor=NULL,
        bool  clear = false,
        float zClearVal = 0.0f
    );

    void RestoreRenderTarget( CPUTRenderParameters &renderParams );
    void Resolve( CPUTRenderParameters &renderParams );

    CPUTTexture               *GetColorTexture()            { return mpColorTexture; }
    CPUTBuffer                *GetColorBuffer()             { return mpColorBuffer; }
    ID3D11ShaderResourceView  *GetColorResourceView()       { return mpColorSRV; }
    ID3D11UnorderedAccessView *GetColorUAV()                { return mpColorUAV; }
    UINT                       GetWidth()                   { return mWidth; }
    UINT                       GetHeight()                  { return mHeight; }
    DXGI_FORMAT                GetColorFormat()             { return mColorFormat; }

    D3D11_MAPPED_SUBRESOURCE  MapRenderTarget(   CPUTRenderParameters &params, eCPUTMapType type, bool wait=true );
    void                      UnmapRenderTarget( CPUTRenderParameters &params );
};

//--------------------------------------------------------------------------------------
class CPUTRenderTargetDepth
{
public:
    static ID3D11DepthStencilView *GetActiveDepthStencilView() { return spActiveDepthStencilView; }
    static void                    SetActiveDepthStencilView(ID3D11DepthStencilView *pView) { spActiveDepthStencilView = pView; }
    static void                    SetActiveWidthHeight( UINT width, UINT height ) {sCurrentWidth = width; sCurrentHeight=height; }
    static UINT                    GetActiveWidth()  {return sCurrentWidth; }
    static UINT                    GetActiveHeight() {return sCurrentHeight; }

protected:
    static UINT                    sCurrentWidth;
    static UINT                    sCurrentHeight;
    static ID3D11DepthStencilView *spActiveDepthStencilView;

    cString                        mName;
    UINT                           mWidth;
    UINT                           mHeight;
    bool                           mRenderTargetSet;
                                  
    DXGI_FORMAT                    mDepthFormat;
    UINT                           mMultiSampleCount;
                                                                 
    CD3D11_TEXTURE2D_DESC          mDepthDesc;
    CPUTTexture                   *mpDepthTexture;
    ID3D11Texture2D               *mpDepthTextureDX;
    ID3D11ShaderResourceView      *mpDepthResourceView;
    ID3D11DepthStencilView        *mpDepthStencilView;

    ID3D11Texture2D               *mpDepthTextureDXStaging;
    eCPUTMapType                   mMappedType;

    UINT                           mSavedWidth;
    UINT                           mSavedHeight;
    ID3D11RenderTargetView        *mpSavedColorRenderTargetView;
    ID3D11DepthStencilView        *mpSavedDepthStencilView;

public:
    CPUTRenderTargetDepth() :
        mWidth(0),
        mHeight(0),
        mRenderTargetSet(false),
        mDepthFormat(DXGI_FORMAT_UNKNOWN),
        mMultiSampleCount(1),
        mpDepthTexture(NULL),
        mpDepthTextureDX(NULL),
        mpDepthResourceView(NULL),
        mpDepthStencilView(NULL),
        mpDepthTextureDXStaging(NULL),
        mSavedWidth(0),
        mSavedHeight(0),
        mpSavedColorRenderTargetView(NULL),
        mpSavedDepthStencilView(NULL)
    {
    }
    ~CPUTRenderTargetDepth();

    HRESULT CreateRenderTarget(
        cString     textureName,
        UINT        width,
        UINT        height,
        DXGI_FORMAT depthFormat,
        UINT        multiSampleCount = 1,
        bool        recreate = false
    );

    HRESULT RecreateRenderTarget(
        UINT        width,
        UINT        height,
        DXGI_FORMAT depthFormat = DXGI_FORMAT_UNKNOWN,
        UINT        multiSampleCount = 1
    );

    void SetRenderTarget(
        CPUTRenderParameters &renderParams,
        DWORD renderTargetIndex = 0,
        float zClearVal = 0.0f,
        bool  clear = false
    );

    void RestoreRenderTarget( CPUTRenderParameters &renderParams );

    ID3D11DepthStencilView   *GetDepthBufferView()   { return mpDepthStencilView; }
    ID3D11ShaderResourceView *GetDepthResourceView() { return mpDepthResourceView; }
    UINT                      GetWidth()             { return mWidth; }
    UINT                      GetHeight()            { return mHeight; }

    D3D11_MAPPED_SUBRESOURCE  MapRenderTarget(   CPUTRenderParameters &params, eCPUTMapType type, bool wait=true );
    void                      UnmapRenderTarget( CPUTRenderParameters &params );
};

#endif // _CPUTRENDERTARGET_H
