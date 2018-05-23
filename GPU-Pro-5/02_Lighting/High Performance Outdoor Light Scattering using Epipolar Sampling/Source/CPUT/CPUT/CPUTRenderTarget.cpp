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

#include "CPUT_DX11.h"
#include "CPUTRenderTarget.h"
#include "CPUTAssetLibrary.h"
#include "CPUTMaterialDX11.h"
#include "CPUTTextureDX11.h"
#include "CPUTBufferDX11.h"

UINT CPUTRenderTargetColor::sCurrentWidth   = 0;
UINT CPUTRenderTargetColor::sCurrentHeight  = 0;
UINT CPUTRenderTargetDepth::sCurrentWidth   = 0;
UINT CPUTRenderTargetDepth::sCurrentHeight  = 0;

ID3D11RenderTargetView *CPUTRenderTargetColor::spActiveRenderTargetView = NULL;
ID3D11DepthStencilView *CPUTRenderTargetDepth::spActiveDepthStencilView = NULL;

//-----------------------------------------------
CPUTRenderTargetColor::~CPUTRenderTargetColor()
{
    SAFE_RELEASE( mpColorTexture );
    SAFE_RELEASE( mpColorUAV );
    SAFE_RELEASE( mpColorBuffer );
    SAFE_RELEASE( mpColorTextureDX );
    SAFE_RELEASE( mpColorTextureDXMSAA );
    SAFE_RELEASE( mpColorSRV );
    SAFE_RELEASE( mpColorRenderTargetView );

    SAFE_RELEASE( mpColorTextureMSAA );
    SAFE_RELEASE( mpColorTextureDXMSAA );
    SAFE_RELEASE( mpColorSRVMSAA );
    SAFE_RELEASE( mpColorTextureDXStaging );
}

//-----------------------------------------------
CPUTRenderTargetDepth::~CPUTRenderTargetDepth()
{
    SAFE_RELEASE( mpDepthTexture );
    SAFE_RELEASE( mpDepthTextureDX );
    SAFE_RELEASE( mpDepthResourceView );
    SAFE_RELEASE( mpDepthStencilView );
}

//-----------------------------------------------
HRESULT CPUTRenderTargetColor::CreateRenderTarget(
    cString     textureName,
    UINT        width,
    UINT        height,
    DXGI_FORMAT colorFormat,
    UINT        multiSampleCount,
    bool        createUAV,
    bool        recreate
)
{
    HRESULT result;

    mName             = textureName;
    mWidth            = width;
    mHeight           = height;
    mColorFormat      = colorFormat;
    mMultiSampleCount = multiSampleCount;

    CPUTAssetLibrary *pAssetLibrary = CPUTAssetLibrary::GetAssetLibrary();
    CPUTOSServices   *pServices     = CPUTOSServices::GetOSServices();

    // Create the color texture
    mColorDesc = CD3D11_TEXTURE2D_DESC(
        colorFormat,
        width,
        height,
        1, // Array Size
        1, // MIP Levels
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
        D3D11_USAGE_DEFAULT,
        0,
        mMultiSampleCount, 0
    );

    ID3D11Device *pD3dDevice = CPUT_DX11::GetDevice();

    // If MSAA enabled, then create MSAA texture
    if( mMultiSampleCount>1 )
    {
        result = pD3dDevice->CreateTexture2D( &mColorDesc, NULL, &mpColorTextureDXMSAA );
        ASSERT( SUCCEEDED(result), _L("Failed creating MSAA render target texture") );

        D3D11_SHADER_RESOURCE_VIEW_DESC srDesc = { colorFormat, D3D11_SRV_DIMENSION_TEXTURE2DMS, 0 };
        srDesc.Texture2D.MipLevels = 1;
        result = pD3dDevice->CreateShaderResourceView( mpColorTextureDXMSAA, &srDesc, &mpColorSRVMSAA );
        ASSERT( SUCCEEDED(result), _L("Failed creating MSAA render target shader resource view") );
		CPUTSetDebugName( mpColorSRVMSAA, textureName + _L(" ColorMSAA") );

        if( !recreate )
        {
            cString msaaName = mName + _L("_MSAA");
            mpColorTextureMSAA = new CPUTTextureDX11( msaaName );

            // If the name starts with a '$', then its an internal texture (has no filesystem path).
            // Otherwise, its a file, so prepend filesystem path
            CPUTAssetLibrary *pAssetLibrary = CPUTAssetLibrary::GetAssetLibrary();
            cString finalName;
            if( mName.at(0) == '$' )
            {
                finalName = msaaName;
            }else
            {
                pServices->ResolveAbsolutePathAndFilename( (pAssetLibrary->GetTextureDirectory() + msaaName), &finalName);
            }
            pAssetLibrary->AddTexture( finalName, mpColorTextureMSAA );
        }
        ((CPUTTextureDX11*)mpColorTextureMSAA)->SetTextureAndShaderResourceView( mpColorTextureDXMSAA, mpColorSRVMSAA );
    }
    
    // Create non-MSAA texture.  If we're MSAA, then we'll resolve into this.  If not, then we'll render directly to this one.
    mColorDesc.SampleDesc.Count = 1;
    result = pD3dDevice->CreateTexture2D( &mColorDesc, NULL, &mpColorTextureDX );
    ASSERT( SUCCEEDED(result), _L("Failed creating render target texture") );

    // Create the shader-resource view from the non-MSAA texture
    D3D11_SHADER_RESOURCE_VIEW_DESC srDesc = { colorFormat, D3D11_SRV_DIMENSION_TEXTURE2D, 0 };
    srDesc.Texture2D.MipLevels = 1;
    result = pD3dDevice->CreateShaderResourceView( mpColorTextureDX, &srDesc, &mpColorSRV );
    ASSERT( SUCCEEDED(result), _L("Failed creating render target shader resource view") );
	CPUTSetDebugName( mpColorSRV, textureName + _L(" Color") );

    mHasUav = createUAV; // Remember, so we know to recreate it (or not) on RecreateRenderTarget()
    if( createUAV )
    {
        // D3D11_SHADER_RESOURCE_VIEW_DESC srDesc = { colorFormat, D3D_SRV_DIMENSION_BUFFER, 0 };
        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
        memset( &uavDesc, 0, sizeof(uavDesc) );
        uavDesc.Format = colorFormat;
        uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Texture2D.MipSlice = 0;
        result = pD3dDevice->CreateUnorderedAccessView( mpColorTextureDX, &uavDesc, &mpColorUAV );
        ASSERT( SUCCEEDED(result), _L("Failed creating render target buffer shader resource view") );
	    CPUTSetDebugName( mpColorUAV, textureName + _L(" Color Buffer") );
    }
    if( !recreate )
    {
        mpColorTexture = new CPUTTextureDX11(mName);
        pAssetLibrary->AddTexture( mName, mpColorTexture );

        mpColorBuffer = new CPUTBufferDX11(mName, NULL, mpColorUAV); // We don't have an ID3D11Buffer, but we want to track this UAV as if the texture was a buffer.
        pAssetLibrary->AddBuffer( mName, mpColorBuffer );
    }
    ((CPUTTextureDX11*)mpColorTexture)->SetTextureAndShaderResourceView( mpColorTextureDX, mpColorSRV);

    // Choose our render target.  If MSAA, then use the MSAA texture, and use resolve to fill the non-MSAA texture.
    ID3D11Texture2D *pColorTexture = (mMultiSampleCount>1) ? mpColorTextureDXMSAA : mpColorTextureDX;
    result = pD3dDevice->CreateRenderTargetView( pColorTexture, NULL, &mpColorRenderTargetView );
    ASSERT( SUCCEEDED(result), _L("Failed creating render target view") );
	CPUTSetDebugName( mpColorRenderTargetView, mName );

    return S_OK;
}

//-----------------------------------------------
HRESULT CPUTRenderTargetColor::RecreateRenderTarget(
    UINT        width,
    UINT        height,
    DXGI_FORMAT colorFormat,
    UINT        multiSampleCount
)
{
    // We don't release these.  Instead, we release their resource views, and change them to the newly-created versions.
    // SAFE_RELEASE( mpColorTexture );
    SAFE_RELEASE( mpColorTextureDX );
    SAFE_RELEASE( mpColorSRV );
    SAFE_RELEASE( mpColorRenderTargetView );
    SAFE_RELEASE( mpColorTextureDXMSAA );
    SAFE_RELEASE( mpColorSRVMSAA );

    // TODO: Complete buffer changes by including them here.


    // Do not release saved resource views since they are not addreff'ed
    //SAFE_RELEASE( mpSavedColorRenderTargetView );
    //SAFE_RELEASE( mpSavedDepthStencilView );

    return CreateRenderTarget( mName, width, height, (DXGI_FORMAT_UNKNOWN != colorFormat) ? colorFormat : mColorFormat, multiSampleCount, mHasUav, true );
}

//-----------------------------------------------
HRESULT CPUTRenderTargetDepth::CreateRenderTarget(
    cString     textureName,
    UINT        width,
    UINT        height,
    DXGI_FORMAT depthFormat,
    UINT        multiSampleCount,
    bool        recreate
)
{
    HRESULT result;

    mName             = textureName;
    mWidth            = width;
    mHeight           = height;
    mDepthFormat      = depthFormat;
    mMultiSampleCount = multiSampleCount;


    // NOTE: The following doesn't work for DX10.0 devices.
    // They don't support binding an MSAA depth texture as 

    // If we have a DX 10.1 or no MSAA, then create a shader resource view, and add a CPUTTexture to the AssetLibrary
    D3D_FEATURE_LEVEL featureLevel = gpSample->GetFeatureLevel();
    bool supportsResourceView = ( featureLevel >= D3D_FEATURE_LEVEL_10_1) || (mMultiSampleCount==1);

    D3D11_TEXTURE2D_DESC depthDesc = {
        width,
        height,
        1, // MIP Levels
        1, // Array Size
        DXGI_FORMAT(depthFormat - 1), //  DXGI_FORMAT_R32_TYPELESS
        mMultiSampleCount, 0,
        D3D11_USAGE_DEFAULT,
        D3D11_BIND_DEPTH_STENCIL | (supportsResourceView ? D3D11_BIND_SHADER_RESOURCE : 0),
        0, // CPU Access flags
        0 // Misc flags
    };

    // Create either a Texture2D, or Texture2DMS, depending on multisample count.
    D3D11_DSV_DIMENSION dsvDimension = (mMultiSampleCount>1) ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D;

    ID3D11Device *pD3dDevice = CPUT_DX11::GetDevice();
    result = pD3dDevice->CreateTexture2D( &depthDesc, NULL, &mpDepthTextureDX );
    ASSERT( SUCCEEDED(result), _L("Failed creating depth texture.\nAre you using MSAA with a DX10.0 GPU?\nOnly DX10.1 and above can create a shader resource view for an MSAA depth texture.") );

    D3D11_DEPTH_STENCIL_VIEW_DESC dsvd = { depthFormat, dsvDimension, 0 };
    result = pD3dDevice->CreateDepthStencilView( mpDepthTextureDX, &dsvd, &mpDepthStencilView );
    ASSERT( SUCCEEDED(result), _L("Failed creating depth stencil view") );
	CPUTSetDebugName( mpDepthStencilView, mName );

    if( supportsResourceView )
    {
        D3D11_SRV_DIMENSION srvDimension = (mMultiSampleCount>1) ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
        // Create the shader-resource view
        D3D11_SHADER_RESOURCE_VIEW_DESC depthRsDesc =
        {
            DXGI_FORMAT(depthFormat + 1),
            srvDimension,
            0
        };
        // TODO: Support optionally creating MIP chain.  Then, support MIP generation (e.g., GenerateMIPS()).
        depthRsDesc.Texture2D.MipLevels = 1;

        result = pD3dDevice->CreateShaderResourceView( mpDepthTextureDX, &depthRsDesc, &mpDepthResourceView );
        ASSERT( SUCCEEDED(result), _L("Failed creating render target shader resource view") );
		CPUTSetDebugName( mpDepthResourceView, textureName + _L(" Depth") );

        if( !recreate )
        {
            CPUTAssetLibrary *pAssetLibrary = CPUTAssetLibrary::GetAssetLibrary();

            mpDepthTexture = new CPUTTextureDX11(mName);
            pAssetLibrary->AddTexture( mName, mpDepthTexture );
        }
        ((CPUTTextureDX11*)mpDepthTexture)->SetTextureAndShaderResourceView(mpDepthTextureDX, mpDepthResourceView);
    }
    return S_OK;
}

//-----------------------------------------------
HRESULT CPUTRenderTargetDepth::RecreateRenderTarget(
    UINT        width,
    UINT        height,
    DXGI_FORMAT depthFormat,
    UINT        multiSampleCount
)
{
    // We don't release these.  Instead, we release their resource views, and change them to the newly-created versions.
    //SAFE_RELEASE( mpDepthTexture );
    SAFE_RELEASE( mpDepthTextureDX );
    SAFE_RELEASE( mpDepthResourceView );
    SAFE_RELEASE( mpDepthStencilView );
    // Do not release saved resource views since they are not addreff'ed
    //SAFE_RELEASE( mpSavedColorRenderTargetView );
    //SAFE_RELEASE( mpSavedDepthStencilView );

    return CreateRenderTarget( mName, width, height, (DXGI_FORMAT_UNKNOWN != depthFormat) ? depthFormat : mDepthFormat, multiSampleCount, true );
}

//-----------------------------------------------
void CPUTRenderTargetColor::SetRenderTarget(
    CPUTRenderParameters   &renderParams,
    CPUTRenderTargetDepth  *pDepthBuffer,
    DWORD                   renderTargetIndex,
    const float            *pClearColor,
    bool                    clear,
    float                   zClearVal
)
{
    // ****************************
    // Save the current render target "state" so we can restore it later.
    // ****************************
    mSavedWidth   = CPUTRenderTargetColor::sCurrentWidth;
    mSavedHeight  = CPUTRenderTargetColor::sCurrentHeight;

    // Save the render target view so we can restore it later.
    mpSavedColorRenderTargetView = CPUTRenderTargetColor::GetActiveRenderTargetView();
    mpSavedDepthStencilView      = CPUTRenderTargetDepth::GetActiveDepthStencilView();

    CPUTRenderTargetColor::SetActiveWidthHeight( mWidth, mHeight );
    CPUTRenderTargetDepth::SetActiveWidthHeight( mWidth, mHeight );

    // TODO: support multiple render target views (i.e., MRT)
    ID3D11DeviceContext *pContext = ((CPUTRenderParametersDX*)&renderParams)->mpContext;

    // Make sure this render target isn't currently bound as a texture.
    static ID3D11ShaderResourceView *pSRV[16] = {0};
    pContext->PSSetShaderResources( 0, 16, pSRV );

    // Clear the shader resources to avoid a hazard warning
    ID3D11ShaderResourceView *pNullResources[16] = {0};
    pContext->PSSetShaderResources(0, 16, pNullResources );
    pContext->VSSetShaderResources(0, 16, pNullResources );

    // ****************************
    // Set the new render target states
    // ****************************
    ID3D11DepthStencilView *pDepthStencilView = pDepthBuffer ? pDepthBuffer->GetDepthBufferView() : NULL;
    pContext->OMSetRenderTargets( 1, &mpColorRenderTargetView, pDepthStencilView );

    CPUTRenderTargetColor::SetActiveRenderTargetView(mpColorRenderTargetView);
    CPUTRenderTargetDepth::SetActiveDepthStencilView(pDepthStencilView);

    if( clear )
    {
        pContext->ClearRenderTargetView( mpColorRenderTargetView, pClearColor );
        if( pDepthStencilView )
        {
            pContext->ClearDepthStencilView( pDepthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, zClearVal, 0 );
        }
    }
    D3D11_VIEWPORT viewport  = { 0.0f, 0.0f, (float)mWidth, (float)mHeight, 0.0f, 1.0f };
    ((CPUTRenderParametersDX*)&renderParams)->mpContext->RSSetViewports( 1, &viewport );

    mRenderTargetSet = true;

    CPUTMaterialDX11::ResetStateTracking();
} // CPUTRenderTargetColor::SetRenderTarget()

//-----------------------------------------------
void CPUTRenderTargetDepth::SetRenderTarget(
    CPUTRenderParameters   &renderParams,
    DWORD                   renderTargetIndex,
    float                   zClearVal,
    bool                    clear
)
{
    // ****************************
    // Save the current render target "state" so we can restore it later.
    // ****************************
    mSavedWidth   = CPUTRenderTargetDepth::GetActiveWidth();
    mSavedHeight  = CPUTRenderTargetDepth::GetActiveHeight();

    CPUTRenderTargetColor::SetActiveWidthHeight( mWidth, mHeight );
    CPUTRenderTargetDepth::SetActiveWidthHeight( mWidth, mHeight );

    // TODO: support multiple render target views (i.e., MRT)
    ID3D11DeviceContext *pContext = ((CPUTRenderParametersDX*)&renderParams)->mpContext;

    // Make sure this render target isn't currently bound as a texture.
    static ID3D11ShaderResourceView *pSRV[16] = {0};
    pContext->PSSetShaderResources( 0, 16, pSRV );

    // Save the color and depth views so we can restore them later.
    mpSavedColorRenderTargetView = CPUTRenderTargetColor::GetActiveRenderTargetView();
    mpSavedDepthStencilView      = CPUTRenderTargetDepth::GetActiveDepthStencilView();

    // Clear the shader resources to avoid a hazard warning
    ID3D11ShaderResourceView *pNullResources[16] = {0};
    pContext->PSSetShaderResources(0, 16, pNullResources );
    pContext->VSSetShaderResources(0, 16, pNullResources );

    // ****************************
    // Set the new render target states
    // ****************************
    ID3D11RenderTargetView *pView[1] = {NULL};
    pContext->OMSetRenderTargets( 1, pView, mpDepthStencilView );

    CPUTRenderTargetColor::SetActiveRenderTargetView( NULL );
    CPUTRenderTargetDepth::SetActiveDepthStencilView( mpDepthStencilView );

    if( clear )
    {
        pContext->ClearDepthStencilView( mpDepthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, zClearVal, 0 );
    }
    D3D11_VIEWPORT viewport  = { 0.0f, 0.0f, (float)mWidth, (float)mHeight, 0.0f, 1.0f };
    ((CPUTRenderParametersDX*)&renderParams)->mpContext->RSSetViewports( 1, &viewport );

    mRenderTargetSet = true;
} // CPUTRenderTargetDepth::SetRenderTarget()

//-----------------------------------------
void CPUTRenderTargetColor::Resolve( CPUTRenderParameters &renderParams )
{
    ID3D11DeviceContext *pContext = ((CPUTRenderParametersDX*)&renderParams)->mpContext;
    pContext->ResolveSubresource( mpColorTextureDX,  0, mpColorTextureDXMSAA, 0, mColorFormat);
}

//-----------------------------------------
void CPUTRenderTargetColor::RestoreRenderTarget( CPUTRenderParameters &renderParams )
{
    ASSERT( mRenderTargetSet, _L("Render target restored without calling SetRenderTarget()"));

    if( mMultiSampleCount>1 )
    {
        Resolve( renderParams );
    }

    ID3D11DeviceContext *pContext = ((CPUTRenderParametersDX*)&renderParams)->mpContext;

    pContext->OMSetRenderTargets( 1, &mpSavedColorRenderTargetView, mpSavedDepthStencilView );

    CPUTRenderTargetColor::SetActiveWidthHeight( mSavedWidth, mSavedHeight );
    CPUTRenderTargetDepth::SetActiveWidthHeight( mSavedWidth, mSavedHeight );
    CPUTRenderTargetColor::SetActiveRenderTargetView( mpSavedColorRenderTargetView );
    CPUTRenderTargetDepth::SetActiveDepthStencilView( mpSavedDepthStencilView );

    // TODO: save/restore original VIEWPORT settings, not assume full-screen viewport.
    D3D11_VIEWPORT viewport  = { 0.0f, 0.0f, (float)mSavedWidth, (float)mSavedHeight, 0.0f, 1.0f };
    ((CPUTRenderParametersDX*)&renderParams)->mpContext->RSSetViewports( 1, &viewport );

    mRenderTargetSet = false;
} // CPUTRenderTarget::RestoreRenderTarget()

//-----------------------------------------
// TODO: This function is exactly the same as the RestoreRenderTargetColor().
void CPUTRenderTargetDepth::RestoreRenderTarget( CPUTRenderParameters &renderParams )
{
    ASSERT( mRenderTargetSet, _L("Render target restored without calling SetRenderTarget()"));

    ID3D11DeviceContext *pContext = ((CPUTRenderParametersDX*)&renderParams)->mpContext;

    pContext->OMSetRenderTargets( 1, &mpSavedColorRenderTargetView, mpSavedDepthStencilView );

    CPUTRenderTargetColor::SetActiveWidthHeight( mSavedWidth, mSavedHeight );
    CPUTRenderTargetDepth::SetActiveWidthHeight( mSavedWidth, mSavedHeight );
    CPUTRenderTargetColor::SetActiveRenderTargetView( mpSavedColorRenderTargetView );
    CPUTRenderTargetDepth::SetActiveDepthStencilView( mpSavedDepthStencilView );

    // TODO: save/restore original VIEWPORT settings, not assume full-screen viewport.
    D3D11_VIEWPORT viewport  = { 0.0f, 0.0f, (float)mSavedWidth, (float)mSavedHeight, 0.0f, 1.0f };
    ((CPUTRenderParametersDX*)&renderParams)->mpContext->RSSetViewports( 1, &viewport );

    mRenderTargetSet = false;
} // CPUTRenderTarget::RestoreRenderTarget()


//-----------------------------------------------------------------------------
D3D11_MAPPED_SUBRESOURCE CPUTRenderTargetColor::MapRenderTarget( CPUTRenderParameters &params, eCPUTMapType type, bool wait )
{
    // Mapping for DISCARD requires dynamic buffer.  Create dynamic copy?
    // Could easily provide input flag.  But, where would we specify? Don't like specifying in the .set file
    // Because mapping is something the application wants to do - it isn't inherent in the data.
    // Could do Clone() and pass dynamic flag to that.
    // But, then we have two.  Could always delete the other.
    // Could support programatic flag - apply to all loaded models in the .set
    // Could support programatic flag on model.  Load model first, then load set.
    // For now, simply support CopyResource mechanism.
    HRESULT hr;
    ID3D11Device *pD3dDevice = CPUT_DX11::GetDevice();
    CPUTRenderParametersDX *pParamsDX11 = (CPUTRenderParametersDX*)&params;
    ID3D11DeviceContext *pContext = pParamsDX11->mpContext;

    if( !mpColorTextureDXStaging )
    {
        CD3D11_TEXTURE2D_DESC desc = mColorDesc;
        // First time.  Create the staging resource
        desc.Usage = D3D11_USAGE_STAGING;
        switch( type )
        {
        case CPUT_MAP_READ:
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc.BindFlags = 0;
            break;
        case CPUT_MAP_READ_WRITE:
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
            desc.BindFlags = 0;
            break;
        case CPUT_MAP_WRITE:
        case CPUT_MAP_WRITE_DISCARD:
        case CPUT_MAP_NO_OVERWRITE:
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            desc.BindFlags = 0;
            break;
        };
        HRESULT hr = pD3dDevice->CreateTexture2D( &desc, NULL, &mpColorTextureDXStaging );
        ASSERT( SUCCEEDED(hr), _L("Failed to create staging texture") );
        CPUTSetDebugName( mpColorTextureDXStaging, _L("Staging buffer") );
    }
    else
    {
        ASSERT( mMappedType == type, _L("Mapping with a different CPU access than creation parameter.") );
    }

    D3D11_MAPPED_SUBRESOURCE info;
    switch( type )
    {
    case CPUT_MAP_READ:
    case CPUT_MAP_READ_WRITE:
        // TODO: Copying and immediately mapping probably introduces a stall.
        // Expose the copy externally?
        // TODO: copy only if vb has changed?
        // Copy only first time?
        // Copy the GPU version before we read from it.
        pContext->CopyResource( mpColorTextureDXStaging, (mMultiSampleCount>1) ? mpColorTextureDXMSAA : mpColorTextureDX);
        break;
    };
    hr = pContext->Map( mpColorTextureDXStaging, wait ? 0 : D3D11_MAP_FLAG_DO_NOT_WAIT, (D3D11_MAP)type, 0, &info );
    mMappedType = type;
    return info;
} // CPUTRenderTargetColor::Map()

//-----------------------------------------------------------------------------
void CPUTRenderTargetColor::UnmapRenderTarget( CPUTRenderParameters &params )
{
    ASSERT( mMappedType != CPUT_MAP_UNDEFINED, _L("Can't unmap a render target that isn't mapped.") );

    CPUTRenderParametersDX *pParamsDX11 = (CPUTRenderParametersDX*)&params;
    ID3D11DeviceContext *pContext = pParamsDX11->mpContext;

    pContext->Unmap( mpColorTextureDXStaging, 0 );

    // If we were mapped for write, then copy staging buffer to GPU
    switch( mMappedType )
    {
    case CPUT_MAP_READ:
        break;
    case CPUT_MAP_READ_WRITE:
    case CPUT_MAP_WRITE:
    case CPUT_MAP_WRITE_DISCARD:
    case CPUT_MAP_NO_OVERWRITE:
        pContext->CopyResource( mpColorTextureDX, mpColorTextureDXStaging );
        break;
    };

    // mMappedType = CPUT_MAP_UNDEFINED;
} // CPUTRenderTargetColor::Unmap()

//-----------------------------------------------------------------------------
D3D11_MAPPED_SUBRESOURCE CPUTRenderTargetDepth::MapRenderTarget( CPUTRenderParameters &params, eCPUTMapType type, bool wait )
{
    // Mapping for DISCARD requires dynamic buffer.  Create dynamic copy?
    // Could easily provide input flag.  But, where would we specify? Don't like specifying in the .set file
    // Because mapping is something the application wants to do - it isn't inherent in the data.
    // Could do Clone() and pass dynamic flag to that.
    // But, then we have two.  Could always delete the other.
    // Could support programatic flag - apply to all loaded models in the .set
    // Could support programatic flag on model.  Load model first, then load set.
    // For now, simply support CopyResource mechanism.
    HRESULT hr;
    ID3D11Device *pD3dDevice = CPUT_DX11::GetDevice();
    CPUTRenderParametersDX *pParamsDX11 = (CPUTRenderParametersDX*)&params;
    ID3D11DeviceContext *pContext = pParamsDX11->mpContext;

    if( !mpDepthTextureDXStaging )
    {
        CD3D11_TEXTURE2D_DESC desc = mDepthDesc;
        // First time.  Create the staging resource
        desc.Usage = D3D11_USAGE_STAGING;
        switch( type )
        {
        case CPUT_MAP_READ:
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc.BindFlags = 0;
            break;
        case CPUT_MAP_READ_WRITE:
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
            desc.BindFlags = 0;
            break;
        case CPUT_MAP_WRITE:
        case CPUT_MAP_WRITE_DISCARD:
        case CPUT_MAP_NO_OVERWRITE:
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            desc.BindFlags = 0;
            break;
        };
        HRESULT hr = pD3dDevice->CreateTexture2D( &desc, NULL, &mpDepthTextureDXStaging );
        ASSERT( SUCCEEDED(hr), _L("Failed to create staging texture") );
        CPUTSetDebugName( mpDepthTextureDXStaging, _L("Staging buffer") );
    }
    else
    {
        ASSERT( mMappedType == type, _L("Mapping with a different CPU access than creation parameter.") );
    }

    D3D11_MAPPED_SUBRESOURCE info;
    switch( type )
    {
    case CPUT_MAP_READ:
    case CPUT_MAP_READ_WRITE:
        // TODO: Copying and immediately mapping probably introduces a stall.
        // Expose the copy externally?
        // TODO: copy only if vb has changed?
        // Copy only first time?
        // Copy the GPU version before we read from it.
        // pContext->CopyResource( mpDepthTextureDXStaging, (mMultiSampleCount>1) ? mpDepthTextureDXMSAA : mpDepthTextureDX);
        // TODO: Do we (can we) support MSAA depth
        pContext->CopyResource( mpDepthTextureDXStaging, mpDepthTextureDX );
        break;
    };
    hr = pContext->Map( mpDepthTextureDXStaging, wait ? 0 : D3D11_MAP_FLAG_DO_NOT_WAIT, (D3D11_MAP)type, 0, &info );
    mMappedType = type;
    return info;
} // CPUTRenderTargetDepth::Map()

//-----------------------------------------------------------------------------
void CPUTRenderTargetDepth::UnmapRenderTarget( CPUTRenderParameters &params )
{
    ASSERT( mMappedType != CPUT_MAP_UNDEFINED, _L("Can't unmap a render target that isn't mapped.") );

    CPUTRenderParametersDX *pParamsDX11 = (CPUTRenderParametersDX*)&params;
    ID3D11DeviceContext *pContext = pParamsDX11->mpContext;

    pContext->Unmap( mpDepthTextureDXStaging, 0 );

    // If we were mapped for write, then copy staging buffer to GPU
    switch( mMappedType )
    {
    case CPUT_MAP_READ:
        break;
    case CPUT_MAP_READ_WRITE:
    case CPUT_MAP_WRITE:
    case CPUT_MAP_WRITE_DISCARD:
    case CPUT_MAP_NO_OVERWRITE:
        pContext->CopyResource( mpDepthTextureDX, mpDepthTextureDXStaging );
        break;
    };

    // mMappedType = CPUT_MAP_UNDEFINED;
} // CPUTRenderTargetDepth::Unmap()
