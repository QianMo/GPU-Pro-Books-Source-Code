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

#include "CPUTBufferDX11.h"

//-----------------------------------------------------------------------------
D3D11_MAPPED_SUBRESOURCE CPUTBufferDX11::MapBuffer( CPUTRenderParameters &params, eCPUTMapType type, bool wait )
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

    if( !mpBufferStaging )
    {
        D3D11_BUFFER_DESC desc;
        mpBuffer->GetDesc( &desc );
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
        hr = pD3dDevice->CreateBuffer( &desc, NULL, &mpBufferStaging );
        ASSERT( SUCCEEDED(hr), _L("Failed to create staging buffer") );
        CPUTSetDebugName( mpBufferStaging, _L("Staging Buffer"));
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
        // TODO: copy only if changed?
        // Copy only first time?
        // Copy the GPU version before we read from it.
        pContext->CopyResource( mpBufferStaging, mpBuffer );
        break;
    };
    hr = pContext->Map( mpBufferStaging, wait ? 0 : D3D11_MAP_FLAG_DO_NOT_WAIT, (D3D11_MAP)type, 0, &info );
    mMappedType = type;
    return info;
} // CPUTBufferDX11::Map()

//-----------------------------------------------------------------------------
void CPUTBufferDX11::UnmapBuffer( CPUTRenderParameters &params )
{
    ASSERT( mMappedType != CPUT_MAP_UNDEFINED, _L("Can't unmap a render target that isn't mapped.") );

    CPUTRenderParametersDX *pParamsDX11 = (CPUTRenderParametersDX*)&params;
    ID3D11DeviceContext *pContext = pParamsDX11->mpContext;

    pContext->Unmap( mpBufferStaging, 0 );

    // If we were mapped for write, then copy staging buffer to GPU
    switch( mMappedType )
    {
    case CPUT_MAP_READ:
        break;
    case CPUT_MAP_READ_WRITE:
    case CPUT_MAP_WRITE:
    case CPUT_MAP_WRITE_DISCARD:
    case CPUT_MAP_NO_OVERWRITE:
        pContext->CopyResource( mpBuffer, mpBufferStaging );
        break;
    };
} // CPUTBufferDX11::Unmap()


