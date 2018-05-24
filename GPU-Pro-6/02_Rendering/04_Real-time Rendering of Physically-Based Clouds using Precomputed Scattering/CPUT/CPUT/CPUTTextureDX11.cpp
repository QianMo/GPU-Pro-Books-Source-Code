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

#include "CPUTTextureDX11.h"

// TODO: Would be nice to find a better place for this decl.  But, not another file just for this.
const cString gDXGIFormatNames[] =
{
    _L("DXGI_FORMAT_UNKNOWN	"),
    _L("DXGI_FORMAT_R32G32B32A32_TYPELESS"),
    _L("DXGI_FORMAT_R32G32B32A32_FLOAT"),
    _L("DXGI_FORMAT_R32G32B32A32_UINT"),
    _L("DXGI_FORMAT_R32G32B32A32_SINT"),
    _L("DXGI_FORMAT_R32G32B32_TYPELESS"),
    _L("DXGI_FORMAT_R32G32B32_FLOAT"),
    _L("DXGI_FORMAT_R32G32B32_UINT"),
    _L("DXGI_FORMAT_R32G32B32_SINT"),
    _L("DXGI_FORMAT_R16G16B16A16_TYPELESS"),
    _L("DXGI_FORMAT_R16G16B16A16_FLOAT"),
    _L("DXGI_FORMAT_R16G16B16A16_UNORM"),
    _L("DXGI_FORMAT_R16G16B16A16_UINT"),
    _L("DXGI_FORMAT_R16G16B16A16_SNORM"),
    _L("DXGI_FORMAT_R16G16B16A16_SINT"),
    _L("DXGI_FORMAT_R32G32_TYPELESS"),
    _L("DXGI_FORMAT_R32G32_FLOAT"),
    _L("DXGI_FORMAT_R32G32_UINT"),
    _L("DXGI_FORMAT_R32G32_SINT"),
    _L("DXGI_FORMAT_R32G8X24_TYPELESS"),
    _L("DXGI_FORMAT_D32_FLOAT_S8X24_UINT"),
    _L("DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS"),
    _L("DXGI_FORMAT_X32_TYPELESS_G8X24_UINT"),
    _L("DXGI_FORMAT_R10G10B10A2_TYPELESS"),
    _L("DXGI_FORMAT_R10G10B10A2_UNORM"),
    _L("DXGI_FORMAT_R10G10B10A2_UINT"),
    _L("DXGI_FORMAT_R11G11B10_FLOAT"),
    _L("DXGI_FORMAT_R8G8B8A8_TYPELESS"),
    _L("DXGI_FORMAT_R8G8B8A8_UNORM"),
    _L("DXGI_FORMAT_R8G8B8A8_UNORM_SRGB"),
    _L("DXGI_FORMAT_R8G8B8A8_UINT"),
    _L("DXGI_FORMAT_R8G8B8A8_SNORM"),
    _L("DXGI_FORMAT_R8G8B8A8_SINT"),
    _L("DXGI_FORMAT_R16G16_TYPELESS"),
    _L("DXGI_FORMAT_R16G16_FLOAT"),
    _L("DXGI_FORMAT_R16G16_UNORM"),
    _L("DXGI_FORMAT_R16G16_UINT"),
    _L("DXGI_FORMAT_R16G16_SNORM"),
    _L("DXGI_FORMAT_R16G16_SINT"),
    _L("DXGI_FORMAT_R32_TYPELESS"),
    _L("DXGI_FORMAT_D32_FLOAT"),
    _L("DXGI_FORMAT_R32_FLOAT"),
    _L("DXGI_FORMAT_R32_UINT"),
    _L("DXGI_FORMAT_R32_SINT"),
    _L("DXGI_FORMAT_R24G8_TYPELESS"),
    _L("DXGI_FORMAT_D24_UNORM_S8_UINT"),
    _L("DXGI_FORMAT_R24_UNORM_X8_TYPELESS"),
    _L("DXGI_FORMAT_X24_TYPELESS_G8_UINT"),
    _L("DXGI_FORMAT_R8G8_TYPELESS"),
    _L("DXGI_FORMAT_R8G8_UNORM"),
    _L("DXGI_FORMAT_R8G8_UINT"),
    _L("DXGI_FORMAT_R8G8_SNORM"),
    _L("DXGI_FORMAT_R8G8_SINT"),
    _L("DXGI_FORMAT_R16_TYPELESS"),
    _L("DXGI_FORMAT_R16_FLOAT"),
    _L("DXGI_FORMAT_D16_UNORM"),
    _L("DXGI_FORMAT_R16_UNORM"),
    _L("DXGI_FORMAT_R16_UINT"),
    _L("DXGI_FORMAT_R16_SNORM"),
    _L("DXGI_FORMAT_R16_SINT"),
    _L("DXGI_FORMAT_R8_TYPELESS"),
    _L("DXGI_FORMAT_R8_UNORM"),
    _L("DXGI_FORMAT_R8_UINT"),
    _L("DXGI_FORMAT_R8_SNORM"),
    _L("DXGI_FORMAT_R8_SINT"),
    _L("DXGI_FORMAT_A8_UNORM"),
    _L("DXGI_FORMAT_R1_UNORM"),
    _L("DXGI_FORMAT_R9G9B9E5_SHAREDEXP"),
    _L("DXGI_FORMAT_R8G8_B8G8_UNORM"),
    _L("DXGI_FORMAT_G8R8_G8B8_UNORM"),
    _L("DXGI_FORMAT_BC1_TYPELESS"),
    _L("DXGI_FORMAT_BC1_UNORM"),
    _L("DXGI_FORMAT_BC1_UNORM_SRGB"),
    _L("DXGI_FORMAT_BC2_TYPELESS"),
    _L("DXGI_FORMAT_BC2_UNORM"),
    _L("DXGI_FORMAT_BC2_UNORM_SRGB"),
    _L("DXGI_FORMAT_BC3_TYPELESS"),
    _L("DXGI_FORMAT_BC3_UNORM"),
    _L("DXGI_FORMAT_BC3_UNORM_SRGB"),
    _L("DXGI_FORMAT_BC4_TYPELESS"),
    _L("DXGI_FORMAT_BC4_UNORM"),
    _L("DXGI_FORMAT_BC4_SNORM"),
    _L("DXGI_FORMAT_BC5_TYPELESS"),
    _L("DXGI_FORMAT_BC5_UNORM"),
    _L("DXGI_FORMAT_BC5_SNORM"),
    _L("DXGI_FORMAT_B5G6R5_UNORM"),
    _L("DXGI_FORMAT_B5G5R5A1_UNORM"),
    _L("DXGI_FORMAT_B8G8R8A8_UNORM"),
    _L("DXGI_FORMAT_B8G8R8X8_UNORM"),
    _L("DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM"),
    _L("DXGI_FORMAT_B8G8R8A8_TYPELESS"),
    _L("DXGI_FORMAT_B8G8R8A8_UNORM_SRGB"),
    _L("DXGI_FORMAT_B8G8R8X8_TYPELESS"),
    _L("DXGI_FORMAT_B8G8R8X8_UNORM_SRGB"),
    _L("DXGI_FORMAT_BC6H_TYPELESS"),
    _L("DXGI_FORMAT_BC6H_UF16"),
    _L("DXGI_FORMAT_BC6H_SF16"),
    _L("DXGI_FORMAT_BC7_TYPELESS"),
    _L("DXGI_FORMAT_BC7_UNORM"),
    _L("DXGI_FORMAT_BC7_UNORM_SRGB")
};
const cString *gpDXGIFormatNames = gDXGIFormatNames;

//-----------------------------------------------------------------------------
CPUTTexture *CPUTTextureDX11::CreateTexture( const cString &name, const cString &absolutePathAndFilename, bool loadAsSRGB )
{
    // TODO:  Delegate to derived class.  We don't currently have CPUTTextureDX11
    ID3D11ShaderResourceView *pShaderResourceView = NULL;
    ID3D11Resource *pTexture = NULL;
    ID3D11Device *pD3dDevice= CPUT_DX11::GetDevice();
    CPUTResult result = CreateNativeTexture( pD3dDevice, absolutePathAndFilename, &pShaderResourceView, &pTexture, loadAsSRGB );
    ASSERT( CPUTSUCCESS(result), _L("Error loading texture: '")+absolutePathAndFilename );

    CPUTTextureDX11 *pNewTexture = new CPUTTextureDX11();
    pNewTexture->mName = name;
    pNewTexture->SetTextureAndShaderResourceView( pTexture, pShaderResourceView );
    pTexture->Release();
    pShaderResourceView->Release();

    CPUTAssetLibrary::GetAssetLibrary()->AddTexture( absolutePathAndFilename, pNewTexture);

    return pNewTexture;
}

//-----------------------------------------------------------------------------
CPUTResult CPUTTextureDX11::CreateNativeTexture(
    ID3D11Device *pD3dDevice,
    const cString &fileName,
    ID3D11ShaderResourceView **ppShaderResourceView,
    ID3D11Resource **ppTexture,
    bool ForceLoadAsSRGB
){
    CPUTResult result;
    HRESULT hr;

    // Set up loading structure
    //
    // Indicate all texture parameters should come from the file
    D3DX11_IMAGE_LOAD_INFO LoadInfo;
    ZeroMemory(&LoadInfo, sizeof(D3DX11_IMAGE_LOAD_INFO));
    LoadInfo.Width          = D3DX11_FROM_FILE;
    LoadInfo.Height         = D3DX11_FROM_FILE;
    LoadInfo.Depth          = D3DX11_FROM_FILE;
    LoadInfo.FirstMipLevel  = D3DX11_FROM_FILE;
    LoadInfo.MipLevels      = D3DX11_FROM_FILE;
    // LoadInfo.Usage          = D3D11_USAGE_IMMUTABLE; // TODO: maintain a "mappable" flag?  Set immutable if not mappable?
    LoadInfo.Usage          = D3D11_USAGE_DEFAULT;
    LoadInfo.BindFlags      = D3D11_BIND_SHADER_RESOURCE;
    LoadInfo.CpuAccessFlags = 0;
    LoadInfo.MiscFlags      = 0;
    LoadInfo.MipFilter      = D3DX11_FROM_FILE;
    LoadInfo.pSrcInfo       = NULL;
    LoadInfo.Format         = (DXGI_FORMAT) D3DX11_FROM_FILE;
    LoadInfo.Filter         = D3DX11_FILTER_NONE;

    // if we're 'forcing' load of sRGB data, we need to verify image is sRGB
    // or determine image format that best matches the non-sRGB source format in hopes that the conversion will be faster
    // and data preserved
    if(true == ForceLoadAsSRGB)
    {
        // get the source image info
        D3DX11_IMAGE_INFO SrcInfo;
        hr = D3DX11GetImageInfoFromFile(fileName.c_str(), NULL, &SrcInfo, NULL);
        ASSERT( SUCCEEDED(hr), _L(" - Error loading texture '")+fileName+_L("'.") );

        // find a closest equivalent sRGB format
        result = GetSRGBEquivalent(SrcInfo.Format, LoadInfo.Format);
        ASSERT( CPUTSUCCESS(result), _L("Error loading texture '")+fileName+_L("'.  It is specified this texture must load as sRGB, but the source image is in a format that cannot be converted to sRGB.\n") );

        // set filtering mode to interpret 'in'-coming data as sRGB, and storing it 'out' on an sRGB surface
        //
        // As it stands, we don't have any tools that support sRGB output in DXT compressed textures.
        // If we later support a format that does provide sRGB, then the filter 'in' flag will need to be removed
        LoadInfo.Filter = D3DX11_FILTER_NONE | D3DX11_FILTER_SRGB_IN | D3DX11_FILTER_SRGB_OUT;
#if 0
        // DWM: TODO:  We want to catch the cases where the loader needs to do work.
        // This happens if the texture's pixel format isn't supported by DXGI.
        // TODO: how to determine?

        // if a runtime conversion must happen report a performance warning error.
        // Note: choosing not to assert here, as this will be a common issue.
        if( SrcInfo.Format != LoadInfo.Format)
        {
            cString dxgiName = GetDXGIFormatString(SrcInfo.Format);
            cString errorString = _T(__FUNCTION__);
            errorString += _L("- PERFORMANCE WARNING: '") + fileName
            +_L("' has an image format ")+dxgiName
            +_L(" but must be run-time converted to ")+GetDXGIFormatString(LoadInfo.Format)
            +_L(" based on requested sRGB target buffer.\n");
            TRACE( errorString.c_str() );
        }
#endif
    }
    hr = D3DX11CreateTextureFromFile( pD3dDevice, fileName.c_str(), &LoadInfo, NULL, ppTexture, NULL );
    ASSERT( SUCCEEDED(hr), _L("Failed to load texture: ") + fileName );
    CPUTSetDebugName( *ppTexture, fileName );

    hr = pD3dDevice->CreateShaderResourceView( *ppTexture, NULL, ppShaderResourceView );
    ASSERT( SUCCEEDED(hr), _L("Failed to create texture shader resource view.") );
    CPUTSetDebugName( *ppShaderResourceView, fileName );

    return CPUT_SUCCESS;
}

//-----------------------------------------------------------------------------
CPUTResult CPUTTextureDX11::GetSRGBEquivalent(DXGI_FORMAT inFormat, DXGI_FORMAT& sRGBFormat)
{
    switch( inFormat )
    {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            sRGBFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            return CPUT_SUCCESS;
        case DXGI_FORMAT_B8G8R8X8_UNORM:
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
            sRGBFormat = DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
            return CPUT_SUCCESS;
        case DXGI_FORMAT_BC1_UNORM:
        case DXGI_FORMAT_BC1_UNORM_SRGB:
            sRGBFormat = DXGI_FORMAT_BC1_UNORM_SRGB;
            return CPUT_SUCCESS;
        case DXGI_FORMAT_BC2_UNORM:
        case DXGI_FORMAT_BC2_UNORM_SRGB:
            sRGBFormat = DXGI_FORMAT_BC2_UNORM_SRGB;
            return CPUT_SUCCESS;
        case DXGI_FORMAT_BC3_UNORM:
        case DXGI_FORMAT_BC3_UNORM_SRGB:
            sRGBFormat = DXGI_FORMAT_BC3_UNORM_SRGB;
            return CPUT_SUCCESS;
        case DXGI_FORMAT_BC7_UNORM:
        case DXGI_FORMAT_BC7_UNORM_SRGB:
            sRGBFormat = DXGI_FORMAT_BC7_UNORM_SRGB;
            return CPUT_SUCCESS;
    };
    return CPUT_ERROR_UNSUPPORTED_SRGB_IMAGE_FORMAT;
}

// This function returns the DXGI string equivalent of the DXGI format for
// error reporting/display purposes
//-----------------------------------------------------------------------------
const cString &CPUTTextureDX11::GetDXGIFormatString(DXGI_FORMAT format)
{
    ASSERT( (format>=0) && (format<=DXGI_FORMAT_BC7_UNORM_SRGB), _L("Invalid DXGI Format.") );
    return gpDXGIFormatNames[format];
}

// Given a certain DXGI texture format, does it even have an equivalent sRGB one
//-----------------------------------------------------------------------------
bool CPUTTextureDX11::DoesExistEquivalentSRGBFormat(DXGI_FORMAT inFormat)
{
    DXGI_FORMAT outFormat;

    if( CPUT_ERROR_UNSUPPORTED_SRGB_IMAGE_FORMAT == GetSRGBEquivalent(inFormat, outFormat) )
    {
        return false;
    }
    return true;
}

//-----------------------------------------------------------------------------
D3D11_MAPPED_SUBRESOURCE CPUTTextureDX11::MapTexture( CPUTRenderParameters &params, eCPUTMapType type, bool wait )
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

    if( !mpTextureStaging )
    {
        // Annoying.  We need to create the texture differently, based on dimension.
        D3D11_RESOURCE_DIMENSION dimension;
        mpTexture->GetType(&dimension);
        switch( dimension )
        {
        case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
            {
                D3D11_TEXTURE1D_DESC desc;
                ((ID3D11Texture1D*)mpTexture)->GetDesc( &desc );
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
                hr = pD3dDevice->CreateTexture1D( &desc, NULL, (ID3D11Texture1D**)&mpTextureStaging );
                ASSERT( SUCCEEDED(hr), _L("Failed to create staging texture") );
                break;
            }
        case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
            {
                D3D11_TEXTURE2D_DESC desc;
                ((ID3D11Texture2D*)mpTexture)->GetDesc( &desc );
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
                hr = pD3dDevice->CreateTexture2D( &desc, NULL, (ID3D11Texture2D**)&mpTextureStaging );
                ASSERT( SUCCEEDED(hr), _L("Failed to create staging texture") );
                break;
            }
        case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
            {
                D3D11_TEXTURE3D_DESC desc;
                ((ID3D11Texture3D*)mpTexture)->GetDesc( &desc );
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
                hr = pD3dDevice->CreateTexture3D( &desc, NULL, (ID3D11Texture3D**)&mpTextureStaging );
                ASSERT( SUCCEEDED(hr), _L("Failed to create staging texture") );
                break;
            }
        default:
            ASSERT(0, _L("Unkown texture dimension") );
            break;
        }
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
        pContext->CopyResource( mpTextureStaging, mpTexture );
        break;
    };
    hr = pContext->Map( mpTextureStaging, wait ? 0 : D3D11_MAP_FLAG_DO_NOT_WAIT, (D3D11_MAP)type, 0, &info );
    mMappedType = type;
    return info;
} // CPUTTextureDX11::Map()

//-----------------------------------------------------------------------------
void CPUTTextureDX11::UnmapTexture( CPUTRenderParameters &params )
{
    ASSERT( mMappedType != CPUT_MAP_UNDEFINED, _L("Can't unmap a render target that isn't mapped.") );

    CPUTRenderParametersDX *pParamsDX11 = (CPUTRenderParametersDX*)&params;
    ID3D11DeviceContext *pContext = pParamsDX11->mpContext;

    pContext->Unmap( mpTextureStaging, 0 );

    // If we were mapped for write, then copy staging buffer to GPU
    switch( mMappedType )
    {
    case CPUT_MAP_READ:
        break;
    case CPUT_MAP_READ_WRITE:
    case CPUT_MAP_WRITE:
    case CPUT_MAP_WRITE_DISCARD:
    case CPUT_MAP_NO_OVERWRITE:
        pContext->CopyResource( mpTexture, mpTextureStaging );
        break;
    };
} // CPUTTextureDX11::Unmap()

