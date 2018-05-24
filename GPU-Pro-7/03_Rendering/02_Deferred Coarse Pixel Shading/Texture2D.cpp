/////////////////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or imlied.
// See the License for the specific language governing permissions and
// limitations under the License.
/////////////////////////////////////////////////////////////////////////////////////////////

#include "Texture2D.h"
#include <assert.h>


Texture2D::Texture2D(ID3D11Device* d3dDevice,
                     int width, int height, DXGI_FORMAT format,
                     UINT bindFlags,
                     int mipLevels)
{
    InternalConstruct(d3dDevice, width, height, format, bindFlags, mipLevels, 1, 1, 0,
        D3D11_RTV_DIMENSION_TEXTURE2D, D3D11_UAV_DIMENSION_TEXTURE2D, D3D11_SRV_DIMENSION_TEXTURE2D);
}

Texture2D::Texture2D(ID3D11Device* d3dDevice,
                     int width, int height, DXGI_FORMAT format,
                     UINT bindFlags,
                     const DXGI_SAMPLE_DESC& sampleDesc)
{
    if (sampleDesc.Count > 1) {
        // UAV's can't point to multisampled resources
        InternalConstruct(d3dDevice, width, height, format, bindFlags, 1, 1, sampleDesc.Count, sampleDesc.Quality,
            D3D11_RTV_DIMENSION_TEXTURE2DMS, D3D11_UAV_DIMENSION_UNKNOWN, D3D11_SRV_DIMENSION_TEXTURE2DMS);
    } else {
        InternalConstruct(d3dDevice, width, height, format, bindFlags, 1, 1, sampleDesc.Count, sampleDesc.Quality,
            D3D11_RTV_DIMENSION_TEXTURE2D, D3D11_UAV_DIMENSION_UNKNOWN, D3D11_SRV_DIMENSION_TEXTURE2D);
    }
}

Texture2D::Texture2D(ID3D11Device* d3dDevice,
                     int width, int height, DXGI_FORMAT format,
                     UINT bindFlags,
                     int mipLevels, int arraySize)
{
    InternalConstruct(d3dDevice, width, height, format, bindFlags, mipLevels, arraySize, 1, 0,
        D3D11_RTV_DIMENSION_TEXTURE2DARRAY, D3D11_UAV_DIMENSION_TEXTURE2DARRAY, D3D11_SRV_DIMENSION_TEXTURE2DARRAY);
}

Texture2D::Texture2D(ID3D11Device* d3dDevice,
                     int width, int height, DXGI_FORMAT format,
                     UINT bindFlags,
                     int arraySize, const DXGI_SAMPLE_DESC& sampleDesc)
{
    if (sampleDesc.Count > 1) {
        // UAV's can't point to multisampled resources
        InternalConstruct(d3dDevice, width, height, format, bindFlags, 1, arraySize, sampleDesc.Count, sampleDesc.Quality,
            D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY, D3D11_UAV_DIMENSION_UNKNOWN, D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY);
    } else {
        InternalConstruct(d3dDevice, width, height, format, bindFlags, 1, arraySize, sampleDesc.Count, sampleDesc.Quality,
            D3D11_RTV_DIMENSION_TEXTURE2DARRAY, D3D11_UAV_DIMENSION_UNKNOWN, D3D11_SRV_DIMENSION_TEXTURE2DARRAY);
    }
}

void Texture2D::InternalConstruct(ID3D11Device* d3dDevice,
                                  int width, int height, DXGI_FORMAT format,
                                  UINT bindFlags, int mipLevels, int arraySize,
                                  int sampleCount, int sampleQuality,
                                  D3D11_RTV_DIMENSION rtvDimension,
                                  D3D11_UAV_DIMENSION uavDimension,
                                  D3D11_SRV_DIMENSION srvDimension)
{
    // Initalize
    mShaderResource = 0;

    CD3D11_TEXTURE2D_DESC desc(
        format,
        width, height, arraySize, mipLevels,
        bindFlags,
        D3D11_USAGE_DEFAULT, 0,
        sampleCount, sampleQuality,
        // If they request mipmap levels, it's nice to be able to autogenerate them.
        (mipLevels != 1 ? D3D11_RESOURCE_MISC_GENERATE_MIPS : 0));

    d3dDevice->CreateTexture2D(&desc, 0, &mTexture);

    // Update with actual mip levels, etc.
    mTexture->GetDesc(&desc);

    if (bindFlags & D3D11_BIND_RENDER_TARGET) {
        for (int i = 0; i < arraySize; ++i) {
            CD3D11_RENDER_TARGET_VIEW_DESC rtvDesc(
                rtvDimension,
                format,
                0,          // Mips
                i, 1        // Array
            );

            ID3D11RenderTargetView* renderTargetView;
            d3dDevice->CreateRenderTargetView(mTexture, &rtvDesc, &renderTargetView);
            mRenderTargetElements.push_back(renderTargetView);
        }
    }

    if (bindFlags & D3D11_BIND_UNORDERED_ACCESS) {
        // UAV's can't point to multisampled resources!
        assert(uavDimension != D3D11_UAV_DIMENSION_UNKNOWN);

        for (int i = 0; i < arraySize; ++i) {
            CD3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc(
                uavDimension,
                format,
                0,          // Mips
                i, 1        // Array
            );

            ID3D11UnorderedAccessView* unorderedAccessView;
            d3dDevice->CreateUnorderedAccessView(mTexture, &uavDesc, &unorderedAccessView);
            mUnorderedAccessElements.push_back(unorderedAccessView);
        }
    }

    if (bindFlags & D3D11_BIND_SHADER_RESOURCE) {
        // Whole resource
        CD3D11_SHADER_RESOURCE_VIEW_DESC srvDesc(
            srvDimension,
            format,
            0, desc.MipLevels,  // Mips
            0, desc.ArraySize   // Array
        );        

        d3dDevice->CreateShaderResourceView(mTexture, &srvDesc, &mShaderResource);

        // Single elements
        for (int i = 0; i < arraySize; ++i) {
            CD3D11_SHADER_RESOURCE_VIEW_DESC srvElementDesc(
                srvDimension,
                format,
                0, 1,  // Mips
                i, 1   // Array
            );

            ID3D11ShaderResourceView* shaderResourceView;
            d3dDevice->CreateShaderResourceView(mTexture, &srvElementDesc, &shaderResourceView);
            mShaderResourceElements.push_back(shaderResourceView);
        }
    }
}


Texture2D::~Texture2D()
{
    for (std::size_t i = 0; i < mRenderTargetElements.size(); ++i) {
        mRenderTargetElements[i]->Release();
    }
    for (std::size_t i = 0; i < mUnorderedAccessElements.size(); ++i) {
        mUnorderedAccessElements[i]->Release();
    }
    for (std::size_t i = 0; i < mShaderResourceElements.size(); ++i) {
        mShaderResourceElements[i]->Release();
    }
    if (mShaderResource) mShaderResource->Release();
    mTexture->Release();
}




Depth2D::Depth2D(ID3D11Device* d3dDevice,
                 int width, int height,
                 UINT bindFlags,
                 bool stencil)
{
    InternalConstruct(d3dDevice, width, height, bindFlags, 1, 1, 0,
        D3D11_DSV_DIMENSION_TEXTURE2D, D3D11_SRV_DIMENSION_TEXTURE2D, stencil);
}

Depth2D::Depth2D(ID3D11Device* d3dDevice,
                 int width, int height,
                 UINT bindFlags,
                 const DXGI_SAMPLE_DESC& sampleDesc,
                 bool stencil)
{
    if (stencil) {
        InternalConstruct(d3dDevice, width, height, bindFlags, 1, sampleDesc.Count, sampleDesc.Quality,
            D3D11_DSV_DIMENSION_TEXTURE2DMS, D3D11_SRV_DIMENSION_TEXTURE2DMS, stencil);
    } else {
        InternalConstruct(d3dDevice, width, height, bindFlags, 1, sampleDesc.Count, sampleDesc.Quality,
            D3D11_DSV_DIMENSION_TEXTURE2D, D3D11_SRV_DIMENSION_TEXTURE2D, stencil);
    }
}

Depth2D::Depth2D(ID3D11Device* d3dDevice,
                 int width, int height,
                 UINT bindFlags,
                 int arraySize,
                 bool stencil)
{
    InternalConstruct(d3dDevice, width, height, bindFlags, arraySize, 1, 0,
        D3D11_DSV_DIMENSION_TEXTURE2DARRAY, D3D11_SRV_DIMENSION_TEXTURE2DARRAY, stencil);
}

Depth2D::Depth2D(ID3D11Device* d3dDevice,
                 int width, int height,
                 UINT bindFlags,
                 int arraySize,
                 const DXGI_SAMPLE_DESC& sampleDesc,
                 bool stencil)
{
    if (stencil) {
        InternalConstruct(d3dDevice, width, height, bindFlags, arraySize, sampleDesc.Count, sampleDesc.Quality,
            D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY, D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY, stencil);
    } else {
        InternalConstruct(d3dDevice, width, height, bindFlags, arraySize, sampleDesc.Count, sampleDesc.Quality,
            D3D11_DSV_DIMENSION_TEXTURE2DARRAY, D3D11_SRV_DIMENSION_TEXTURE2DARRAY, stencil);
    }
}
    
void Depth2D::InternalConstruct(ID3D11Device* d3dDevice,
                                int width, int height,
                                UINT bindFlags, int arraySize,
                                int sampleCount, int sampleQuality,
                                D3D11_DSV_DIMENSION dsvDimension,
                                D3D11_SRV_DIMENSION srvDimension,
                                bool stencil)
{
    // Initalize
    mShaderResource = 0;

    CD3D11_TEXTURE2D_DESC desc(
        stencil ? DXGI_FORMAT_R32G8X24_TYPELESS : DXGI_FORMAT_R32_TYPELESS,
        width, height, arraySize, 1,
        bindFlags,
        D3D11_USAGE_DEFAULT, 0,
        sampleCount, sampleQuality);

    d3dDevice->CreateTexture2D(&desc, 0, &mTexture);

    if (bindFlags & D3D11_BIND_DEPTH_STENCIL) {
        for (int i = 0; i < arraySize; ++i) {
            CD3D11_DEPTH_STENCIL_VIEW_DESC depthStencilDesc(
                dsvDimension,
                stencil ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT : DXGI_FORMAT_D32_FLOAT,
                0,          // Mips
                i, 1        // Array
            );

            ID3D11DepthStencilView* depthStencilView;
            d3dDevice->CreateDepthStencilView(mTexture, &depthStencilDesc, &depthStencilView);
            mDepthStencilElements.push_back(depthStencilView);
        }
    }

    if (bindFlags & D3D11_BIND_SHADER_RESOURCE) {
        CD3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceDesc(
            srvDimension,
            stencil ? DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS : DXGI_FORMAT_R32_FLOAT,
            0, 1,           // Mips
            0, arraySize    // Array
        );

        d3dDevice->CreateShaderResourceView(mTexture, &shaderResourceDesc, &mShaderResource);
    }
}


Depth2D::~Depth2D()
{
    for (std::size_t i = 0; i < mDepthStencilElements.size(); ++i) {
        mDepthStencilElements[i]->Release();
    }
    if (mShaderResource) mShaderResource->Release();
    mTexture->Release();
}
