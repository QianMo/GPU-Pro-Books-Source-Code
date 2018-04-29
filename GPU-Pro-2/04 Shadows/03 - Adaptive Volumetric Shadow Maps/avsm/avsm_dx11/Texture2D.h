// Copyright 2010 Intel Corporation
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

#pragma once

#include <d3d11.h>
#include <vector>

class Texture2D
{
public:
    // Construct a Texture2D
    Texture2D(ID3D11Device* d3dDevice,
              int width, int height, DXGI_FORMAT format,
              UINT bindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
              int mipLevels = 1);

    // Construct a Texture2DMS
    Texture2D(ID3D11Device* d3dDevice,
              int width, int height, DXGI_FORMAT format,
              UINT bindFlags,
              const DXGI_SAMPLE_DESC& sampleDesc);

    // Construct a Texture2DArray
    Texture2D(ID3D11Device* d3dDevice,
              int width, int height, DXGI_FORMAT format,
              UINT bindFlags,
              int mipLevels, int arraySize);

    // Construct a Texture2DMSArray
    Texture2D(ID3D11Device* d3dDevice,
              int width, int height, DXGI_FORMAT format,
              UINT bindFlags,
              int arraySize, const DXGI_SAMPLE_DESC& sampleDesc);

    // Construct a generic texture
    Texture2D(ID3D11Device* d3dDevice,
              int width, int height, DXGI_FORMAT format,
              UINT bindFlags,
              int arraySize, const DXGI_SAMPLE_DESC& sampleDesc,
              D3D11_RTV_DIMENSION rtvDim, 
              D3D11_UAV_DIMENSION uavDim, 
              D3D11_SRV_DIMENSION srvDim);

    ~Texture2D();

    ID3D11Texture2D* GetTexture() { return mTexture; }
    ID3D11RenderTargetView* GetRenderTarget(std::size_t arrayIndex = 0) { return mRenderTargetElements[arrayIndex]; }
    ID3D11UnorderedAccessView* GetUnorderedAccessArray() { return mUnorderedAccessArray; }
    // Treat these like render targets for now... i.e. only access a slice
    ID3D11UnorderedAccessView* GetUnorderedAccess(std::size_t arrayIndex = 0) { return mUnorderedAccessElements[arrayIndex]; }
    ID3D11ShaderResourceView* GetShaderResource() { return mShaderResource; }

private:
    void InternalConstruct(ID3D11Device* d3dDevice,
                           int width, int height, DXGI_FORMAT format,
                           UINT bindFlags, int mipLevels, int arraySize,
                           int sampleCount, int sampleQuality,
                           D3D11_RTV_DIMENSION rtvDimension,
                           D3D11_UAV_DIMENSION uavDimension,
                           D3D11_SRV_DIMENSION srvDimension);

    // Not implemented
    Texture2D(const Texture2D&);
    Texture2D& operator=(const Texture2D&);

    ID3D11Texture2D* mTexture;
    ID3D11ShaderResourceView*  mShaderResource;
    // One per array element
    std::vector<ID3D11RenderTargetView*> mRenderTargetElements;
    std::vector<ID3D11UnorderedAccessView*> mUnorderedAccessElements;
    // UAV array
    ID3D11UnorderedAccessView* mUnorderedAccessArray;
};


// Currently always float 32 as this one works best with sampling
class Depth2D
{
public:
    // Construct a Texture2D depth buffer
    Depth2D(ID3D11Device* d3dDevice,
            int width, int height,
            UINT bindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE);

    // Construct a Texture2DMS depth buffer
    Depth2D(ID3D11Device* d3dDevice,
            int width, int height,
            UINT bindFlags,
            const DXGI_SAMPLE_DESC& sampleDesc);

    // Construct a Texture2DArray depth buffer
    Depth2D(ID3D11Device* d3dDevice,
            int width, int height,
            UINT bindFlags,
            int arraySize);

    // Construct a Texture2DMSArray depth buffer
    Depth2D(ID3D11Device* d3dDevice,
            int width, int height,
            UINT bindFlags,
            int arraySize, const DXGI_SAMPLE_DESC& sampleDesc);

    ~Depth2D();

    ID3D11Texture2D* GetTexture() { return mTexture; }
    ID3D11DepthStencilView* GetDepthStencil(std::size_t arrayIndex = 0) { return mDepthStencilElements[arrayIndex]; }
    ID3D11ShaderResourceView* GetShaderResource() { return mShaderResource; }

private:
    void InternalConstruct(ID3D11Device* d3dDevice,
                           int width, int height,
                           UINT bindFlags, int arraySize,
                           int sampleCount, int sampleQuality,
                           D3D11_DSV_DIMENSION dsvDimension,
                           D3D11_SRV_DIMENSION srvDimension);

    // Not implemented
    Depth2D(const Depth2D&);
    Depth2D& operator=(const Depth2D&);

    ID3D11Texture2D* mTexture;
    ID3D11ShaderResourceView* mShaderResource;
    // One per array element
    std::vector<ID3D11DepthStencilView*> mDepthStencilElements;
};
