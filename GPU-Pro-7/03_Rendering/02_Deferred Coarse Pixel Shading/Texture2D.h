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

    ~Texture2D();

    ID3D11Texture2D* GetTexture() { return mTexture; }

    ID3D11RenderTargetView* GetRenderTarget(std::size_t arrayIndex = 0) { return mRenderTargetElements[arrayIndex]; }

    // Treat these like render targets for now... i.e. only access a slice
    ID3D11UnorderedAccessView* GetUnorderedAccess(std::size_t arrayIndex = 0) { return mUnorderedAccessElements[arrayIndex]; }

    // Get a full view of the resource
    ID3D11ShaderResourceView* GetShaderResource() { return mShaderResource; }

    // Get a view to the top mip of a single array element
    ID3D11ShaderResourceView* GetShaderResource(std::size_t arrayIndex) { return mShaderResourceElements[arrayIndex]; }

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
    ID3D11ShaderResourceView* mShaderResource;

    // One per array element
    std::vector<ID3D11RenderTargetView*> mRenderTargetElements;
    std::vector<ID3D11UnorderedAccessView*> mUnorderedAccessElements;
    std::vector<ID3D11ShaderResourceView*> mShaderResourceElements;
};


// Currently always float 32 as this one works best with sampling
// Optionally supports adding 8-bit stencil, but SRVs will only reference the 32-bit float part
class Depth2D
{
public:
    // Construct a Texture2D depth buffer
    Depth2D(ID3D11Device* d3dDevice,
            int width, int height,
            UINT bindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE,
            bool stencil = false);

    // Construct a Texture2DMS depth buffer
    Depth2D(ID3D11Device* d3dDevice,
            int width, int height,
            UINT bindFlags,
            const DXGI_SAMPLE_DESC& sampleDesc,
            bool stencil = false);

    // Construct a Texture2DArray depth buffer
    Depth2D(ID3D11Device* d3dDevice,
            int width, int height,
            UINT bindFlags,
            int arraySize,
            bool stencil = false);

    // Construct a Texture2DMSArray depth buffer
    Depth2D(ID3D11Device* d3dDevice,
            int width, int height,
            UINT bindFlags,
            int arraySize, const DXGI_SAMPLE_DESC& sampleDesc,
            bool stencil = false);

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
                           D3D11_SRV_DIMENSION srvDimension,
                           bool stencil);

    // Not implemented
    Depth2D(const Depth2D&);
    Depth2D& operator=(const Depth2D&);

    ID3D11Texture2D* mTexture;
    ID3D11ShaderResourceView* mShaderResource;
    // One per array element
    std::vector<ID3D11DepthStencilView*> mDepthStencilElements;
};
