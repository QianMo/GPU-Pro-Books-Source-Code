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

// NOTE: Ensure that T is exactly the same size/layout as the shader structure!
template <typename T>
class StructuredBuffer
{
public:
    // Construct a structured buffer
    StructuredBuffer(ID3D11Device* d3dDevice, int elements,
                     UINT bindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
                     bool dynamic = false);
    
    ~StructuredBuffer();

    ID3D11Buffer* GetBuffer() { return mBuffer; }
    ID3D11UnorderedAccessView* GetUnorderedAccess() { return mUnorderedAccess; }
    ID3D11ShaderResourceView* GetShaderResource() { return mShaderResource; }

    // Only valid for dynamic buffers
    // TODO: Support NOOVERWRITE ring buffer?
    T* MapDiscard(ID3D11DeviceContext* d3dDeviceContext);
    void Unmap(ID3D11DeviceContext* d3dDeviceContext);

private:
    // Not implemented
    StructuredBuffer(const StructuredBuffer&);
    StructuredBuffer& operator=(const StructuredBuffer&);

    int mElements;
    ID3D11Buffer* mBuffer;
    ID3D11ShaderResourceView* mShaderResource;
    ID3D11UnorderedAccessView* mUnorderedAccess;
};


template <typename T>
StructuredBuffer<T>::StructuredBuffer(ID3D11Device* d3dDevice, int elements,
                                      UINT bindFlags, bool dynamic)
    : mElements(elements), mShaderResource(0), mUnorderedAccess(0)
{
    CD3D11_BUFFER_DESC desc(sizeof(T) * elements, bindFlags,
        dynamic ? D3D11_USAGE_DYNAMIC : D3D11_USAGE_DEFAULT,
        dynamic ? D3D11_CPU_ACCESS_WRITE : 0,
        D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
        sizeof(T));

	d3dDevice->CreateBuffer(&desc, 0, &mBuffer);

    if (bindFlags & D3D11_BIND_UNORDERED_ACCESS) {
        d3dDevice->CreateUnorderedAccessView(mBuffer, 0, &mUnorderedAccess);
    }

    if (bindFlags & D3D11_BIND_SHADER_RESOURCE) {
        d3dDevice->CreateShaderResourceView(mBuffer, 0, &mShaderResource);
    }
}


template <typename T>
StructuredBuffer<T>::~StructuredBuffer()
{
    if (mUnorderedAccess) mUnorderedAccess->Release();
    if (mShaderResource) mShaderResource->Release();
    mBuffer->Release();
}


template <typename T>
T* StructuredBuffer<T>::MapDiscard(ID3D11DeviceContext* d3dDeviceContext)
{
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    d3dDeviceContext->Map(mBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    return static_cast<T*>(mappedResource.pData);
}


template <typename T>
void StructuredBuffer<T>::Unmap(ID3D11DeviceContext* d3dDeviceContext)
{
    d3dDeviceContext->Unmap(mBuffer, 0);
}


// TODO: Constant buffers
