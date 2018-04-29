/*
 * Copyright (C) 2010 Jorge Jimenez (jim@unizar.es)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must display the name 'Jorge Jimenez' as
 *    'Real-Time Rendering R&D' in the credits of the application, if such
 *    credits exist. The author of this work must be notified via email
 *    (jim@unizar.es) in this case of redistribution.
 *
 * 3. Neither the name of copyright holders nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
 * IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <DXUT.h>
#include "RenderTarget.h"


const DXGI_SAMPLE_DESC RenderTarget::NO_MSAA = { 1, 0 };


RenderTarget::RenderTarget(ID3D10Device *device, int width, int height, DXGI_FORMAT format, const DXGI_SAMPLE_DESC &sampleDesc)
        : device(device), width(width), height(height) {
    HRESULT hr;

    D3D10_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc = sampleDesc;
    desc.Usage = D3D10_USAGE_DEFAULT;
    desc.BindFlags = D3D10_BIND_RENDER_TARGET | D3D10_BIND_SHADER_RESOURCE;
    V(device->CreateTexture2D(&desc, NULL, &texture2D));

    D3D10_RENDER_TARGET_VIEW_DESC rtdesc;
    rtdesc.Format = desc.Format;
    if (sampleDesc.Count == 1) {
        rtdesc.ViewDimension = D3D10_RTV_DIMENSION_TEXTURE2D;
    } else {
        rtdesc.ViewDimension = D3D10_RTV_DIMENSION_TEXTURE2DMS;
    }
    rtdesc.Texture2D.MipSlice = 0;
    V(device->CreateRenderTargetView(texture2D, &rtdesc, &renderTargetView));

    D3D10_SHADER_RESOURCE_VIEW_DESC srdesc;
    srdesc.Format = desc.Format;
    if (sampleDesc.Count == 1) {
        srdesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
    } else {
        srdesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2DMS;
    }
    srdesc.Texture2D.MostDetailedMip = 0;
    srdesc.Texture2D.MipLevels = 1;
    V(device->CreateShaderResourceView(texture2D, &srdesc, &shaderResourceView));
}


RenderTarget::~RenderTarget() {
    SAFE_RELEASE(texture2D);
    SAFE_RELEASE(renderTargetView);
    SAFE_RELEASE(shaderResourceView);
}


void RenderTarget::setViewport(float minDepth, float maxDepth) const {
    D3D10_VIEWPORT viewport;
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width = width;
    viewport.Height = height;
    viewport.MinDepth = minDepth;
    viewport.MaxDepth = maxDepth;
    device->RSSetViewports(1, &viewport);
}


const DXGI_SAMPLE_DESC DepthStencil::NO_MSAA = { 1, 0 };


DepthStencil::DepthStencil(ID3D10Device *device, int width, int height, DXGI_FORMAT texture2DFormat,
                           const DXGI_FORMAT depthStencilViewFormat, DXGI_FORMAT shaderResourceViewFormat,
                           const DXGI_SAMPLE_DESC &sampleDesc)
        : device(device), width(width), height(height) {
    HRESULT hr;

    D3D10_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = texture2DFormat;
    desc.SampleDesc = sampleDesc;
    desc.Usage = D3D10_USAGE_DEFAULT;
    if (sampleDesc.Count == 1) {
        desc.BindFlags = D3D10_BIND_DEPTH_STENCIL | D3D10_BIND_SHADER_RESOURCE;
    } else {
        desc.BindFlags = D3D10_BIND_DEPTH_STENCIL;
    }
    V(device->CreateTexture2D(&desc, NULL, &texture2D));

    D3D10_DEPTH_STENCIL_VIEW_DESC dsdesc;
    dsdesc.Format = depthStencilViewFormat;
    if (sampleDesc.Count == 1) {
        dsdesc.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2D;
    } else {
        dsdesc.ViewDimension = D3D10_DSV_DIMENSION_TEXTURE2DMS;
    }
    dsdesc.Texture2D.MipSlice = 0;
    V(device->CreateDepthStencilView(texture2D, &dsdesc, &depthStencilView));

    if (sampleDesc.Count == 1) {
        D3D10_SHADER_RESOURCE_VIEW_DESC srdesc;
        srdesc.Format = shaderResourceViewFormat;
        srdesc.ViewDimension = D3D10_SRV_DIMENSION_TEXTURE2D;
        srdesc.Texture2D.MostDetailedMip = 0;
        srdesc.Texture2D.MipLevels = 1;
        V(device->CreateShaderResourceView(texture2D, &srdesc, &shaderResourceView));
    } else {
        shaderResourceView = NULL;
    }
}


DepthStencil::~DepthStencil() {
    SAFE_RELEASE(texture2D);
    SAFE_RELEASE(depthStencilView);
    SAFE_RELEASE(shaderResourceView);
}


void DepthStencil::setViewport(float minDepth, float maxDepth) const {
    D3D10_VIEWPORT viewport;
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width = width;
    viewport.Height = height;
    viewport.MinDepth = minDepth;
    viewport.MaxDepth = maxDepth;
    device->RSSetViewports(1, &viewport);
}


struct FSVertex{
    D3DXVECTOR3 position;
    D3DXVECTOR2 texcoord;
};


Quad::Quad(ID3D10Device *device, const D3D10_PASS_DESC &desc) 
        : device(device) {
    D3D10_BUFFER_DESC BufDesc;
    D3D10_SUBRESOURCE_DATA SRData;
    FSVertex vertices[4];

    vertices[0].position = D3DXVECTOR3(-1.0f, -1.0f, 1.0f);
    vertices[1].position = D3DXVECTOR3(-1.0f,  1.0f, 1.0f);
    vertices[2].position = D3DXVECTOR3( 1.0f, -1.0f, 1.0f);
    vertices[3].position = D3DXVECTOR3( 1.0f,  1.0f, 1.0f);

    vertices[0].texcoord = D3DXVECTOR2(0, 1);
    vertices[1].texcoord = D3DXVECTOR2(0, 0);
    vertices[2].texcoord = D3DXVECTOR2(1, 1);
    vertices[3].texcoord = D3DXVECTOR2(1, 0);

    BufDesc.ByteWidth = sizeof(FSVertex) * 4;
    BufDesc.Usage = D3D10_USAGE_DEFAULT;
    BufDesc.BindFlags = D3D10_BIND_VERTEX_BUFFER;
    BufDesc.CPUAccessFlags = 0;
    BufDesc.MiscFlags = 0;

    SRData.pSysMem = vertices;
    SRData.SysMemPitch = 0;
    SRData.SysMemSlicePitch = 0;

    HRESULT hr;
    V(device->CreateBuffer(&BufDesc, &SRData, &buffer));

    const D3D10_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D10_APPEND_ALIGNED_ELEMENT, D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD",  0, DXGI_FORMAT_R32G32_FLOAT,    0, D3D10_APPEND_ALIGNED_ELEMENT, D3D10_INPUT_PER_VERTEX_DATA, 0 },
    };
    UINT numElements = sizeof(layout) / sizeof(D3D10_INPUT_ELEMENT_DESC);

    V(device->CreateInputLayout(layout, numElements, desc.pIAInputSignature, desc.IAInputSignatureSize, &vertexLayout));
}


Quad::~Quad() {
    SAFE_RELEASE(buffer);
    SAFE_RELEASE(vertexLayout);
}


void Quad::draw() {
    const UINT offset = 0;
    const UINT stride = sizeof(FSVertex);
    device->IASetVertexBuffers(0, 1, &buffer, &stride, &offset);
    device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    device->Draw(4, 0);
}


SaveViewportScope::SaveViewportScope(ID3D10Device *device) : device(device) {
    UINT numViewports = 1;
    device->RSGetViewports(&numViewports, &viewport);
}


SaveViewportScope::~SaveViewportScope() {
    device->RSSetViewports(1, &viewport);
}


D3D10_VIEWPORT Utils::viewportFromView(ID3D10View *view) {
    ID3D10Texture2D *texture2D;
    view->GetResource(reinterpret_cast<ID3D10Resource **>(&texture2D));
    D3D10_VIEWPORT viewport = viewportFromTexture2D(texture2D);
    texture2D->Release();
    return viewport;
}

D3D10_VIEWPORT Utils::viewportFromTexture2D(ID3D10Texture2D *texture2D) {
    D3D10_TEXTURE2D_DESC desc;
    texture2D->GetDesc(&desc);

    D3D10_VIEWPORT viewport;
    viewport.TopLeftX = 0;
    viewport.TopLeftY = 0;
    viewport.Width = desc.Width;
    viewport.Height = desc.Height;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    return viewport;
}
