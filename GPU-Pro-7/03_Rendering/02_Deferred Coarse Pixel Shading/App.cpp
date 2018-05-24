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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified by StephanieB5 to remove dependencies on DirectX SDK in 2017
//
/////////////////////////////////////////////////////////////////////////////////////////////

#include "App.h"
#include "ColorUtil.h"
#include "ShaderDefines.h"
#include <limits>
#include <sstream>
#include <random>
#include <algorithm>

using std::tr1::shared_ptr;
using namespace DirectX;

// NOTE: Must match layout of shader constant buffers

__declspec(align(16))
struct PerFrameConstants
{
    XMFLOAT4X4 mCameraWorldViewProj;
    XMFLOAT4X4 mCameraWorldView;
    XMFLOAT4X4 mCameraViewProj;
    XMFLOAT4X4 mCameraProj;
    XMFLOAT4 mCameraNearFar;

    unsigned int mFramebufferDimensionsX;
    unsigned int mFramebufferDimensionsY;
    unsigned int mFramebufferDimensionsZ;
    unsigned int mFramebufferDimensionsW;

    UIConstants mUI;
};

#ifdef _PERF
float GetGPUCounterSeconds( ID3D11DeviceContext* d3dDeviceContext, ID3D11Query** query )
{
    // Get GPU counters
    UINT64 queryTimeA, queryTimeB;
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT queryTimeData;

    while( S_OK != d3dDeviceContext->GetData(query[0], &queryTimeA, 
        sizeof(UINT64), 0) ) {}
    while( S_OK != d3dDeviceContext->GetData(query[1], &queryTimeB, 
        sizeof(UINT64), 0) ) {}
    while( S_OK != d3dDeviceContext->GetData(query[2], &queryTimeData, 
        sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT), 0) ) {}

    if (0 == queryTimeData.Disjoint) {
        UINT64 deltaTimeTicks = queryTimeB - queryTimeA;
        return (float)deltaTimeTicks / (float)queryTimeData.Frequency;  
    } else {
        return 0.0f;
    }
}
#endif // _PERF


App::App(ID3D11Device *d3dDevice, unsigned int activeLights, unsigned int msaaSamples)
    : mMSAASamples(msaaSamples)
    , mTotalTime(0.0f)
    , mActiveLights(0)
    , mLightBuffer(0)
    , mDepthBufferReadOnlyDSV(0)
{
    std::string msaaSamplesStr;
    {
        std::ostringstream oss;
        oss << mMSAASamples;
        msaaSamplesStr = oss.str();
    }

    // Set up macros
    D3D10_SHADER_MACRO defines[] = {
        {"MSAA_SAMPLES", msaaSamplesStr.c_str()},
        {0, 0}
    };

    // Create shaders
    mGeometryVS = new VertexShader(d3dDevice, L"Rendering.hlsl", "GeometryVS", defines);

    mGBufferPS = new PixelShader(d3dDevice, L"GBuffer.hlsl", "GBufferPS", defines);
    mGBufferAlphaTestPS = new PixelShader(d3dDevice, L"GBuffer.hlsl", "GBufferAlphaTestPS", defines);

    mForwardPS = new PixelShader(d3dDevice, L"Forward.hlsl", "ForwardPS", defines);
    mForwardAlphaTestPS = new PixelShader(d3dDevice, L"Forward.hlsl", "ForwardAlphaTestPS", defines);
    mForwardAlphaTestOnlyPS = new PixelShader(d3dDevice, L"Forward.hlsl", "ForwardAlphaTestOnlyPS", defines);
    
    mFullScreenTriangleVS = new VertexShader(d3dDevice, L"Rendering.hlsl", "FullScreenTriangleVS", defines);

    mSkyboxVS = new VertexShader(d3dDevice, L"SkyboxToneMap.hlsl", "SkyboxVS", defines);
    mSkyboxPS = new PixelShader(d3dDevice, L"SkyboxToneMap.hlsl", "SkyboxPS", defines);
    
    mRequiresPerSampleShadingPS = new PixelShader(d3dDevice, L"GBuffer.hlsl", "RequiresPerSampleShadingPS", defines);

    mBasicLoopPS = new PixelShader(d3dDevice, L"BasicLoop.hlsl", "BasicLoopPS", defines);
    mBasicLoopPerSamplePS = new PixelShader(d3dDevice, L"BasicLoop.hlsl", "BasicLoopPerSamplePS", defines);
    mComputeShaderTileCS = new ComputeShader(d3dDevice, NULL, L"ComputeShaderTileCS.fxo");

    mGPUQuadVS = new VertexShader(d3dDevice, L"GPUQuad.hlsl", "GPUQuadVS", defines);
    mGPUQuadGS = new GeometryShader(d3dDevice, L"GPUQuad.hlsl", "GPUQuadGS", defines);
    mGPUQuadPS = new PixelShader(d3dDevice, L"GPUQuad.hlsl", "GPUQuadPS", defines);
    mGPUQuadPerSamplePS = new PixelShader(d3dDevice, L"GPUQuad.hlsl", "GPUQuadPerSamplePS", defines);

    mGPUQuadDLPS = new PixelShader(d3dDevice, L"GPUQuadDL.hlsl", "GPUQuadDLPS", defines);
    mGPUQuadDLPerSamplePS = new PixelShader(d3dDevice, L"GPUQuadDL.hlsl", "GPUQuadDLPerSamplePS", defines);

    mGPUQuadDLResolvePS = new PixelShader(d3dDevice, L"GPUQuadDL.hlsl", "GPUQuadDLResolvePS", defines);
    mGPUQuadDLResolvePerSamplePS = new PixelShader(d3dDevice, L"GPUQuadDL.hlsl", "GPUQuadDLResolvePerSamplePS", defines);

    // Create input layout
    {
        // We need the vertex shader bytecode for this... rather than try to wire that all through the
        // shader interface, just recompile the vertex shader.
        UINT shaderFlags = D3D10_SHADER_ENABLE_STRICTNESS | D3D10_SHADER_PACK_MATRIX_ROW_MAJOR;
        ID3D10Blob *bytecode = 0;
        HRESULT hr = D3DCompileFromFile(L"Rendering.hlsl", defines, D3D_COMPILE_STANDARD_FILE_INCLUDE, "GeometryVS", "vs_5_0", shaderFlags, 0, &bytecode, 0);
        if (FAILED(hr)) {
            assert(false);      // It worked earlier...
        }

        const D3D11_INPUT_ELEMENT_DESC layout[] =
        {
            {"position",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"normal",    0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"texCoord",  0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        
        d3dDevice->CreateInputLayout( 
            layout, ARRAYSIZE(layout), 
            bytecode->GetBufferPointer(),
            bytecode->GetBufferSize(), 
            &mMeshVertexLayout);

        bytecode->Release();
    }

    // Create standard rasterizer state
    {
        CD3D11_RASTERIZER_DESC desc(D3D11_DEFAULT);
        d3dDevice->CreateRasterizerState(&desc, &mRasterizerState);

        desc.CullMode = D3D11_CULL_NONE;
        d3dDevice->CreateRasterizerState(&desc, &mDoubleSidedRasterizerState);
    }
    
    {
        CD3D11_DEPTH_STENCIL_DESC desc(D3D11_DEFAULT);
        // NOTE: Complementary Z => GREATER test
        desc.DepthFunc = D3D11_COMPARISON_GREATER_EQUAL;
        d3dDevice->CreateDepthStencilState(&desc, &mDepthState);
    }

    // Stencil states for MSAA
    {
        CD3D11_DEPTH_STENCIL_DESC desc(
            FALSE, D3D11_DEPTH_WRITE_MASK_ZERO, D3D11_COMPARISON_GREATER_EQUAL,   // Depth
            TRUE, 0xFF, 0xFF,                                                     // Stencil
            D3D11_STENCIL_OP_REPLACE, D3D11_STENCIL_OP_REPLACE, D3D11_STENCIL_OP_REPLACE, D3D11_COMPARISON_ALWAYS, // Front face stencil
            D3D11_STENCIL_OP_REPLACE, D3D11_STENCIL_OP_REPLACE, D3D11_STENCIL_OP_REPLACE, D3D11_COMPARISON_ALWAYS  // Back face stencil
            );
        d3dDevice->CreateDepthStencilState(&desc, &mWriteStencilState);
    }
    {
        CD3D11_DEPTH_STENCIL_DESC desc(
            TRUE, D3D11_DEPTH_WRITE_MASK_ZERO, D3D11_COMPARISON_GREATER_EQUAL,    // Depth
            TRUE, 0xFF, 0xFF,                                                     // Stencil
            D3D11_STENCIL_OP_KEEP, D3D11_STENCIL_OP_KEEP, D3D11_STENCIL_OP_KEEP, D3D11_COMPARISON_EQUAL, // Front face stencil
            D3D11_STENCIL_OP_KEEP, D3D11_STENCIL_OP_KEEP, D3D11_STENCIL_OP_KEEP, D3D11_COMPARISON_EQUAL  // Back face stencil
            );
        d3dDevice->CreateDepthStencilState(&desc, &mEqualStencilState);
    }

    // Create geometry phase blend state
    {
        CD3D11_BLEND_DESC desc(D3D11_DEFAULT);
        d3dDevice->CreateBlendState(&desc, &mGeometryBlendState);
    }

    // Create lighting phase blend state
    {
        CD3D11_BLEND_DESC desc(D3D11_DEFAULT);
        // Additive blending
        desc.RenderTarget[0].BlendEnable = true;
        desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
        desc.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
        desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
        desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        d3dDevice->CreateBlendState(&desc, &mLightingBlendState);
    }

    // Create constant buffers
    {
        CD3D11_BUFFER_DESC desc(
            sizeof(PerFrameConstants),
            D3D11_BIND_CONSTANT_BUFFER,
            D3D11_USAGE_DYNAMIC,
            D3D11_CPU_ACCESS_WRITE);

        d3dDevice->CreateBuffer(&desc, 0, &mPerFrameConstants);
    }

    // Create sampler state
    {
        CD3D11_SAMPLER_DESC desc(D3D11_DEFAULT);
        desc.Filter = D3D11_FILTER_ANISOTROPIC;
        desc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
        desc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
        desc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        desc.MaxAnisotropy = 16;
        d3dDevice->CreateSamplerState(&desc, &mDiffuseSampler);
        desc.MipLODBias = 1;
        d3dDevice->CreateSamplerState(&desc, &mDPSDiffuseSampler);
    }

    // Create skybox mesh
    mSkyboxMesh.Create(d3dDevice, L"Media\\Skybox\\Skybox.sdkmesh");

#ifdef _PERF
    // Initialize queries for timers.
    D3D11_QUERY_DESC desc0 = {D3D11_QUERY_TIMESTAMP, 0};
    D3D11_QUERY_DESC desc1 = {D3D11_QUERY_TIMESTAMP_DISJOINT, 0};
    for (UINT i = 0; i < TIMER_COUNT; ++i) {
        d3dDevice->CreateQuery(&desc0, &m_pTimers[i][T_BEGIN]);
        d3dDevice->CreateQuery(&desc0, &m_pTimers[i][T_END]);
        d3dDevice->CreateQuery(&desc1, &m_pTimers[i][T_VALID]);
    }
#endif // _PERF

    // StephanieB5: removed unused parameter ID3D11Device* d3dDevice
    InitializeLightParameters();
    SetActiveLights(d3dDevice, activeLights);
}


App::~App() 
{
    mSkyboxMesh.Destroy();

#ifdef _PERF
    for (UINT i = 0; i < TIMER_COUNT; ++i) {
        SAFE_RELEASE(m_pTimers[i][T_BEGIN]);
        SAFE_RELEASE(m_pTimers[i][T_END]);
        SAFE_RELEASE(m_pTimers[i][T_VALID]);
    }
#endif // _PERF

    SAFE_RELEASE(mDepthBufferReadOnlyDSV);
    delete mLightBuffer;
    SAFE_RELEASE(mDiffuseSampler);
    SAFE_RELEASE(mDPSDiffuseSampler);
    SAFE_RELEASE(mPerFrameConstants);
    SAFE_RELEASE(mLightingBlendState);
    SAFE_RELEASE(mGeometryBlendState);
    SAFE_RELEASE(mEqualStencilState);
    SAFE_RELEASE(mWriteStencilState);
    SAFE_RELEASE(mDepthState);
    SAFE_RELEASE(mDoubleSidedRasterizerState);
    SAFE_RELEASE(mRasterizerState);
    SAFE_RELEASE(mMeshVertexLayout);
    delete mSkyboxPS;
    delete mSkyboxVS;
    delete mComputeShaderTileCS;
    delete mGPUQuadDLResolvePerSamplePS;
    delete mGPUQuadDLResolvePS;
    delete mGPUQuadDLPerSamplePS;
    delete mGPUQuadDLPS;
    delete mGPUQuadPerSamplePS;
    delete mGPUQuadPS;
    delete mGPUQuadGS;
    delete mGPUQuadVS;
    delete mRequiresPerSampleShadingPS;
    delete mBasicLoopPerSamplePS;
    delete mBasicLoopPS;
    delete mFullScreenTriangleVS;
    delete mForwardAlphaTestOnlyPS;
    delete mForwardAlphaTestPS;
    delete mForwardPS;
    delete mGBufferAlphaTestPS;
    delete mGBufferPS;
    delete mGeometryVS;
}


void App::OnD3D11ResizedSwapChain(ID3D11Device* d3dDevice,
                                  const DXGI_SURFACE_DESC* backBufferDesc)
{
    mGBufferWidth = backBufferDesc->Width;
    mGBufferHeight = backBufferDesc->Height;

    // Create/recreate any textures related to screen size
    mGBuffer.resize(0);
    mGBufferRTV.resize(0);
    mGBufferSRV.resize(0);
    mLitBufferPS = 0;
    mLitBufferCS = 0;
    mDeferredLightingAccumBuffer = 0;
    mDepthBuffer = 0;
    SAFE_RELEASE(mDepthBufferReadOnlyDSV);

    DXGI_SAMPLE_DESC sampleDesc;
    sampleDesc.Count = mMSAASamples;
    sampleDesc.Quality = 0;

    // standard depth/stencil buffer
    mDepthBuffer = shared_ptr<Depth2D>(new Depth2D(
        d3dDevice, mGBufferWidth, mGBufferHeight,
        D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE,
        sampleDesc,
        mMSAASamples > 1    // Include stencil if using MSAA
        ));

    // read-only depth stencil view
    {
        D3D11_DEPTH_STENCIL_VIEW_DESC desc;
        mDepthBuffer->GetDepthStencil()->GetDesc(&desc);
        desc.Flags = D3D11_DSV_READ_ONLY_DEPTH;

        d3dDevice->CreateDepthStencilView(mDepthBuffer->GetTexture(), &desc, &mDepthBufferReadOnlyDSV);
    }

    // NOTE: The next set of buffers are not all needed at the same time... a given technique really only needs one of them.
    // We allocate them all up front for quick swapping between techniques and to keep the code as simple as possible.

    // lit buffers
    mLitBufferPS = shared_ptr<Texture2D>(new Texture2D(
        d3dDevice, mGBufferWidth, mGBufferHeight, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        sampleDesc));

    mLitBufferCS = shared_ptr< StructuredBuffer<FramebufferFlatElement> >(new StructuredBuffer<FramebufferFlatElement>(
        d3dDevice, mGBufferWidth * mGBufferHeight * mMSAASamples,
        D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE));

    // deferred lighting accumulation buffer
    mDeferredLightingAccumBuffer = shared_ptr<Texture2D>(new Texture2D(
        d3dDevice, mGBufferWidth, mGBufferHeight, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        sampleDesc));


    // G-Buffer

    // normal_specular
    mGBuffer.push_back(shared_ptr<Texture2D>(new Texture2D(
        d3dDevice, mGBufferWidth, mGBufferHeight, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        sampleDesc)));

    // albedo
    mGBuffer.push_back(shared_ptr<Texture2D>(new Texture2D(
        d3dDevice, mGBufferWidth, mGBufferHeight, DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        sampleDesc)));

    // sampled w/ biased sampler.
    mGBuffer.push_back(shared_ptr<Texture2D>(new Texture2D(
        d3dDevice, mGBufferWidth, mGBufferHeight, DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        sampleDesc)));

    // positionZgrad
    mGBuffer.push_back(shared_ptr<Texture2D>(new Texture2D(
        d3dDevice, mGBufferWidth, mGBufferHeight, DXGI_FORMAT_R16G16_FLOAT,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        sampleDesc)));

    // view space Z
    mGBuffer.push_back(shared_ptr<Texture2D>(new Texture2D(
        d3dDevice, mGBufferWidth, mGBufferHeight, DXGI_FORMAT_R32_FLOAT,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        sampleDesc)));


    // Set up GBuffer resource list
    mGBufferRTV.resize(mGBuffer.size(), 0);
    mGBufferSRV.resize(mGBuffer.size(), 0);
    for (std::size_t i = 0; i < mGBuffer.size(); ++i) {
        mGBufferRTV[i] = mGBuffer[i]->GetRenderTarget();
        mGBufferSRV[i] = mGBuffer[i]->GetShaderResource();
    }
//     // Depth buffer is the last SRV that we use for reading
//     mGBufferSRV.back() = mDepthBuffer->GetShaderResource();
}

// StephanieB5: removed unused parameter ID3D11Device* d3dDevice
void App::InitializeLightParameters()
{
    mPointLightParameters.resize(MAX_LIGHTS);
    mLightInitialTransform.resize(MAX_LIGHTS);
    mPointLightPositionWorld.resize(MAX_LIGHTS);

    // Use a constant seed for consistency
    std::tr1::mt19937 rng(1337);

    std::tr1::uniform_real<float> radiusNormDist(0.0f, 1.0f);
    const float maxRadius = 100.0f;
    std::tr1::uniform_real<float> angleDist(0.0f, 2.0f * XM_PI); 
    std::tr1::uniform_real<float> heightDist(0.0f, 20.0f);
    std::tr1::uniform_real<float> animationSpeedDist(2.0f, 20.0f);
    std::tr1::uniform_int<int> animationDirection(0, 1);
    std::tr1::uniform_real<float> hueDist(0.0f, 1.0f);
    std::tr1::uniform_real<float> intensityDist(0.1f, 0.5f);
    std::tr1::uniform_real<float> attenuationDist(2.0f, 15.0f);
    const float attenuationStartFactor = 0.8f;
    
    for (unsigned int i = 0; i < MAX_LIGHTS; ++i) {
        PointLight& params = mPointLightParameters[i];
        PointLightInitTransform& init = mLightInitialTransform[i];

        init.radius = std::sqrt(radiusNormDist(rng)) * maxRadius;
        init.angle = angleDist(rng);
        init.height = heightDist(rng);
        // Normalize by arc length
        init.animationSpeed = (animationDirection(rng) * 2 - 1) * animationSpeedDist(rng) / init.radius;
        
        // HSL->RGB, vary light hue
        XMVECTOR color = intensityDist(rng) * HueToRGB(hueDist(rng));
        XMStoreFloat3(&params.color, color);
        params.attenuationEnd = attenuationDist(rng);
        params.attenuationBegin = attenuationStartFactor * params.attenuationEnd;
    }
}


void App::SetActiveLights(ID3D11Device* d3dDevice, unsigned int activeLights)
{
    mActiveLights = activeLights;

    delete mLightBuffer;
    mLightBuffer = new StructuredBuffer<PointLight>(d3dDevice, activeLights, D3D11_BIND_SHADER_RESOURCE, true);
    
    // Make sure all the active lights are set up
    Move(0.0f);
}


void App::Move(float elapsedTime)
{
    mTotalTime += elapsedTime;

    // Update positions of active lights
    for (unsigned int i = 0; i < mActiveLights; ++i) {
        const PointLightInitTransform& initTransform = mLightInitialTransform[i];
        float angle = initTransform.angle + mTotalTime * initTransform.animationSpeed;
        mPointLightPositionWorld[i] = XMFLOAT3(
            initTransform.radius * std::cos(angle),
            initTransform.height,
            initTransform.radius * std::sin(angle));
    }
}


void App::Render(ID3D11DeviceContext* d3dDeviceContext, 
                 ID3D11RenderTargetView* backBuffer,
                 CDXUTSDKMesh& mesh_opaque,
                 CDXUTSDKMesh& mesh_alpha,
                 ID3D11ShaderResourceView* skybox,
                 const CXMMATRIX worldMatrix,
                 const CFirstPersonCamera* viewerCamera,
                 const D3D11_VIEWPORT* viewport,
                 const UIConstants* ui)
{
    XMMATRIX cameraProj = viewerCamera->GetProjMatrix();
    XMMATRIX cameraView = viewerCamera->GetViewMatrix();
    
    XMMATRIX cameraViewInv = XMMatrixInverse(nullptr, cameraView);
        
    // Compute composite matrices
    XMMATRIX cameraViewProj = cameraView * cameraProj;
    XMMATRIX cameraWorldViewProj = worldMatrix * cameraViewProj;

    // Fill in frame constants
    {
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        d3dDeviceContext->Map(mPerFrameConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
        PerFrameConstants* constants = static_cast<PerFrameConstants *>(mappedResource.pData);

        XMStoreFloat4x4(&constants->mCameraWorldViewProj, cameraWorldViewProj);
        XMStoreFloat4x4(&constants->mCameraWorldView, worldMatrix * cameraView);
        XMStoreFloat4x4(&constants->mCameraViewProj, cameraViewProj);
        XMStoreFloat4x4(&constants->mCameraProj, cameraProj);
        // NOTE: Complementary Z => swap near/far back
        constants->mCameraNearFar = XMFLOAT4(viewerCamera->GetFarClip(), viewerCamera->GetNearClip(), 0.0f, 0.0f);

        constants->mFramebufferDimensionsX = mGBufferWidth;
        constants->mFramebufferDimensionsY = mGBufferHeight;
        constants->mFramebufferDimensionsZ = 0;     // Unused
        constants->mFramebufferDimensionsW = 0;     // Unused

        constants->mUI = *ui;
        
        d3dDeviceContext->Unmap(mPerFrameConstants, 0);
    }

    // Geometry phase
    if (mesh_opaque.IsLoaded()) {
        mesh_opaque.ComputeInFrustumFlags(cameraWorldViewProj);
    }
    if (mesh_alpha.IsLoaded()) {
        mesh_alpha.ComputeInFrustumFlags(cameraWorldViewProj);
    }

    // Setup lights
    ID3D11ShaderResourceView *lightBufferSRV = SetupLights(d3dDeviceContext, cameraView);

    // Forward rendering takes a different path here
    // StephanieB5 removed unnecessary parameters from RenderForward and RenderGBuffer
    // The parameters were CFirstPersonCamera* viewerCamera and UIConstants* ui
    if (ui->lightCullTechnique == CULL_FORWARD_NONE) {
        RenderForward(d3dDeviceContext, mesh_opaque, mesh_alpha, lightBufferSRV, viewport, false);
    } else if (ui->lightCullTechnique == CULL_FORWARD_PREZ_NONE) {
        RenderForward(d3dDeviceContext, mesh_opaque, mesh_alpha, lightBufferSRV, viewport, true);
    } else {
        RenderGBuffer(d3dDeviceContext, mesh_opaque, mesh_alpha, viewport);
        ComputeLighting(d3dDeviceContext, lightBufferSRV, viewport, ui);
    }

    // Render skybox and tonemap
    RenderSkyboxAndToneMap(d3dDeviceContext, backBuffer, skybox,
        mDepthBuffer->GetShaderResource(), viewport, ui);

#ifdef _PERF
    m_frameStats.m_totalGBuffGen = 1000 * GetGPUCounterSeconds(d3dDeviceContext, &m_pTimers[TIMER_GBUFFER_GEN][0]);
    m_frameStats.m_totalShadingTime= 1000 * GetGPUCounterSeconds(d3dDeviceContext, &m_pTimers[TIMER_SHADING][0]);
    m_frameStats.m_totalSkyBox = 1000 * GetGPUCounterSeconds(d3dDeviceContext, &m_pTimers[TIMER_SKYBOX][0]);
#endif // _PERF

}


ID3D11ShaderResourceView * App::SetupLights(ID3D11DeviceContext* d3dDeviceContext,
                                            const CXMMATRIX cameraView)
{
    // Transform light world positions into view space and store in our parameters array
    XMVector3TransformCoordStream(&mPointLightParameters[0].positionView, sizeof(PointLight),
        &mPointLightPositionWorld[0], sizeof(XMFLOAT3), mActiveLights, cameraView);
    
    // Copy light list into shader buffer
    {
        PointLight* light = mLightBuffer->MapDiscard(d3dDeviceContext);
        for (unsigned int i = 0; i < mActiveLights; ++i) {
            light[i] = mPointLightParameters[i];
        }
        mLightBuffer->Unmap(d3dDeviceContext);
    }
    
    return mLightBuffer->GetShaderResource();
}


// StephanieB5: removed unused parameters CFirstPersonCamera* viewerCamera & UIConstants* ui
ID3D11ShaderResourceView * App::RenderForward(ID3D11DeviceContext* d3dDeviceContext,
                                              CDXUTSDKMesh& mesh_opaque,
                                              CDXUTSDKMesh& mesh_alpha,
                                              ID3D11ShaderResourceView *lightBufferSRV,
                                              const D3D11_VIEWPORT* viewport,
                                              bool doPreZ)
{
    // Clear lit and depth buffer
    const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    d3dDeviceContext->ClearRenderTargetView(mLitBufferPS->GetRenderTarget(), zeros);
    // NOTE: Complementary Z buffer: clear to 0 (far)!
    d3dDeviceContext->ClearDepthStencilView(mDepthBuffer->GetDepthStencil(), D3D11_CLEAR_DEPTH, 0.0f, 0);

    d3dDeviceContext->IASetInputLayout(mMeshVertexLayout);

    d3dDeviceContext->VSSetConstantBuffers(0, 1, &mPerFrameConstants);
    d3dDeviceContext->VSSetShader(mGeometryVS->GetShader(), 0, 0);
    
    d3dDeviceContext->GSSetShader(0, 0, 0);

    d3dDeviceContext->RSSetViewports(1, viewport);

    d3dDeviceContext->PSSetConstantBuffers(0, 1, &mPerFrameConstants);
    d3dDeviceContext->PSSetShaderResources(5, 1, &lightBufferSRV);
    d3dDeviceContext->PSSetSamplers(0, 1, &mDiffuseSampler);
    // Diffuse texture set per-material by DXUT mesh routines

    d3dDeviceContext->OMSetDepthStencilState(mDepthState, 0);
    
    // Pre-Z pass if requested
    if (doPreZ) {
        d3dDeviceContext->OMSetRenderTargets(0, 0, mDepthBuffer->GetDepthStencil());
            
        // Render opaque geometry
        if (mesh_opaque.IsLoaded()) {
            d3dDeviceContext->RSSetState(mRasterizerState);
            d3dDeviceContext->PSSetShader(0, 0, 0);
            mesh_opaque.Render(d3dDeviceContext, 0);
        }

        // Render alpha tested geometry
        if (mesh_alpha.IsLoaded()) {
            d3dDeviceContext->RSSetState(mDoubleSidedRasterizerState);
            // NOTE: Use simplified alpha test shader that only clips
            d3dDeviceContext->PSSetShader(mForwardAlphaTestOnlyPS->GetShader(), 0, 0);
            mesh_alpha.Render(d3dDeviceContext, 0);
        }
    }

    // Set up render targets
    ID3D11RenderTargetView *renderTargets[1] = {mLitBufferPS->GetRenderTarget()};
    d3dDeviceContext->OMSetRenderTargets(1, renderTargets, mDepthBuffer->GetDepthStencil());
    d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);
    
    // Render opaque geometry
    if (mesh_opaque.IsLoaded()) {
        d3dDeviceContext->RSSetState(mRasterizerState);
        d3dDeviceContext->PSSetShader(mForwardPS->GetShader(), 0, 0);
        mesh_opaque.Render(d3dDeviceContext, 0);
    }

    // Render alpha tested geometry
    if (mesh_alpha.IsLoaded()) {
        d3dDeviceContext->RSSetState(mDoubleSidedRasterizerState);
        d3dDeviceContext->PSSetShader(mForwardAlphaTestPS->GetShader(), 0, 0);
        mesh_alpha.Render(d3dDeviceContext, 0);
    }

    // Cleanup (aka make the runtime happy)
    d3dDeviceContext->OMSetRenderTargets(0, 0, 0);

    return mLitBufferPS->GetShaderResource();
}

// StephanieB5: removed unused parameters CFirstPersonCamera* viewerCamera & UIConstants* ui
void App::RenderGBuffer(ID3D11DeviceContext* d3dDeviceContext,
                        CDXUTSDKMesh& mesh_opaque,
                        CDXUTSDKMesh& mesh_alpha,
                        const D3D11_VIEWPORT* viewport)
{
#ifdef _PERF
    d3dDeviceContext->Begin(m_pTimers[TIMER_GBUFFER_GEN][T_VALID]);
    d3dDeviceContext->End(m_pTimers[TIMER_GBUFFER_GEN][T_BEGIN]);
#endif // _PERF

    // Clear GBuffer
    // NOTE: We actually only need to clear the depth buffer here since we replace unwritten (i.e. far plane) samples
    // with the skybox. We use the depth buffer to reconstruct position and only in-frustum positions are shaded.
    // NOTE: Complementary Z buffer: clear to 0 (far)!
    d3dDeviceContext->ClearDepthStencilView(mDepthBuffer->GetDepthStencil(), D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0.0f, 0);

    d3dDeviceContext->IASetInputLayout(mMeshVertexLayout);

    d3dDeviceContext->VSSetConstantBuffers(0, 1, &mPerFrameConstants);
    d3dDeviceContext->VSSetShader(mGeometryVS->GetShader(), 0, 0);
    
    d3dDeviceContext->GSSetShader(0, 0, 0);

    d3dDeviceContext->RSSetViewports(1, viewport);

    d3dDeviceContext->PSSetConstantBuffers(0, 1, &mPerFrameConstants);
    d3dDeviceContext->PSSetSamplers(0, 1, &mDiffuseSampler);
    d3dDeviceContext->PSSetSamplers(1, 1, &mDPSDiffuseSampler);

    // Diffuse texture set per-material by DXUT mesh routines

    // Set up render GBuffer render targets
    d3dDeviceContext->OMSetDepthStencilState(mDepthState, 0);
    d3dDeviceContext->OMSetRenderTargets(static_cast<UINT>(mGBufferRTV.size()), &mGBufferRTV.front(), mDepthBuffer->GetDepthStencil());
    d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);
    
    // Render opaque geometry
    if (mesh_opaque.IsLoaded()) {
        d3dDeviceContext->RSSetState(mRasterizerState);
        d3dDeviceContext->PSSetShader(mGBufferPS->GetShader(), 0, 0);
        mesh_opaque.Render(d3dDeviceContext, 0);
    }

    // Render alpha tested geometry
    if (mesh_alpha.IsLoaded()) {
        d3dDeviceContext->RSSetState(mDoubleSidedRasterizerState);
        d3dDeviceContext->PSSetShader(mGBufferAlphaTestPS->GetShader(), 0, 0);
        mesh_alpha.Render(d3dDeviceContext, 0);
    }

    // Cleanup (aka make the runtime happy)
    d3dDeviceContext->OMSetRenderTargets(0, 0, 0);
#ifdef _PERF
    d3dDeviceContext->End(m_pTimers[TIMER_GBUFFER_GEN][T_END]);
    d3dDeviceContext->End(m_pTimers[TIMER_GBUFFER_GEN][T_VALID]);
#endif // _PERF
}


void App::ComputeLighting(ID3D11DeviceContext* d3dDeviceContext,
                          ID3D11ShaderResourceView *lightBufferSRV,
                          const D3D11_VIEWPORT* viewport,
                          const UIConstants* ui)
{
    // TODO: Clean up the branchiness here a bit... refactor into small functions
         
    switch (ui->lightCullTechnique) {

    case CULL_COMPUTE_SHADER_TILE:
    {
#ifdef _PERF
        d3dDeviceContext->Begin(m_pTimers[TIMER_SHADING][T_VALID]);
        d3dDeviceContext->End(m_pTimers[TIMER_SHADING][T_BEGIN]);
#endif // _PERF

        // No need to clear, we write all pixels
        
        // Compute shader setup (always does all the lights at once)
        d3dDeviceContext->CSSetConstantBuffers(0, 1, &mPerFrameConstants);
        d3dDeviceContext->CSSetShaderResources(0, static_cast<UINT>(mGBufferSRV.size()), &mGBufferSRV.front());
        d3dDeviceContext->CSSetShaderResources(5, 1, &lightBufferSRV);

        ID3D11UnorderedAccessView *litBufferUAV = mLitBufferCS->GetUnorderedAccess();
        d3dDeviceContext->CSSetUnorderedAccessViews(0, 1, &litBufferUAV, 0);
        d3dDeviceContext->CSSetShader(mComputeShaderTileCS->GetShader(), 0, 0);

        // Dispatch
        unsigned int dispatchWidth = (mGBufferWidth + COMPUTE_SHADER_TILE_GROUP_DIM - 1) / COMPUTE_SHADER_TILE_GROUP_DIM;
        unsigned int dispatchHeight = (mGBufferHeight + COMPUTE_SHADER_TILE_GROUP_DIM - 1) / COMPUTE_SHADER_TILE_GROUP_DIM;
        d3dDeviceContext->Dispatch(dispatchWidth, dispatchHeight, 1);
#ifdef _PERF
        d3dDeviceContext->End(m_pTimers[TIMER_SHADING][T_END]);
        d3dDeviceContext->End(m_pTimers[TIMER_SHADING][T_VALID]);
#endif // _PERF

    }
    break;

    case CULL_QUAD:
    case CULL_QUAD_DEFERRED_LIGHTING: {
        bool deferredLighting = (ui->lightCullTechnique == CULL_QUAD_DEFERRED_LIGHTING);
        std::tr1::shared_ptr<Texture2D> &accumulateBuffer = deferredLighting ? mDeferredLightingAccumBuffer : mLitBufferPS;

        // Clear
        const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        d3dDeviceContext->ClearRenderTargetView(accumulateBuffer->GetRenderTarget(), zeros);
        
        if (mMSAASamples > 1) {
            // Full screen triangle setup
            d3dDeviceContext->IASetInputLayout(0);
            d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            d3dDeviceContext->IASetVertexBuffers(0, 0, 0, 0, 0);

            d3dDeviceContext->VSSetShader(mFullScreenTriangleVS->GetShader(), 0, 0);
            d3dDeviceContext->GSSetShader(0, 0, 0);

            d3dDeviceContext->RSSetState(mRasterizerState);
            d3dDeviceContext->RSSetViewports(1, viewport);

            d3dDeviceContext->PSSetConstantBuffers(0, 1, &mPerFrameConstants);
            d3dDeviceContext->PSSetShaderResources(0, static_cast<UINT>(mGBufferSRV.size()), &mGBufferSRV.front());
            d3dDeviceContext->PSSetShaderResources(5, 1, &lightBufferSRV);

            // Set stencil mask for samples that require per-sample shading
            d3dDeviceContext->PSSetShader(mRequiresPerSampleShadingPS->GetShader(), 0, 0);
            d3dDeviceContext->OMSetDepthStencilState(mWriteStencilState, 1);
            d3dDeviceContext->OMSetRenderTargets(0, 0, mDepthBufferReadOnlyDSV);
            d3dDeviceContext->Draw(3, 0);
        }

        // Point primitives expanded into quads in the geometry shader
        d3dDeviceContext->IASetInputLayout(0);
        d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
        d3dDeviceContext->IASetVertexBuffers(0, 0, 0, 0, 0);

        d3dDeviceContext->VSSetConstantBuffers(0, 1, &mPerFrameConstants);
        d3dDeviceContext->VSSetShaderResources(5, 1, &lightBufferSRV);
        d3dDeviceContext->VSSetShader(mGPUQuadVS->GetShader(), 0, 0);

        d3dDeviceContext->GSSetShader(mGPUQuadGS->GetShader(), 0, 0);

        d3dDeviceContext->RSSetState(mRasterizerState);
        d3dDeviceContext->RSSetViewports(1, viewport);

        d3dDeviceContext->PSSetConstantBuffers(0, 1, &mPerFrameConstants);
        d3dDeviceContext->PSSetShaderResources(0, static_cast<UINT>(mGBufferSRV.size()), &mGBufferSRV.front());
        d3dDeviceContext->PSSetShaderResources(5, 1, &lightBufferSRV);

        // Additively blend into lit buffer        
        ID3D11RenderTargetView * renderTargets[1] = {accumulateBuffer->GetRenderTarget()};
        // Use depth buffer for culling but no writes (use the read-only DSV)
        d3dDeviceContext->OMSetRenderTargets(1, renderTargets, mDepthBufferReadOnlyDSV);
        d3dDeviceContext->OMSetBlendState(mLightingBlendState, 0, 0xFFFFFFFF);
        
        // Dispatch one point per light

        // Do pixel frequency shading
        d3dDeviceContext->PSSetShader(deferredLighting ? mGPUQuadDLPS->GetShader() : mGPUQuadPS->GetShader(), 0, 0);
        d3dDeviceContext->OMSetDepthStencilState(mEqualStencilState, 0);
        d3dDeviceContext->Draw(mActiveLights, 0);
        
        if (mMSAASamples > 1) {
            // Do sample frequency shading
            d3dDeviceContext->PSSetShader(deferredLighting ? mGPUQuadDLPerSamplePS->GetShader() : mGPUQuadPerSamplePS->GetShader(), 0, 0);
            d3dDeviceContext->OMSetDepthStencilState(mEqualStencilState, 1);
            d3dDeviceContext->Draw(mActiveLights, 0);
        }

        if (deferredLighting) {
            // Final screen-space pass to combine diffuse and specular
            d3dDeviceContext->IASetInputLayout(0);
            d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            d3dDeviceContext->IASetVertexBuffers(0, 0, 0, 0, 0);

            d3dDeviceContext->VSSetShader(mFullScreenTriangleVS->GetShader(), 0, 0);
            d3dDeviceContext->GSSetShader(0, 0, 0);
            
            ID3D11RenderTargetView * resolveRenderTargets[1] = {mLitBufferPS->GetRenderTarget()};
            d3dDeviceContext->OMSetRenderTargets(1, resolveRenderTargets, mDepthBufferReadOnlyDSV);
            d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);

            ID3D11ShaderResourceView * accumulateBufferSRV = accumulateBuffer->GetShaderResource();
            d3dDeviceContext->PSSetShaderResources(7, 1, &accumulateBufferSRV);

            // Do pixel frequency resolve
            d3dDeviceContext->PSSetShader(mGPUQuadDLResolvePS->GetShader(), 0, 0);
            d3dDeviceContext->OMSetDepthStencilState(mEqualStencilState, 0);
            d3dDeviceContext->Draw(3, 0);

            if (mMSAASamples > 1) {
                // Do sample frequency resolve
                d3dDeviceContext->PSSetShader(mGPUQuadDLResolvePerSamplePS->GetShader(), 0, 0);
                d3dDeviceContext->OMSetDepthStencilState(mEqualStencilState, 1);
                d3dDeviceContext->Draw(3, 0);
            }
        }
    }
    break;
    
    case CULL_DEFERRED_NONE:
    {
        // Clear
        const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        d3dDeviceContext->ClearRenderTargetView(mLitBufferPS->GetRenderTarget(), zeros);
        
        // Full screen triangle setup
        d3dDeviceContext->IASetInputLayout(0);
        d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        d3dDeviceContext->IASetVertexBuffers(0, 0, 0, 0, 0);

        d3dDeviceContext->VSSetShader(mFullScreenTriangleVS->GetShader(), 0, 0);
        d3dDeviceContext->GSSetShader(0, 0, 0);

        d3dDeviceContext->RSSetState(mRasterizerState);
        d3dDeviceContext->RSSetViewports(1, viewport);

        d3dDeviceContext->PSSetConstantBuffers(0, 1, &mPerFrameConstants);
        d3dDeviceContext->PSSetShaderResources(0, static_cast<UINT>(mGBufferSRV.size()), &mGBufferSRV.front());
        d3dDeviceContext->PSSetShaderResources(5, 1, &lightBufferSRV);

        if (mMSAASamples > 1) {
            // Set stencil mask for samples that require per-sample shading
            d3dDeviceContext->PSSetShader(mRequiresPerSampleShadingPS->GetShader(), 0, 0);
            d3dDeviceContext->OMSetDepthStencilState(mWriteStencilState, 1);
            d3dDeviceContext->OMSetRenderTargets(0, 0, mDepthBufferReadOnlyDSV);
            d3dDeviceContext->Draw(3, 0);
        }
                
        // Additively blend into back buffer
        ID3D11RenderTargetView * renderTargets[1] = {mLitBufferPS->GetRenderTarget()};
        d3dDeviceContext->OMSetRenderTargets(1, renderTargets, mDepthBufferReadOnlyDSV);
        d3dDeviceContext->OMSetBlendState(mLightingBlendState, 0, 0xFFFFFFFF);

        // Do pixel frequency shading
        d3dDeviceContext->PSSetShader(mBasicLoopPS->GetShader(), 0, 0);
        d3dDeviceContext->OMSetDepthStencilState(mEqualStencilState, 0);
        d3dDeviceContext->Draw(3, 0);

        if (mMSAASamples > 1) {
            // Do sample frequency shading
            d3dDeviceContext->PSSetShader(mBasicLoopPerSamplePS->GetShader(), 0, 0);
            d3dDeviceContext->OMSetDepthStencilState(mEqualStencilState, 1);
            d3dDeviceContext->Draw(3, 0);
        }
    }
    break;

    };  // switch

    // Cleanup (aka make the runtime happy)
    d3dDeviceContext->VSSetShader(0, 0, 0);
    d3dDeviceContext->GSSetShader(0, 0, 0);
    d3dDeviceContext->PSSetShader(0, 0, 0);
    d3dDeviceContext->OMSetRenderTargets(0, 0, 0);
    ID3D11ShaderResourceView* nullSRV[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    d3dDeviceContext->VSSetShaderResources(0, 8, nullSRV);
    d3dDeviceContext->PSSetShaderResources(0, 8, nullSRV);
    d3dDeviceContext->CSSetShaderResources(0, 8, nullSRV);
    ID3D11UnorderedAccessView *nullUAV[1] = {0};
    d3dDeviceContext->CSSetUnorderedAccessViews(0, 1, nullUAV, 0);
}


void App::RenderSkyboxAndToneMap(ID3D11DeviceContext* d3dDeviceContext,
                                 ID3D11RenderTargetView* backBuffer,
                                 ID3D11ShaderResourceView* skybox,
                                 ID3D11ShaderResourceView* depthSRV,
                                 const D3D11_VIEWPORT* viewport,
                                 const UIConstants* ui)
{
#ifdef _PERF
    d3dDeviceContext->Begin(m_pTimers[TIMER_SKYBOX][T_VALID]);
    d3dDeviceContext->End(m_pTimers[TIMER_SKYBOX][T_BEGIN]);
#endif // _PERF

    D3D11_VIEWPORT skyboxViewport(*viewport);
    skyboxViewport.MinDepth = 1.0f;
    skyboxViewport.MaxDepth = 1.0f;

    d3dDeviceContext->IASetInputLayout(mMeshVertexLayout);

    d3dDeviceContext->VSSetConstantBuffers(0, 1, &mPerFrameConstants);
    d3dDeviceContext->VSSetShader(mSkyboxVS->GetShader(), 0, 0);

    d3dDeviceContext->RSSetState(mDoubleSidedRasterizerState);
    d3dDeviceContext->RSSetViewports(1, &skyboxViewport);

    d3dDeviceContext->PSSetConstantBuffers(0, 1, &mPerFrameConstants);
    d3dDeviceContext->PSSetSamplers(0, 1, &mDiffuseSampler);
    d3dDeviceContext->PSSetShader(mSkyboxPS->GetShader(), 0, 0);

    d3dDeviceContext->PSSetShaderResources(5, 1, &skybox);
    d3dDeviceContext->PSSetShaderResources(6, 1, &depthSRV);

    // Bind the appropriate lit buffer depending on the technique
    ID3D11ShaderResourceView* litViews[2] = {0, 0};
    switch (ui->lightCullTechnique) {
    // Compute-shader based techniques use the flattened MSAA buffer
    case CULL_COMPUTE_SHADER_TILE:
        litViews[1] = mLitBufferCS->GetShaderResource();
        break;
    default:
        litViews[0] = mLitBufferPS->GetShaderResource();
        break;
    }
    d3dDeviceContext->PSSetShaderResources(7, 2, litViews);

    d3dDeviceContext->OMSetRenderTargets(1, &backBuffer, 0);
    d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);
    
    mSkyboxMesh.Render(d3dDeviceContext);

    // Cleanup (aka make the runtime happy)
    d3dDeviceContext->OMSetRenderTargets(0, 0, 0);
    ID3D11ShaderResourceView* nullViews[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    d3dDeviceContext->PSSetShaderResources(0, 10, nullViews);
#ifdef _PERF
    d3dDeviceContext->End(m_pTimers[TIMER_SKYBOX][T_END]);
    d3dDeviceContext->End(m_pTimers[TIMER_SKYBOX][T_VALID]);
#endif // _PERF
}
