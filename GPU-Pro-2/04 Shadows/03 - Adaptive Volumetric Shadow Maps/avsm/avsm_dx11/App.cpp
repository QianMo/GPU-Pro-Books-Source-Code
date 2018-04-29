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

#include "App.h"
#include "AVSM_def.h"
#include "ListTexture.h"
#include "Partitions.h"
#include "HairListTexture.h"
#include "ShaderUtils.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

using std::tr1::shared_ptr;

using namespace ShaderUtils;

static const float EMPTY_NODE = 65504.0f; // max half prec

App::App(ID3D11Device *d3dDevice,
         ID3D11DeviceContext* d3dDeviceContext,
         unsigned int nodeCount,
         unsigned int shadowTextureDim,
         unsigned int avsmShadowTextureDim)
    : mShadowTextureDim(shadowTextureDim)
    , mAVSMShadowTextureDim(avsmShadowTextureDim)
    , mLisTexNodeCount(1 << 24)
    // 50% coverage of screen (assuming resolution of 1680x1050
    // with average depth complexity of 50. Should be more than plenty.
    , mHairLTMaxNodeCount(unsigned int(1680 * 1050 * 0.5f * 19)) 
    , mHair(NULL)
    , mLastTime((float)DXUTGetGlobalTimer()->GetAbsoluteTime())
    , mHairMeshDirty(true)
    , mDumpTransmittanceCurve(false)
    , mDumpTransmittanceCurveIndex(0)
    , mDrawTransmittanceMaxNodes(10000)
{
    SetNodeCount(nodeCount);

    UINT shaderFlags = D3D10_SHADER_ENABLE_STRICTNESS | D3D10_SHADER_PACK_MATRIX_ROW_MAJOR;

    // Set up macros
    std::string avsmNodeCountStr;
    {
        std::ostringstream oss;
        oss << mAVSMNodeCount;
        avsmNodeCountStr = oss.str();
    }

    D3D10_SHADER_MACRO shaderDefines[] = {
        {"AVSM_NODE_COUNT", avsmNodeCountStr.c_str()},
        {0, 0}
    };

    // Create geometry pass vertex shaders and input layout
    {
        HRESULT hr;

        // Vertex shader
        ID3D10Blob *vertexShaderBlob = 0;
        hr = D3DX11CompileFromFile(L"Rendering.hlsl", shaderDefines, 0,
                                   "GeometryVS", "vs_5_0",
                                   shaderFlags,
                                   0, 0, &vertexShaderBlob, 0, 0);
        // Do something better here on shader compile failure...
        assert(SUCCEEDED(hr));
        
        hr = d3dDevice->CreateVertexShader(vertexShaderBlob->GetBufferPointer(),
                                           vertexShaderBlob->GetBufferSize(),
                                           0,
                                           &mGeometryVS);
        assert(SUCCEEDED(hr));

        // Create input layout
        const D3D11_INPUT_ELEMENT_DESC layout[] =
        {
            {"position",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"normal",    0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"texCoord",  0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        
        d3dDevice->CreateInputLayout( 
            layout, ARRAYSIZE(layout), 
            vertexShaderBlob->GetBufferPointer(),
            vertexShaderBlob->GetBufferSize(), 
            &mMeshVertexLayout);

        D3DReflect(vertexShaderBlob->GetBufferPointer(),
                   vertexShaderBlob->GetBufferSize(), 
                   IID_ID3D11ShaderReflection, 
                   (void**) &mGeometryVSReflector);

        vertexShaderBlob->Release();      
    }

    // Create geometry pass pixel shader
    CreatePixelShaderFromCompiledObj(d3dDevice,
                                     L"ShaderObjs\\GeometryPS.fxo",
                                     &mGeometryPS,
                                     &mGeometryPSReflector);

    // Create lighting pass shaders
    CreateVertexShaderFromCompiledObj(d3dDevice,
                                      L"ShaderObjs\\FullScreenTriangleVS.fxo",
                                      &mFullScreenTriangleVS,
                                      &mFullScreenTriangleVSReflector);

    // Create AVSM shaders
    CreatePixelShaderFromCompiledObj(d3dDevice,
                                     L"ShaderObjs\\ParticleAVSMCapturePS.fxo",
                                     &mParticleAVSMCapturePS,
                                     &mParticleAVSMCapturePSReflector);

    CreatePixelShadersFromCompiledObjs(d3dDevice, MAX_SHADER_VARIATIONS, 
                                       "ShaderObjs\\AVSM", 
                                       "UnsortedResolvePS.fxo", 
                                       mAVSMUnsortedResolvePS, 
                                       mAVSMUnsortedResolvePSReflector);

    CreatePixelShadersFromCompiledObjs(d3dDevice, MAX_SHADER_VARIATIONS, 
                                       "ShaderObjs\\AVSM", 
                                       "InsertionSortResolvePS.fxo", 
                                       mAVSMInsertionSortResolvePS, 
                                       mAVSMInsertionSortResolvePSReflector);

    // Create other volume shadowing techniques shaders
    CreateComputeShaderFromCompiledObj(d3dDevice,
                                       L"ShaderObjs\\ComputeVisibilityCurveCS.fxo",
                                       &mComputeVisibilityCurveCS,
                                       &mComputeVisibilityCurveCSReflector); 


    CreateVertexShaderFromCompiledObj(d3dDevice,
                                      L"ShaderObjs\\DynamicParticlesShadingVS.fxo",
                                      &mParticleShadingVS,
                                      &mParticleShadingVSReflector); 

    CreatePixelShadersFromCompiledObjs(d3dDevice, MAX_SHADER_VARIATIONS, 
                                       "ShaderObjs\\DynamicParticlesShading", 
                                       "PS.fxo", 
                                       mParticleShadingPS, 
                                       mParticleShadingPSReflector);

    CreatePixelShadersFromCompiledObjs(d3dDevice, MAX_SHADER_VARIATIONS, 
                                       "ShaderObjs\\Lighting", 
                                       "PS.fxo", 
                                       mLightingPS, 
                                       mLightingPSReflector);

    CreatePixelShadersFromCompiledObjs(d3dDevice, MAX_SHADER_VARIATIONS, 
                                       "ShaderObjs\\ParticleAVSM", 
                                       "SinglePassInsertPS.fxo", 
                                       mAVSMSinglePassInsertPS, 
                                       mAVSMSinglePassInsertPSReflector);

    CreatePixelShaderFromCompiledObj(d3dDevice,
                                    L"ShaderObjs\\AVSMClearStructuredBufPS.fxo",
                                    &mAVSMClearStructuredBufPS,
                                    &mAVSMClearStructuredBufPSReflector); 

    CreatePixelShaderFromCompiledObj(d3dDevice,
                                    L"ShaderObjs\\AVSMConvertSUAVtoTex2DPS.fxo",
                                    &mAVSMConvertSUAVtoTex2DPS,
                                    &mAVSMConvertSUAVtoTex2DPSReflector); 

    // Create visualization shaders
    CreatePixelShaderFromCompiledObj(d3dDevice,
                                     L"ShaderObjs\\VisualizeListTexFirstNodeOffsetPS.fxo",
                                     &mVisualizeListTexFirstNodeOffsetPS,
                                     &mVisualizeListTexFirstNodeOffsetPSReflector); 

    CreatePixelShaderFromCompiledObj(d3dDevice,
                                     L"ShaderObjs\\VisualizeAVSMPS.fxo",
                                     &mVisualizeAVSMPS,
                                     &mVisualizeAVSMPSReflector); 

    // Shaders for capturing hair in main camera view
    CreateVertexShaderFromCompiledObj(d3dDevice,
                                      L"ShaderObjs\\HairVS.fxo",
                                      &mHairVS,
                                      &mHairVSReflector);

    CreateGeometryShaderFromCompiledObj(d3dDevice,
                                        L"ShaderObjs\\HairGS.fxo",
                                        &mHairGS,
                                        &mHairGSReflector);
  
    CreatePixelShadersFromCompiledObjs(d3dDevice, MAX_SHADER_VARIATIONS, 
                                       "ShaderObjs\\CameraHairCapture", 
                                       "PS.fxo", 
                                       mCameraHairCapturePS, 
                                       mCameraHairCapturePSReflector); 

    // Shaders for rendering hair in the main camera view
    CreatePixelShaderFromCompiledObj(d3dDevice,
                                     L"ShaderObjs\\CameraHairRenderPS.fxo",
                                     &mCameraHairRenderPS,
                                     &mCameraHairRenderPSReflector); 
    
    CreatePixelShaderFromCompiledObj(d3dDevice,
                                     L"ShaderObjs\\ShadowHairCapturePS.fxo",
                                     &mShadowHairCapturePS,
                                     &mShadowHairCapturePSReflector);
  
    // Hair render with no OIT
    CreatePixelShadersFromCompiledObjs(d3dDevice, MAX_SHADER_VARIATIONS,
                                       "ShaderObjs\\StandardCameraHairRender",
                                       "PS.fxo",
                                       mStandardCameraHairRenderPS,
                                       mStandardCameraHairRenderPSReflector); 

    // Create skybox shaders
    CreateVertexShaderFromCompiledObj(d3dDevice,
                                      L"ShaderObjs\\SkyboxVS.fxo",
                                      &mSkyboxVS,
                                      &mSkyboxVSReflector); 

    CreatePixelShaderFromCompiledObj(d3dDevice,
                                     L"ShaderObjs\\SkyboxPS.fxo",
                                     &mSkyboxPS,
                                     &mSkyboxPSReflector); 
 
    CreatePixelShaderFromCompiledObj(d3dDevice,
                                     L"ShaderObjs\\DrawTransmittancePS.fxo",
                                     &mDrawTransmittancePS,
                                     &mDrawTransmittancePSReflector); 

    {
        HRESULT hr;

        // Vertex shader
        ID3D10Blob *vertexShaderBlob = 0;
        hr = D3DX11CompileFromFile(L"DrawTransmittance.hlsl", shaderDefines, 0,
                                   "DrawTransmittanceVS", "vs_5_0",
                                   shaderFlags,
                                   0, 0, &vertexShaderBlob, 0, 0);
        // Do something better here on shader compile failure...
        assert(SUCCEEDED(hr));
        
        hr = d3dDevice->CreateVertexShader(vertexShaderBlob->GetBufferPointer(),
                                           vertexShaderBlob->GetBufferSize(),
                                           0,
                                           &mDrawTransmittanceVS);
        assert(SUCCEEDED(hr));

        // Create input layout
        const D3D11_INPUT_ELEMENT_DESC layout[] =
        {
            {"POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0},
        };
        
        d3dDevice->CreateInputLayout( 
            layout, ARRAYSIZE(layout), 
            vertexShaderBlob->GetBufferPointer(),
            vertexShaderBlob->GetBufferSize(), 
            &mDrawTransmittanceLayout);

        D3DReflect(vertexShaderBlob->GetBufferPointer(),
                   vertexShaderBlob->GetBufferSize(), 
                   IID_ID3D11ShaderReflection, 
                   (void**) &mDrawTransmittanceVSReflector);

        vertexShaderBlob->Release();      
    }
   
    // Create standard rasterizer state
    {
        CD3D11_RASTERIZER_DESC desc(D3D11_DEFAULT);
        d3dDevice->CreateRasterizerState(&desc, &mRasterizerState);
    }

    // Create double-sized standard rasterizer state
    {
        CD3D11_RASTERIZER_DESC desc(D3D11_DEFAULT);
        desc.CullMode = D3D11_CULL_NONE;
        d3dDevice->CreateRasterizerState(&desc, &mDoubleSidedRasterizerState);
    }

    // Shadow rasterizer state has no back-face culling and multisampling enabled
    {
        CD3D11_RASTERIZER_DESC desc(D3D11_DEFAULT);
        desc.CullMode = D3D11_CULL_NONE;
        desc.MultisampleEnable = true;
        desc.DepthClipEnable = false;
        d3dDevice->CreateRasterizerState(&desc, &mShadowRasterizerState);
    }

    // Create particle rasterizer state
    {
        CD3D11_RASTERIZER_DESC desc(D3D11_DEFAULT);
        desc.CullMode = D3D11_CULL_NONE;
        desc.DepthClipEnable = false;
        d3dDevice->CreateRasterizerState(&desc, &mParticleRasterizerState);
    }

    // Create hair rasterizer state
    {
        CD3D11_RASTERIZER_DESC desc(D3D11_DEFAULT);
        desc.CullMode = D3D11_CULL_NONE;
        desc.DepthClipEnable = false;
        d3dDevice->CreateRasterizerState(&desc, &mHairRasterizerState);
    }

    // Create default depth-stencil state
    {
        CD3D11_DEPTH_STENCIL_DESC desc(D3D11_DEFAULT);
        // We need LESS_EQUAL for the skybox phase, so we just set it on everything
        // as it doesn't hurt.
        desc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
        d3dDevice->CreateDepthStencilState(&desc, &mDefaultDepthStencilState);
    }

    // Create particle depth-stencil state
    {
        CD3D11_DEPTH_STENCIL_DESC desc(D3D11_DEFAULT);
        desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        d3dDevice->CreateDepthStencilState(&desc, &mParticleDepthStencilState);
    }

    // Create particle depth-stencil state
    {
        CD3D11_DEPTH_STENCIL_DESC desc(D3D11_DEFAULT);
        desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        desc.DepthEnable    = false;
        d3dDevice->CreateDepthStencilState(&desc, &mAVSMCaptureDepthStencilState);
    }

    // Create hair capture depth-stencil state
    {
        CD3D11_DEPTH_STENCIL_DESC desc(D3D11_DEFAULT);
        desc.DepthEnable = false;
        desc.StencilEnable = false;
        d3dDevice->CreateDepthStencilState(&desc,
                                           &mHairCaptureDepthStencilState);
    }

    // Create hair render depth-stencil state
    {
        CD3D11_DEPTH_STENCIL_DESC desc(D3D11_DEFAULT);
        desc.DepthEnable = true;
        desc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
        d3dDevice->CreateDepthStencilState(&desc,
                                           &mHairRenderDepthStencilState);
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
        d3dDevice->CreateBlendState(&desc, &mLightingBlendState);
    }

    // Create alpha phase blend state
    {
        CD3D11_BLEND_DESC desc(D3D11_DEFAULT);
        desc.RenderTarget[0].BlendEnable = true;
        desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
        desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        d3dDevice->CreateBlendState(&desc, &mParticleBlendState);
    }

    // Create hair render blend state
    {
        CD3D11_BLEND_DESC desc(D3D11_DEFAULT);
        desc.RenderTarget[0].BlendEnable = true;
        desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
        desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        d3dDevice->CreateBlendState(&desc, &mHairRenderBlendState);
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

    {
        CD3D11_BUFFER_DESC desc(
            sizeof(ParticlePerFrameConstants),
            D3D11_BIND_CONSTANT_BUFFER,
            D3D11_USAGE_DYNAMIC,
            D3D11_CPU_ACCESS_WRITE);

        d3dDevice->CreateBuffer(&desc, 0, &mParticlePerFrameConstants);
    }
    {
        CD3D11_BUFFER_DESC desc(
            sizeof(ParticlePerPassConstants),
            D3D11_BIND_CONSTANT_BUFFER,
            D3D11_USAGE_DYNAMIC,
            D3D11_CPU_ACCESS_WRITE);

        d3dDevice->CreateBuffer(&desc, 0, &mParticlePerPassConstants);
    }

    {
        CD3D11_BUFFER_DESC desc(
            sizeof(LT_Constants),
            D3D11_BIND_CONSTANT_BUFFER,
            D3D11_USAGE_DYNAMIC,
            D3D11_CPU_ACCESS_WRITE);

        d3dDevice->CreateBuffer(&desc, 0, &mListTextureConstants);
    }
    {
        CD3D11_BUFFER_DESC desc(
            sizeof(AVSMConstants),
            D3D11_BIND_CONSTANT_BUFFER,
            D3D11_USAGE_DYNAMIC,
            D3D11_CPU_ACCESS_WRITE);

        d3dDevice->CreateBuffer(&desc, 0, &mAVSMConstants);
    }

    {
        CD3D11_BUFFER_DESC desc(
            sizeof(VolumeShadowConstants),
            D3D11_BIND_CONSTANT_BUFFER,
            D3D11_USAGE_DYNAMIC,
            D3D11_CPU_ACCESS_WRITE);

        d3dDevice->CreateBuffer(&desc, 0, &mVolumeShadowConstants);
    }

    {
        CD3D11_BUFFER_DESC desc(
            sizeof(HairConstants),
            D3D11_BIND_CONSTANT_BUFFER,
            D3D11_USAGE_DYNAMIC,
            D3D11_CPU_ACCESS_WRITE);

        d3dDevice->CreateBuffer(&desc, 0, &mHairConstants);
    }

    {
        CD3D11_BUFFER_DESC desc(
            sizeof(HairLTConstants),
            D3D11_BIND_CONSTANT_BUFFER,
            D3D11_USAGE_DYNAMIC,
            D3D11_CPU_ACCESS_WRITE);

        d3dDevice->CreateBuffer(&desc, 0, &mHairLTConstants);
    }

    {
        // Create vertex buffer for transmittance line graph
        const size_t bufferSize = 
            sizeof(PointVertex) * mDrawTransmittanceMaxNodes;
        D3D11_BUFFER_DESC bd;
        ZeroMemory(&bd, sizeof(bd));
        bd.Usage = D3D11_USAGE_DYNAMIC;
        bd.ByteWidth = static_cast<UINT>(bufferSize);
        bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        bd.MiscFlags = 0;
        d3dDevice->CreateBuffer(&bd, NULL, &mDrawTransmittanceVB);
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
    }
    {
        CD3D11_SAMPLER_DESC desc(D3D11_DEFAULT);
        desc.Filter = D3D11_FILTER_ANISOTROPIC;
        desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.MaxAnisotropy = 16;
        d3dDevice->CreateSamplerState(&desc, &mShadowSampler);
    }
    {
        CD3D11_SAMPLER_DESC desc(D3D11_DEFAULT);
        desc.Filter = D3D11_FILTER_ANISOTROPIC;
        desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.MaxAnisotropy = 16;
        desc.MaxLOD = 0.0f;
        d3dDevice->CreateSamplerState(&desc, &mShadowOnParticlesSampler);
    }
    {
        CD3D11_SAMPLER_DESC desc(D3D11_DEFAULT);
        desc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
        desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        d3dDevice->CreateSamplerState(&desc, &mAVSMSampler);
    }

    // Create AVSM textures and viewport
    {
        DXGI_SAMPLE_DESC sampleDesc;
        sampleDesc.Count = 1;
        sampleDesc.Quality = 0;
        mAVSMTextures = shared_ptr<Texture2D>(new Texture2D(
            d3dDevice, mAVSMShadowTextureDim, mAVSMShadowTextureDim, DXGI_FORMAT_R32G32B32A32_FLOAT,
            D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, MAX_AVSM_RT_COUNT, sampleDesc,
            D3D11_RTV_DIMENSION_TEXTURE2DARRAY, D3D11_UAV_DIMENSION_TEXTURE2DARRAY, D3D11_SRV_DIMENSION_TEXTURE2DARRAY));

        mAVSMShadowViewport.Width    = static_cast<float>(mAVSMShadowTextureDim);
        mAVSMShadowViewport.Height   = static_cast<float>(mAVSMShadowTextureDim);
        mAVSMShadowViewport.MinDepth = 0.0f;
        mAVSMShadowViewport.MaxDepth = 1.0f;
        mAVSMShadowViewport.TopLeftX = 0.0f;
        mAVSMShadowViewport.TopLeftY = 0.0f;
    }

    // Create AVSM debug textures
    {
        HRESULT hr;
        CD3D11_TEXTURE2D_DESC desc(
            DXGI_FORMAT_R32G32B32A32_FLOAT,         
            mAVSMShadowTextureDim,
            mAVSMShadowTextureDim,
            MAX_AVSM_RT_COUNT,
            1,
            0,
            D3D11_USAGE_STAGING,
            D3D10_CPU_ACCESS_READ);
        V(d3dDevice->CreateTexture2D(&desc, 0, &mAVSMTexturesDebug));
    }

    // Need to create a structured buffer to store AVSM data
    // This buffer is required by sw rasterization and RTR passes, due to the fact that is not possible
    // to perform reads to a vector typed UAV. Data is copied from the structured buffer to
    // the AVSM UAV upon insertion blocker completion.
    {
        HRESULT hr;
        UINT structSize = sizeof(float) * 2 * mAVSMNodeCount;

        CD3D11_BUFFER_DESC desc(
            structSize * mAVSMShadowTextureDim * mAVSMShadowTextureDim,            
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            D3D11_USAGE_DEFAULT,
            0,
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
            structSize);
        V(d3dDevice->CreateBuffer(&desc, 0, &mAVSMStructBuf));

        CD3D11_UNORDERED_ACCESS_VIEW_DESC unorderedAccessResourceDesc(
            D3D11_UAV_DIMENSION_BUFFER,
            DXGI_FORMAT_UNKNOWN,
            0, mAVSMShadowTextureDim * mAVSMShadowTextureDim, 1, 0);
        V(d3dDevice->CreateUnorderedAccessView(mAVSMStructBuf, &unorderedAccessResourceDesc, &mAVSMStructBufUAV));

        CD3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceDesc(
            D3D11_SRV_DIMENSION_BUFFER,
            DXGI_FORMAT_UNKNOWN,
            0, mAVSMShadowTextureDim * mAVSMShadowTextureDim, 1);
        V(d3dDevice->CreateShaderResourceView(mAVSMStructBuf, &shaderResourceDesc, &mAVSMStructBufSRV));
    }

    // Create List texture first offset debug texture
    {
        HRESULT hr;
        CD3D11_TEXTURE2D_DESC desc(
            DXGI_FORMAT_R32_UINT,         
            mAVSMShadowTextureDim,
            mAVSMShadowTextureDim,
            1,
            1,
            0,
            D3D11_USAGE_STAGING,
            D3D10_CPU_ACCESS_READ);
        V(d3dDevice->CreateTexture2D(&desc, 0, &mListTexFirstOffsetDebug));
    }

    // Create List Texture first segment node offset texture
    {
        DXGI_SAMPLE_DESC sampleDesc;
        sampleDesc.Count = 1;
        sampleDesc.Quality = 0;
        mListTexFirstSegmentNodeOffset = shared_ptr<Texture2D>(new Texture2D(
            d3dDevice, mAVSMShadowTextureDim, mAVSMShadowTextureDim, DXGI_FORMAT_R32_UINT,
            D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE, 1, sampleDesc,
            D3D11_RTV_DIMENSION_UNKNOWN, D3D11_UAV_DIMENSION_TEXTURE2D, D3D11_SRV_DIMENSION_TEXTURE2D));
    }

    // Create List Texture first visibility node offset texture
    {
        DXGI_SAMPLE_DESC sampleDesc;
        sampleDesc.Count = 1;
        sampleDesc.Quality = 0;
        mListTexFirstVisibilityNodeOffset = shared_ptr<Texture2D>(new Texture2D(
            d3dDevice, mAVSMShadowTextureDim, mAVSMShadowTextureDim, DXGI_FORMAT_R32_UINT,
            D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE, 1, sampleDesc,
            D3D11_RTV_DIMENSION_UNKNOWN, D3D11_UAV_DIMENSION_TEXTURE2D, D3D11_SRV_DIMENSION_TEXTURE2D));
    }

    // Create List Texture segment nodes buffer for AVSM data capture
    {
        HRESULT hr;
        UINT structSize = sizeof(SegmentNode);

        CD3D11_BUFFER_DESC desc(
            structSize * mLisTexNodeCount,            
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            D3D11_USAGE_DEFAULT,
            0,
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
            structSize);
        V(d3dDevice->CreateBuffer(&desc, 0, &mListTexSegmentNodes));

        CD3D11_UNORDERED_ACCESS_VIEW_DESC unorderedAccessResourceDesc(
            D3D11_UAV_DIMENSION_BUFFER,
            DXGI_FORMAT_UNKNOWN,
            0, mLisTexNodeCount, 1, D3D11_BUFFER_UAV_FLAG_COUNTER);
        V(d3dDevice->CreateUnorderedAccessView(mListTexSegmentNodes, &unorderedAccessResourceDesc, &mListTexSegmentNodesUAV));

        CD3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceDesc(
            D3D11_SRV_DIMENSION_BUFFER,
            DXGI_FORMAT_UNKNOWN,
            0, mLisTexNodeCount, 1);
        V(d3dDevice->CreateShaderResourceView(mListTexSegmentNodes, &shaderResourceDesc, &mListTexSegmentNodesSRV));
    }

    // Create List Texture visibility nodes buffer for AVSM data capture
    {
        HRESULT hr;
        UINT structSize = sizeof(VisibilityNode);

        CD3D11_BUFFER_DESC desc(
            structSize * mLisTexNodeCount,            
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            D3D11_USAGE_DEFAULT,
            0,
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
            structSize);
        V(d3dDevice->CreateBuffer(&desc, 0, &mListTexVisibilityNodes));

        CD3D11_UNORDERED_ACCESS_VIEW_DESC unorderedAccessResourceDesc(
            D3D11_UAV_DIMENSION_BUFFER,
            DXGI_FORMAT_UNKNOWN,
            0, mLisTexNodeCount, 1, D3D11_BUFFER_UAV_FLAG_COUNTER);
        V(d3dDevice->CreateUnorderedAccessView(mListTexVisibilityNodes, &unorderedAccessResourceDesc, &mListTexVisibilityNodesUAV));

        CD3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceDesc(
            D3D11_SRV_DIMENSION_BUFFER,
            DXGI_FORMAT_UNKNOWN,
            0, mLisTexNodeCount, 1);
        V(d3dDevice->CreateShaderResourceView(mListTexVisibilityNodes, &shaderResourceDesc, &mListTexVisibilityNodesSRV));
    }

    // Create Hair List Texture nodes buffer
    {
        HRESULT hr;
        UINT structSize = sizeof(HairLTNode);

        CD3D11_BUFFER_DESC desc(
            structSize * mHairLTMaxNodeCount,            
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            D3D11_USAGE_DEFAULT,
            0,
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
            structSize);
        V(d3dDevice->CreateBuffer(&desc, 0, &mHairLTNodes));

        CD3D11_UNORDERED_ACCESS_VIEW_DESC unorderedAccessResourceDesc(
            D3D11_UAV_DIMENSION_BUFFER,
            DXGI_FORMAT_UNKNOWN,
            0, mHairLTMaxNodeCount, 1, D3D11_BUFFER_UAV_FLAG_COUNTER);
        V(d3dDevice->CreateUnorderedAccessView(mHairLTNodes, 
                                               &unorderedAccessResourceDesc, 
                                               &mHairLTNodesUAV));

        CD3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceDesc(
            D3D11_SRV_DIMENSION_BUFFER,
            DXGI_FORMAT_UNKNOWN,
            0, mHairLTMaxNodeCount, 1);
        V(d3dDevice->CreateShaderResourceView(mHairLTNodes, 
                                              &shaderResourceDesc, 
                                              &mHairLTNodesSRV));
    }

    // Create List Texture segment nodes debug buffer
    {
        HRESULT hr;
        UINT structSize = sizeof(SegmentNode);

        CD3D11_BUFFER_DESC desc(
            structSize * mLisTexNodeCount,            
            0,
            D3D11_USAGE_STAGING,
            D3D10_CPU_ACCESS_READ,
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
            structSize);
        V(d3dDevice->CreateBuffer(&desc, 0, &mListTexSegmentNodesDebug));
    }

    // Create List Texture segment nodes debug buffer
    {
        HRESULT hr;
        UINT structSize = sizeof(VisibilityNode);

        CD3D11_BUFFER_DESC desc(
            structSize * mLisTexNodeCount,            
            0,
            D3D11_USAGE_STAGING,
            D3D10_CPU_ACCESS_READ,
            D3D11_RESOURCE_MISC_BUFFER_STRUCTURED,
            structSize);
        V(d3dDevice->CreateBuffer(&desc, 0, &mListTexVisibilityNodesDebug));
    }

    // Create skybox mesh and texture
    {
        mSkyboxMesh.Create(d3dDevice, L"..\\media\\Skybox\\Skybox.sdkmesh");
        HRESULT hr;
        hr = D3DX11CreateTextureFromFile(
            d3dDevice, 
            L"..\\media\\Skybox\\Clouds.dds", 
            //L"..\\media\\private\\valve\\sky_l4d_c5_1.dds",
            0, 0, (ID3D11Resource**)&mSkyboxTexture, 0);

        assert(SUCCEEDED(hr));

        d3dDevice->CreateShaderResourceView(mSkyboxTexture, 0, &mSkyboxSRV);
    }

    // Create noise texture for particle density/opacity modulation
    {
        HRESULT hr;
        hr = D3DX11CreateTextureFromFile(
            d3dDevice, 
            //L"..\\media\\Skybox\\Clouds.dds", 
            L"..\\media\\ParticleNoise\\Noise02.dds",
            0, 0, (ID3D11Resource**)&mParticleOpacityNoiseTexture, 0);
        assert(SUCCEEDED(hr));

        d3dDevice->CreateShaderResourceView(mParticleOpacityNoiseTexture, 0, &mParticleOpacityNoiseTextureSRV);
    }


    mHair = new Hair(d3dDevice, d3dDeviceContext, "Media/hair/dark.hair", shaderDefines);
}

App::~App() 
{
    SAFE_RELEASE(mSkyboxSRV);
    SAFE_RELEASE(mSkyboxTexture);
    mSkyboxMesh.Destroy();

    SAFE_RELEASE(mParticleOpacityNoiseTexture);
    SAFE_RELEASE(mParticleOpacityNoiseTextureSRV);

    SAFE_RELEASE(mParticleDepthStencilState);
    SAFE_RELEASE(mDefaultDepthStencilState);
    SAFE_RELEASE(mAVSMCaptureDepthStencilState);
    SAFE_RELEASE(mShadowSampler);
    SAFE_RELEASE(mShadowOnParticlesSampler);
    SAFE_RELEASE(mAVSMSampler);
    SAFE_RELEASE(mDiffuseSampler);
    SAFE_RELEASE(mPerFrameConstants);
    SAFE_RELEASE(mParticlePerFrameConstants);
    SAFE_RELEASE(mParticlePerPassConstants);
    SAFE_RELEASE(mListTextureConstants);
    SAFE_RELEASE(mAVSMConstants);
    SAFE_RELEASE(mVolumeShadowConstants);
    SAFE_RELEASE(mHairConstants);
    SAFE_RELEASE(mHairLTConstants);
    SAFE_RELEASE(mLightingBlendState);
    SAFE_RELEASE(mGeometryBlendState);
    SAFE_RELEASE(mParticleBlendState);
    SAFE_RELEASE(mShadowRasterizerState);
    SAFE_RELEASE(mRasterizerState);
    SAFE_RELEASE(mDoubleSidedRasterizerState);
    SAFE_RELEASE(mParticleRasterizerState);
    SAFE_RELEASE(mHairRasterizerState);
    SAFE_RELEASE(mMeshVertexLayout);
    SAFE_RELEASE(mSkyboxPS);
    SAFE_RELEASE(mSkyboxPSReflector);
    SAFE_RELEASE(mSkyboxVS);
    SAFE_RELEASE(mSkyboxVSReflector);
    SAFE_RELEASE(mVisualizeListTexFirstNodeOffsetPS);
    SAFE_RELEASE(mVisualizeAVSMPS);
    SAFE_RELEASE(mVisualizeAVSMPSReflector);
    SAFE_RELEASE(mVisualizeListTexFirstNodeOffsetPSReflector);
    SAFE_RELEASE(mFullScreenTriangleVS);
    SAFE_RELEASE(mFullScreenTriangleVSReflector);
    SAFE_RELEASE(mGeometryPS);
    SAFE_RELEASE(mGeometryPSReflector);
    SAFE_RELEASE(mGeometryVS);
    SAFE_RELEASE(mGeometryVSReflector);
    SAFE_RELEASE(mParticleShadingVS);
    SAFE_RELEASE(mParticleShadingVSReflector);
    SAFE_RELEASE(mAVSMClearStructuredBufPS);
    SAFE_RELEASE(mAVSMClearStructuredBufPSReflector);
    SAFE_RELEASE(mAVSMConvertSUAVtoTex2DPS);
    SAFE_RELEASE(mAVSMConvertSUAVtoTex2DPSReflector);
    SAFE_RELEASE(mParticleAVSMCapturePS);
    SAFE_RELEASE(mParticleAVSMCapturePSReflector);
    SAFE_RELEASE(mComputeVisibilityCurveCS);
    SAFE_RELEASE(mComputeVisibilityCurveCSReflector);
    SAFE_RELEASE(mListTexSegmentNodes);
    SAFE_RELEASE(mListTexSegmentNodesDebug);
    SAFE_RELEASE(mListTexSegmentNodesUAV);
    SAFE_RELEASE(mListTexSegmentNodesSRV);
    SAFE_RELEASE(mListTexVisibilityNodes);
    SAFE_RELEASE(mListTexVisibilityNodesDebug);
    SAFE_RELEASE(mListTexVisibilityNodesUAV);
    SAFE_RELEASE(mListTexVisibilityNodesSRV);
    SAFE_RELEASE(mAVSMStructBuf);
    SAFE_RELEASE(mAVSMTexturesDebug);
    SAFE_RELEASE(mAVSMStructBufUAV);
    SAFE_RELEASE(mAVSMStructBufSRV);
    SAFE_RELEASE(mListTexFirstOffsetDebug);

    for (int i = 0; i < MAX_SHADER_VARIATIONS; ++i) {
        SAFE_RELEASE(mAVSMUnsortedResolvePS[i]);
        SAFE_RELEASE(mAVSMUnsortedResolvePSReflector[i]);
        SAFE_RELEASE(mAVSMInsertionSortResolvePS[i]);
        SAFE_RELEASE(mAVSMInsertionSortResolvePSReflector[i]);
        SAFE_RELEASE(mAVSMSinglePassInsertPS[i]);
        SAFE_RELEASE(mAVSMSinglePassInsertPSReflector[i]);
        SAFE_RELEASE(mLightingPS[i]);
        SAFE_RELEASE(mLightingPSReflector[i]);
        SAFE_RELEASE(mParticleShadingPS[i]);
        SAFE_RELEASE(mParticleShadingPSReflector[i]);
        SAFE_RELEASE(mCameraHairCapturePS[i]);
        SAFE_RELEASE(mCameraHairCapturePSReflector[i]);
        SAFE_RELEASE(mStandardCameraHairRenderPS[i]);
        SAFE_RELEASE(mStandardCameraHairRenderPSReflector[i]);
    }

    // Hair
    SAFE_RELEASE(mHairVS);
    SAFE_RELEASE(mHairVSReflector);
    SAFE_RELEASE(mHairGS);
    SAFE_RELEASE(mHairGSReflector);
    SAFE_RELEASE(mCameraHairRenderPS);
    SAFE_RELEASE(mCameraHairRenderPSReflector);
    SAFE_RELEASE(mShadowHairCapturePS);
    SAFE_RELEASE(mShadowHairCapturePSReflector);
    SAFE_RELEASE(mHairLTNodes);
    SAFE_RELEASE(mHairLTNodesUAV);
    SAFE_RELEASE(mHairLTNodesSRV);
    SAFE_RELEASE(mHairCaptureDepthStencilState);
    SAFE_RELEASE(mHairRenderDepthStencilState);
    SAFE_RELEASE(mHairRenderBlendState);
    SAFE_RELEASE(mDrawTransmittancePS);
    SAFE_RELEASE(mDrawTransmittancePSReflector);
    SAFE_RELEASE(mDrawTransmittanceVB);
    SAFE_RELEASE(mDrawTransmittanceVS);
    SAFE_RELEASE(mDrawTransmittanceVSReflector);
    SAFE_RELEASE(mDrawTransmittanceLayout);

    delete mHair;

}

void App::OnD3D11ResizedSwapChain(ID3D11Device* d3dDevice,
                                  const DXGI_SURFACE_DESC* backBufferDesc)
{
    // Create/recreate GBuffer textures
    mGBuffer.clear();

    // standard depth buffer
    mDepthBuffer = shared_ptr<Depth2D>(new Depth2D(
        d3dDevice, backBufferDesc->Width, backBufferDesc->Height));

    // viewSpaceZ
    mGBuffer.push_back(shared_ptr<Texture2D>(new Texture2D(
        d3dDevice, backBufferDesc->Width, backBufferDesc->Height, DXGI_FORMAT_R32_FLOAT)));

    // normals
    mGBuffer.push_back(shared_ptr<Texture2D>(new Texture2D(
        d3dDevice, backBufferDesc->Width, backBufferDesc->Height, DXGI_FORMAT_R16G16B16A16_FLOAT)));
    
    // albedo
    mGBuffer.push_back(shared_ptr<Texture2D>(new Texture2D(
        d3dDevice, backBufferDesc->Width, backBufferDesc->Height, DXGI_FORMAT_R8G8B8A8_UNORM)));

    // Create Hair List Texture first node offset texture
    {
        DXGI_SAMPLE_DESC sampleDesc;
        sampleDesc.Count = 1;
        sampleDesc.Quality = 0;
        mHairLTFirstNodeOffset = 
            shared_ptr<Texture2D>(
                new Texture2D(d3dDevice, 
                              backBufferDesc->Width, 
                              backBufferDesc->Height, 
                              DXGI_FORMAT_R32_UINT,
                              D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE, 
                              1, 
                              sampleDesc,
                              D3D11_RTV_DIMENSION_UNKNOWN, 
                              D3D11_UAV_DIMENSION_TEXTURE2D, 
                              D3D11_SRV_DIMENSION_TEXTURE2D));
    }
}


// Cleanup (aka make the runtime happy)
void Cleanup(ID3D11DeviceContext* d3dDeviceContext)
{
    d3dDeviceContext->GSSetShader(NULL, 0, 0);

    d3dDeviceContext->OMSetRenderTargets(0, 0, 0);       
    d3dDeviceContext->OMSetRenderTargetsAndUnorderedAccessViews(0, 0, 0, 0, 0, 0, 0);

    ID3D11ShaderResourceView* nullViews[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0, 0};
    d3dDeviceContext->VSSetShaderResources(0, 16, nullViews);
    d3dDeviceContext->GSSetShaderResources(0, 16, nullViews);
    d3dDeviceContext->PSSetShaderResources(0, 16, nullViews);
    d3dDeviceContext->CSSetShaderResources(0, 16, nullViews);

    ID3D11UnorderedAccessView* nullUAViews[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    d3dDeviceContext->CSSetUnorderedAccessViews(0, 8, nullUAViews, 0);
                                               
}

void App::UpdateHairMesh()
{
    mHairMeshDirty = true;
}

void App::SetNodeCount(int nodeCount)
{
    mAVSMNodeCount = nodeCount;
    mShaderIdx = nodeCount / 4 - 1;
}

static void
TransformBBox(D3DXVECTOR3 *min,
              D3DXVECTOR3 *max,
              const D3DXMATRIXA16 &m)
{
    const D3DXVECTOR3 &minCorner = *min;
    const D3DXVECTOR3 &maxCorner = *max;
    D3DXVECTOR3 corners[8];
    // Bottom corners
    corners[0] = D3DXVECTOR3(minCorner.x, minCorner.y, minCorner.z);
    corners[1] = D3DXVECTOR3(maxCorner.x, minCorner.y, minCorner.z);
    corners[2] = D3DXVECTOR3(maxCorner.x, minCorner.y, maxCorner.z);
    corners[3] = D3DXVECTOR3(minCorner.x, minCorner.y, maxCorner.z);
    // Top corners
    corners[4] = D3DXVECTOR3(minCorner.x, maxCorner.y, minCorner.z);
    corners[5] = D3DXVECTOR3(maxCorner.x, maxCorner.y, minCorner.z);
    corners[6] = D3DXVECTOR3(maxCorner.x, maxCorner.y, maxCorner.z);
    corners[7] = D3DXVECTOR3(minCorner.x, maxCorner.y, maxCorner.z);

    D3DXVECTOR3 newCorners[8];
    for (int i = 0; i < 8; i++) {
        D3DXVec3TransformCoord(&newCorners[i], &corners[i], &m);
    }

    D3DXVECTOR3 newMin, newMax;

    // Initialize
    for (int i = 0; i < 3; ++i) {
        newMin[i] = std::numeric_limits<float>::max();
        newMax[i] = std::numeric_limits<float>::min();
    }

    // Find new min and max corners
    for (int i = 0; i < 8; i++) {
        D3DXVec3Minimize(&newMin, &newMin, &newCorners[i]);
        D3DXVec3Maximize(&newMax, &newMax, &newCorners[i]);
    }

    *min = newMin;
    *max = newMax;
}

void App::Render(const Options &options,
                 ID3D11DeviceContext* d3dDeviceContext, 
                 ID3D11RenderTargetView* backBuffer,
                 CDXUTSDKMesh* mesh,
                 ParticleSystem *particleSystem,
                 const D3DXMATRIXA16& worldMatrix,
                 CFirstPersonCamera* viewerCamera,
                 CFirstPersonCamera* lightCamera,
                 D3D11_VIEWPORT* viewport,
                 UIConstants* ui)
{
    FrameMatrices frameMatx;
    frameMatx.worldMatrix = worldMatrix;

    if (mHairMeshDirty) {
        mHair->UpdateHairMesh(d3dDeviceContext, 
                              options.hairX, options.hairY, options.hairZ);
        mHairMeshDirty = false;
    }

    assert((options.enableParticles && particleSystem) ||
           (options.enableParticles == false));

    frameMatx.cameraProj = *viewerCamera->GetProjMatrix();
    frameMatx.cameraView = *viewerCamera->GetViewMatrix();
    D3DXVECTOR4 cameraPos  = D3DXVECTOR4(*viewerCamera->GetEyePt(), 1.0f);
    
    D3DXMatrixInverse(&frameMatx.cameraViewInv, 0, &frameMatx.cameraView);

    // We only use the view direction from the camera object
    // We then center the directional light on the camera frustum and set the
    // extents to completely cover it.
    frameMatx.lightView = *lightCamera->GetViewMatrix();
    {
        // NOTE: We don't include the projection matrix here, since we want to just get back the
        // raw view-space extents and use that to *build* the bounding projection matrix
        D3DXVECTOR3 min, max;
        ComputeFrustumExtents(frameMatx.cameraViewInv, frameMatx.cameraProj,
                              viewerCamera->GetNearClip(), viewerCamera->GetFarClip(),
                              frameMatx.lightView, &min, &max);

        // First adjust the light matrix to be centered on the extents in x/y and behind everything in z
        D3DXVECTOR3 center = 0.5f * (min + max);
        D3DXMATRIXA16 centerTransform;
        D3DXMatrixTranslation(&centerTransform, -center.x, -center.y, 0.0f);
        frameMatx.lightView *= centerTransform;

        // Now create a projection matrix that covers the extents when centered
        // TODO: Again use scene AABB to decide on light far range - this one can actually clip out
        // any objects further away than the frustum can see if desired.
        D3DXVECTOR3 dimensions = max - min;
        D3DXMatrixOrthoLH(&frameMatx.lightProj, dimensions.x, dimensions.y, 0.0f, 1000.0f);
    }

    // Compute composite matrices
    frameMatx.cameraViewProj = frameMatx.cameraView * frameMatx.cameraProj;
    frameMatx.cameraWorldViewProj = frameMatx.worldMatrix * frameMatx.cameraViewProj;
    frameMatx.cameraWorldView = frameMatx.worldMatrix * frameMatx.cameraView;
    frameMatx.lightViewProj = frameMatx.lightView * frameMatx.lightProj;
    frameMatx.lightWorldViewProj = frameMatx.worldMatrix * frameMatx.lightViewProj;
    frameMatx.cameraViewToLightProj = frameMatx.cameraViewInv * frameMatx.lightViewProj;
    frameMatx.cameraViewToLightView = frameMatx.cameraViewInv * frameMatx.lightView;

    frameMatx.avsmLightProj = *lightCamera->GetProjMatrix();
    frameMatx.avsmLightView = *lightCamera->GetViewMatrix();

    float hairDepthBounds[2];
    if (options.enableAutoBoundsAVSM) {
        // Get bounding boxes from transparent geometry
        const float bigNumf = 1e10f;
        D3DXVECTOR3 maxBB(-bigNumf, -bigNumf, -bigNumf);
        D3DXVECTOR3 minBB(+bigNumf, +bigNumf, +bigNumf);

        if (options.enableHair ||
            options.enableParticles) {
            // Initialize minBB, maxBB
            for (size_t p = 0; p < 3; ++p) {
                minBB[p] = std::numeric_limits<float>::max();
                maxBB[p] = std::numeric_limits<float>::min();
            }

            if (options.enableHair) {
                D3DXVECTOR3 hairMin, hairMax;
                mHair->GetBBox(&hairMin, &hairMax);

                for (size_t p = 0; p < 3; ++p) {
                    minBB[p] = std::min(minBB[p], hairMin[p]);
                    maxBB[p] = std::max(maxBB[p], hairMax[p]);
                }
            } 

            if (options.enableParticles) {
                D3DXVECTOR3 particleMin, particleMax;
                particleSystem->GetBBox(&particleMin, &particleMax);

                for (size_t p = 0; p < 3; ++p) {
                    minBB[p] = std::min(minBB[p], particleMin[p]);
                    maxBB[p] = std::max(maxBB[p], particleMax[p]);
                }
            }

            TransformBBox(&minBB, &maxBB, frameMatx.avsmLightView);
        }

        // First adjust the light matrix to be centered on the extents in x/y and behind everything in z
        D3DXVECTOR3 center = 0.5f * (minBB + maxBB);
        D3DXMATRIXA16 centerTransform;
        D3DXMatrixTranslation(&centerTransform, -center.x, -center.y, -minBB.z);
        frameMatx.avsmLightView *= centerTransform;

        hairDepthBounds[0] = 0.0f;
        hairDepthBounds[1] = maxBB.z - minBB.z;

        // Now create a projection matrix that covers the extents when centered
        // TODO: Again use scene AABB to decide on light far range - this one can actually clip out
        // any objects further away than the frustum can see if desired.
        D3DXVECTOR3 dimensions = maxBB - minBB;
        D3DXMatrixOrthoLH(&frameMatx.avsmLightProj, dimensions.x, dimensions.y, 0, dimensions.z);
    }
    
    // Compute composite matrices;
    frameMatx.avsmLightViewProj = frameMatx.avsmLightView * frameMatx.avsmLightProj;
    frameMatx.avmsLightWorldViewProj = frameMatx.worldMatrix * frameMatx.avsmLightViewProj;
    frameMatx.cameraViewToAvsmLightProj = frameMatx.cameraViewInv * frameMatx.avsmLightViewProj;
    frameMatx.cameraViewToAvsmLightView = frameMatx.cameraViewInv * frameMatx.avsmLightView;


    float particleDepthBounds[2];
    if (options.enableParticles) {
        if (ui->pauseParticleAnimaton == false) {
            // Update particles 
            const float currTime = static_cast<float>(DXUTGetGlobalTimer()->GetAbsoluteTime());
            float deltaTime = 0.009; //currTime - mLastTime;
            mLastTime = currTime;
            particleSystem->UpdateParticles(viewerCamera, lightCamera, deltaTime);
        }
                
        particleSystem->SortParticles(particleDepthBounds, &frameMatx.avsmLightView, false, 1);
        particleSystem->PopulateVertexBuffers(d3dDeviceContext);

        // Fill in particle emitter (per frame) constants
        {
            D3D11_MAPPED_SUBRESOURCE mappedResource;
            d3dDeviceContext->Map(mParticlePerFrameConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
            ParticlePerFrameConstants* constants = static_cast<ParticlePerFrameConstants *>(mappedResource.pData);

            constants->mScale                        = 1.0f;
            constants->mParticleSize                 = ((float)ui->particleSize / 3.0f);
            constants->mParticleAlpha                = 1.0f;
            constants->mbSoftParticles               = 1.0f;
            constants->mParticleOpacity              = 0.8f * ((float)ui->particleOpacity / 33.0f);
            constants->mSoftParticlesSaturationDepth = 1.0f;

            d3dDeviceContext->Unmap(mParticlePerFrameConstants, 0);
        }
    }

    if (options.enableHair && options.enableParticles) {
        float minDepth = std::min(particleDepthBounds[0], hairDepthBounds[0]);
        float maxDepth = std::max(particleDepthBounds[1], hairDepthBounds[1]);
        particleDepthBounds[0] = hairDepthBounds[0] = minDepth;
        particleDepthBounds[1] = hairDepthBounds[1] = maxDepth;
    }

    FillInFrameConstants(d3dDeviceContext, frameMatx, cameraPos, lightCamera,  ui);

    // Render the GBuffer
    mesh->SetInFrustumFlags(true);
    RenderGeometryPhase(options, d3dDeviceContext, backBuffer, mesh, viewport);
    mesh->SetInFrustumFlags(true);

    // Set up render GBuffer textures
    std::vector<ID3D11ShaderResourceView*> gbufferTextures(mGBuffer.size());
    for (std::size_t i = 0; i < mGBuffer.size(); ++i) {
        gbufferTextures[i] = mGBuffer[i]->GetShaderResource();
    }

    unsigned int savedVolMethod = ui->volumeShadowMethod;
    {
        // This phase computes a visibility representation for our shadow volume
        if (ui->volumeShadowMethod && options.enableVolumeShadowCreation) 
        {
            // Setup constants buffers
            FillListTextureConstants(d3dDeviceContext);
            FillAVSMConstants(d3dDeviceContext);

            ClearShadowBuffers(d3dDeviceContext, ui);

            // First pass, capture all fragments
            if (options.enableHair) {
                HairConstants hairConstants;
                hairConstants.mHairProj = frameMatx.avsmLightProj;
                hairConstants.mHairWorldView = frameMatx.avsmLightView;
                hairConstants.mHairWorldViewProj = frameMatx.avsmLightViewProj;
                FillHairConstants(d3dDeviceContext, hairConstants);
                CaptureHair(d3dDeviceContext, ui, true);
            } 
            
            if (options.enableParticles) {
                FillParticleRendererConstants(d3dDeviceContext, 
                                              lightCamera, 
                                              frameMatx.avsmLightView, 
                                              frameMatx.avsmLightViewProj);
                CaptureFragments(d3dDeviceContext, particleSystem, ui, !options.enableHair);
            }

            // Second pass, generate visibility curves (AVSM, uncompressed, deep shadow maps, etc..)
            GenerateVisibilityCurve(d3dDeviceContext, ui);                       
        }

        if (!ui->enableVolumeShadowLookup)
        {
            ui->volumeShadowMethod = VOL_SHADOW_NO_SHADOW;
            FillInFrameConstants(d3dDeviceContext, frameMatx, cameraPos, lightCamera,  ui);
        }

        // Lighting accumulation phase
        ShadeGBuffer(d3dDeviceContext, backBuffer, viewport, gbufferTextures, ui);  


        if (!ui->enableVolumeShadowLookup) {
            ui->volumeShadowMethod = savedVolMethod;
            FillInFrameConstants(d3dDeviceContext, frameMatx, cameraPos, lightCamera,  ui);
        }

        // Skybox
        RenderSkybox(d3dDeviceContext, backBuffer, viewport);
    }

    // Reset frustum culling
    mesh->SetInFrustumFlags(true);

    if (!ui->enableVolumeShadowLookup) {
        ui->volumeShadowMethod = VOL_SHADOW_NO_SHADOW;
        FillInFrameConstants(d3dDeviceContext, frameMatx, cameraPos, lightCamera,  ui);
    }

    if (options.enableHair) {
        HairConstants hairConstants;
        hairConstants.mHairProj = frameMatx.cameraProj;
        hairConstants.mHairWorldView = frameMatx.cameraView;
        hairConstants.mHairWorldViewProj = frameMatx.cameraViewProj;
        FillHairConstants(d3dDeviceContext, hairConstants);
        RenderHair(d3dDeviceContext, backBuffer, viewport, 
                   frameMatx.cameraWorldView, ui, options);

    }

    if (options.enableParticles) {
        // Particle Alpha Pass
        // Update particles
        particleSystem->SortParticles(NULL, &frameMatx.cameraView);
        particleSystem->PopulateVertexBuffers(d3dDeviceContext);

        // Fill particle renderer constants and shade particles
        FillParticleRendererConstants(d3dDeviceContext, viewerCamera, frameMatx.cameraView, frameMatx.cameraViewProj);            
        ShadeParticles(d3dDeviceContext, particleSystem, backBuffer, viewport, 
                       gbufferTextures, ui);            
    }

    if (!ui->enableVolumeShadowLookup) {
        ui->volumeShadowMethod = savedVolMethod;
        FillInFrameConstants(d3dDeviceContext, frameMatx, cameraPos, lightCamera,  ui);
    }

    if (options.enableShadowPicking) {
        switch (ui->volumeShadowMethod) {
        case VOL_SHADOW_UNCOMPRESSED:
        case VOL_SHADOW_DSM:
            VisualizeFirstNode(d3dDeviceContext, backBuffer, viewport);
            break;
        case VOL_SHADOW_AVSM:
        case VOL_SHADOW_AVSM_BOX_4:
        case VOL_SHADOW_AVSM_GAUSS_7:
            VisualizeAVSM(d3dDeviceContext, backBuffer, viewport);
            break;
        }
    }

    if (mDumpTransmittanceCurve || options.enableTransmittanceCurve) {
        DumpOrDrawTransmittance(options, *ui,
                                d3dDeviceContext, backBuffer, viewport, 
                                options.enableHair ? hairDepthBounds : particleDepthBounds,
                                options.pickedX, options.pickedY);
    }
}

float saturate(float value)
{
    if (value < 0.0f)
        return 0;
    else if (value > 1.0f)
        return 1;
    else
        return value;
}

float linstep(float min, float max, float v)
{
    return saturate((v - min) / (max - min));
}

float LT_Interp(float d0, float d1, float t0, float t1, float r)
{
    float depth = linstep(d0, d1, r);
    return t0 + (t1 - t0) * depth;
}

std::string App::GetSceneName(SCENE_SELECTION scene)
{
    std::string sceneName("unknown");
    switch (scene) {
    case POWER_PLANT_SCENE: sceneName = "powerplant"; break;
    case GROUND_PLANE_SCENE: sceneName = "groundplane"; break;
    }
    return sceneName;
}

SCENE_SELECTION App::GetSceneEnum(const std::string &name)
{

    if (name == "powerplant") {
        return POWER_PLANT_SCENE;
    } else if (name == "groundplane") {
        return GROUND_PLANE_SCENE;
    } else {
        assert(0);
        return POWER_PLANT_SCENE;
    }
}

std::string App::GetShadowMethod(const UIConstants &ui)
{
    char buffer[1024];
    std::string shadowMethod("unknown");
    switch (ui.volumeShadowMethod) {
        case VOL_SHADOW_UNCOMPRESSED: shadowMethod = "uncompressed"; break;
        case VOL_SHADOW_DSM: shadowMethod = "dsm"; break;
        case VOL_SHADOW_AVSM: {            
            sprintf_s(buffer, sizeof(buffer), "avsm%d", mAVSMNodeCount);
            shadowMethod = buffer;
            if (ui.volumeShadowMethod == VOL_SHADOW_DSM) {
                sprintf_s(buffer, sizeof(buffer), "dsmError%f", ui.dsmError);
                shadowMethod += std::string("_") + buffer;
            }
            break;
        }
    }
    return shadowMethod;
}

void App::DumpOrDrawTransmittance(const Options &options,
                                  const UIConstants &ui,
                                  ID3D11DeviceContext* d3dDeviceContext,
                                  ID3D11RenderTargetView* backBuffer,
                                  D3D11_VIEWPORT* viewport,
                                  float depthBounds[2],
                                  int x, int y)
{
    HRESULT hr;
    std::vector<float> depth;
    std::vector<float> transmission;

    switch (ui.volumeShadowMethod) {
        case VOL_SHADOW_UNCOMPRESSED: {
            D3D11_MAPPED_SUBRESOURCE mappedResource;
            d3dDeviceContext->CopyResource(mListTexFirstOffsetDebug, 
                                           mListTexFirstSegmentNodeOffset->GetTexture());
            d3dDeviceContext->Map(mListTexFirstOffsetDebug, 0, 
                                  D3D11_MAP_READ, 0, &mappedResource);
            unsigned int *offset0 = static_cast<unsigned int*>(mappedResource.pData);

            d3dDeviceContext->CopyResource(mListTexSegmentNodesDebug, 
                                           mListTexSegmentNodes);
            d3dDeviceContext->Map(mListTexSegmentNodesDebug, 0, 
                                  D3D11_MAP_READ, 0, &mappedResource);
            SegmentNode *nodes = static_cast<SegmentNode*>(mappedResource.pData);

            const int index = mAVSMShadowTextureDim * y + x;
            unsigned int offset = offset0[index];

            std::vector<float> segmentDepth;
            std::vector<float> segmentTrans;
            while (offset != 0xFFFFFFFFUL && 
                   // Make sure we don't go off the reservation
                   segmentDepth.size() < (size_t)mDrawTransmittanceMaxNodes) {
                const SegmentNode &node = nodes[offset];
                segmentDepth.push_back(node.depth[0]);
                segmentTrans.push_back(node.trans);
                segmentDepth.push_back(node.depth[1]);
                segmentTrans.push_back(node.trans);
                offset = node.next;
            }

            // Compute transmittance for each end point of the segment and
            // insert depth and computed transmittance into list of
            // depths/transmittance values.
            for (size_t i = 0; i < segmentDepth.size(); ++i) {
                const float receiverDepth = segmentDepth[i];
                float transmittance = 1.0f;
                for (size_t j = 0; j < segmentDepth.size(); j += 2) {
                    transmittance *= 
                        LT_Interp(segmentDepth[j], segmentDepth[j + 1], 1.0f, 
                                  segmentTrans[j], receiverDepth);
                }
                depth.push_back(segmentDepth[i]);
                transmission.push_back(transmittance);
            }

            // Sort by depth
            for (size_t i = 0; i < depth.size(); ++i) {
                for (size_t j = i + 1; j < depth.size(); ++j) {
                    if (depth[i] > depth[j]) {
                        std::swap(depth[i], depth[j]);
                        std::swap(transmission[i], transmission[j]);
                    }
                }
            }

            d3dDeviceContext->Unmap(mListTexFirstOffsetDebug, 0);
            d3dDeviceContext->Unmap(mListTexSegmentNodesDebug, 0);
            break;
        }

        case VOL_SHADOW_DSM: {
            D3D11_MAPPED_SUBRESOURCE mappedResource;
            d3dDeviceContext->CopyResource(mListTexFirstOffsetDebug, 
                                           mListTexFirstVisibilityNodeOffset->GetTexture());
            d3dDeviceContext->Map(mListTexFirstOffsetDebug, 0, 
                                  D3D11_MAP_READ, 0, &mappedResource);
            unsigned int *offset0 = static_cast<unsigned int*>(mappedResource.pData);

            d3dDeviceContext->CopyResource(mListTexVisibilityNodesDebug,
                                           mListTexVisibilityNodes);
            d3dDeviceContext->Map(mListTexVisibilityNodesDebug, 0, 
                                  D3D11_MAP_READ, 0, &mappedResource);
            VisibilityNode *nodes = static_cast<VisibilityNode*>(mappedResource.pData);

            const int index = mAVSMShadowTextureDim * y + x;
            unsigned int offset = offset0[index];

            while (offset != 0xFFFFFFFFUL && 
                   // Make sure we don't go off the reservation
                   depth.size() < (size_t)mDrawTransmittanceMaxNodes) {
                const VisibilityNode &node = nodes[offset];
                depth.push_back(node.depth);
                transmission.push_back(node.trans);
                offset = node.next;
            }

            d3dDeviceContext->Unmap(mListTexFirstOffsetDebug, 0);
            d3dDeviceContext->Unmap(mListTexVisibilityNodesDebug, 0);
            break;
        }

        case VOL_SHADOW_AVSM:
        case VOL_SHADOW_AVSM_BOX_4:
        case VOL_SHADOW_AVSM_GAUSS_7: {
            d3dDeviceContext->CopyResource(mAVSMTexturesDebug, mAVSMTextures->GetTexture());
            D3D11_MAPPED_SUBRESOURCE mappedDepth;
            D3D11_MAPPED_SUBRESOURCE mappedTrans;
            const int avsmRTCount = mAVSMNodeCount / 2;
            const int transStart = avsmRTCount / 2;
            const int index = 4 * (mAVSMShadowTextureDim * y + x);
            for (int i = 0; i < avsmRTCount / 2; ++i) {
                d3dDeviceContext->Map(mAVSMTexturesDebug, 
                                      i, D3D11_MAP_READ, 0, &mappedDepth);
                d3dDeviceContext->Map(mAVSMTexturesDebug, 
                                      i + transStart, D3D11_MAP_READ, 0, &mappedTrans);
                const float *depthData = 
                    static_cast<const float*>(mappedDepth.pData) + index;
                const float *transData = 
                    static_cast<const float*>(mappedTrans.pData) + index;
                for (int j = 0; j < 4; ++j) {
                    if (*depthData != EMPTY_NODE) {
                        depth.push_back(*depthData);
                        transmission.push_back(*transData);
                    }
                    ++depthData;
                    ++transData;
                }
            }
            // Clean up
            for (int i = 0; i < avsmRTCount; ++i) {
                d3dDeviceContext->Unmap(mAVSMTexturesDebug, i);
            }
            break;
        }
    }

    if (!depth.empty()) {
        float minDepth = std::numeric_limits<float>::max();
        float maxDepth = std::numeric_limits<float>::min();
        for (size_t i = 0; i < depth.size(); ++i) {
            minDepth = std::min(depth[i], minDepth);
            maxDepth = std::max(depth[i], maxDepth);
        }

        // Create vertex buffer
        const int displayWidth = 256;
        const int displayHeight = 256;

        // Populate vertices based on nodes
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        hr = d3dDeviceContext->Map(mDrawTransmittanceVB, 0, 
                                   D3D11_MAP_WRITE_DISCARD, 0, 
                                   &mappedResource);
        PointVertex *vertexBufferData =
            reinterpret_cast<PointVertex*>(mappedResource.pData);
        float depthRange = maxDepth - minDepth;
        // XXX: max nodes in vertex buffer
        assert(depth.size() < (size_t)mDrawTransmittanceMaxNodes);
        for (size_t i = 0; i < depth.size(); ++i) {
            vertexBufferData[i].x = (depth[i] - minDepth) / depthRange * displayWidth;
            vertexBufferData[i].x /= (displayWidth / 2.0f);
            vertexBufferData[i].x -= 1.0f;

            vertexBufferData[i].y = transmission[i] / 2.0f - 0.5f;
            vertexBufferData[i].z = 0.0f;

        }
        d3dDeviceContext->Unmap(mDrawTransmittanceVB, 0);

        if (mDumpTransmittanceCurve) {
            std::string sceneName = GetSceneName(options.scene);
            std::string shadowMethod = GetShadowMethod(ui);

            char fileName[1024];
            sprintf_s(fileName, sizeof(fileName),
                      "transmittance_#%d_%s_%s_x%d_y%d.csv", 
                      mDumpTransmittanceCurveIndex,
                      sceneName.c_str(), 
                      shadowMethod.c_str(),
                      x, y);
            std::ofstream stream(fileName);
            stream << shadowMethod << "\nDepth, Transmittance\n";
            for (size_t i = 0; i < depth.size(); ++i) {
                stream << depth[i];
                stream << ",";
                stream << transmission[i];
                stream << std::endl;
            }
            stream.close();

            mDumpTransmittanceCurve = false;
        }

        if (options.enableTransmittanceCurve) {
            D3D11_VIEWPORT drawViewport;
            drawViewport.Width = displayWidth;
            drawViewport.Height = displayHeight;
            drawViewport.TopLeftX = std::floor(viewport->TopLeftX + 10.0f);
            drawViewport.TopLeftY = std::floor(viewport->TopLeftY +
                                               viewport->Height - drawViewport.Height - 10.0f);
            drawViewport.MinDepth = 0.0f;
            drawViewport.MaxDepth = 1.0f;        

            d3dDeviceContext->IASetInputLayout(mDrawTransmittanceLayout);
            d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP);
            const UINT strides = sizeof(PointVertex), offsets = 0;
            d3dDeviceContext->IASetVertexBuffers(0, 1, &mDrawTransmittanceVB, &strides, &offsets);
            d3dDeviceContext->RSSetViewports(1, &drawViewport);
            d3dDeviceContext->OMSetRenderTargets(1, &backBuffer, 0);
            d3dDeviceContext->VSSetShader(mDrawTransmittanceVS, 0, 0);
            d3dDeviceContext->PSSetShader(mDrawTransmittancePS, 0, 0);
            d3dDeviceContext->Draw((UINT) depth.size(), 0);
        }
    }
}

void App::FillHairConstants(ID3D11DeviceContext* d3dDeviceContext,
                            const HairConstants &hairConstants)
{
    // Update Constant Buffers
    {
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        d3dDeviceContext->Map(mHairConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, 
                              &mappedResource);
        HairConstants* constants = 
            static_cast<HairConstants*>(mappedResource.pData);
        *constants = hairConstants;
        d3dDeviceContext->Unmap(mHairConstants, 0);
    }
}

void App::RenderHair(ID3D11DeviceContext* d3dDeviceContext,
                     ID3D11RenderTargetView* backBuffer,
                     D3D11_VIEWPORT* viewport,
                     const D3DXMATRIXA16 &cameraWorldView,
                     const UIConstants* ui,
                     const Options &options)
{
    if (options.hairSortEnabled) {
        switch (options.hairSort) {
        default:
        case HAIR_SORT_NONE:
            mHair->ResetSort(d3dDeviceContext);
            break;
        case HAIR_SORT_PER_PIXEL:
            break;
        case HAIR_SORT_PER_LINE:
            mHair->SortPerLine(d3dDeviceContext, cameraWorldView);
            break;
        }
    }

    if (options.hairSort == HAIR_SORT_PER_PIXEL) {
        RenderHairPerPixelSort(d3dDeviceContext, backBuffer, viewport, ui, options);
    } else {
        RenderHairNoPerPixelSort(d3dDeviceContext, backBuffer, viewport, ui, options);
    }
}

void App::RenderHairPerPixelSort(ID3D11DeviceContext* d3dDeviceContext,
                                 ID3D11RenderTargetView* backBuffer,
                                 D3D11_VIEWPORT* viewport,
                                 const UIConstants* ui,
                                 const Options &options)
{
    // Update Constant Buffers
    {
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        d3dDeviceContext->Map(mHairLTConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, 
                              &mappedResource);
        HairLTConstants* constants = 
            static_cast<HairLTConstants*>(mappedResource.pData);
        constants->mMaxNodeCount = mHairLTMaxNodeCount;
        d3dDeviceContext->Unmap(mHairLTConstants, 0);
    }

    // Capture hair
    {
        const UINT clearValues[4] = {
            0xFFFFFFFFUL, 
            0xFFFFFFFFUL, 
            0xFFFFFFFFUL, 
            0xFFFFFFFFUL
        };
        ID3D11UnorderedAccessView* hairLTFirstNodeOffsetUAV = 
            mHairLTFirstNodeOffset->GetUnorderedAccess();
        d3dDeviceContext->ClearUnorderedAccessViewUint(hairLTFirstNodeOffsetUAV, 
                                                       clearValues);
        d3dDeviceContext->ClearUnorderedAccessViewUint(mHairLTNodesUAV, 
                                                       clearValues);

        VSSetConstantBuffers(d3dDeviceContext,
                             mHairVSReflector,
                             "HairConstants",
                             1, 
                             &mHairConstants);
        d3dDeviceContext->VSSetShader(mHairVS, 0, 0);

        GSSetConstantBuffers(d3dDeviceContext,
                             mHairGSReflector,
                             "PerFrameConstants",
                             1,
                             &mPerFrameConstants);
        GSSetConstantBuffers(d3dDeviceContext,
                             mHairGSReflector,
                             "HairConstants",
                             1, 
                             &mHairConstants);

        d3dDeviceContext->GSSetShader(mHairGS, 0, 0);

        d3dDeviceContext->RSSetState(mHairRasterizerState);
        d3dDeviceContext->RSSetViewports(1, viewport);

        ID3D11ShaderResourceView* asvmTextureSRV = 
            mAVSMTextures->GetShaderResource();
        ID3D11ShaderResourceView* listTexFirstSegmentNodeOffsetSRV = 
            mListTexFirstSegmentNodeOffset->GetShaderResource();
        ID3D11ShaderResourceView* listTexFirstVisibilityNodeOffsetSRV = 
            mListTexFirstVisibilityNodeOffset->GetShaderResource();

        PSSetConstantBuffers(d3dDeviceContext,
                             mCameraHairCapturePSReflector[mShaderIdx],
                             "PerFrameConstants",
                             1,
                             &mPerFrameConstants);
        PSSetConstantBuffers(d3dDeviceContext,
                             mCameraHairCapturePSReflector[mShaderIdx],
                             "HairLTConstants",
                             1,
                             &mHairLTConstants);
        PSSetConstantBuffers(d3dDeviceContext,
                             mCameraHairCapturePSReflector[mShaderIdx],
                             "AVSMConstants",
                             1,
                             &mAVSMConstants);
        PSSetShaderResources(d3dDeviceContext,
                             mCameraHairCapturePSReflector[mShaderIdx],
                             "gAVSMTexture",
                             1, 
                             &asvmTextureSRV);
        PSSetSamplers(d3dDeviceContext, 
                      mCameraHairCapturePSReflector[mShaderIdx],
                      "gAVSMSampler",
                      1, 
                      &mAVSMSampler);

        PSSetShaderResources(d3dDeviceContext,
                             mCameraHairCapturePSReflector[mShaderIdx],
                             "gListTexFirstSegmentNodeAddressSRV",
                             1, 
                             &listTexFirstSegmentNodeOffsetSRV);
        PSSetShaderResources(d3dDeviceContext,
                             mCameraHairCapturePSReflector[mShaderIdx],
                             "gListTexSegmentNodesSRV",
                             1, 
                             &mListTexSegmentNodesSRV);
        PSSetShaderResources(d3dDeviceContext,
                             mCameraHairCapturePSReflector[mShaderIdx],
                             "gListTexFirstVisibilityNodeAddressSRV",
                             1, 
                             &listTexFirstVisibilityNodeOffsetSRV);
        PSSetShaderResources(d3dDeviceContext,
                             mCameraHairCapturePSReflector[mShaderIdx],
                             "gListTexVisibilityNodesSRV",
                             1, 
                             &mListTexVisibilityNodesSRV);
        PSSetConstantBuffers(d3dDeviceContext,
                             mCameraHairCapturePSReflector[mShaderIdx],
                             "LT_Constants",
                             1, 
                             &mListTextureConstants);

        d3dDeviceContext->PSSetShader(mCameraHairCapturePS[mShaderIdx], 0, 0);

        static const char *paramUAVs[] = {
            "gHairLTFirstNodeOffsetUAV",
            "gHairLTNodeUAV",
        };
        const UINT numUAVs = sizeof(paramUAVs) / sizeof(paramUAVs[0]);
        const UINT firstUAVIndex =
            GetStartBindIndex(mCameraHairCapturePSReflector[mShaderIdx], paramUAVs, numUAVs);
        UINT pUAVInitialCounts[numUAVs] = {0, 0};
        ID3D11UnorderedAccessView* pUAVs[numUAVs] = {
            hairLTFirstNodeOffsetUAV, 
            mHairLTNodesUAV,
        };
        d3dDeviceContext->OMSetRenderTargetsAndUnorderedAccessViews(
            0, NULL, // render targets
            NULL,    // depth-stencil
            firstUAVIndex, numUAVs, pUAVs, pUAVInitialCounts);
        d3dDeviceContext->OMSetDepthStencilState(mHairCaptureDepthStencilState, 0x0);

        mHair->Draw(d3dDeviceContext);

        Cleanup(d3dDeviceContext);
    }

    // Render hair: full screen pass
    {
        ID3D11ShaderResourceView* hairLTFirstNodeOffsetSRV = 
            mHairLTFirstNodeOffset->GetShaderResource();
        ID3D11ShaderResourceView* depthBufferSRV =
            mDepthBuffer->GetShaderResource();

        d3dDeviceContext->IASetInputLayout(0);
        d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        d3dDeviceContext->IASetVertexBuffers(0, 0, 0, 0, 0);

        d3dDeviceContext->VSSetShader(mFullScreenTriangleVS, 0, 0);

        d3dDeviceContext->RSSetState(mHairRasterizerState);
        d3dDeviceContext->RSSetViewports(1, viewport);     

        PSSetShaderResources(d3dDeviceContext,
                             mCameraHairRenderPSReflector,
                             "gHairLTFirstNodeOffsetSRV",
                             1,
                             &hairLTFirstNodeOffsetSRV);
        PSSetShaderResources(d3dDeviceContext,
                             mCameraHairRenderPSReflector,
                             "gHairLTNodesSRV",
                             1,
                             &mHairLTNodesSRV);
        PSSetShaderResources(d3dDeviceContext,
                             mCameraHairRenderPSReflector,
                             "depthBufferSRV",
                             1,
                             &depthBufferSRV);
        d3dDeviceContext->PSSetShader(mCameraHairRenderPS, 0, 0);

        d3dDeviceContext->OMSetDepthStencilState(mHairRenderDepthStencilState, 0x0);
        d3dDeviceContext->OMSetBlendState(mHairRenderBlendState, 0, 0xffffffff);
        d3dDeviceContext->OMSetRenderTargets(1, 
                                             &backBuffer, 
                                             NULL);
        d3dDeviceContext->Draw(3, 0);

        Cleanup(d3dDeviceContext);
    }
}

void App::RenderHairNoPerPixelSort(ID3D11DeviceContext* d3dDeviceContext,
                                   ID3D11RenderTargetView* backBuffer,
                                   D3D11_VIEWPORT* viewport,
                                   const UIConstants* ui,
                                   const Options &options)
{
    {
        VSSetConstantBuffers(d3dDeviceContext,
                             mHairVSReflector,
                             "HairConstants",
                             1, 
                             &mHairConstants);

        d3dDeviceContext->VSSetShader(mHairVS, 0, 0);

        GSSetConstantBuffers(d3dDeviceContext,
                             mHairGSReflector,
                             "PerFrameConstants",
                             1,
                             &mPerFrameConstants);
        GSSetConstantBuffers(d3dDeviceContext,
                             mHairGSReflector,
                             "HairConstants",
                             1, 
                             &mHairConstants);

        d3dDeviceContext->GSSetShader(mHairGS, 0, 0);

        d3dDeviceContext->RSSetState(mRasterizerState);
        d3dDeviceContext->RSSetViewports(1, viewport);     

        ID3D11ShaderResourceView* asvmTextureSRV = 
            mAVSMTextures->GetShaderResource();
        ID3D11ShaderResourceView* listTexFirstSegmentNodeOffsetSRV = 
            mListTexFirstSegmentNodeOffset->GetShaderResource();
        ID3D11ShaderResourceView* listTexFirstVisibilityNodeOffsetSRV = 
            mListTexFirstVisibilityNodeOffset->GetShaderResource();

        PSSetConstantBuffers(d3dDeviceContext,
                             mStandardCameraHairRenderPSReflector[mShaderIdx],
                             "PerFrameConstants",
                             1,
                             &mPerFrameConstants);
        PSSetConstantBuffers(d3dDeviceContext,
                             mStandardCameraHairRenderPSReflector[mShaderIdx],
                             "AVSMConstants",
                             1,
                             &mAVSMConstants);
        PSSetShaderResources(d3dDeviceContext,
                             mStandardCameraHairRenderPSReflector[mShaderIdx],
                             "gAVSMTexture",
                             1, 
                             &asvmTextureSRV);
        PSSetSamplers(d3dDeviceContext, 
                      mStandardCameraHairRenderPSReflector[mShaderIdx],
                      "gAVSMSampler",
                      1, 
                      &mAVSMSampler);

        PSSetShaderResources(d3dDeviceContext,
                             mStandardCameraHairRenderPSReflector[mShaderIdx],
                             "gListTexFirstSegmentNodeAddressSRV",
                             1, 
                             &listTexFirstSegmentNodeOffsetSRV);
        PSSetShaderResources(d3dDeviceContext,
                             mStandardCameraHairRenderPSReflector[mShaderIdx],
                             "gListTexSegmentNodesSRV",
                             1, 
                             &mListTexSegmentNodesSRV);
        PSSetShaderResources(d3dDeviceContext,
                             mStandardCameraHairRenderPSReflector[mShaderIdx],
                             "gListTexFirstVisibilityNodeAddressSRV",
                             1, 
                             &listTexFirstVisibilityNodeOffsetSRV);
        PSSetShaderResources(d3dDeviceContext,
                             mStandardCameraHairRenderPSReflector[mShaderIdx],
                             "gListTexVisibilityNodesSRV",
                             1, 
                             &mListTexVisibilityNodesSRV);
        PSSetConstantBuffers(d3dDeviceContext,
                             mStandardCameraHairRenderPSReflector[mShaderIdx],
                             "LT_Constants",
                             1, 
                             &mListTextureConstants);

        d3dDeviceContext->PSSetShader(mStandardCameraHairRenderPS[mShaderIdx],
                                      0, 0);

        if (options.hairBlending) {
            d3dDeviceContext->OMSetDepthStencilState(mHairRenderDepthStencilState, 0x0);
            d3dDeviceContext->OMSetBlendState(mHairRenderBlendState, 0, 0xffffffff);
        } else {
            d3dDeviceContext->OMSetDepthStencilState(mDefaultDepthStencilState, 0x0);
            d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xffffffff);
        }

        d3dDeviceContext->PSSetShader(mStandardCameraHairRenderPS[mShaderIdx], 0, 0);
        d3dDeviceContext->OMSetRenderTargets(1, 
                                             &backBuffer, 
                                             mDepthBuffer->GetDepthStencil());
        mHair->Draw(d3dDeviceContext);
        Cleanup(d3dDeviceContext);
    }
}

// Geometry phase
void App::RenderGeometryPhase(const Options &options,
                              ID3D11DeviceContext* d3dDeviceContext,
                              ID3D11RenderTargetView* backBuffer,
                              CDXUTSDKMesh* mesh,
                              D3D11_VIEWPORT* viewport)
{
    // Clear GBuffer, back buffer and depth buffer
    float floatMax = std::numeric_limits<float>::max();
    float viewSpaceZClear[4] = {floatMax, floatMax, floatMax, floatMax};
    float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float bgColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    d3dDeviceContext->ClearRenderTargetView(mGBuffer[0]->GetRenderTarget(), viewSpaceZClear);      // viewSpaceZ
    d3dDeviceContext->ClearRenderTargetView(mGBuffer[1]->GetRenderTarget(), zeros);                // normals
    d3dDeviceContext->ClearRenderTargetView(mGBuffer[2]->GetRenderTarget(), zeros);                // albedo
    d3dDeviceContext->ClearRenderTargetView(backBuffer, bgColor);
    d3dDeviceContext->ClearDepthStencilView(mDepthBuffer->GetDepthStencil(), D3D11_CLEAR_DEPTH, 1.0f, 0);

    d3dDeviceContext->IASetInputLayout(mMeshVertexLayout);

    VSSetConstantBuffers(d3dDeviceContext, 
                         mGeometryVSReflector,
                         "PerFrameConstants",
                         1, 
                         &mPerFrameConstants);
    d3dDeviceContext->VSSetShader(mGeometryVS, 0, 0);

    // Uses double-sided polygons (no culling) if requested
    d3dDeviceContext->RSSetState(mRasterizerState);

    d3dDeviceContext->RSSetViewports(1, viewport);

    PSSetConstantBuffers(d3dDeviceContext,
                         mGeometryPSReflector,
                         "PerFrameConstants",
                         1, 
                         &mPerFrameConstants);
    PSSetSamplers(d3dDeviceContext,
                  mGeometryPSReflector,
                  "gDiffuseSampler",
                  1, 
                  &mDiffuseSampler);
    // Diffuse texture set per-material by DXUT mesh routines
    d3dDeviceContext->PSSetShader(mGeometryPS, 0, 0);

    // Set up render GBuffer render targets
    std::vector<ID3D11RenderTargetView*> renderTargets(mGBuffer.size() + 1);
    renderTargets[0] = backBuffer;
    for (std::size_t i = 0; i < mGBuffer.size(); ++i) {
        renderTargets[i + 1] = mGBuffer[i]->GetRenderTarget();
    }
    d3dDeviceContext->OMSetRenderTargets(static_cast<UINT>(renderTargets.size()), &renderTargets.front(), mDepthBuffer->GetDepthStencil());
    d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);
    d3dDeviceContext->OMSetDepthStencilState(mDefaultDepthStencilState, 0x0);
    
    UINT diffuseTextureIndex;
    GetBindIndex(mGeometryPSReflector, 
                 "gDiffuseTexture",
                 &diffuseTextureIndex);
    mesh->Render(d3dDeviceContext, diffuseTextureIndex);

    // Cleanup (aka make the runtime happy)
    Cleanup(d3dDeviceContext);
}

void App::RenderSkybox(ID3D11DeviceContext* d3dDeviceContext,
                       ID3D11RenderTargetView* backBuffer,
                       const D3D11_VIEWPORT* viewport)
{
    D3D11_VIEWPORT skyboxViewport(*viewport);
    skyboxViewport.MinDepth = 1.0f;
    skyboxViewport.MaxDepth = 1.0f;

    d3dDeviceContext->IASetInputLayout(mMeshVertexLayout);

    VSSetConstantBuffers(d3dDeviceContext,
                         mSkyboxVSReflector,
                         "PerFrameConstants",
                         1, 
                         &mPerFrameConstants);
    
    d3dDeviceContext->VSSetShader(mSkyboxVS, 0, 0);

    d3dDeviceContext->RSSetState(mDoubleSidedRasterizerState);
    d3dDeviceContext->RSSetViewports(1, &skyboxViewport);

    PSSetSamplers(d3dDeviceContext,
                  mSkyboxPSReflector,
                  "gDiffuseSampler",
                  1, 
                  &mDiffuseSampler);

    d3dDeviceContext->PSSetShader(mSkyboxPS, 0, 0);

    // Set skybox texture
    PSSetShaderResources(d3dDeviceContext,
                         mSkyboxPSReflector,
                         "gSkyboxTexture",
                         1,
                         &mSkyboxSRV);

    d3dDeviceContext->OMSetRenderTargets(1, &backBuffer, mDepthBuffer->GetDepthStencil());
    d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);
    d3dDeviceContext->OMSetDepthStencilState(mDefaultDepthStencilState, 0x0);
    
    mSkyboxMesh.Render(d3dDeviceContext);

    // Cleanup (aka make the runtime happy)
    Cleanup(d3dDeviceContext);
}

void App::CaptureFragments(ID3D11DeviceContext* d3dDeviceContext, 
                           ParticleSystem *particleSystem,
                           const UIConstants* ui,
                           bool initCounter)
{
    switch(ui->volumeShadowMethod) {
        case VOL_SHADOW_UNCOMPRESSED:
        case VOL_SHADOW_DSM:
        case VOL_SHADOW_AVSM:
        case VOL_SHADOW_AVSM_BOX_4:
        case VOL_SHADOW_AVSM_GAUSS_7:
        {
            VSSetConstantBuffers(d3dDeviceContext,
                                 mParticleShadingVSReflector,
                                 "ParticlePerFrameConstants",
                                 1, 
                                 &mParticlePerFrameConstants);
            VSSetConstantBuffers(d3dDeviceContext,
                                 mParticleShadingVSReflector,
                                 "ParticlePerPassConstants",
                                 1, 
                                 &mParticlePerPassConstants);
            d3dDeviceContext->VSSetShader(mParticleShadingVS, 0, 0);

            d3dDeviceContext->RSSetState(mParticleRasterizerState);
            d3dDeviceContext->RSSetViewports(1, &mAVSMShadowViewport);


            PSSetConstantBuffers(d3dDeviceContext,
                                 mParticleAVSMCapturePSReflector,
                                 "ParticlePerFrameConstants",
                                 1, 
                                 &mParticlePerFrameConstants);
            PSSetConstantBuffers(d3dDeviceContext,
                                 mParticleAVSMCapturePSReflector,
                                 "LT_Constants",
                                 1, 
                                 &mListTextureConstants);
            d3dDeviceContext->PSSetShader(mParticleAVSMCapturePS, 0, 0);

            ID3D11UnorderedAccessView* listTexFirstSegmentNodeOffsetUAV = 
                mListTexFirstSegmentNodeOffset->GetUnorderedAccess();

            // Set List Texture UAVs (we don't need any RT!)
            static const char *paramUAVs[] = {
                "gListTexFirstSegmentNodeAddressUAV",
                "gListTexSegmentNodesUAV",
            };
            const UINT numUAVs = sizeof(paramUAVs) / sizeof(paramUAVs[0]);
            const UINT firstUAVIndex =
                GetStartBindIndex(mParticleAVSMCapturePSReflector, 
                                  paramUAVs, numUAVs);           
            UINT pUAVInitialCounts[numUAVs] = {0, 0};
            ID3D11UnorderedAccessView* pUAVs[numUAVs] = {
                listTexFirstSegmentNodeOffsetUAV, 
                mListTexSegmentNodesUAV
            };

            d3dDeviceContext->OMSetRenderTargetsAndUnorderedAccessViews(
                0, NULL, // render targets
                NULL,    // depth-stencil
                firstUAVIndex, numUAVs, pUAVs, initCounter ? pUAVInitialCounts : NULL);

            d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);
            d3dDeviceContext->OMSetDepthStencilState(mAVSMCaptureDepthStencilState, 0x0);

            particleSystem->Draw(d3dDeviceContext, NULL, 0, particleSystem->GetParticleCount());

            // Cleanup (aka make the runtime happy)
            Cleanup(d3dDeviceContext);

            break;
        }
    }
}

void App::ClearShadowBuffers(ID3D11DeviceContext* d3dDeviceContext, const UIConstants* ui)
{
    switch(ui->volumeShadowMethod) {
        case VOL_SHADOW_NO_SHADOW:
        case VOL_SHADOW_UNCOMPRESSED:
        case VOL_SHADOW_DSM:
        case VOL_SHADOW_AVSM:
        case VOL_SHADOW_AVSM_BOX_4:
        case VOL_SHADOW_AVSM_GAUSS_7:
        {
            ID3D11UnorderedAccessView* listTexFirstSegmentNodeOffsetUAV = 
                mListTexFirstSegmentNodeOffset->GetUnorderedAccess();

            // Initialize the first node offset RW UAV with a NULL offset (end of the list)
            UINT clearValues[4] = {
                0xFFFFFFFFUL, 
                0xFFFFFFFFUL, 
                0xFFFFFFFFUL, 
                0xFFFFFFFFUL
            };

            d3dDeviceContext->ClearUnorderedAccessViewUint(
                listTexFirstSegmentNodeOffsetUAV, clearValues);
            break;
        }
    }
}

void App::CreatePixelShadersFromCompiledObjs(ID3D11Device* d3dDevice,
                                             int shaderCount,
                                             const char* prefix, 
                                             const char* suffix, 
                                             ID3D11PixelShader** shaderArray,
                                             ID3D11ShaderReflection** shaderReflArray)
{
    for (int i = 0; i < shaderCount; i++) 
    {
        char cNodeCount[32];
        _itoa_s((i + 1) * 4, cNodeCount, sizeof(cNodeCount), 10);  

        {
            std::string objName = std::string(prefix) + 
                                  std::string(cNodeCount) + 
                                  std::string(suffix);
            std::wstring objWName(objName.length(), L'');
            std::copy(objName.begin(), objName.end(), objWName.begin());
            CreatePixelShaderFromCompiledObj(d3dDevice,
                                            (LPCTSTR)objWName.c_str(),
                                            &shaderArray[i],
                                            &shaderReflArray[i]);
        }
    }
}   

void App::CaptureHair(ID3D11DeviceContext* d3dDeviceContext, 
                      const UIConstants* ui,
                      bool initCounter)
{

    VSSetConstantBuffers(d3dDeviceContext,
                         mHairVSReflector,
                         "HairConstants",
                         1, 
                         &mHairConstants);
    d3dDeviceContext->VSSetShader(mHairVS, 0, 0);

    GSSetConstantBuffers(d3dDeviceContext,
                         mHairGSReflector,
                         "PerFrameConstants",
                         1,
                         &mPerFrameConstants);
    GSSetConstantBuffers(d3dDeviceContext,
                         mHairGSReflector,
                         "HairConstants",
                         1, 
                         &mHairConstants);

    d3dDeviceContext->GSSetShader(mHairGS, 0, 0);

    d3dDeviceContext->RSSetState(mHairRasterizerState);
    d3dDeviceContext->RSSetViewports(1, &mAVSMShadowViewport);
    

    PSSetConstantBuffers(d3dDeviceContext,
                         mShadowHairCapturePSReflector,
                         "PerFrameConstants",
                         1, 
                         &mPerFrameConstants);

    PSSetConstantBuffers(d3dDeviceContext,
                         mShadowHairCapturePSReflector,
                         "LT_Constants",
                         1, 
                         &mListTextureConstants);

    d3dDeviceContext->PSSetShader(mShadowHairCapturePS, 0, 0);

    ID3D11UnorderedAccessView* listTexFirstSegmentNodeOffsetUAV = 
        mListTexFirstSegmentNodeOffset->GetUnorderedAccess();

    // Set List Texture UAVs (we don't need any RT!)
    static const char *paramUAVs[] = {
        "gListTexFirstSegmentNodeAddressUAV",
        "gListTexSegmentNodesUAV",
    };
    const UINT numUAVs = sizeof(paramUAVs) / sizeof(paramUAVs[0]);
    const UINT firstUAVIndex =
        GetStartBindIndex(mShadowHairCapturePSReflector, 
                          paramUAVs, numUAVs);
    UINT pUAVInitialCounts[numUAVs] = {0, 0};
    ID3D11UnorderedAccessView* pUAVs[numUAVs] = {
        listTexFirstSegmentNodeOffsetUAV, 
        mListTexSegmentNodesUAV
    };

    d3dDeviceContext->OMSetRenderTargetsAndUnorderedAccessViews(
        0, NULL, // render targets
        NULL,    // depth-stencil
        firstUAVIndex, numUAVs, pUAVs, initCounter ? pUAVInitialCounts : NULL);

    d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);
    d3dDeviceContext->OMSetDepthStencilState(mAVSMCaptureDepthStencilState, 0x0);

    mHair->Draw(d3dDeviceContext);

    // Cleanup (aka make the runtime happy)
    Cleanup(d3dDeviceContext);
}

void App::GenerateVisibilityCurve(ID3D11DeviceContext* d3dDeviceContext,
                                 const UIConstants* ui)
{
    ID3D11ShaderResourceView*  listTexFirstSegmentNodeOffsetSRV = mListTexFirstSegmentNodeOffset->GetShaderResource();

    if (VOL_SHADOW_AVSM == ui->volumeShadowMethod ||
        VOL_SHADOW_AVSM_BOX_4 == ui->volumeShadowMethod ||
        VOL_SHADOW_AVSM_GAUSS_7 == ui->volumeShadowMethod) {

        // Second (full screen) pass, sort fragments and insert them in our AVSM texture(s)
        d3dDeviceContext->IASetInputLayout(0);
        d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        d3dDeviceContext->IASetVertexBuffers(0, 0, 0, 0, 0);

        d3dDeviceContext->VSSetShader(mFullScreenTriangleVS, 0, 0);

        d3dDeviceContext->RSSetState(mRasterizerState);
        d3dDeviceContext->RSSetViewports(1, &mAVSMShadowViewport);     
        
        ID3D11ShaderReflection *shaderReflector = NULL;
        switch (ui->avsmSortingMethod) {
            case 0:
                shaderReflector = mAVSMUnsortedResolvePSReflector[mShaderIdx];
                d3dDeviceContext->PSSetShader(mAVSMUnsortedResolvePS[mShaderIdx], 0, 0); 
                break;
            case 1:
                shaderReflector = mAVSMInsertionSortResolvePSReflector[mShaderIdx];
                d3dDeviceContext->PSSetShader(mAVSMInsertionSortResolvePS[mShaderIdx], 0, 0); 
                break;
            default: 
                break;
        }


        PSSetConstantBuffers(d3dDeviceContext,
                             shaderReflector,
                             "AVSMConstants",
                             1, 
                             &mAVSMConstants);

        PSSetShaderResources(d3dDeviceContext,
                             shaderReflector,
                             "gListTexFirstSegmentNodeAddressSRV",
                             1, 
                             &listTexFirstSegmentNodeOffsetSRV);
        PSSetShaderResources(d3dDeviceContext,
                             shaderReflector,
                             "gListTexSegmentNodesSRV",
                             1, 
                             &mListTexSegmentNodesSRV);
        PSSetShaderResources(d3dDeviceContext,
                             shaderReflector,
                             "gListTexFirstSegmentNodeAddressSRV",
                             1, 
                             &listTexFirstSegmentNodeOffsetSRV);
        
        ID3D11RenderTargetView* pRTs[16];
        const int avsmRTCount = mAVSMNodeCount / 2;
        for (int i = 0; i < avsmRTCount; ++i) {
            pRTs[i] = mAVSMTextures->GetRenderTarget(i);
        }
        d3dDeviceContext->OMSetRenderTargets(avsmRTCount, pRTs, 0);
                                                                    
        d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);
        d3dDeviceContext->OMSetDepthStencilState(mDefaultDepthStencilState, 0x0);

        // Full-screen triangle
        d3dDeviceContext->Draw(3, 0);

        // Cleanup (aka make the runtime happy)
        Cleanup(d3dDeviceContext);

    } else if (VOL_SHADOW_DSM == ui->volumeShadowMethod){   

        // Fill in general visibility/volume shadow constants
        {
            D3D11_MAPPED_SUBRESOURCE mappedResource;
            d3dDeviceContext->Map(mVolumeShadowConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
            VolumeShadowConstants* constants = static_cast<VolumeShadowConstants *>(mappedResource.pData);

            constants->mDSMError = ui->dsmError;
            
            d3dDeviceContext->Unmap(mVolumeShadowConstants, 0);
        }

        // This pass handles unbounded visiblity rapresentations (deep shadow maps, etc..)
        ID3D11UnorderedAccessView* listTexFirstVisibilityNodeOffsetUAV = mListTexFirstVisibilityNodeOffset->GetUnorderedAccess();

        // Initialize the first node offset RW UAV with a NULL offset (end of the list)
        UINT clearValues[4] = {0xFFFFFFFFUL, 0xFFFFFFFFUL, 0xFFFFFFFFUL, 0xFFFFFFFFUL};
        d3dDeviceContext->ClearUnorderedAccessViewUint(listTexFirstVisibilityNodeOffsetUAV, clearValues);

        CSSetConstantBuffers(d3dDeviceContext,
                             mComputeVisibilityCurveCSReflector,
                             "LT_Constants",
                             1, 
                             &mListTextureConstants);
        CSSetConstantBuffers(d3dDeviceContext,
                             mComputeVisibilityCurveCSReflector,
                             "VolumeShadowConstants",
                             1, 
                             &mVolumeShadowConstants);
        CSSetShaderResources(d3dDeviceContext,
                             mComputeVisibilityCurveCSReflector,
                             "gListTexFirstSegmentNodeAddressSRV",
                             1, 
                             &listTexFirstSegmentNodeOffsetSRV);
        CSSetShaderResources(d3dDeviceContext,
                             mComputeVisibilityCurveCSReflector,
                             "gListTexSegmentNodesSRV",
                             1, 
                             &mListTexSegmentNodesSRV);
        d3dDeviceContext->CSSetShader(mComputeVisibilityCurveCS, 0, 0);

        static const char *paramUAVs[] = {
            "gListTexFirstVisibilityNodeAddressUAV",
            "gListTexVisibilityNodesUAV",
        };
        const UINT numUAVs = sizeof(paramUAVs) / sizeof(paramUAVs[0]);
        const UINT firstUAVIndex =
            GetStartBindIndex(mComputeVisibilityCurveCSReflector, 
                              paramUAVs, numUAVs);

        UINT pUAVInitialCounts[2] = {0, 0};
        ID3D11UnorderedAccessView* pUAVs[2] = {listTexFirstVisibilityNodeOffsetUAV, mListTexVisibilityNodesUAV};
        d3dDeviceContext->CSSetUnorderedAccessViews(
            firstUAVIndex, numUAVs, pUAVs, pUAVInitialCounts); 

        d3dDeviceContext->Dispatch(mAVSMShadowTextureDim / 2, mAVSMShadowTextureDim / 4, 1);

        Cleanup(d3dDeviceContext);
    }
}

void App::ShadeGBuffer(ID3D11DeviceContext* d3dDeviceContext,
                       ID3D11RenderTargetView* backBuffer,
                       D3D11_VIEWPORT* viewport,
                       std::vector<ID3D11ShaderResourceView*> &gbufferTextures,
                       const UIConstants* ui)
{
    d3dDeviceContext->IASetInputLayout(0);
    d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    d3dDeviceContext->IASetVertexBuffers(0, 0, 0, 0, 0);

    d3dDeviceContext->VSSetShader(mFullScreenTriangleVS, 0, 0);

    d3dDeviceContext->RSSetState(mRasterizerState);
    d3dDeviceContext->RSSetViewports(1, viewport);

    ID3D11ShaderResourceView* asvmTextureSRV = mAVSMTextures->GetShaderResource();

    ID3D11ShaderResourceView*  listTexFirstSegmentNodeOffsetSRV = mListTexFirstSegmentNodeOffset->GetShaderResource();
    ID3D11ShaderResourceView*  listTexFirstVisibilityNodeOffsetSRV = mListTexFirstVisibilityNodeOffset->GetShaderResource();

    PSSetConstantBuffers(d3dDeviceContext,
                         mLightingPSReflector[mShaderIdx],
                         "PerFrameConstants",
                         1, 
                         &mPerFrameConstants);
    PSSetShaderResources(d3dDeviceContext,
                         mLightingPSReflector[mShaderIdx],
                         "gGBufferTextures[0]",
                         static_cast<UINT>(gbufferTextures.size()), 
                         &gbufferTextures.front());
    PSSetShaderResources(d3dDeviceContext,
                         mLightingPSReflector[mShaderIdx],
                         "gAVSMTexture",
                         1, 
                         &asvmTextureSRV);
    PSSetSamplers(d3dDeviceContext,
                  mLightingPSReflector[mShaderIdx],
                  "gAVSMSampler",
                  1, 
                  &mAVSMSampler);
 
    PSSetShaderResources(d3dDeviceContext,
                         mLightingPSReflector[mShaderIdx],
                         "gListTexFirstSegmentNodeAddressSRV",
                         1, 
                         &listTexFirstSegmentNodeOffsetSRV);
    PSSetShaderResources(d3dDeviceContext,
                         mLightingPSReflector[mShaderIdx],
                         "gListTexSegmentNodesSRV",
                         1, 
                         &mListTexSegmentNodesSRV);
    PSSetShaderResources(d3dDeviceContext,
                         mLightingPSReflector[mShaderIdx],
                         "gListTexFirstVisibilityNodeAddressSRV",
                         1, 
                         &listTexFirstVisibilityNodeOffsetSRV);
    PSSetShaderResources(d3dDeviceContext,
                         mLightingPSReflector[mShaderIdx],
                         "gListTexVisibilityNodesSRV",
                         1, 
                         &mListTexVisibilityNodesSRV);
    PSSetConstantBuffers(d3dDeviceContext,
                         mLightingPSReflector[mShaderIdx],
                         "LT_Constants",
                         1, 
                         &mListTextureConstants);
    PSSetConstantBuffers(d3dDeviceContext,
                         mLightingPSReflector[mShaderIdx],
                         "AVSMConstants",
                         1, 
                         &mAVSMConstants);

    d3dDeviceContext->PSSetShader(mLightingPS[mShaderIdx], 0, 0);

    // Additively blend into back buffer
    d3dDeviceContext->OMSetRenderTargets(1, &backBuffer, 0);
    d3dDeviceContext->OMSetBlendState(mLightingBlendState, 0, 0xFFFFFFFF);

    // Full-screen triangle
    d3dDeviceContext->Draw(3, 0);

    // Cleanup (aka make the runtime happy)
    Cleanup(d3dDeviceContext);
}

void App::ShadeParticles(ID3D11DeviceContext* d3dDeviceContext,
                         ParticleSystem *particleSystem,
                         ID3D11RenderTargetView* backBuffer,
                         D3D11_VIEWPORT* viewport,
                         std::vector<ID3D11ShaderResourceView*> &gbufferTextures,
                         const UIConstants* ui)
{
    VSSetConstantBuffers(d3dDeviceContext,
                         mParticleShadingVSReflector,
                         "ParticlePerFrameConstants",
                         1, 
                         &mParticlePerFrameConstants);
    VSSetConstantBuffers(d3dDeviceContext,
                         mParticleShadingVSReflector,
                         "ParticlePerPassConstants",
                         1, 
                         &mParticlePerPassConstants);
    d3dDeviceContext->VSSetShader(mParticleShadingVS, 0, 0);

    d3dDeviceContext->RSSetState(mParticleRasterizerState);
    d3dDeviceContext->RSSetViewports(1, viewport);

    ID3D11ShaderResourceView* asvmTextureSRV = mAVSMTextures->GetShaderResource();

    PSSetConstantBuffers(d3dDeviceContext,
                         mParticleShadingPSReflector[mShaderIdx],
                         "PerFrameConstants",
                         1, 
                         &mPerFrameConstants);
    PSSetConstantBuffers(d3dDeviceContext,
                         mParticleShadingPSReflector[mShaderIdx],
                         "ParticlePerFrameConstants",
                         1, 
                         &mParticlePerFrameConstants);

    ID3D11ShaderResourceView*  listTexFirstSegmentNodeOffsetSRV = mListTexFirstSegmentNodeOffset->GetShaderResource();
    ID3D11ShaderResourceView*  listTexFirstVisibilityNodeOffsetSRV = mListTexFirstVisibilityNodeOffset->GetShaderResource();

    PSSetShaderResources(d3dDeviceContext,
                         mParticleShadingPSReflector[mShaderIdx],
                         "gGBufferTextures[0]",
                         static_cast<UINT>(gbufferTextures.size()), 
                         &gbufferTextures.front());

    PSSetShaderResources(d3dDeviceContext,
                         mParticleShadingPSReflector[mShaderIdx],
                         "gAVSMTexture",
                         1, 
                         &asvmTextureSRV);
    PSSetSamplers(d3dDeviceContext,
                  mParticleShadingPSReflector[mShaderIdx],
                  "gAVSMSampler",
                  1, 
                  &mAVSMSampler);

    PSSetShaderResources(d3dDeviceContext,
                         mParticleShadingPSReflector[mShaderIdx],
                         "gListTexFirstSegmentNodeAddressSRV",
                         1, 
                         &listTexFirstSegmentNodeOffsetSRV);
    PSSetShaderResources(d3dDeviceContext,
                         mParticleShadingPSReflector[mShaderIdx],
                         "gListTexSegmentNodesSRV",
                         1, 
                         &mListTexSegmentNodesSRV);
    PSSetShaderResources(d3dDeviceContext,
                         mParticleShadingPSReflector[mShaderIdx],
                         "gListTexFirstVisibilityNodeAddressSRV",
                         1, 
                         &listTexFirstVisibilityNodeOffsetSRV);
    PSSetShaderResources(d3dDeviceContext,
                         mParticleShadingPSReflector[mShaderIdx],
                         "gListTexVisibilityNodesSRV",
                         1, 
                         &mListTexVisibilityNodesSRV);
    PSSetConstantBuffers(d3dDeviceContext,
                         mParticleShadingPSReflector[mShaderIdx],
                         "LT_Constants",
                         1, 
                         &mListTextureConstants);
    PSSetConstantBuffers(d3dDeviceContext,
                         mParticleShadingPSReflector[mShaderIdx],
                         "AVSMConstants",
                         1, 
                         &mAVSMConstants);

    d3dDeviceContext->PSSetShader(mParticleShadingPS[mShaderIdx], 0, 0);

    // Additively blend into back buffer
    d3dDeviceContext->OMSetRenderTargets(1, &backBuffer, mDepthBuffer->GetDepthStencil());
    d3dDeviceContext->OMSetBlendState(mParticleBlendState, 0, 0xFFFFFFFF);
    d3dDeviceContext->OMSetDepthStencilState(mParticleDepthStencilState, 0x0);

    particleSystem->Draw(d3dDeviceContext, NULL, 0, particleSystem->GetParticleCount());

    // Cleanup (aka make the runtime happy)
    Cleanup(d3dDeviceContext);
}

// Fill in frame constants
void App::FillInFrameConstants(ID3D11DeviceContext* d3dDeviceContext,
                               const FrameMatrices &m,
                               D3DXVECTOR4 cameraPos, 
                               CFirstPersonCamera* lightCamera,
                               const UIConstants* ui)
{
    // Compute light direction in view space
    D3DXVECTOR3 lightPosWorld = *lightCamera->GetEyePt();
    D3DXVECTOR3 lightTargetWorld = *lightCamera->GetLookAtPt();
    D3DXVECTOR3 lightPosView;
    D3DXVec3TransformCoord(&lightPosView, &lightPosWorld, &m.cameraView);
    D3DXVECTOR3 lightTargetView;
    D3DXVec3TransformCoord(&lightTargetView, &lightTargetWorld, &m.cameraView);
    D3DXVECTOR3 lightDirView = lightTargetView - lightPosView;
    D3DXVec3Normalize(&lightDirView, &lightDirView);

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    d3dDeviceContext->Map(mPerFrameConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    PerFrameConstants* constants = static_cast<PerFrameConstants *>(mappedResource.pData);

    // No world matrix for now...
    constants->mCameraWorldViewProj = m.cameraWorldViewProj;
    constants->mCameraWorldView = m.worldMatrix * m.cameraView;
    constants->mCameraViewProj = m.cameraViewProj;
    constants->mCameraProj = m.cameraProj;
    constants->mCameraPos = cameraPos;
    constants->mLightWorldViewProj = m.lightWorldViewProj;
    constants->mAvsmLightWorldViewProj = m.avmsLightWorldViewProj;
    constants->mCameraViewToLightProj = m.cameraViewToLightProj;
    constants->mCameraViewToLightView = m.cameraViewToLightView;
    constants->mCameraViewToAvsmLightProj = m.cameraViewToAvsmLightProj;
    constants->mCameraViewToAvsmLightView = m.cameraViewToAvsmLightView;
    constants->mLightDir = D3DXVECTOR4(lightDirView, 0.0f);

    constants->mUI = *ui;
    
    d3dDeviceContext->Unmap(mPerFrameConstants, 0);
}


void App::FillParticleRendererConstants(ID3D11DeviceContext* d3dDeviceContext,
                                        CFirstPersonCamera* camera,
                                        const D3DXMATRIXA16 &cameraView,
                                        const D3DXMATRIXA16 &cameraViewProj)
{
    // Particle renderer constants
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    d3dDeviceContext->Map(mParticlePerPassConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    ParticlePerPassConstants* constants = static_cast<ParticlePerPassConstants *>(mappedResource.pData);

    D3DXMATRIXA16 lightProj = *camera->GetProjMatrix();
    D3DXMATRIXA16 lightView = *camera->GetViewMatrix();

    constants->mParticleWorldViewProj = cameraViewProj;
    constants->mParticleWorldView     = cameraView;
    constants->mEyeRight              = *camera->GetWorldRight();
    constants->mEyeUp                 = *camera->GetWorldUp();
    
    d3dDeviceContext->Unmap(mParticlePerPassConstants, 0);
}

void App::FillListTextureConstants(ID3D11DeviceContext* d3dDeviceContext)
{
    // List texture related constants  
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    d3dDeviceContext->Map(mListTextureConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    LT_Constants* constants = static_cast<LT_Constants*>(mappedResource.pData);

    constants->mMaxNodes = mLisTexNodeCount;
    constants->mFirstNodeMapSize = (float)mAVSMShadowTextureDim;
    
    d3dDeviceContext->Unmap(mListTextureConstants, 0);
}

void App::FillAVSMConstants(ID3D11DeviceContext* d3dDeviceContext)
{
    // AVSM related constants  
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    d3dDeviceContext->Map(mAVSMConstants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    AVSMConstants* constants = static_cast<AVSMConstants*>(mappedResource.pData);

    constants->mMask0 = D3DXVECTOR4( 0.0f,  1.0f,  2.0f,  3.0f);
    constants->mMask1 = D3DXVECTOR4( 4.0f,  5.0f,  6.0f,  7.0f);
    constants->mMask2 = D3DXVECTOR4( 8.0f,  9.0f, 10.0f, 11.0f);
    constants->mMask3 = D3DXVECTOR4(12.0f, 13.0f, 14.0f, 15.0f);
    constants->mMask4 = D3DXVECTOR4(16.0f, 17.0f, 18.0f, 19.0f);
    constants->mEmptyNode = EMPTY_NODE;
    constants->mOpaqueNodeTrans = 1E-4f;
    constants->mShadowMapSize = (float)mAVSMShadowTextureDim;
    
    d3dDeviceContext->Unmap(mAVSMConstants, 0);
}

void App::VisualizeFirstNode(ID3D11DeviceContext* d3dDeviceContext,
                             ID3D11RenderTargetView* backBuffer,
                             D3D11_VIEWPORT* viewport)
{
    D3D11_VIEWPORT visualizationViewport;
    visualizationViewport.Width = mAVSMShadowViewport.Width;
    visualizationViewport.Height = mAVSMShadowViewport.Height;
    visualizationViewport.TopLeftX = 0.0f;//std::floor(viewport->TopLeftX);// + 10.0f);
    visualizationViewport.TopLeftY = 0.0f;//std::floor(viewport->TopLeftY + viewport->Height - visualizationViewport.Height - 10.0f);
    visualizationViewport.MinDepth = 0.0f;
    visualizationViewport.MaxDepth = 1.0f;        

    d3dDeviceContext->IASetInputLayout(0);
    d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    d3dDeviceContext->IASetVertexBuffers(0, 0, 0, 0, 0);

    d3dDeviceContext->VSSetShader(mFullScreenTriangleVS, 0, 0);

    d3dDeviceContext->RSSetState(mRasterizerState);
    d3dDeviceContext->RSSetViewports(1, &visualizationViewport);

    PSSetConstantBuffers(d3dDeviceContext,
                         mVisualizeListTexFirstNodeOffsetPSReflector,
                         "AVSMConstants",
                         1, 
                         &mAVSMConstants);

    
    ID3D11ShaderResourceView* listTexFirstNodeOffsetSRV = mListTexFirstSegmentNodeOffset->GetShaderResource();
    PSSetShaderResources(d3dDeviceContext,
                         mVisualizeListTexFirstNodeOffsetPSReflector,
                         "gListTexFirstSegmentNodeAddressSRV",
                         1, 
                         &listTexFirstNodeOffsetSRV);
    d3dDeviceContext->PSSetShader(mVisualizeListTexFirstNodeOffsetPS, 0, 0);

    d3dDeviceContext->OMSetRenderTargets(1, &backBuffer, 0);

    // Additively blend into back buffer
    d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);

    // Full-screen triangle
    d3dDeviceContext->Draw(3, 0);

    // Cleanup (aka make the runtime happy)
    Cleanup(d3dDeviceContext);
}

void App::VisualizeAVSM(ID3D11DeviceContext* d3dDeviceContext,
                        ID3D11RenderTargetView* backBuffer,
                        D3D11_VIEWPORT* viewport)
{
    D3D11_VIEWPORT visualizationViewport;
    visualizationViewport.Width = mAVSMShadowViewport.Width;;
    visualizationViewport.Height = mAVSMShadowViewport.Height;
    visualizationViewport.TopLeftX = 0.0f;//std::floor(viewport->TopLeftX + 10.0f);
    visualizationViewport.TopLeftY = 0.0f;//std::floor(viewport->TopLeftY + viewport->Height - visualizationViewport.Height - 10.0f);
    visualizationViewport.MinDepth = 0.0f;
    visualizationViewport.MaxDepth = 1.0f;        

    d3dDeviceContext->IASetInputLayout(0);
    d3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    d3dDeviceContext->IASetVertexBuffers(0, 0, 0, 0, 0);

    d3dDeviceContext->VSSetShader(mFullScreenTriangleVS, 0, 0);

    d3dDeviceContext->RSSetState(mRasterizerState);
    d3dDeviceContext->RSSetViewports(1, &visualizationViewport);

    PSSetConstantBuffers(d3dDeviceContext,
                         mVisualizeAVSMPSReflector,
                         "AVSMConstants",
                         1, 
                         &mAVSMConstants);

    ID3D11ShaderResourceView* asvmTextureSRV = mAVSMTextures->GetShaderResource();
    PSSetShaderResources(d3dDeviceContext,
                         mVisualizeAVSMPSReflector,
                         "gAVSMTexture",
                         1, 
                         &asvmTextureSRV);

    d3dDeviceContext->PSSetShader(mVisualizeAVSMPS, 0, 0);

    d3dDeviceContext->OMSetRenderTargets(1, &backBuffer, 0);

    // Additively blend into back buffer
    d3dDeviceContext->OMSetBlendState(mGeometryBlendState, 0, 0xFFFFFFFF);

    // Full-screen triangle
    d3dDeviceContext->Draw(3, 0);

    // Cleanup (aka make the runtime happy)
    Cleanup(d3dDeviceContext);
}
