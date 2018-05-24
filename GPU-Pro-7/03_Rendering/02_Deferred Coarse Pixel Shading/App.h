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

#pragma once

#include "DXUT.h"
#include "DXUTcamera.h"
#include "SDKMesh.h"
#include "Texture2D.h"
#include "Shader.h"
#include "Buffer.h"
#include <vector>
#include <memory>
#include "IntelPowerGadgetLib.h"
#define _PERF 1

enum LightCullTechnique {
    CULL_FORWARD_NONE = 0,
    CULL_FORWARD_PREZ_NONE,
    CULL_DEFERRED_NONE,
    CULL_QUAD,
    CULL_QUAD_DEFERRED_LIGHTING,
    CULL_COMPUTE_SHADER_TILE,
};

// NOTE: Must match shader equivalent structure
__declspec(align(16))
struct UIConstants
{
    unsigned int forcePerPixel;
    unsigned int lightingOnly;
    unsigned int faceNormals;
    unsigned int visualizeLightCount;
    unsigned int visualizePerSampleShading;
    unsigned int lightCullTechnique;
};

// NOTE: Must match shader equivalent structure
struct PointLight
{
    DirectX::XMFLOAT3 positionView;
    float attenuationBegin;
    DirectX::XMFLOAT3 color;
    float attenuationEnd;
};

// Host-side world-space initial transform data
struct PointLightInitTransform
{
    // Cylindrical coordinates
    float radius;
    float angle;
    float height;
    float animationSpeed;
};

// Flat framebuffer RGBA16-encoded
struct FramebufferFlatElement
{
    unsigned int rb;
    unsigned int ga;
};

#ifdef _PERF
enum DPS_TIMER {
    TIMER_GBUFFER_GEN = 0,
    TIMER_SHADING,
    TIMER_SKYBOX,
    TIMER_COUNT,
};

enum GPU_TIMER_TYPE {
    T_BEGIN,
    T_END,
    T_VALID,
    T_NUM
};

struct FrameStats
{
    FrameStats() {
        memset(this, 0, sizeof(*this));
    }

    void Accumulate(const FrameStats &stats) {
        m_totalShadingTime      += stats.m_totalShadingTime;
        m_totalGBuffGen         += stats.m_totalGBuffGen;
        m_totalSkyBox           += stats.m_totalSkyBox;
    }

    void Normalize(UINT numFrames) {
        assert(numFrames != 0);
        m_totalShadingTime      /= numFrames;
        m_totalGBuffGen         /= numFrames;
        m_totalSkyBox           /= numFrames;
    }

    float   m_totalShadingTime;
    float   m_totalSkyBox;
    float   m_totalGBuffGen;
};

float GetGPUCounterSeconds(ID3D11DeviceContext* d3dDeviceContext, ID3D11Query** query);
#endif // _PERF


class AverageReading
{
	int Index;
	int mSize;
	float Readings[64];
public:
	AverageReading()
	{
		Index = 0;
		mSize = 64;
		for(int i=0;i<mSize;i++)
			Readings[i] = -1.0f;

	}

	AverageReading(int Size)
	{
		Index = 0;
		mSize = Size;
		for(int i=0;i<mSize;i++)
			Readings[i] = -1.0f;

	}
	void Set(float NewReading)
	{
		Readings[Index]=NewReading;
		Index++;
		Index = Index%mSize;
	}
	float Get()
	{
		float Result=0;
		int dataGathered = 0;
		for(int i=0;i<mSize;i++)
		{
			if(Readings[i]!=-1.0f)
			{
				dataGathered++;
				Result+=Readings[i];
			}
		}
		if( dataGathered == 0 )
			Result = 0.0f;
		else
			Result /= (float)dataGathered;

		return Result;
	}
};
class App
{
public:
    App(ID3D11Device* d3dDevice, unsigned int activeLights, unsigned int msaaSamples);

    ~App();
    
    void OnD3D11ResizedSwapChain(ID3D11Device* d3dDevice,
                                 const DXGI_SURFACE_DESC* backBufferDesc);

    void Move(float elapsedTime);

    void Render(ID3D11DeviceContext* d3dDeviceContext,
                ID3D11RenderTargetView* backBuffer,
                CDXUTSDKMesh& mesh_opaque,
                CDXUTSDKMesh& mesh_alpha,
                ID3D11ShaderResourceView* skybox,
                const DirectX::CXMMATRIX worldMatrix,
                const CFirstPersonCamera* viewerCamera,
                const D3D11_VIEWPORT* viewport,
                const UIConstants* ui);

    void SetActiveLights(ID3D11Device* d3dDevice, unsigned int activeLights);
    unsigned int GetActiveLights() const { return mActiveLights; }
#ifdef _PERF
    FrameStats GetFrameStats() const { return m_frameStats; }
#endif // _PERF

private:
    // StephanieB5: removed unused parameter ID3D11Device* d3dDevice
    void InitializeLightParameters();

    // Notes: 
    // - Most of these functions should all be called after initializing per frame/pass constants, etc.
    //   as the shaders that they invoke bind those constant buffers.

    // Set up shader light buffer
    ID3D11ShaderResourceView * SetupLights(ID3D11DeviceContext* d3dDeviceContext,
                                           const DirectX::CXMMATRIX cameraView);

    // StephanieB5 removed unused parameters CFirstPersonCamera* viewerCamera & UIConstants* ui
    // Forward rendering of geometry into
    ID3D11ShaderResourceView * RenderForward(ID3D11DeviceContext* d3dDeviceContext,
                                             CDXUTSDKMesh& mesh_opaque,
                                             CDXUTSDKMesh& mesh_alpha,
                                             ID3D11ShaderResourceView *lightBufferSRV,
                                             const D3D11_VIEWPORT* viewport,
                                             bool doPreZ);
    
    // StephanieB5: removed unused parameters CFirstPersonCamera* viewerCamera & UIConstants* ui
    // Draws geometry into G-buffer
    void RenderGBuffer(ID3D11DeviceContext* d3dDeviceContext,
                       CDXUTSDKMesh& mesh_opaque,
                       CDXUTSDKMesh& mesh_alpha,
                       const D3D11_VIEWPORT* viewport);    

    // Handles skybox, tone mapping, etc
    void RenderSkyboxAndToneMap(ID3D11DeviceContext* d3dDeviceContext,
                                ID3D11RenderTargetView* backBuffer,
                                ID3D11ShaderResourceView* skybox,
                                ID3D11ShaderResourceView* depthSRV,
                                const D3D11_VIEWPORT* viewport,
                                const UIConstants* ui);
    
    void ComputeLighting(ID3D11DeviceContext* d3dDeviceContext,
                         ID3D11ShaderResourceView *lightBufferSRV,
                         const D3D11_VIEWPORT* viewport,
                         const UIConstants* ui);
    
    unsigned int mMSAASamples;
    float mTotalTime;

    ID3D11InputLayout* mMeshVertexLayout;

    VertexShader* mGeometryVS;

    PixelShader* mGBufferPS;
    PixelShader* mGBufferAlphaTestPS;

    PixelShader* mForwardPS;
    PixelShader* mForwardAlphaTestPS;
    PixelShader* mForwardAlphaTestOnlyPS;

    CDXUTSDKMesh mSkyboxMesh;
    VertexShader* mSkyboxVS;
    PixelShader* mSkyboxPS;
    
    VertexShader* mFullScreenTriangleVS;

    PixelShader* mRequiresPerSampleShadingPS;

    PixelShader* mBasicLoopPS;
    PixelShader* mBasicLoopPerSamplePS;

    ComputeShader* mComputeShaderTileCS;

    VertexShader* mGPUQuadVS;
    GeometryShader* mGPUQuadGS;
    PixelShader* mGPUQuadPS;
    PixelShader* mGPUQuadPerSamplePS;

    PixelShader* mGPUQuadDLPS;
    PixelShader* mGPUQuadDLPerSamplePS;

    PixelShader* mGPUQuadDLResolvePS;
    PixelShader* mGPUQuadDLResolvePerSamplePS;
    
    ID3D11Buffer* mPerFrameConstants;
    
    ID3D11RasterizerState* mRasterizerState;
    ID3D11RasterizerState* mDoubleSidedRasterizerState;

    ID3D11DepthStencilState* mDepthState;
    ID3D11DepthStencilState* mWriteStencilState;
    ID3D11DepthStencilState* mEqualStencilState;

    ID3D11BlendState* mGeometryBlendState;
    ID3D11BlendState* mLightingBlendState;

    ID3D11SamplerState* mDiffuseSampler;
    ID3D11SamplerState* mDPSDiffuseSampler;

    std::vector< std::tr1::shared_ptr<Texture2D> > mGBuffer;
    // Handy cache of list of RT pointers for G-buffer
    std::vector<ID3D11RenderTargetView*> mGBufferRTV;
    // Handy cache of list of SRV pointers for the G-buffer
    std::vector<ID3D11ShaderResourceView*> mGBufferSRV;
    unsigned int mGBufferWidth;
    unsigned int mGBufferHeight;

    // We use a different lit buffer (different bind flags and MSAA handling) depending on whether we
    // write to it from the pixel shader (render target) or compute shader (UAV)
    std::tr1::shared_ptr<Texture2D> mLitBufferPS;
    std::tr1::shared_ptr<StructuredBuffer<FramebufferFlatElement> > mLitBufferCS;

    // A temporary accumulation buffer used for deferred lighting
    std::tr1::shared_ptr<Texture2D> mDeferredLightingAccumBuffer;

    std::tr1::shared_ptr<Depth2D> mDepthBuffer;
    // We also need a read-only depth stencil view for techniques that read the G-buffer while also using Z-culling
    ID3D11DepthStencilView* mDepthBufferReadOnlyDSV;

    // Lighting state
    unsigned int mActiveLights;
    std::vector<PointLightInitTransform> mLightInitialTransform;
    std::vector<PointLight> mPointLightParameters;
    std::vector<DirectX::XMFLOAT3> mPointLightPositionWorld;
    
    StructuredBuffer<PointLight>* mLightBuffer;

#ifdef _PERF
    // Queries
    ID3D11Query*                m_pTimers[TIMER_COUNT][T_NUM];
    FrameStats                  m_frameStats;
#endif // _PERF
};
