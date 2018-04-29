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

#ifndef H_APP
#define H_APP

//--------------------------------------------------------------------------------------
// Includes
//--------------------------------------------------------------------------------------

#include "DXUT.h"
#include "DXUTcamera.h"
#include "SDKMesh.h"
#include "AVSM_def.h"
#include "AppShaderConstants.h"
#include "Hair.h"
#include "Texture2D.h"
#include "ParticleSystem.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

//--------------------------------------------------------------------------------------
// Defines
//--------------------------------------------------------------------------------------

#define MAX_AVSM_RT_COUNT   (8)
#define MAX_SHADER_VARIATIONS (MAX_AVSM_RT_COUNT / 2)


//--------------------------------------------------------------------------------------
// Enums
//--------------------------------------------------------------------------------------
enum SCENE_SELECTION {
    POWER_PLANT_SCENE        = 0,
    GROUND_PLANE_SCENE       = 1,
    HAIR_AND_PARTICLES_SCENE = 2,
};

//--------------------------------------------------------------------------------------
// Classes
//--------------------------------------------------------------------------------------

class App
{
public:
    struct Options {
        Options() : 
            scene(POWER_PLANT_SCENE),
            enableHair(false),
            enableParticles(true),
            enableAutoBoundsAVSM(true),
            enableShadowPicking(false),
            NodeCount(8),
            enableTransmittanceCurve(false),
            enableVolumeShadowCreation(true),
            pickedX(0),
            pickedY(0),
            hairX(50),
            hairY(50),
            hairZ(50),
            hairSort(HAIR_SORT_NONE),
            hairSortEnabled(true),
            hairBlending(true)
        {}

        SCENE_SELECTION scene;
        bool enableHair;
        bool enableParticles;
        bool enableAutoBoundsAVSM;
        bool enableShadowPicking;
        unsigned int NodeCount;
        bool enableTransmittanceCurve;
        bool enableVolumeShadowCreation;
        int pickedX;
        int pickedY;
        float hairX;
        float hairY;
        float hairZ;
        unsigned int hairSort;
        bool hairSortEnabled;
        bool hairBlending;
    };

    App(ID3D11Device* d3dDevice,
        ID3D11DeviceContext* d3dDeviceContext,
        unsigned int nodeCount,
        unsigned int shadowTextureDim,     
        unsigned int avsmShadowTextureDim);

    ~App();
    
    void OnD3D11ResizedSwapChain(ID3D11Device* d3dDevice,
                                 const DXGI_SURFACE_DESC* backBufferDesc);

    void Render(const Options &options,
                ID3D11DeviceContext* d3dDeviceContext,
                ID3D11RenderTargetView* backBuffer,
                CDXUTSDKMesh* mesh,
                ParticleSystem *particleSystem,
                const D3DXMATRIXA16& worldMatrix,
                CFirstPersonCamera* viewerCamera,
                CFirstPersonCamera* lightCamera,
                D3D11_VIEWPORT* viewport,
                UIConstants* ui);

    void DumpTransmittanceCurve() {
        mDumpTransmittanceCurve = true;
        ++mDumpTransmittanceCurveIndex;
    }

    void UpdateHairMesh();

    void SetNodeCount(int nodeCount);

private:
    struct PointVertex {
        float x, y, z;
    };

    struct FrameMatrices
    {    
        D3DXMATRIXA16 worldMatrix;

        D3DXMATRIXA16 cameraProj;
        D3DXMATRIXA16 cameraView;
        D3DXMATRIXA16 cameraViewInv;

        D3DXMATRIXA16 cameraWorldViewProj;
        D3DXMATRIXA16 cameraWorldView;

        D3DXMATRIXA16 cameraViewProj;
        D3DXMATRIXA16 cameraViewToLightProj;
        D3DXMATRIXA16 cameraViewToLightView;
        D3DXMATRIXA16 cameraViewToAvsmLightProj;
        D3DXMATRIXA16 cameraViewToAvsmLightView;

        D3DXMATRIXA16 lightProj;
        D3DXMATRIXA16 lightView;
        D3DXMATRIXA16 lightViewProj;
        D3DXMATRIXA16 lightWorldViewProj;

        D3DXMATRIXA16 avsmLightProj;
        D3DXMATRIXA16 avsmLightView;
        D3DXMATRIXA16 avsmLightViewProj;
        D3DXMATRIXA16 avmsLightWorldViewProj;
    };


    void DumpOrDrawTransmittance(const Options &options,
                                 const UIConstants &ui,
                                 ID3D11DeviceContext* d3dDeviceContext, 
                                 ID3D11RenderTargetView* backBuffer,
                                 D3D11_VIEWPORT* viewport,
                                 float depthBounds[2],
                                 int x, int y);

    void RenderGeometryPhase(const Options &options,
                             ID3D11DeviceContext* d3dDeviceContext,
                             ID3D11RenderTargetView* backBuffer,
                             CDXUTSDKMesh* mesh,
                             D3D11_VIEWPORT* viewport);

     void RenderSkybox(ID3D11DeviceContext* d3dDeviceContext,
                      ID3D11RenderTargetView* backBuffer,
                      const D3D11_VIEWPORT* viewport);

 
    void CaptureFragments(ID3D11DeviceContext* d3dDeviceContext,
                          ParticleSystem *particleSystem,
                          const UIConstants* ui,
                          bool initCounter);

    void CaptureHair(ID3D11DeviceContext* d3dDeviceContext, 
                     const UIConstants* ui,
                     bool initCounter);

    void GenerateVisibilityCurve(ID3D11DeviceContext* d3dDeviceContext,
                                 const UIConstants* ui);




    void ShadeGBuffer(ID3D11DeviceContext* d3dDeviceContext,
                      ID3D11RenderTargetView* backBuffer,
                      D3D11_VIEWPORT* viewport,
                      std::vector<ID3D11ShaderResourceView*> &gbufferTextures,
                      const UIConstants* ui);

    void ShadeParticles(ID3D11DeviceContext* d3dDeviceContext,
                        ParticleSystem *particleSystem,
                        ID3D11RenderTargetView* backBuffer,
                        D3D11_VIEWPORT* viewport,
                        std::vector<ID3D11ShaderResourceView*> &gbufferTextures,
                        const UIConstants* ui);

    void FillInFrameConstants(ID3D11DeviceContext* d3dDeviceContext,
                              const FrameMatrices &m,
                              D3DXVECTOR4 cameraPos,
                              CFirstPersonCamera* lightCamera,
                              const UIConstants* ui);

    void FillParticleRendererConstants(ID3D11DeviceContext* d3dDeviceContext,
                                       CFirstPersonCamera* lightCamera,
                                       const D3DXMATRIXA16 &cameraView,
                                       const D3DXMATRIXA16 &cameraViewProj);
    void FillListTextureConstants(ID3D11DeviceContext* d3dDeviceContext);
    void FillAVSMConstants(ID3D11DeviceContext* d3dDeviceContext);
    void FillHairConstants(ID3D11DeviceContext* d3dDeviceContext,
                           const HairConstants &hairConstants);

    // A few debug functions
    void VisualizeFirstNode(ID3D11DeviceContext* d3dDeviceContext,
                            ID3D11RenderTargetView* backBuffer,
                            D3D11_VIEWPORT* viewport);
    void VisualizeAVSM(ID3D11DeviceContext* d3dDeviceContext,
                       ID3D11RenderTargetView* backBuffer,
                       D3D11_VIEWPORT* viewport);

    void RenderHair(ID3D11DeviceContext* d3dDeviceContext,
                    ID3D11RenderTargetView* backBuffer,
                    D3D11_VIEWPORT* viewport,
                    const D3DXMATRIXA16 &cameraWorldView,
                    const UIConstants* ui,
                    const Options &options);
    void RenderHairPerPixelSort(ID3D11DeviceContext* d3dDeviceContext,
                                ID3D11RenderTargetView* backBuffer,
                                D3D11_VIEWPORT* viewport,
                                const UIConstants* ui,
                                const Options &options);
    void RenderHairNoPerPixelSort(ID3D11DeviceContext* d3dDeviceContext,
                                  ID3D11RenderTargetView* backBuffer,
                                  D3D11_VIEWPORT* viewport,
                                  const UIConstants* ui,
                                  const Options &options);

    void ClearShadowBuffers(ID3D11DeviceContext* d3dDeviceContext, const UIConstants* ui);

    void CreatePixelShadersFromCompiledObjs(ID3D11Device* d3dDevice,
                                           int shaderCount,
                                           const char* prefix, 
                                           const char* suffix, 
                                           ID3D11PixelShader** shaderArray,
                                           ID3D11ShaderReflection** shaderReflArray);
    static std::string GetSceneName(SCENE_SELECTION scene);
    static SCENE_SELECTION GetSceneEnum(const std::string &name);
    std::string GetShadowMethod(const UIConstants &ui);

    unsigned int mShadowTextureDim;
    unsigned int mAVSMShadowTextureDim;
    unsigned int mLisTexNodeCount;

    ID3D11InputLayout*  mMeshVertexLayout;

    ID3D11VertexShader* mGeometryVS;
    ID3D11ShaderReflection* mGeometryVSReflector;
    ID3D11PixelShader*  mGeometryPS;
    ID3D11ShaderReflection* mGeometryPSReflector;

    CDXUTSDKMesh mSkyboxMesh;
    ID3D11Texture2D* mSkyboxTexture;
    ID3D11ShaderResourceView* mSkyboxSRV;
    ID3D11VertexShader* mSkyboxVS;
    ID3D11ShaderReflection* mSkyboxVSReflector;
    ID3D11PixelShader* mSkyboxPS;
    ID3D11ShaderReflection* mSkyboxPSReflector;

    ID3D11Texture2D *mParticleOpacityNoiseTexture;
    ID3D11ShaderResourceView* mParticleOpacityNoiseTextureSRV;

    ID3D11VertexShader* mFullScreenTriangleVS;
    ID3D11ShaderReflection* mFullScreenTriangleVSReflector;
    ID3D11PixelShader*  mLightingPS[MAX_SHADER_VARIATIONS];
    ID3D11ShaderReflection* mLightingPSReflector[MAX_SHADER_VARIATIONS];

    ID3D11PixelShader*  mVisualizeListTexFirstNodeOffsetPS;
    ID3D11ShaderReflection* mVisualizeListTexFirstNodeOffsetPSReflector;
    ID3D11PixelShader*  mVisualizeAVSMPS;
    ID3D11ShaderReflection* mVisualizeAVSMPSReflector;

    // Particle and AVSM shaders
    ID3D11VertexShader*  mParticleShadingVS;
    ID3D11ShaderReflection* mParticleShadingVSReflector;
    ID3D11PixelShader*   mParticleShadingPS[MAX_SHADER_VARIATIONS];
    ID3D11ShaderReflection* mParticleShadingPSReflector[MAX_SHADER_VARIATIONS];
    ID3D11PixelShader*   mParticleAVSMCapturePS; // Capture-all-fragments shaders
    ID3D11ShaderReflection* mParticleAVSMCapturePSReflector;
    ID3D11PixelShader*   mAVSMUnsortedResolvePS[MAX_SHADER_VARIATIONS];
    ID3D11ShaderReflection* mAVSMUnsortedResolvePSReflector[MAX_SHADER_VARIATIONS];
    ID3D11PixelShader*   mAVSMInsertionSortResolvePS[MAX_SHADER_VARIATIONS];
    ID3D11ShaderReflection* mAVSMInsertionSortResolvePSReflector[MAX_SHADER_VARIATIONS];
    ID3D11ComputeShader* mComputeVisibilityCurveCS;
    ID3D11ShaderReflection* mComputeVisibilityCurveCSReflector;
    ID3D11PixelShader*   mAVSMSinglePassInsertPS[MAX_SHADER_VARIATIONS]; // Fake render-target-read shaders
    ID3D11ShaderReflection* mAVSMSinglePassInsertPSReflector[MAX_SHADER_VARIATIONS];
    ID3D11PixelShader*   mAVSMClearStructuredBufPS;
    ID3D11ShaderReflection* mAVSMClearStructuredBufPSReflector;
    ID3D11PixelShader*   mAVSMConvertSUAVtoTex2DPS;
    ID3D11ShaderReflection* mAVSMConvertSUAVtoTex2DPSReflector;

    ID3D11Buffer* mPerFrameConstants;
    ID3D11Buffer* mParticlePerFrameConstants;
    ID3D11Buffer* mParticlePerPassConstants;
    ID3D11Buffer* mListTextureConstants;
    ID3D11Buffer* mAVSMConstants;

    ID3D11Buffer* mVolumeShadowConstants;

    ID3D11RasterizerState* mRasterizerState;
    ID3D11RasterizerState* mDoubleSidedRasterizerState;
    ID3D11RasterizerState* mShadowRasterizerState;
    ID3D11RasterizerState* mParticleRasterizerState;
    ID3D11RasterizerState* mHairRasterizerState;

    ID3D11DepthStencilState* mDefaultDepthStencilState;
    ID3D11DepthStencilState* mParticleDepthStencilState;
    ID3D11DepthStencilState* mAVSMCaptureDepthStencilState;
    ID3D11DepthStencilState* mHairCaptureDepthStencilState;
    ID3D11DepthStencilState* mHairRenderDepthStencilState;

    ID3D11BlendState* mGeometryBlendState;
    ID3D11BlendState* mLightingBlendState;
    ID3D11BlendState* mParticleBlendState;
    ID3D11BlendState* mHairRenderBlendState;

    ID3D11SamplerState* mDiffuseSampler;
    ID3D11SamplerState* mShadowSampler;
    ID3D11SamplerState* mShadowOnParticlesSampler;
    ID3D11SamplerState* mAVSMSampler;

    std::vector< std::tr1::shared_ptr<Texture2D> > mGBuffer;
    std::tr1::shared_ptr<Depth2D> mDepthBuffer;

    // AVSM 
    D3D11_VIEWPORT                  mAVSMShadowViewport;
    int                             mAVSMNodeCount;
    int                             mShaderIdx;
    std::tr1::shared_ptr<Texture2D> mAVSMTextures;
    ID3D11Texture2D*                mAVSMTexturesDebug;
    ID3D11Buffer*                   mAVSMStructBuf;
    ID3D11UnorderedAccessView*      mAVSMStructBufUAV;
    ID3D11ShaderResourceView*       mAVSMStructBufSRV;

    // List texture 
    std::tr1::shared_ptr<Texture2D> mListTexFirstSegmentNodeOffset;
    std::tr1::shared_ptr<Texture2D> mListTexFirstVisibilityNodeOffset;
    ID3D11Buffer*                   mListTexSegmentNodes;
    ID3D11Buffer*                   mListTexSegmentNodesDebug;
    ID3D11Buffer*                   mListTexVisibilityNodes;
    ID3D11Buffer*                   mListTexVisibilityNodesDebug;
    ID3D11UnorderedAccessView*      mListTexSegmentNodesUAV;
    ID3D11ShaderResourceView*       mListTexSegmentNodesSRV;
    ID3D11UnorderedAccessView*      mListTexVisibilityNodesUAV;
    ID3D11ShaderResourceView*       mListTexVisibilityNodesSRV;
    ID3D11Texture2D*                mListTexFirstOffsetDebug;

    // Hair constants
    unsigned int                    mHairLTMaxNodeCount;

    // Hair constant buffers
    ID3D11Buffer*                   mHairConstants;
    ID3D11Buffer*                   mHairLTConstants;

    // Hair shaders
    ID3D11VertexShader*             mHairVS;
    ID3D11GeometryShader*           mHairGS;
    ID3D11PixelShader*              mCameraHairCapturePS[MAX_SHADER_VARIATIONS];
    ID3D11PixelShader*              mCameraHairRenderPS;
    ID3D11PixelShader*              mStandardCameraHairRenderPS[MAX_SHADER_VARIATIONS];
    ID3D11PixelShader*              mShadowHairCapturePS;
    ID3D11ShaderReflection*         mHairVSReflector;
    ID3D11ShaderReflection*         mHairGSReflector;
    ID3D11ShaderReflection*         mCameraHairCapturePSReflector[MAX_SHADER_VARIATIONS];
    ID3D11ShaderReflection*         mCameraHairRenderPSReflector;
    ID3D11ShaderReflection*         mStandardCameraHairRenderPSReflector[MAX_SHADER_VARIATIONS];
    ID3D11ShaderReflection*         mShadowHairCapturePSReflector;
 
    // Hair List Texture
    std::tr1::shared_ptr<Texture2D> mHairLTFirstNodeOffset;
    ID3D11Buffer*                   mHairLTNodes;
    ID3D11UnorderedAccessView*      mHairLTNodesUAV;
    ID3D11ShaderResourceView*       mHairLTNodesSRV;

    // Hair renderer
    Hair*                           mHair;
    bool                            mHairMeshDirty;

    ID3D11VertexShader*             mDrawTransmittanceVS;
    ID3D11ShaderReflection*         mDrawTransmittanceVSReflector;
    ID3D11PixelShader*              mDrawTransmittancePS;
    ID3D11ShaderReflection*         mDrawTransmittancePSReflector;
    ID3D11Buffer*                   mDrawTransmittanceVB;
    ID3D11InputLayout*              mDrawTransmittanceLayout;
    bool                            mDumpTransmittanceCurve;
    int                             mDumpTransmittanceCurveIndex;
    int                             mDrawTransmittanceMaxNodes;

    float                           mLastTime;
};

#endif // H_APP
