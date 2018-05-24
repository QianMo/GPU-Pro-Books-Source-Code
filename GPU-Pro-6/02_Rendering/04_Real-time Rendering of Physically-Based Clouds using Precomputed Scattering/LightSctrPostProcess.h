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

#pragma once

#include <D3DX11.h>
#include <D3DX10math.h>

#include "RenderTechnique.h"

#include "structures.fxh"
struct SFrameAttribs
{
    ID3D11Device *pd3dDevice;
    ID3D11DeviceContext *pd3dDeviceContext;
    
    double dElapsedTime;

    SLightAttribs *pLightAttribs;
    ID3D11Buffer *pcbLightAttribs;
    ID3D11Buffer *pcbCameraAttribs;

    //SCameraAttribs CameraAttribs;
    
    ID3D11ShaderResourceView *ptex2DSrcColorBufferSRV;
    ID3D11RenderTargetView   *ptex2DSrcColorBufferRTV;
    ID3D11DepthStencilView   *ptex2DSrcDepthBufferDSV;
    ID3D11ShaderResourceView *ptex2DSrcDepthBufferSRV;
    ID3D11ShaderResourceView *ptex2DShadowMapSRV;
    ID3D11RenderTargetView *pDstRTV;
    ID3D11ShaderResourceView *ptex2DScrSpaceCloudTransparencySRV;
    ID3D11ShaderResourceView *ptex2DScrSpaceCloudMinMaxDistSRV;
    ID3D11ShaderResourceView *ptex2DScrSpaceCloudColorSRV;
    ID3D11ShaderResourceView *ptex2DLiSpCloudTransparencySRV;
    ID3D11ShaderResourceView *ptex2DLiSpCloudMinMaxDepthSRV;
};

#undef float4
#undef float3
#undef float2

#include <atlcomcli.h>

class CLightSctrPostProcess
{
public:
    CLightSctrPostProcess();
    ~CLightSctrPostProcess();

    HRESULT OnCreateDevice(ID3D11Device* in_pd3dDevice, 
                           ID3D11DeviceContext *in_pd3dDeviceContext);
    void OnDestroyDevice();


    HRESULT OnResizedSwapChain(ID3D11Device* pd3dDevice, UINT uiBackBufferWidth, UINT uiBackBufferHeight);

    void PerformPostProcessing(SFrameAttribs &FrameAttribs,
                               SPostProcessingAttribs &PPAttribs);

    void ComputeSunColor(const D3DXVECTOR3 &vDirectionOnSun,
                         const D3DXVECTOR4 &f4ExtraterrestrialSunColor,
                         D3DXVECTOR4 &f4SunColorAtGround,
                         D3DXVECTOR4 &f4AmbientLight);
    
    void RenderSun(SFrameAttribs &FrameAttribs);
    ID3D11Buffer *GetMediaAttribsCB(){return m_pcbMediaAttribs;}
    ID3D11ShaderResourceView* GetPrecomputedNetDensitySRV(){return m_ptex2DOccludedNetDensityToAtmTopSRV;}
    ID3D11ShaderResourceView* GetAmbientSkyLightSRV(ID3D11Device *pDevice, ID3D11DeviceContext *pContext);

private:
    void ReconstructCameraSpaceZ(SFrameAttribs &FrameAttribs);
    void RenderSliceEndpoints(SFrameAttribs &FrameAttribs);
    void RenderCoordinateTexture(SFrameAttribs &FrameAttribs);
    void RenderCoarseUnshadowedInsctr(SFrameAttribs &FrameAttribs);
    void RefineSampleLocations(SFrameAttribs &FrameAttribs);
    void MarkRayMarchingSamples(SFrameAttribs &FrameAttribs);
    void RenderSliceUVDirAndOrig(SFrameAttribs &FrameAttribs);
    void Build1DMinMaxMipMap(SFrameAttribs &FrameAttribs, int iCascadeIndex);
    void RenderCldDensEpipolarScan(SFrameAttribs &FrameAttribs, int iCascadeIndex);
    void DoRayMarching(SFrameAttribs &FrameAttribs, UINT uiMaxStepsAlongRay, const SShadowMapAttribs &SMAttribs, int iCascadeIndex);
    void InterpolateInsctrIrradiance(SFrameAttribs &FrameAttribs);
    void UnwarpEpipolarScattering(SFrameAttribs &FrameAttribs, bool bRenderLuminance);
    void UpdateAverageLuminance(SFrameAttribs &FrameAttribs);
    void FixInscatteringAtDepthBreaks(SFrameAttribs &FrameAttribs, UINT uiMaxStepsAlongRay, const SShadowMapAttribs &SMAttribs, bool bRenderLuminance);
    void RenderSampleLocations(SFrameAttribs &FrameAttribs);
    HRESULT CreatePrecomputedOpticalDepthTexture(ID3D11Device *pDevice, ID3D11DeviceContext *pContext);
    HRESULT CreatePrecomputedScatteringLUT(ID3D11Device *pDevice, ID3D11DeviceContext *pContext);
    void CreateRandomSphereSamplingTexture(ID3D11Device *pDevice);
    void CreateLowResLuminanceTexture(ID3D11Device *pDevice);
    void CreateAmbientSkyLightTexture(ID3D11Device *pDevice, ID3D11DeviceContext *pContext);

    void DefineMacros(class CD3DShaderMacroHelper &Macros);
    
    SPostProcessingAttribs m_PostProcessingAttribs;
    bool m_bUseCombinedMinMaxTexture;
    UINT m_uiSampleRefinementCSThreadGroupSize;
    UINT m_uiSampleRefinementCSMinimumThreadGroupSize;

    CComPtr<ID3D11ShaderResourceView> m_ptex2DSliceEndpointsSRV;
    CComPtr<ID3D11RenderTargetView> m_ptex2DSliceEndpointsRTV;

    CComPtr<ID3D11ShaderResourceView> m_ptex2DCoordinateTextureSRV;
    CComPtr<ID3D11RenderTargetView> m_ptex2DCoordinateTextureRTV;
    CComPtr<ID3D11DepthStencilView> m_ptex2DEpipolarImageDSV;

    CComPtr<ID3D11ShaderResourceView> m_ptex2DEpipolarCamSpaceZSRV;
    CComPtr<ID3D11RenderTargetView> m_ptex2DEpipolarCamSpaceZRTV;

    CComPtr<ID3D11ShaderResourceView> m_ptex2DEpipolarCloudTranspSRV;
    CComPtr<ID3D11RenderTargetView> m_ptex2DEpipolarCloudTranspRTV;

    CComPtr<ID3D11ShaderResourceView> m_ptex2DInterpolationSourcesSRV;
    CComPtr<ID3D11UnorderedAccessView> m_ptex2DInterpolationSourcesUAV;

    CComPtr<ID3D11ShaderResourceView> m_ptex2DInitialScatteredLightSRV;
    CComPtr<ID3D11RenderTargetView> m_ptex2DInitialScatteredLightRTV;

    CComPtr<ID3D11ShaderResourceView> m_ptex2DEpipolarInscatteringSRV, m_ptex2DEpipolarExtinctionSRV;
    CComPtr<ID3D11RenderTargetView> m_ptex2DEpipolarInscatteringRTV, m_ptex2DEpipolarExtinctionRTV;

    CComPtr<ID3D11ShaderResourceView> m_ptex2DCameraSpaceZSRV;
    CComPtr<ID3D11RenderTargetView> m_ptex2DCameraSpaceZRTV;
    
    CComPtr<ID3D11ShaderResourceView> m_ptex2DSliceUVDirAndOriginSRV;
    CComPtr<ID3D11RenderTargetView> m_ptex2DSliceUVDirAndOriginRTV;

    CComPtr<ID3D11ShaderResourceView> m_ptex2DMinMaxShadowMapSRV[2];
    CComPtr<ID3D11RenderTargetView> m_ptex2DMinMaxShadowMapRTV[2];

    CComPtr<ID3D11ShaderResourceView> m_ptex2DCldDensEpipolarScanSRV[2];
    CComPtr<ID3D11RenderTargetView> m_ptex2DCldDensEpipolarScanRTV[2];

    static const int sm_iNumPrecomputedHeights = 1024;
    static const int sm_iNumPrecomputedAngles = 1024;
    CComPtr<ID3D11ShaderResourceView> m_ptex2DOccludedNetDensityToAtmTopSRV;
    CComPtr<ID3D11RenderTargetView>   m_ptex2DOccludedNetDensityToAtmTopRTV;

    
    static const int sm_iPrecomputedSctrUDim = 32/2;
    static const int sm_iPrecomputedSctrVDim = 128;
    static const int sm_iPrecomputedSctrWDim = 64/2;
    static const int sm_iPrecomputedSctrQDim = 16;
    CComPtr<ID3D11ShaderResourceView> m_ptex3DSingleScatteringSRV;
    CComPtr<ID3D11ShaderResourceView> m_ptex3DHighOrderScatteringSRV;
    CComPtr<ID3D11ShaderResourceView> m_ptex3DMultipleScatteringSRV;
    
    static const int sm_iNumRandomSamplesOnSphere = 128;
    CComPtr<ID3D11ShaderResourceView> m_ptex2DSphereRandomSamplingSRV;

    HRESULT CreateTextures(ID3D11Device* pd3dDevice);
    HRESULT CreateMinMaxShadowMap(ID3D11Device* pd3dDevice);
    HRESULT CreateCldDensEpipolarScanTex(ID3D11Device* pd3dDevice);
    CComPtr<ID3D11DepthStencilView> m_ptex2DScreenSizeDSV;

    static const int sm_iLowResLuminanceMips = 7; // 64x64
    CComPtr<ID3D11RenderTargetView> m_ptex2DLowResLuminanceRTV;
    CComPtr<ID3D11ShaderResourceView> m_ptex2DLowResLuminanceSRV;
    
    CComPtr<ID3D11RenderTargetView> m_ptex2DAverageLuminanceRTV;
    CComPtr<ID3D11ShaderResourceView> m_ptex2DAverageLuminanceSRV;

    static const int sm_iAmbientSkyLightTexDim = 1024;
    CComPtr<ID3D11RenderTargetView> m_ptex2DAmbientSkyLightRTV;
    CComPtr<ID3D11ShaderResourceView> m_ptex2DAmbientSkyLightSRV;

    UINT m_uiBackBufferWidth, m_uiBackBufferHeight;
    LPCTSTR m_strEffectPath;

    CComPtr<ID3D11VertexShader> m_pGenerateScreenSizeQuadVS;

    CRenderTechnique m_ReconstrCamSpaceZTech;
    CRenderTechnique m_RendedSliceEndpointsTech;
    CRenderTechnique m_RendedCoordTexTech;
    CRenderTechnique m_RefineSampleLocationsTech;
    CRenderTechnique m_RenderCoarseUnshadowedInsctrTech;
    CRenderTechnique m_MarkRayMarchingSamplesInStencilTech;
    CRenderTechnique m_RenderSliceUVDirInSMTech;
    CRenderTechnique m_InitializeMinMaxShadowMapTech;
    CRenderTechnique m_ComputeMinMaxSMLevelTech;
    CRenderTechnique m_InitializeCldDensEpipolarScanTech;
    CRenderTechnique m_ComputeCldDensEpiScanLevelTech;
    CRenderTechnique m_DoRayMarchTech[2]; // 0 - min/max optimization disabled; 1 - min/max optimization enabled
    CRenderTechnique m_InterpolateIrradianceTech;
    CRenderTechnique m_UnwarpEpipolarSctrImgTech;
    CRenderTechnique m_UnwarpAndRenderLuminanceTech;
    CRenderTechnique m_UpdateAverageLuminanceTech;
    CRenderTechnique m_FixInsctrAtDepthBreaksTech[4]; // bit 0: 0 - perform ray marching, 1 - perform ray marching and attenuate background
                                                      // bit 1: 0 - perform tone mapping, 1 - render luminance only
    CRenderTechnique m_RenderSampleLocationsTech;
    CRenderTechnique m_RenderSunTech;
    CRenderTechnique m_PrecomputeSingleSctrTech;
    CRenderTechnique m_ComputeSctrRadianceTech;
    CRenderTechnique m_ComputeScatteringOrderTech;
    CRenderTechnique m_AddScatteringOrderTech;

    CComPtr<ID3D11SamplerState> m_psamLinearClamp;
    CComPtr<ID3D11SamplerState> m_psamLinearBorder0;
    CComPtr<ID3D11SamplerState> m_psamComparison;
    CComPtr<ID3D11SamplerState> m_psamPointClamp;

    CComPtr<ID3D11DepthStencilState> m_pEnableDepthCmpEqDS;
    CComPtr<ID3D11DepthStencilState> m_pDisableDepthTestDS;
    CComPtr<ID3D11DepthStencilState> m_pDisableDepthTestIncrStencilDS;
    CComPtr<ID3D11DepthStencilState> m_pNoDepth_StEqual_IncrStencilDS;
    CComPtr<ID3D11DepthStencilState> m_pNoDepth_StEqual_KeepStencilDS;

    CComPtr<ID3D11RasterizerState> m_pSolidFillNoCullRS;

    CComPtr<ID3D11BlendState> m_pDefaultBS, m_pAdditiveBlendBS, m_pAlphaBlendBS;

    void ComputeScatteringCoefficients(ID3D11DeviceContext *pDeviceCtx = NULL);
    
    const float m_fTurbidity;
    SAirScatteringAttribs m_MediaParams;

    CComPtr<ID3D11Buffer> m_pcbPostProcessingAttribs;
    CComPtr<ID3D11Buffer> m_pcbMediaAttribs;
    CComPtr<ID3D11Buffer> m_pcbMiscParams;
};
