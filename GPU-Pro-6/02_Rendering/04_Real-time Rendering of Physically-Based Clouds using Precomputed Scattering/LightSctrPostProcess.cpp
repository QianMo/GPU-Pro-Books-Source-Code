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
#include "StdAfx.h"
#include "LightSctrPostProcess.h"
#include <atlcomcli.h>
#include <cassert>
#include <stdio.h>
#include "ShaderMacroHelper.h"

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef V
#define V(x)           { hr = (x); assert( SUCCEEDED(hr) ); }
#endif
#ifndef V_RETURN
#define V_RETURN(x)    { hr = (x); assert( SUCCEEDED(hr) ); if( FAILED(hr) ) { return hr; } }
#endif

CLightSctrPostProcess :: CLightSctrPostProcess() : 
    m_uiSampleRefinementCSThreadGroupSize(0),
    // Using small group size is inefficient because a lot of SIMD lanes become idle
    m_uiSampleRefinementCSMinimumThreadGroupSize(128),// Must be greater than 32
    m_fTurbidity(1.02f),
    m_strEffectPath( L"fx\\LightScattering.fx" ),
    m_bUseCombinedMinMaxTexture(false)
{
    ComputeScatteringCoefficients();
}

CLightSctrPostProcess :: ~CLightSctrPostProcess()
{

}

HRESULT CLightSctrPostProcess :: OnCreateDevice(ID3D11Device* in_pd3dDevice, 
                                                ID3D11DeviceContext *in_pd3dDeviceContext)
{
    

    HRESULT hr;


    // Create depth stencil states

    D3D11_DEPTH_STENCIL_DESC EnableDepthTestDSDesc;
    ZeroMemory(&EnableDepthTestDSDesc, sizeof(EnableDepthTestDSDesc));
    EnableDepthTestDSDesc.DepthEnable = TRUE;
    EnableDepthTestDSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    EnableDepthTestDSDesc.DepthFunc = D3D11_COMPARISON_EQUAL;
    V_RETURN( in_pd3dDevice->CreateDepthStencilState(  &EnableDepthTestDSDesc, &m_pEnableDepthCmpEqDS) );
    
    // Disable depth testing
    D3D11_DEPTH_STENCIL_DESC DisableDepthTestDSDesc;
    ZeroMemory(&DisableDepthTestDSDesc, sizeof(DisableDepthTestDSDesc));
    DisableDepthTestDSDesc.DepthEnable = FALSE;
    DisableDepthTestDSDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    V_RETURN( in_pd3dDevice->CreateDepthStencilState(  &DisableDepthTestDSDesc, &m_pDisableDepthTestDS) );
    
    // Disable depth testing and always increment stencil value
    // This depth stencil state is used to mark samples which will undergo further processing
    // Pixel shader discards pixels which should not be further processed, thus keeping the
    // stencil value untouched
    // For instance, pixel shader performing epipolar coordinates generation discards all 
    // sampes, whoose coordinates are outside the screen [-1,1]x[-1,1] area
    D3D11_DEPTH_STENCIL_DESC DisbaleDepthIncrStencilDSSDesc = DisableDepthTestDSDesc;
    DisbaleDepthIncrStencilDSSDesc.StencilEnable = TRUE;
    DisbaleDepthIncrStencilDSSDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
    DisbaleDepthIncrStencilDSSDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_INCR;
    DisbaleDepthIncrStencilDSSDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    DisbaleDepthIncrStencilDSSDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    DisbaleDepthIncrStencilDSSDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
    DisbaleDepthIncrStencilDSSDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_INCR;
    DisbaleDepthIncrStencilDSSDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    DisbaleDepthIncrStencilDSSDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    DisbaleDepthIncrStencilDSSDesc.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK;
    DisbaleDepthIncrStencilDSSDesc.StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK;
    V_RETURN( in_pd3dDevice->CreateDepthStencilState(  &DisbaleDepthIncrStencilDSSDesc, &m_pDisableDepthTestIncrStencilDS) );


    // Disable depth testing, stencil testing function equal, increment stencil
    // This state is used to process only these pixels that were marked at the previous pass
    // All pixels whith different stencil value are discarded from further processing as well
    // as some pixels can also be discarded during the draw call
    // For instance, pixel shader marking ray marching samples processes only these pixels which are inside
    // the screen. It also discards all but these samples which are interpolated from themselves
    D3D11_DEPTH_STENCIL_DESC DisbaleDepthStencilEqualIncrStencilDSSDesc = DisbaleDepthIncrStencilDSSDesc;
    DisbaleDepthStencilEqualIncrStencilDSSDesc.FrontFace.StencilFunc = D3D11_COMPARISON_EQUAL;
    DisbaleDepthStencilEqualIncrStencilDSSDesc.BackFace.StencilFunc = D3D11_COMPARISON_EQUAL;
    V_RETURN( in_pd3dDevice->CreateDepthStencilState(  &DisbaleDepthStencilEqualIncrStencilDSSDesc, &m_pNoDepth_StEqual_IncrStencilDS) );

    D3D11_DEPTH_STENCIL_DESC DisbaleDepthStencilEqualKeepStencilDSSDesc = DisbaleDepthStencilEqualIncrStencilDSSDesc;
    DisbaleDepthStencilEqualKeepStencilDSSDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    DisbaleDepthStencilEqualKeepStencilDSSDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    V_RETURN( in_pd3dDevice->CreateDepthStencilState(  &DisbaleDepthStencilEqualKeepStencilDSSDesc, &m_pNoDepth_StEqual_KeepStencilDS) );
    

    // Create rasterizer state
    D3D11_RASTERIZER_DESC SolidFillCullBackRSDesc;
    ZeroMemory(&SolidFillCullBackRSDesc, sizeof(SolidFillCullBackRSDesc));
    SolidFillCullBackRSDesc.FillMode = D3D11_FILL_SOLID;
    SolidFillCullBackRSDesc.CullMode = D3D11_CULL_NONE;
    V_RETURN( in_pd3dDevice->CreateRasterizerState( &SolidFillCullBackRSDesc, &m_pSolidFillNoCullRS) );

    // Create default blend state
    D3D11_BLEND_DESC DefaultBlendStateDesc;
    ZeroMemory(&DefaultBlendStateDesc, sizeof(DefaultBlendStateDesc));
    DefaultBlendStateDesc.IndependentBlendEnable = FALSE;
    for(int i=0; i< _countof(DefaultBlendStateDesc.RenderTarget); i++)
        DefaultBlendStateDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    V_RETURN( in_pd3dDevice->CreateBlendState( &DefaultBlendStateDesc, &m_pDefaultBS) );

    D3D11_BLEND_DESC AdditiveBlendStateDesc;
    ZeroMemory(&AdditiveBlendStateDesc, sizeof(AdditiveBlendStateDesc));
    AdditiveBlendStateDesc.IndependentBlendEnable = FALSE;
    for(int i=0; i< _countof(AdditiveBlendStateDesc.RenderTarget); i++)
        AdditiveBlendStateDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    AdditiveBlendStateDesc.RenderTarget[0].BlendEnable = TRUE;
    AdditiveBlendStateDesc.RenderTarget[0].BlendOp     = D3D11_BLEND_OP_ADD;
    AdditiveBlendStateDesc.RenderTarget[0].BlendOpAlpha= D3D11_BLEND_OP_ADD;
    AdditiveBlendStateDesc.RenderTarget[0].DestBlend   = D3D11_BLEND_ONE;
    AdditiveBlendStateDesc.RenderTarget[0].DestBlendAlpha= D3D11_BLEND_ONE;
    AdditiveBlendStateDesc.RenderTarget[0].SrcBlend     = D3D11_BLEND_ONE;
    AdditiveBlendStateDesc.RenderTarget[0].SrcBlendAlpha= D3D11_BLEND_ONE;
    V_RETURN( in_pd3dDevice->CreateBlendState( &AdditiveBlendStateDesc, &m_pAdditiveBlendBS) );

    D3D11_BLEND_DESC AlphaBlendStateDesc = AdditiveBlendStateDesc;
    AlphaBlendStateDesc.RenderTarget[0].SrcBlend      = D3D11_BLEND_SRC_ALPHA;
    AlphaBlendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
    AlphaBlendStateDesc.RenderTarget[0].DestBlend      = D3D11_BLEND_INV_SRC_ALPHA;
    AlphaBlendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    V_RETURN( in_pd3dDevice->CreateBlendState( &AlphaBlendStateDesc, &m_pAlphaBlendBS) );
    

    

    // Create samplers

    D3D11_SAMPLER_DESC SamLinearBorder0Desc = 
    {
        D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        D3D11_TEXTURE_ADDRESS_BORDER,
        D3D11_TEXTURE_ADDRESS_BORDER,
        D3D11_TEXTURE_ADDRESS_BORDER,
        0, //FLOAT MipLODBias;
        0, //UINT MaxAnisotropy;
        D3D11_COMPARISON_NEVER, // D3D11_COMPARISON_FUNC ComparisonFunc;
        {0.f, 0.f, 0.f, 0.f}, //FLOAT BorderColor[ 4 ];
        -FLT_MAX, //FLOAT MinLOD;
        +FLT_MAX //FLOAT MaxLOD;
    };
    V_RETURN( in_pd3dDevice->CreateSamplerState( &SamLinearBorder0Desc, &m_psamLinearBorder0) );

    D3D11_SAMPLER_DESC SamLinearClampDesc = 
    {
        D3D11_FILTER_MIN_MAG_MIP_LINEAR,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        D3D11_TEXTURE_ADDRESS_CLAMP,
        0, //FLOAT MipLODBias;
        0, //UINT MaxAnisotropy;
        D3D11_COMPARISON_NEVER, // D3D11_COMPARISON_FUNC ComparisonFunc;
        {0.f, 0.f, 0.f, 0.f}, //FLOAT BorderColor[ 4 ];
        -FLT_MAX, //FLOAT MinLOD;
        +FLT_MAX //FLOAT MaxLOD;
    };
    V_RETURN( in_pd3dDevice->CreateSamplerState( &SamLinearClampDesc, &m_psamLinearClamp) );

    D3D11_SAMPLER_DESC SamPointClampDesc = SamLinearClampDesc;
    SamPointClampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    V_RETURN( in_pd3dDevice->CreateSamplerState( &SamPointClampDesc, &m_psamPointClamp) );

    D3D11_SAMPLER_DESC SamComparisonDesc = 
    {
        D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
        D3D11_TEXTURE_ADDRESS_BORDER,
        D3D11_TEXTURE_ADDRESS_BORDER,
        D3D11_TEXTURE_ADDRESS_BORDER,
        0, //FLOAT MipLODBias;
        0, //UINT MaxAnisotropy;
        D3D11_COMPARISON_GREATER, // D3D11_COMPARISON_FUNC ComparisonFunc;
        {0.f, 0.f, 0.f, 0.f}, //FLOAT BorderColor[ 4 ];
        -FLT_MAX, //FLOAT MinLOD;
        +FLT_MAX //FLOAT MaxLOD;
    };
    V_RETURN( in_pd3dDevice->CreateSamplerState( &SamComparisonDesc, &m_psamComparison) );


    
    // Create constant buffers

    D3D11_BUFFER_DESC CBDesc = 
    {
        sizeof(SPostProcessingAttribs),
        D3D11_USAGE_DYNAMIC,
        D3D11_BIND_CONSTANT_BUFFER,
        D3D11_CPU_ACCESS_WRITE, //UINT CPUAccessFlags
        0, //UINT MiscFlags;
        0, //UINT StructureByteStride;
    };
    V_RETURN( in_pd3dDevice->CreateBuffer( &CBDesc, NULL, &m_pcbPostProcessingAttribs) );

    CBDesc.ByteWidth = sizeof(SMiscDynamicParams);
    V_RETURN( in_pd3dDevice->CreateBuffer( &CBDesc, NULL, &m_pcbMiscParams) );

    CBDesc.ByteWidth = sizeof(SAirScatteringAttribs);
    CBDesc.Usage = D3D11_USAGE_DEFAULT;
    CBDesc.CPUAccessFlags = 0;
    D3D11_SUBRESOURCE_DATA InitData = 
    {
        &m_MediaParams,
        0, // UINT SysMemPitch
        0  // UINT SysMemSlicePitch
    };
    V_RETURN( in_pd3dDevice->CreateBuffer( &CBDesc, &InitData, &m_pcbMediaAttribs) );

    CRenderTechnique GenerateScreenSizeQuadTech;
    GenerateScreenSizeQuadTech.SetDeviceAndContext( in_pd3dDevice, in_pd3dDeviceContext );
    V( GenerateScreenSizeQuadTech.CreateVertexShaderFromFile(m_strEffectPath, "GenerateScreenSizeQuadVS", NULL ) );
    m_pGenerateScreenSizeQuadVS = GenerateScreenSizeQuadTech.GetVS();

    CreatePrecomputedOpticalDepthTexture(in_pd3dDevice, in_pd3dDeviceContext);

    return S_OK;
}

void CLightSctrPostProcess :: OnDestroyDevice()
{
    m_ptex2DSliceEndpointsSRV.Release();
    m_ptex2DSliceEndpointsRTV.Release();
    m_ptex2DCoordinateTextureSRV.Release();
    m_ptex2DCoordinateTextureRTV.Release();
    m_ptex2DEpipolarImageDSV.Release();
    m_ptex2DInterpolationSourcesSRV.Release();
    m_ptex2DInterpolationSourcesUAV.Release();
    m_ptex2DEpipolarCamSpaceZSRV.Release();
    m_ptex2DEpipolarCamSpaceZRTV.Release();
    m_ptex2DEpipolarCloudTranspSRV.Release();
    m_ptex2DEpipolarCloudTranspRTV.Release();
    m_ptex2DEpipolarInscatteringSRV.Release();
    m_ptex2DEpipolarInscatteringRTV.Release();
    m_ptex2DEpipolarExtinctionSRV.Release();
    m_ptex2DEpipolarExtinctionRTV.Release();
    m_ptex2DInitialScatteredLightSRV.Release();
    m_ptex2DInitialScatteredLightRTV.Release();
    m_ptex2DScreenSizeDSV.Release();
    m_ptex2DCameraSpaceZRTV.Release();
    m_ptex2DCameraSpaceZSRV.Release();
    for(int i=0; i < _countof(m_ptex2DMinMaxShadowMapSRV); ++i)
        m_ptex2DMinMaxShadowMapSRV[i].Release();
    for(int i=0; i < _countof(m_ptex2DMinMaxShadowMapRTV); ++i)
        m_ptex2DMinMaxShadowMapRTV[i].Release();
    for(int i=0; i < _countof(m_ptex2DCldDensEpipolarScanSRV); ++i)
        m_ptex2DCldDensEpipolarScanSRV[i].Release();
    for(int i=0; i < _countof(m_ptex2DCldDensEpipolarScanRTV); ++i)
        m_ptex2DCldDensEpipolarScanRTV[i].Release();
    m_ptex2DSliceUVDirAndOriginSRV.Release();
    m_ptex2DSliceUVDirAndOriginRTV.Release();

    m_ptex2DOccludedNetDensityToAtmTopSRV.Release();
    m_ptex2DOccludedNetDensityToAtmTopRTV.Release();
    m_ptex3DSingleScatteringSRV.Release();
    m_ptex3DHighOrderScatteringSRV.Release();
    m_ptex3DMultipleScatteringSRV.Release();
    
    m_ptex2DLowResLuminanceSRV.Release();
    m_ptex2DLowResLuminanceRTV.Release();
    m_ptex2DAverageLuminanceRTV.Release();
    m_ptex2DAverageLuminanceSRV.Release();

    m_ptex2DAmbientSkyLightRTV.Release();
    m_ptex2DAmbientSkyLightSRV.Release();

    m_ptex2DSphereRandomSamplingSRV.Release();

    m_psamLinearClamp.Release();
    m_psamLinearBorder0.Release();
    m_psamComparison.Release();
    m_psamPointClamp.Release();

    m_ReconstrCamSpaceZTech.Release();
    m_RendedSliceEndpointsTech.Release();
    m_RendedCoordTexTech.Release();
    m_RefineSampleLocationsTech.Release();
    m_RenderCoarseUnshadowedInsctrTech.Release();
    m_MarkRayMarchingSamplesInStencilTech.Release();
    m_RenderSliceUVDirInSMTech.Release();
    m_InitializeMinMaxShadowMapTech.Release();
    m_ComputeMinMaxSMLevelTech.Release();
    m_InitializeCldDensEpipolarScanTech.Release();
    m_ComputeCldDensEpiScanLevelTech.Release();
    m_DoRayMarchTech[0].Release();
    m_DoRayMarchTech[1].Release();
    m_InterpolateIrradianceTech.Release();
    m_UnwarpEpipolarSctrImgTech.Release();
    m_UnwarpAndRenderLuminanceTech.Release();
    m_UpdateAverageLuminanceTech.Release();
    for(size_t i=0; i<_countof(m_FixInsctrAtDepthBreaksTech); ++i)
        m_FixInsctrAtDepthBreaksTech[i].Release();
    m_RenderSampleLocationsTech.Release();
    m_RenderSunTech.Release();
    m_PrecomputeSingleSctrTech.Release();
    m_ComputeSctrRadianceTech.Release();
    m_ComputeScatteringOrderTech.Release();
    m_AddScatteringOrderTech.Release();

    m_pGenerateScreenSizeQuadVS.Release();

    m_pEnableDepthCmpEqDS.Release();
    m_pDisableDepthTestDS.Release();
    m_pDisableDepthTestIncrStencilDS.Release();
    m_pNoDepth_StEqual_IncrStencilDS.Release();
    m_pNoDepth_StEqual_KeepStencilDS.Release();

    m_pSolidFillNoCullRS.Release();

    m_pDefaultBS.Release();
    m_pAdditiveBlendBS.Release();
    m_pAlphaBlendBS.Release();

    m_pcbPostProcessingAttribs.Release();
    m_pcbMediaAttribs.Release();
    m_pcbMiscParams.Release();
}

HRESULT CLightSctrPostProcess :: OnResizedSwapChain(ID3D11Device* pd3dDevice, UINT uiBackBufferWidth, UINT uiBackBufferHeight)
{
    m_uiBackBufferWidth = uiBackBufferWidth;
    m_uiBackBufferHeight = uiBackBufferHeight;
    D3D11_TEXTURE2D_DESC ScreenSizeDepthStencilTexDesc = 
    {
        uiBackBufferWidth,                  //UINT Width;
        uiBackBufferHeight,                 //UINT Height;
        1,                                  //UINT MipLevels;
        1,                                  //UINT ArraySize;
        DXGI_FORMAT_D24_UNORM_S8_UINT,      //DXGI_FORMAT Format;
        {1,0},                              //DXGI_SAMPLE_DESC SampleDesc;
        D3D11_USAGE_DEFAULT,                //D3D11_USAGE Usage;
        D3D11_BIND_DEPTH_STENCIL,           //UINT BindFlags;
        0,                                  //UINT CPUAccessFlags;
        0,                                  //UINT MiscFlags;
    };

    m_ptex2DScreenSizeDSV.Release();
    CComPtr<ID3D11Texture2D> ptex2DScreenSizeDepthStencil;
    // Create 2-D texture, shader resource and target view buffers on the device
    HRESULT hr;
    V_RETURN( pd3dDevice->CreateTexture2D( &ScreenSizeDepthStencilTexDesc, NULL, &ptex2DScreenSizeDepthStencil) );
    V_RETURN( pd3dDevice->CreateDepthStencilView( ptex2DScreenSizeDepthStencil, NULL, &m_ptex2DScreenSizeDSV)  );

    m_ptex2DCameraSpaceZRTV.Release();
    m_ptex2DCameraSpaceZSRV.Release();
    D3D11_TEXTURE2D_DESC CamSpaceZTexDesc = ScreenSizeDepthStencilTexDesc;
    CamSpaceZTexDesc.Format = DXGI_FORMAT_R32_FLOAT;
    CamSpaceZTexDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    CComPtr<ID3D11Texture2D> ptex2DCamSpaceZ;
    V_RETURN( pd3dDevice->CreateTexture2D( &CamSpaceZTexDesc, NULL, &ptex2DCamSpaceZ) );
    V_RETURN( pd3dDevice->CreateShaderResourceView( ptex2DCamSpaceZ, NULL, &m_ptex2DCameraSpaceZSRV)  );
    V_RETURN( pd3dDevice->CreateRenderTargetView( ptex2DCamSpaceZ, NULL, &m_ptex2DCameraSpaceZRTV)  );

    m_RendedSliceEndpointsTech.Release();
    m_RendedCoordTexTech.Release();
    m_RenderSliceUVDirInSMTech.Release();
    m_RenderSampleLocationsTech.Release();
    m_UnwarpEpipolarSctrImgTech.Release();
    m_UnwarpAndRenderLuminanceTech.Release();
    return S_OK;
}

static void UnbindResources(ID3D11DeviceContext *pDeviceCtx)
{
    ID3D11ShaderResourceView *pDummySRVs[20]={NULL};
    ID3D11UnorderedAccessView *pDummyUAVs[8]={NULL};
    pDeviceCtx->PSSetShaderResources(0, _countof(pDummySRVs), pDummySRVs);
    pDeviceCtx->VSSetShaderResources(0, _countof(pDummySRVs), pDummySRVs);
    pDeviceCtx->GSSetShaderResources(0, _countof(pDummySRVs), pDummySRVs);
    pDeviceCtx->CSSetShaderResources(0, _countof(pDummySRVs), pDummySRVs);
    pDeviceCtx->CSSetUnorderedAccessViews(0, _countof(pDummyUAVs), pDummyUAVs, NULL);
}

void RenderQuad(ID3D11DeviceContext *pd3dDeviceCtx, 
                CRenderTechnique &State, 
                int iWidth = 0, int iHeight = 0,
                int iTopLeftX = 0, int iTopLeftY = 0,
                int iNumInstances = 1)
{
    // Initialize the viewport
    if( !iWidth && !iHeight )
    {
        assert( iTopLeftX == 0 && iTopLeftY == 0 );
        CComPtr<ID3D11RenderTargetView> pRTV;
        CComPtr<ID3D11DepthStencilView> pDSV;
        pd3dDeviceCtx->OMGetRenderTargets(1, &pRTV, &pDSV);
        CComPtr<ID3D11Resource> pDstTex;
        if( pRTV )
            pRTV->GetResource( &pDstTex );
        else if(pDSV)
            pDSV->GetResource( &pDstTex );
        D3D11_TEXTURE2D_DESC DstTexDesc;
        CComQIPtr<ID3D11Texture2D>(pDstTex)->GetDesc( &DstTexDesc );
        iWidth = DstTexDesc.Width;
        iHeight = DstTexDesc.Height;
    }
    
    D3D11_VIEWPORT NewViewPort;
    NewViewPort.TopLeftX = static_cast<float>( iTopLeftX );
    NewViewPort.TopLeftY = static_cast<float>( iTopLeftY );
    NewViewPort.Width  = static_cast<float>( iWidth );
    NewViewPort.Height = static_cast<float>( iHeight );
    NewViewPort.MinDepth = 0;
    NewViewPort.MaxDepth = 1;
    // Set the viewport
    pd3dDeviceCtx->RSSetViewports(1, &NewViewPort);  

    UINT offset[1] = {0};
    UINT stride[1] = {0};
    ID3D11Buffer *ppBuffers[1] = {0};
    pd3dDeviceCtx->IASetVertexBuffers(0, 1, ppBuffers, stride, offset);
    // There is no input-layout object and the primitive topology is triangle strip
    pd3dDeviceCtx->IASetInputLayout(NULL);
    pd3dDeviceCtx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    State.Apply();
    if( iNumInstances == 1 )
    {
        // Draw 4 vertices (two triangles )
        pd3dDeviceCtx->Draw(4, 0);
    }
    else
    {
        // Draw 4 vertices (two triangles ) x number of instances
        pd3dDeviceCtx->DrawInstanced(4, iNumInstances, 0, 0);
    }
    
    // Unbind resources
    UnbindResources( pd3dDeviceCtx );
}

void CLightSctrPostProcess :: DefineMacros(class CD3DShaderMacroHelper &Macros)
{
    Macros.AddShaderMacro("NUM_EPIPOLAR_SLICES", m_PostProcessingAttribs.m_uiNumEpipolarSlices);
    Macros.AddShaderMacro("MAX_SAMPLES_IN_SLICE", m_PostProcessingAttribs.m_uiMaxSamplesInSlice);
    Macros.AddShaderMacro("OPTIMIZE_SAMPLE_LOCATIONS", m_PostProcessingAttribs.m_bOptimizeSampleLocations);
    Macros.AddShaderMacro("USE_COMBINED_MIN_MAX_TEXTURE", m_bUseCombinedMinMaxTexture );
    Macros.AddShaderMacro("EXTINCTION_EVAL_MODE", m_PostProcessingAttribs.m_uiExtinctionEvalMode );
    Macros.AddShaderMacro("ENABLE_LIGHT_SHAFTS", m_PostProcessingAttribs.m_bEnableLightShafts);
    Macros.AddShaderMacro("MULTIPLE_SCATTERING_MODE", m_PostProcessingAttribs.m_uiMultipleScatteringMode);
    Macros.AddShaderMacro("SINGLE_SCATTERING_MODE", m_PostProcessingAttribs.m_uiSingleScatteringMode);
    Macros.AddShaderMacro("ENABLE_CLOUDS", m_PostProcessingAttribs.m_bEnableClouds);
    Macros.AddShaderMacro("SHAFTS_FROM_CLOUDS_MODE", m_PostProcessingAttribs.m_uiShaftsFromCloudsMode);

    {
        std::stringstream ss;
        ss<<"float2("<<m_uiBackBufferWidth<<","<<m_uiBackBufferHeight<<")";
        Macros.AddShaderMacro("SCREEN_RESLOUTION", ss.str());
    }

    {
        std::stringstream ss;
        ss<<"float4("<<sm_iPrecomputedSctrUDim<<","
                     <<sm_iPrecomputedSctrVDim<<","
                     <<sm_iPrecomputedSctrWDim<<","
                     <<sm_iPrecomputedSctrQDim<<")";
        Macros.AddShaderMacro("PRECOMPUTED_SCTR_LUT_DIM", ss.str());
    }

    Macros.AddShaderMacro("EARTH_RADIUS",   m_MediaParams.fEarthRadius);
    Macros.AddShaderMacro("ATM_TOP_HEIGHT", m_MediaParams.fAtmTopHeight);
    Macros.AddShaderMacro("ATM_TOP_RADIUS", m_MediaParams.fAtmTopRadius);
    
    {
        std::stringstream ss;
        ss<<"float2("<<m_MediaParams.f2ParticleScaleHeight.x<<","<<m_MediaParams.f2ParticleScaleHeight.y<<")";
        Macros.AddShaderMacro("PARTICLE_SCALE_HEIGHT", ss.str());
    }
}

HRESULT CLightSctrPostProcess::CreatePrecomputedOpticalDepthTexture(ID3D11Device* in_pd3dDevice, 
                                                                    ID3D11DeviceContext *in_pd3dDeviceContext)
{
    HRESULT hr;

    CRenderTechnique PrecomputeNetDensityToAtmTopTech;
    PrecomputeNetDensityToAtmTopTech.SetDeviceAndContext(in_pd3dDevice, in_pd3dDeviceContext);
    CD3DShaderMacroHelper Macros;
    Macros.Finalize();
    PrecomputeNetDensityToAtmTopTech.CreatePixelShaderFromFile( m_strEffectPath, "PrecomputeNetDensityToAtmTopPS", Macros );
    PrecomputeNetDensityToAtmTopTech.SetVS( m_pGenerateScreenSizeQuadVS );
    PrecomputeNetDensityToAtmTopTech.SetDS( m_pDisableDepthTestDS, 0 );
    PrecomputeNetDensityToAtmTopTech.SetRS( m_pSolidFillNoCullRS );
    PrecomputeNetDensityToAtmTopTech.SetBS( m_pDefaultBS );

    D3D11_TEXTURE2D_DESC NetDensityToAtmTopTexDesc = 
    {
        sm_iNumPrecomputedHeights,                              //UINT Width;
        sm_iNumPrecomputedAngles,                               //UINT Height;
        1,                                                      //UINT MipLevels;
        1,                                                      //UINT ArraySize;
        DXGI_FORMAT_R32G32_FLOAT,                               //DXGI_FORMAT Format;
        {1,0},                                                  //DXGI_SAMPLE_DESC SampleDesc;
        D3D11_USAGE_DEFAULT,                                    //D3D11_USAGE Usage;
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,  //UINT BindFlags;
        0,                                                      //UINT CPUAccessFlags;
        0,                                                      //UINT MiscFlags;
    };

    m_ptex2DOccludedNetDensityToAtmTopSRV.Release();
    m_ptex2DOccludedNetDensityToAtmTopRTV.Release();

    CComPtr<ID3D11Texture2D> ptex2DOccludedNetDensityToAtmTop;
    V_RETURN( in_pd3dDevice->CreateTexture2D( &NetDensityToAtmTopTexDesc, NULL, &ptex2DOccludedNetDensityToAtmTop) );
    V_RETURN( in_pd3dDevice->CreateShaderResourceView( ptex2DOccludedNetDensityToAtmTop, NULL, &m_ptex2DOccludedNetDensityToAtmTopSRV)  );
    V_RETURN( in_pd3dDevice->CreateRenderTargetView( ptex2DOccludedNetDensityToAtmTop, NULL, &m_ptex2DOccludedNetDensityToAtmTopRTV)  );

    ID3D11RenderTargetView *pRTVs[] = { m_ptex2DOccludedNetDensityToAtmTopRTV };
    in_pd3dDeviceContext->OMSetRenderTargets(_countof(pRTVs), pRTVs, NULL);

    ID3D11Buffer *pCBs[] = {m_pcbMediaAttribs};
    in_pd3dDeviceContext->PSSetConstantBuffers(1, _countof(pCBs), pCBs);

    RenderQuad(in_pd3dDeviceContext, PrecomputeNetDensityToAtmTopTech);
    
    in_pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);

    return S_OK;
}

void CLightSctrPostProcess :: CreateRandomSphereSamplingTexture(ID3D11Device *pDevice)
{
    D3D11_TEXTURE2D_DESC RandomSphereSamplingTexDesc = 
    {
        sm_iNumRandomSamplesOnSphere,   //UINT Width;
        1,                              //UINT Height;
        1,                              //UINT MipLevels;
        1,                              //UINT ArraySize;
        DXGI_FORMAT_R32G32B32A32_FLOAT, //DXGI_FORMAT Format;
        {1,0},                          //DXGI_SAMPLE_DESC SampleDesc;
        D3D11_USAGE_IMMUTABLE,          //D3D11_USAGE Usage;
        D3D11_BIND_SHADER_RESOURCE,     //UINT BindFlags;
        0,                              //UINT CPUAccessFlags;
        0,                              //UINT MiscFlags;
    };
    std::vector<D3DXVECTOR4> SphereSampling(sm_iNumRandomSamplesOnSphere);
    for(int iSample = 0; iSample < sm_iNumRandomSamplesOnSphere; ++iSample)
    {
        D3DXVECTOR4 &f4Sample = SphereSampling[iSample];
        f4Sample.z = ((float)rand()/(float)RAND_MAX) * 2.f - 1.f;
        float t = ((float)rand()/(float)RAND_MAX) * 2.f * PI;
        float r = sqrt( max(1 - f4Sample.z*f4Sample.z, 0.f) );
        f4Sample.x = r * cos(t);
        f4Sample.y = r * sin(t);
        f4Sample.w = 0;
    }
    D3D11_SUBRESOURCE_DATA InitData = 
    {
        &SphereSampling[0],
        sm_iNumRandomSamplesOnSphere*sizeof(D3DXVECTOR4), // UINT SysMemPitch
        0  // UINT SysMemSlicePitch
    };
    HRESULT hr;
    CComPtr<ID3D11Texture2D> ptex2DSphereRandomSampling;
    V( pDevice->CreateTexture2D( &RandomSphereSamplingTexDesc, &InitData, &ptex2DSphereRandomSampling) );
    V( pDevice->CreateShaderResourceView( ptex2DSphereRandomSampling, NULL, &m_ptex2DSphereRandomSamplingSRV) );
}

HRESULT CLightSctrPostProcess :: CreatePrecomputedScatteringLUT(ID3D11Device *pDevice, ID3D11DeviceContext *pContext)
{
    HRESULT hr;

    if( !m_PrecomputeSingleSctrTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();
        m_PrecomputeSingleSctrTech.SetDeviceAndContext(pDevice, pContext);
        m_PrecomputeSingleSctrTech.CreatePixelShaderFromFile( m_strEffectPath, "PrecomputeSingleScatteringPS", Macros );
        m_PrecomputeSingleSctrTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_PrecomputeSingleSctrTech.SetDS( m_pDisableDepthTestDS );
        m_PrecomputeSingleSctrTech.SetRS( m_pSolidFillNoCullRS );
        m_PrecomputeSingleSctrTech.SetBS( m_pDefaultBS );
    }
    
    if( !m_ComputeSctrRadianceTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro( "NUM_RANDOM_SPHERE_SAMPLES", sm_iNumRandomSamplesOnSphere );
        Macros.Finalize();
        m_ComputeSctrRadianceTech.SetDeviceAndContext(pDevice, pContext);
        m_ComputeSctrRadianceTech.CreatePixelShaderFromFile( m_strEffectPath, "ComputeSctrRadiancePS", Macros );
        m_ComputeSctrRadianceTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_ComputeSctrRadianceTech.SetDS( m_pDisableDepthTestDS );
        m_ComputeSctrRadianceTech.SetRS( m_pSolidFillNoCullRS );
        m_ComputeSctrRadianceTech.SetBS( m_pDefaultBS );
    }
    
    if( !m_ComputeScatteringOrderTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();
        m_ComputeScatteringOrderTech.SetDeviceAndContext(pDevice, pContext);
        m_ComputeScatteringOrderTech.CreatePixelShaderFromFile( m_strEffectPath, "ComputeScatteringOrderPS", Macros );
        m_ComputeScatteringOrderTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_ComputeScatteringOrderTech.SetDS( m_pDisableDepthTestDS );
        m_ComputeScatteringOrderTech.SetRS( m_pSolidFillNoCullRS );
        m_ComputeScatteringOrderTech.SetBS( m_pDefaultBS );
    }

    if( !m_AddScatteringOrderTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();
        m_AddScatteringOrderTech.SetDeviceAndContext(pDevice, pContext);
        m_AddScatteringOrderTech.CreatePixelShaderFromFile( m_strEffectPath, "AddScatteringOrderPS", Macros );
        m_AddScatteringOrderTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_AddScatteringOrderTech.SetDS( m_pDisableDepthTestDS );
        m_AddScatteringOrderTech.SetRS( m_pSolidFillNoCullRS );
        m_AddScatteringOrderTech.SetBS( m_pAdditiveBlendBS );
    }
        

    if( !m_ptex2DSphereRandomSamplingSRV )
        CreateRandomSphereSamplingTexture(pDevice);

    m_ptex3DSingleScatteringSRV.Release();
    m_ptex3DHighOrderScatteringSRV.Release();
    m_ptex3DMultipleScatteringSRV.Release();

    D3D11_TEXTURE3D_DESC PrecomputedSctrTexDesc = 
    {
        sm_iPrecomputedSctrUDim, //UINT Width;
        sm_iPrecomputedSctrVDim, //UINT Height;
        sm_iPrecomputedSctrWDim * sm_iPrecomputedSctrQDim, //UINT Depth;
        1, //UINT MipLevels;
        DXGI_FORMAT_R16G16B16A16_FLOAT,//DXGI_FORMAT Format;
        D3D11_USAGE_DEFAULT, //D3D11_USAGE Usage;
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,//UINT BindFlags;
        0,//UINT CPUAccessFlags;
        0//UINT MiscFlags;
    };

    CComPtr<ID3D11Texture3D> ptex3DSingleSctr, ptex3DHighOrderSctr, ptex3DMultipleSctr;
    V_RETURN(pDevice->CreateTexture3D(&PrecomputedSctrTexDesc, NULL, &ptex3DSingleSctr));
    V_RETURN(pDevice->CreateShaderResourceView(ptex3DSingleSctr, NULL, &m_ptex3DSingleScatteringSRV));
    V_RETURN(pDevice->CreateTexture3D(&PrecomputedSctrTexDesc, NULL, &ptex3DHighOrderSctr));
    V_RETURN(pDevice->CreateShaderResourceView(ptex3DHighOrderSctr, NULL, &m_ptex3DHighOrderScatteringSRV));
    V_RETURN(pDevice->CreateTexture3D(&PrecomputedSctrTexDesc, NULL, &ptex3DMultipleSctr));
    V_RETURN(pDevice->CreateShaderResourceView(ptex3DMultipleSctr, NULL, &m_ptex3DMultipleScatteringSRV));
    std::vector< CComPtr<ID3D11RenderTargetView> > ptex3DHighOrderSctrRTVs(PrecomputedSctrTexDesc.Depth);
    std::vector< CComPtr<ID3D11RenderTargetView> > ptex3DMultipleSctrRTVs(PrecomputedSctrTexDesc.Depth);

    // Precompute single scattering
    for(UINT uiDepthSlice = 0; uiDepthSlice < PrecomputedSctrTexDesc.Depth; ++uiDepthSlice)
    {
        D3D11_RENDER_TARGET_VIEW_DESC CurrSliceRTVDesc;
        CurrSliceRTVDesc.Format = PrecomputedSctrTexDesc.Format;
        CurrSliceRTVDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
        CurrSliceRTVDesc.Texture3D.MipSlice = 0;
        CurrSliceRTVDesc.Texture3D.FirstWSlice = uiDepthSlice;
        CurrSliceRTVDesc.Texture3D.WSize = 1;

        CComPtr<ID3D11RenderTargetView> ptex3DCurrDepthSliceRTV;
        V_RETURN(pDevice->CreateRenderTargetView(ptex3DSingleSctr, &CurrSliceRTVDesc, &ptex3DCurrDepthSliceRTV));
        
        V_RETURN(pDevice->CreateRenderTargetView(ptex3DHighOrderSctr, &CurrSliceRTVDesc, &ptex3DHighOrderSctrRTVs[uiDepthSlice]));
        float Zero[] = {0.f, 0.f, 0.f, 0.f};
        pContext->ClearRenderTargetView(ptex3DHighOrderSctrRTVs[uiDepthSlice].p, Zero);

        V_RETURN(pDevice->CreateRenderTargetView(ptex3DMultipleSctr, &CurrSliceRTVDesc, &ptex3DMultipleSctrRTVs[uiDepthSlice]));

        pContext->OMSetRenderTargets(1, &ptex3DCurrDepthSliceRTV.p, NULL);

        ID3D11ShaderResourceView *pSRVs[] = 
        {
            m_ptex2DOccludedNetDensityToAtmTopSRV  // Texture2D<float2> g_tex2DOccludedNetDensityToAtmTop : register( t5 );
        };
        pContext->PSSetShaderResources(5, _countof(pSRVs), pSRVs);

        // Set sun zenith and sun view angles
        SMiscDynamicParams MiscDynamicParams = {NULL};
        UINT uiW = uiDepthSlice % sm_iPrecomputedSctrWDim;
        UINT uiQ = uiDepthSlice / sm_iPrecomputedSctrWDim;
        MiscDynamicParams.f2WQ.x = ((float)uiW + 0.5f) / (float)sm_iPrecomputedSctrWDim;
        assert(0 < MiscDynamicParams.f2WQ.x && MiscDynamicParams.f2WQ.x < 1);
        MiscDynamicParams.f2WQ.y = ((float)uiQ + 0.5f) / (float)sm_iPrecomputedSctrQDim;
        assert(0 < MiscDynamicParams.f2WQ.y && MiscDynamicParams.f2WQ.y < 1);
        UpdateConstantBuffer(pContext, m_pcbMiscParams, &MiscDynamicParams, sizeof(MiscDynamicParams));
        //cbuffer cbMiscDynamicParams : register( b4 )
        pContext->PSSetConstantBuffers(4, 1, &m_pcbMiscParams.p);

        RenderQuad( pContext, 
                    m_PrecomputeSingleSctrTech,
                    PrecomputedSctrTexDesc.Width, PrecomputedSctrTexDesc.Height );
    }

    // Precompute multiple scattering
    // We need higher precision to store intermediate data
    PrecomputedSctrTexDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    CComPtr<ID3D11Texture3D> ptex3DSctrRadiance, ptex3DInsctrOrder;
    CComPtr<ID3D11ShaderResourceView> ptex3DSctrRadianceSRV, ptex3DInsctrOrderSRV;
    V_RETURN(pDevice->CreateTexture3D(&PrecomputedSctrTexDesc, NULL, &ptex3DSctrRadiance));
    V_RETURN(pDevice->CreateTexture3D(&PrecomputedSctrTexDesc, NULL, &ptex3DInsctrOrder));
    V_RETURN(pDevice->CreateShaderResourceView(ptex3DSctrRadiance, NULL, &ptex3DSctrRadianceSRV));
    V_RETURN(pDevice->CreateShaderResourceView(ptex3DInsctrOrder, NULL, &ptex3DInsctrOrderSRV));

    std::vector< CComPtr<ID3D11RenderTargetView> > ptex3DSctrRadianceRTVs(PrecomputedSctrTexDesc.Depth), ptex3DInsctrOrderRTVs(PrecomputedSctrTexDesc.Depth);
    for(UINT uiDepthSlice = 0; uiDepthSlice < PrecomputedSctrTexDesc.Depth; ++uiDepthSlice)
    {
        D3D11_RENDER_TARGET_VIEW_DESC CurrSliceRTVDesc;
        CurrSliceRTVDesc.Format = PrecomputedSctrTexDesc.Format;
        CurrSliceRTVDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
        CurrSliceRTVDesc.Texture3D.MipSlice = 0;
        CurrSliceRTVDesc.Texture3D.FirstWSlice = uiDepthSlice;
        CurrSliceRTVDesc.Texture3D.WSize = 1;

        V_RETURN(pDevice->CreateRenderTargetView(ptex3DSctrRadiance, &CurrSliceRTVDesc, &ptex3DSctrRadianceRTVs[uiDepthSlice]));
        V_RETURN(pDevice->CreateRenderTargetView(ptex3DInsctrOrder, &CurrSliceRTVDesc, &ptex3DInsctrOrderRTVs[uiDepthSlice]));
    }

    const int iNumScatteringOrders = 4;
    for(int iSctrOrder = 1; iSctrOrder < iNumScatteringOrders; ++iSctrOrder)
    {
        for(int iPass = 0; iPass < 3; ++iPass)
        {
            pContext->OMSetRenderTargets(0, NULL, NULL);
            // Pass 0: compute differential in-scattering
            // Pass 1: integrate differential in-scattering
            // Pass 2: accumulate total multiple scattering

            ID3D11ShaderResourceView *pSRVs[2] = {nullptr};
            
            CRenderTechnique *pRenderTech = nullptr;
            std::vector< CComPtr<ID3D11RenderTargetView> > *pRTVs;
            switch(iPass)
            {
                case 0:
                    // Pre-compute the radiance of light scattered at a given point in given direction.
                    pRenderTech = &m_ComputeSctrRadianceTech; 
                    pRTVs = &ptex3DSctrRadianceRTVs;
                    pSRVs[0] = (iSctrOrder == 1) ? m_ptex3DSingleScatteringSRV : ptex3DInsctrOrderSRV;
                    pSRVs[1] = m_ptex2DSphereRandomSamplingSRV;
                    break;

                case 1:
                    // Compute in-scattering order for a given point and direction
                    pRenderTech = &m_ComputeScatteringOrderTech; 
                    pRTVs = &ptex3DInsctrOrderRTVs;
                    pSRVs[0] = ptex3DSctrRadianceSRV;
                    break;

                case 2:
                    // Accumulate in-scattering
                    pRenderTech = &m_AddScatteringOrderTech; 
                    pRTVs = &ptex3DHighOrderSctrRTVs;
                    pSRVs[0] = ptex3DInsctrOrderSRV;
                    break;
            }
            
            for(UINT uiDepthSlice = 0; uiDepthSlice < PrecomputedSctrTexDesc.Depth; ++uiDepthSlice)
            {
                pContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

                // Set sun zenith and sun view angles
                SMiscDynamicParams MiscDynamicParams = {NULL};
                MiscDynamicParams.uiDepthSlice = uiDepthSlice;
                UINT uiW = uiDepthSlice % sm_iPrecomputedSctrWDim;
                UINT uiQ = uiDepthSlice / sm_iPrecomputedSctrWDim;
                MiscDynamicParams.f2WQ.x = ((float)uiW + 0.5f) / (float)sm_iPrecomputedSctrWDim;
                assert(0 < MiscDynamicParams.f2WQ.x && MiscDynamicParams.f2WQ.x < 1);
                MiscDynamicParams.f2WQ.y = ((float)uiQ + 0.5f) / (float)sm_iPrecomputedSctrQDim;
                assert(0 < MiscDynamicParams.f2WQ.y && MiscDynamicParams.f2WQ.y < 1);
                UpdateConstantBuffer(pContext, m_pcbMiscParams, &MiscDynamicParams, sizeof(MiscDynamicParams));
                //cbuffer cbMiscDynamicParams : register( b4 )
                pContext->PSSetConstantBuffers(4, 1, &m_pcbMiscParams.p);

                auto *pRTV = (ID3D11RenderTargetView*)(*pRTVs)[uiDepthSlice];
                pContext->OMSetRenderTargets(1, &pRTV, NULL);
                
                RenderQuad( pContext, 
                            *pRenderTech,
                            PrecomputedSctrTexDesc.Width, PrecomputedSctrTexDesc.Height );
            }
        }
    }
    pContext->OMSetRenderTargets(0, NULL, NULL);

    // Combine single scattering and higher order scattering into single texture
    pContext->CopyResource(ptex3DMultipleSctr, ptex3DSingleSctr);
    for(UINT uiDepthSlice = 0; uiDepthSlice < PrecomputedSctrTexDesc.Depth; ++uiDepthSlice)
    {
        ID3D11ShaderResourceView *pSRVs[1] = {m_ptex3DHighOrderScatteringSRV};
        pContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

        SMiscDynamicParams MiscDynamicParams = {NULL};
        MiscDynamicParams.uiDepthSlice = uiDepthSlice;
        UpdateConstantBuffer(pContext, m_pcbMiscParams, &MiscDynamicParams, sizeof(MiscDynamicParams));
        //cbuffer cbMiscDynamicParams : register( b4 )
        pContext->PSSetConstantBuffers(4, 1, &m_pcbMiscParams.p);

        auto *pRTV = (ID3D11RenderTargetView*)ptex3DMultipleSctrRTVs[uiDepthSlice];
        pContext->OMSetRenderTargets(1, &pRTV, NULL);
                
        RenderQuad( pContext, 
                    m_AddScatteringOrderTech,
                    PrecomputedSctrTexDesc.Width, PrecomputedSctrTexDesc.Height );
    }

    pContext->OMSetRenderTargets(0, NULL, NULL);

    return S_OK;
}

void CLightSctrPostProcess :: ReconstructCameraSpaceZ(SFrameAttribs &FrameAttribs)
{
    if( !m_ReconstrCamSpaceZTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();
        m_ReconstrCamSpaceZTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_ReconstrCamSpaceZTech.CreatePixelShaderFromFile( m_strEffectPath, "ReconstructCameraSpaceZPS", Macros );
        m_ReconstrCamSpaceZTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_ReconstrCamSpaceZTech.SetDS( m_pDisableDepthTestDS );
        m_ReconstrCamSpaceZTech.SetRS( m_pSolidFillNoCullRS );
        m_ReconstrCamSpaceZTech.SetBS( m_pDefaultBS );
    }

    // Depth buffer is non-linear and cannot be interpolated directly
    // We have to reconstruct camera space z to be able to use bilinear filtering
    
    // Set render target first, because depth buffer is still bound on output and it must be unbound 
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(1, &m_ptex2DCameraSpaceZRTV.p, NULL);
    // Texture2D<float> g_tex2DCamSpaceZ : register( t0 );
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(0, 1, &FrameAttribs.ptex2DSrcDepthBufferSRV);
    RenderQuad( FrameAttribs.pd3dDeviceContext, 
                m_ReconstrCamSpaceZTech,
                m_uiBackBufferWidth, m_uiBackBufferHeight );
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: RenderSliceEndpoints(SFrameAttribs &FrameAttribs)
{
    if( !m_RendedSliceEndpointsTech.IsValid() )
    {
        m_RendedSliceEndpointsTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();
        m_RendedSliceEndpointsTech.CreatePixelShaderFromFile( m_strEffectPath, "GenerateSliceEndpointsPS", Macros );
        m_RendedSliceEndpointsTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_RendedSliceEndpointsTech.SetDS( m_pDisableDepthTestDS );
        m_RendedSliceEndpointsTech.SetRS( m_pSolidFillNoCullRS );
        m_RendedSliceEndpointsTech.SetBS( m_pDefaultBS );
    }
    ID3D11RenderTargetView *ppRTVs[] = {m_ptex2DSliceEndpointsRTV};
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(_countof(ppRTVs), ppRTVs, NULL);

    RenderQuad( FrameAttribs.pd3dDeviceContext, m_RendedSliceEndpointsTech,
                m_PostProcessingAttribs.m_uiNumEpipolarSlices, 1 );
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: RenderCoordinateTexture(SFrameAttribs &FrameAttribs)
{
    if( !m_RendedCoordTexTech.IsValid() )
    {
        m_RendedCoordTexTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();
        m_RendedCoordTexTech.CreatePixelShaderFromFile( m_strEffectPath, "GenerateCoordinateTexturePS", Macros );
        m_RendedCoordTexTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_RendedCoordTexTech.SetDS( m_pDisableDepthTestIncrStencilDS );
        m_RendedCoordTexTech.SetRS( m_pSolidFillNoCullRS );
        m_RendedCoordTexTech.SetBS( m_pDefaultBS );
    }
    // Coordinate texture is a texture with dimensions [Total Samples X Num Slices]
    // Texel t[i,j] contains projection-space screen cooridantes of the i-th sample in j-th epipolar slice
    ID3D11RenderTargetView *ppRTVs[] = {m_ptex2DCoordinateTextureRTV, m_ptex2DEpipolarCamSpaceZRTV, m_ptex2DEpipolarCloudTranspRTV};
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(_countof(ppRTVs), ppRTVs, m_ptex2DEpipolarImageDSV);

    static const float fInvalidCoordinate = -1e+30f; // Both coord texture and epipolar CamSpaceZ are 32-bit float
    float InvalidCoords[] = {fInvalidCoordinate, fInvalidCoordinate, fInvalidCoordinate, fInvalidCoordinate};
    // Clear both render targets with values that can't be correct projection space coordinates and camera space Z:
    FrameAttribs.pd3dDeviceContext->ClearRenderTargetView(m_ptex2DCoordinateTextureRTV, InvalidCoords);
    FrameAttribs.pd3dDeviceContext->ClearRenderTargetView(m_ptex2DEpipolarCamSpaceZRTV, InvalidCoords);
    // Clear depth stencil view. Since we use stencil part only, there is no need to clear depth
    // Set stencil value to 0
    FrameAttribs.pd3dDeviceContext->ClearDepthStencilView(m_ptex2DEpipolarImageDSV, D3D11_CLEAR_STENCIL, 1.0f, 0);

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        m_ptex2DCameraSpaceZSRV,           // Texture2D<float>  g_tex2DCamSpaceZ              : register( t0 );
        NULL,                              // Unused                                          : register( t1 )
        NULL,                              // Unused                                          : register( t2 )
        NULL,                              // Unused                                          : register( t3 )
        m_ptex2DSliceEndpointsSRV,         // Texture2D<float4> g_tex2DSliceEndPoints         : register( t4 );
    };
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

    //Texture2D<float>  g_tex2DScrSpaceCloudTransparency : register( t11 );
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(11, 1, &FrameAttribs.ptex2DScrSpaceCloudTransparencySRV);

    // Depth stencil state is configured to always increment stencil value. If coordinates are outside the screen,
    // the pixel shader discards the pixel and stencil value is left untouched. All such pixels will be skipped from
    // further processing
    RenderQuad( FrameAttribs.pd3dDeviceContext, m_RendedCoordTexTech,
                m_PostProcessingAttribs.m_uiMaxSamplesInSlice, m_PostProcessingAttribs.m_uiNumEpipolarSlices );
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: RenderCoarseUnshadowedInsctr(SFrameAttribs &FrameAttribs)
{
    if( !m_RenderCoarseUnshadowedInsctrTech.IsValid() )
    {
        m_RenderCoarseUnshadowedInsctrTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();
       
        m_RenderCoarseUnshadowedInsctrTech.CreatePixelShaderFromFile( m_strEffectPath, "RenderCoarseUnshadowedInsctrPS", Macros );
        m_RenderCoarseUnshadowedInsctrTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_RenderCoarseUnshadowedInsctrTech.SetDS( m_pNoDepth_StEqual_KeepStencilDS, 1);
        m_RenderCoarseUnshadowedInsctrTech.SetRS( m_pSolidFillNoCullRS );
        m_RenderCoarseUnshadowedInsctrTech.SetBS( m_pDefaultBS );
    }

    if( m_PostProcessingAttribs.m_uiExtinctionEvalMode == EXTINCTION_EVAL_MODE_EPIPOLAR && 
        !m_ptex2DEpipolarExtinctionSRV )
    {
        D3D11_TEXTURE2D_DESC EpipolarExtinctionTexDesc = 
        {
            m_PostProcessingAttribs.m_uiMaxSamplesInSlice,          //UINT Width;
            m_PostProcessingAttribs.m_uiNumEpipolarSlices,          //UINT Height;
            1,                                                     //UINT MipLevels;
            1,                                                     //UINT ArraySize;
            DXGI_FORMAT_R8G8B8A8_UNORM,                              //DXGI_FORMAT Format;
            {1,0},                                                 //DXGI_SAMPLE_DESC SampleDesc;
            D3D11_USAGE_DEFAULT,                                   //D3D11_USAGE Usage;
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, //UINT BindFlags;
            0,                                                     //UINT CPUAccessFlags;
            0,                                                     //UINT MiscFlags;
        };

        CComPtr<ID3D11Texture2D> ptex2DEpipolarExtinction;
        // Create 2-D texture, shader resource and target view buffers on the device
        HRESULT hr;
        V( FrameAttribs.pd3dDevice->CreateTexture2D( &EpipolarExtinctionTexDesc, NULL, &ptex2DEpipolarExtinction) );
        V( FrameAttribs.pd3dDevice->CreateShaderResourceView( ptex2DEpipolarExtinction, NULL, &m_ptex2DEpipolarExtinctionSRV)  );
        V( FrameAttribs.pd3dDevice->CreateRenderTargetView( ptex2DEpipolarExtinction, NULL, &m_ptex2DEpipolarExtinctionRTV)  );
    }

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        m_ptex2DCoordinateTextureSRV,  //Texture2D<float2> g_tex2DCoordinates       : register( t1 );
        m_ptex2DEpipolarCamSpaceZSRV,  //Texture2D<float> g_tex2DEpipolarCamSpaceZ  : register( t2 );
        nullptr,                       // t3
        nullptr,                       // t4
        m_ptex2DOccludedNetDensityToAtmTopSRV,  // Texture2D<float2> g_tex2DOccludedNetDensityToAtmTop : register( t5 );
        nullptr,                             // t6 
        m_ptex3DSingleScatteringSRV,         // Texture3D<float3> g_tex3DSingleSctrLUT      : register( t7 );
        m_ptex3DHighOrderScatteringSRV,      // Texture3D<float3> g_tex3DHighOrderSctrLUT   : register( t8 );
        m_ptex3DMultipleScatteringSRV,       // Texture3D<float3> g_tex3DMultipleSctrLUT    : register( t9 );
        nullptr,                             // t10
        FrameAttribs.ptex2DScrSpaceCloudTransparencySRV,// Texture2D<float>  g_tex2DScrSpaceCloudTransparency : register( t11 );
        FrameAttribs.ptex2DScrSpaceCloudMinMaxDistSRV,  // Texture2D<float2> g_tex2DScrSpaceCloudMinMaxDist      : register( t12 );
        nullptr,                             // t13
        nullptr,                             // t14
        nullptr,                             // t15
        m_ptex2DEpipolarCloudTranspSRV       // Texture2D<float> g_tex2DEpipolarCloudTransparency : register( t16 );
    };
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(1, _countof(pSRVs), pSRVs);
    
    float flt16max = 65504.f; // Epipolar Inscattering is 16-bit float
    const float InvalidInsctr[] = {-flt16max, -flt16max, -flt16max, -flt16max};
    if( m_ptex2DEpipolarInscatteringRTV )
        FrameAttribs.pd3dDeviceContext->ClearRenderTargetView(m_ptex2DEpipolarInscatteringRTV, InvalidInsctr);
    const float One[] = {1, 1, 1, 1};
    if( m_ptex2DEpipolarExtinctionRTV )
        FrameAttribs.pd3dDeviceContext->ClearRenderTargetView(m_ptex2DEpipolarExtinctionRTV, One);
    ID3D11RenderTargetView *pRTVs[] = {m_ptex2DEpipolarInscatteringRTV, m_ptex2DEpipolarExtinctionRTV};
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(_countof(pRTVs),pRTVs, m_ptex2DEpipolarImageDSV);

    RenderQuad( FrameAttribs.pd3dDeviceContext, 
                m_RenderCoarseUnshadowedInsctrTech,
                m_PostProcessingAttribs.m_uiMaxSamplesInSlice, m_PostProcessingAttribs.m_uiNumEpipolarSlices );
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: RefineSampleLocations(SFrameAttribs &FrameAttribs)
{
    if( !m_RefineSampleLocationsTech.IsValid() )
    {
        m_RefineSampleLocationsTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        
        // Thread group size must be at least as large as initial sample step
        m_uiSampleRefinementCSThreadGroupSize = max( m_uiSampleRefinementCSMinimumThreadGroupSize, m_PostProcessingAttribs.m_uiInitialSampleStepInSlice );
        // Thread group size cannot be larger than the total number of samples in slice
        m_uiSampleRefinementCSThreadGroupSize = min( m_uiSampleRefinementCSThreadGroupSize, m_PostProcessingAttribs.m_uiMaxSamplesInSlice );

        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("INITIAL_SAMPLE_STEP", m_PostProcessingAttribs.m_uiInitialSampleStepInSlice);
        Macros.AddShaderMacro("THREAD_GROUP_SIZE"  , m_uiSampleRefinementCSThreadGroupSize );
        Macros.AddShaderMacro("REFINEMENT_CRITERION", m_PostProcessingAttribs.m_uiRefinementCriterion );
        Macros.AddShaderMacro("AUTO_EXPOSURE",        m_PostProcessingAttribs.m_bAutoExposure);
        Macros.Finalize();

        m_RefineSampleLocationsTech.CreateComputeShaderFromFile( L"fx\\RefineSampleLocations.fx", "RefineSampleLocationsCS", Macros );
    }

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        m_ptex2DCoordinateTextureSRV,  //Texture2D<float2> g_tex2DCoordinates       : register( t1 );
        m_ptex2DEpipolarCamSpaceZSRV,  //Texture2D<float> g_tex2DEpipolarCamSpaceZ  : register( t2 );
        m_ptex2DEpipolarInscatteringSRV,        // Texture2D<float3> g_tex2DScatteredColor   : register( t3 );
        nullptr,                                // t4
        nullptr,                                // t5
        nullptr,                                // t6
        nullptr,                                // t7
        nullptr,                                // t8
        nullptr,                                // t9
        m_ptex2DAverageLuminanceSRV,            // Texture2D<float>  g_tex2DAverageLuminance  : register( t10 );
        FrameAttribs.ptex2DScrSpaceCloudTransparencySRV,// Texture2D<float>  g_tex2DScrSpaceCloudTransparency : register( t11 );
        FrameAttribs.ptex2DScrSpaceCloudMinMaxDistSRV,  // Texture2D<float2> g_tex2DScrSpaceCloudMinMaxDist      : register( t12 );
        nullptr,                                // t13
        nullptr,                                // t14
        nullptr,                                // t15
        m_ptex2DEpipolarCloudTranspSRV          // Texture2D<float> g_tex2DEpipolarCloudTransparency : register( t16 );
    };
    FrameAttribs.pd3dDeviceContext->CSSetShaderResources(1, _countof(pSRVs), pSRVs);
    FrameAttribs.pd3dDeviceContext->CSSetUnorderedAccessViews(0, 1, &m_ptex2DInterpolationSourcesUAV.p, NULL);
    // Using small group size is inefficient since a lot of SIMD lanes become idle
    m_RefineSampleLocationsTech.Apply();
    FrameAttribs.pd3dDeviceContext->Dispatch( m_PostProcessingAttribs.m_uiMaxSamplesInSlice/m_uiSampleRefinementCSThreadGroupSize,
                                              m_PostProcessingAttribs.m_uiNumEpipolarSlices,
                                              1);
    UnbindResources( FrameAttribs.pd3dDeviceContext );
}

void CLightSctrPostProcess :: MarkRayMarchingSamples(SFrameAttribs &FrameAttribs)
{
    if( !m_MarkRayMarchingSamplesInStencilTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_MarkRayMarchingSamplesInStencilTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_MarkRayMarchingSamplesInStencilTech.CreatePixelShaderFromFile( m_strEffectPath, "MarkRayMarchingSamplesInStencilPS", Macros );
        m_MarkRayMarchingSamplesInStencilTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_MarkRayMarchingSamplesInStencilTech.SetDS( m_pNoDepth_StEqual_IncrStencilDS, 1 );
        m_MarkRayMarchingSamplesInStencilTech.SetRS( m_pSolidFillNoCullRS );
        m_MarkRayMarchingSamplesInStencilTech.SetBS( m_pDefaultBS );
    }

    // Mark ray marching samples in the stencil
    // The depth stencil state is configured to pass only pixels, whose stencil value equals 1. Thus all epipolar samples with 
    // coordinates outsied the screen (generated on the previous pass) are automatically discarded. The pixel shader only
    // passes samples which are interpolated from themselves, the rest are discarded. Thus after this pass all ray
    // marching samples will be marked with 2 in stencil
    ID3D11RenderTargetView *pDummyRTV = NULL;
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(1, &pDummyRTV, m_ptex2DEpipolarImageDSV);
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(6, 1, &m_ptex2DInterpolationSourcesSRV.p); // Texture2D<uint2> g_tex2DInterpolationSource : register( t6 );
    RenderQuad( FrameAttribs.pd3dDeviceContext, 
                m_MarkRayMarchingSamplesInStencilTech,
                m_PostProcessingAttribs.m_uiMaxSamplesInSlice, m_PostProcessingAttribs.m_uiNumEpipolarSlices );
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: RenderSliceUVDirAndOrig(SFrameAttribs &FrameAttribs)
{
    if( !m_RenderSliceUVDirInSMTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_RenderSliceUVDirInSMTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_RenderSliceUVDirInSMTech.CreatePixelShaderFromFile( m_strEffectPath, "RenderSliceUVDirInShadowMapTexturePS", Macros );
        m_RenderSliceUVDirInSMTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_RenderSliceUVDirInSMTech.SetDS( m_pDisableDepthTestDS );
        m_RenderSliceUVDirInSMTech.SetRS( m_pSolidFillNoCullRS );
        m_RenderSliceUVDirInSMTech.SetBS( m_pDefaultBS );
    }

    if( !m_ptex2DSliceUVDirAndOriginRTV || !m_ptex2DSliceUVDirAndOriginSRV )
    {
        D3D11_TEXTURE2D_DESC SliceUVDirAndOriginDesc = 
        {
            m_PostProcessingAttribs.m_uiNumEpipolarSlices,         //UINT Width;
            m_PostProcessingAttribs.m_iNumCascades,                //UINT Height;
            1,                                                     //UINT MipLevels;
            1,                                                     //UINT ArraySize;
            DXGI_FORMAT_R32G32B32A32_FLOAT,                        //DXGI_FORMAT Format;
            {1,0},                                                 //DXGI_SAMPLE_DESC SampleDesc;
            D3D11_USAGE_DEFAULT,                                   //D3D11_USAGE Usage;
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, //UINT BindFlags;
            0,                                                     //UINT CPUAccessFlags;
            0,                                                     //UINT MiscFlags;
        };

        CComPtr<ID3D11Texture2D> ptex2DSliceUVDirInShadowMap;
        // Create 2-D texture, shader resource and target view buffers on the device
        HRESULT hr;
        V( FrameAttribs.pd3dDevice->CreateTexture2D( &SliceUVDirAndOriginDesc, NULL, &ptex2DSliceUVDirInShadowMap) );
        V( FrameAttribs.pd3dDevice->CreateShaderResourceView( ptex2DSliceUVDirInShadowMap, NULL, &m_ptex2DSliceUVDirAndOriginSRV)  );
        V( FrameAttribs.pd3dDevice->CreateRenderTargetView( ptex2DSliceUVDirInShadowMap, NULL, &m_ptex2DSliceUVDirAndOriginRTV)  );
    }
    // Render [Num Slices x 1] texture containing slice direction in shadow map UV space
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets( 1, &m_ptex2DSliceUVDirAndOriginRTV.p, NULL);
    ID3D11ShaderResourceView *pSRVs[] = 
    {
        m_ptex2DCameraSpaceZSRV,            // Texture2D<float>  g_tex2DCamSpaceZ              : register( t0 );
        NULL,                               // Unused                                          : register( t1 )
        NULL,                               // Unused                                          : register( t2 )
        NULL,                               // Unused                                          : register( t3 )
        m_ptex2DSliceEndpointsSRV,          // Texture2D<float4> g_tex2DSliceEndPoints         : register( t4 );
    };
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

    RenderQuad( FrameAttribs.pd3dDeviceContext, 
                m_RenderSliceUVDirInSMTech,
                m_PostProcessingAttribs.m_uiNumEpipolarSlices, m_PostProcessingAttribs.m_iNumCascades - m_PostProcessingAttribs.m_iFirstCascade,
                0, m_PostProcessingAttribs.m_iFirstCascade);
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: Build1DMinMaxMipMap(SFrameAttribs &FrameAttribs, 
                                                  int iCascadeIndex)
{
    if( !m_InitializeMinMaxShadowMapTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("IS_32BIT_MIN_MAX_MAP", m_PostProcessingAttribs.m_bIs32BitMinMaxMipMap);
        Macros.Finalize();

        m_InitializeMinMaxShadowMapTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_InitializeMinMaxShadowMapTech.CreatePixelShaderFromFile( m_strEffectPath, "InitializeMinMaxShadowMapPS", Macros );
        m_InitializeMinMaxShadowMapTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_InitializeMinMaxShadowMapTech.SetDS( m_pDisableDepthTestDS );
        m_InitializeMinMaxShadowMapTech.SetRS( m_pSolidFillNoCullRS );
        m_InitializeMinMaxShadowMapTech.SetBS( m_pDefaultBS );
    }

    if( !m_ComputeMinMaxSMLevelTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_ComputeMinMaxSMLevelTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_ComputeMinMaxSMLevelTech.CreatePixelShaderFromFile( m_strEffectPath, "ComputeMinMaxShadowMapLevelPS", Macros );
        m_ComputeMinMaxSMLevelTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_ComputeMinMaxSMLevelTech.SetDS( m_pDisableDepthTestDS );
        m_ComputeMinMaxSMLevelTech.SetRS( m_pSolidFillNoCullRS );
        m_ComputeMinMaxSMLevelTech.SetBS( m_pDefaultBS );
    }
        
    int iMinMaxTexHeight = m_PostProcessingAttribs.m_uiNumEpipolarSlices;
    if( m_bUseCombinedMinMaxTexture )
        iMinMaxTexHeight *= (m_PostProcessingAttribs.m_iNumCascades - m_PostProcessingAttribs.m_iFirstCascade);

    // Computing min/max mip map using compute shader is much slower because a lot of threads are idle
    UINT uiXOffset = 0;
    UINT uiPrevXOffset = 0;
    UINT uiParity = 0;
    CComPtr<ID3D11Resource> presMinMaxShadowMap0, presMinMaxShadowMap1;
    m_ptex2DMinMaxShadowMapRTV[0]->GetResource(&presMinMaxShadowMap0);
    m_ptex2DMinMaxShadowMapRTV[1]->GetResource(&presMinMaxShadowMap1);
#ifdef _DEBUG
    {
        D3D11_TEXTURE2D_DESC MinMaxShadowMapTexDesc;
        CComQIPtr<ID3D11Texture2D>(presMinMaxShadowMap0)->GetDesc(&MinMaxShadowMapTexDesc);
        assert( MinMaxShadowMapTexDesc.Width == m_PostProcessingAttribs.m_uiMinMaxShadowMapResolution );
        assert( MinMaxShadowMapTexDesc.Height == iMinMaxTexHeight );
    }
#endif
    // Note that we start rendering min/max shadow map from step == 2
    for(UINT iStep = 2; iStep <= (UINT)m_PostProcessingAttribs.m_fMaxShadowMapStep; iStep *=2, uiParity = (uiParity+1)%2 )
    {
        // Use two buffers which are in turn used as the source and destination
        FrameAttribs.pd3dDeviceContext->OMSetRenderTargets( 1, &m_ptex2DMinMaxShadowMapRTV[uiParity].p, NULL);

        // Set source and destination min/max data offsets:
        SMiscDynamicParams MiscDynamicParams = {NULL};
        MiscDynamicParams.uiSrcMinMaxLevelXOffset = uiPrevXOffset;
        MiscDynamicParams.uiDstMinMaxLevelXOffset = uiXOffset;
        MiscDynamicParams.fCascadeInd = static_cast<float>(iCascadeIndex);
        UpdateConstantBuffer(FrameAttribs.pd3dDeviceContext, m_pcbMiscParams, &MiscDynamicParams, sizeof(MiscDynamicParams));
        //cbuffer cbMiscDynamicParams : register( b4 )
        FrameAttribs.pd3dDeviceContext->PSSetConstantBuffers(4, 1, &m_pcbMiscParams.p);

        if( iStep == 2 )
        {
            // At the initial pass, the shader gathers 8 depths which will be used for
            // PCF filtering at the sample location and its next neighbor along the slice 
            // and outputs min/max depths

            ID3D11ShaderResourceView *pSRVs[] = 
            {
                FrameAttribs.ptex2DShadowMapSRV, // Texture2D<float2> g_tex2DLightSpaceDepthMap    : register( t3 );
                nullptr,                         // t4
                nullptr,                         // t5
                m_ptex2DSliceUVDirAndOriginSRV   // Texture2D<float4> g_tex2DSliceUVDirAndOrigin   : register( t6 );
            };
            FrameAttribs.pd3dDeviceContext->PSSetShaderResources( 3, _countof(pSRVs), pSRVs );
        }
        else
        {
            // At the subsequent passes, the shader loads two min/max values from the next finer level 
            // to compute next level of the binary tree

            // Texture2D<float2> g_tex2DMinMaxLightSpaceDepth  : register( t4 );
            FrameAttribs.pd3dDeviceContext->PSSetShaderResources( 4, 1, &m_ptex2DMinMaxShadowMapSRV[ (uiParity+1)%2 ].p );
        }
            

        RenderQuad( FrameAttribs.pd3dDeviceContext, 
                    (iStep>2) ? m_ComputeMinMaxSMLevelTech : m_InitializeMinMaxShadowMapTech,
                    m_PostProcessingAttribs.m_uiMinMaxShadowMapResolution / iStep, iMinMaxTexHeight, 
                    uiXOffset, 0 );

        // All the data must reside in 0-th texture, so copy current level, if necessary, from 1-st texture
        if( uiParity == 1 )
        {
            D3D11_BOX SrcBox;
            SrcBox.left = uiXOffset;
            SrcBox.right = uiXOffset + m_PostProcessingAttribs.m_uiMinMaxShadowMapResolution / iStep;
            SrcBox.top = 0;
            SrcBox.bottom = iMinMaxTexHeight;
            SrcBox.front = 0;
            SrcBox.back = 1;
            FrameAttribs.pd3dDeviceContext->CopySubresourceRegion(presMinMaxShadowMap0, 0, uiXOffset, 0, 0,
                                                                    presMinMaxShadowMap1, 0, &SrcBox);
        }

        uiPrevXOffset = uiXOffset;
        uiXOffset += m_PostProcessingAttribs.m_uiMinMaxShadowMapResolution / iStep;
    }
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: RenderCldDensEpipolarScan(SFrameAttribs &FrameAttribs, int iCascadeIndex)
{
    if( !m_InitializeCldDensEpipolarScanTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_InitializeCldDensEpipolarScanTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_InitializeCldDensEpipolarScanTech.CreatePixelShaderFromFile( m_strEffectPath, "InitCldDensEpipolarScanPS", Macros );
        m_InitializeCldDensEpipolarScanTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_InitializeCldDensEpipolarScanTech.SetDS( m_pDisableDepthTestDS );
        m_InitializeCldDensEpipolarScanTech.SetRS( m_pSolidFillNoCullRS );
        m_InitializeCldDensEpipolarScanTech.SetBS( m_pDefaultBS );
    }

    if( !m_ComputeCldDensEpiScanLevelTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();

        m_ComputeCldDensEpiScanLevelTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_ComputeCldDensEpiScanLevelTech.CreatePixelShaderFromFile( m_strEffectPath, "ComputeCldDensEpipolarScanLevelPS", Macros );
        m_ComputeCldDensEpiScanLevelTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_ComputeCldDensEpiScanLevelTech.SetDS( m_pDisableDepthTestDS );
        m_ComputeCldDensEpiScanLevelTech.SetRS( m_pSolidFillNoCullRS );
        m_ComputeCldDensEpiScanLevelTech.SetBS( m_pDefaultBS );
    }
        
    int iTexHeight = m_PostProcessingAttribs.m_uiNumEpipolarSlices;
    if( m_bUseCombinedMinMaxTexture )
        iTexHeight *= (m_PostProcessingAttribs.m_iNumCascades - m_PostProcessingAttribs.m_iFirstCascade);

    // Computing min/max mip map using compute shader is much slower because a lot of threads are idle
    UINT uiXOffset = 0;
    UINT uiPrevXOffset = 0;
    UINT uiParity = 0;
    CComPtr<ID3D11Resource> presCldDensEpipolarScan0, presCldDensEpipolarScan1;
    m_ptex2DCldDensEpipolarScanRTV[0]->GetResource(&presCldDensEpipolarScan0);
    m_ptex2DCldDensEpipolarScanRTV[1]->GetResource(&presCldDensEpipolarScan1);
#ifdef _DEBUG
    {
        D3D11_TEXTURE2D_DESC TexDesc;
        CComQIPtr<ID3D11Texture2D>(presCldDensEpipolarScan0)->GetDesc(&TexDesc);
        assert( TexDesc.Width == m_PostProcessingAttribs.m_uiMinMaxShadowMapResolution );
        assert( TexDesc.Height == iTexHeight );
    }
#endif

    UINT uiMaxStep = (UINT)m_PostProcessingAttribs.m_fMaxShadowMapStep;
    // Note that we start rendering min/max shadow map from step == 2
    for(UINT iStep = 2; iStep <= uiMaxStep; iStep *=2, uiParity = (uiParity+1)%2 )
    {
        // Use two buffers which are in turn used as the source and destination
        FrameAttribs.pd3dDeviceContext->OMSetRenderTargets( 1, &m_ptex2DCldDensEpipolarScanRTV[uiParity].p, NULL);

        // Set source and destination min/max data offsets:
        SMiscDynamicParams MiscDynamicParams = {NULL};
        MiscDynamicParams.uiSrcMinMaxLevelXOffset = uiPrevXOffset;
        MiscDynamicParams.uiDstMinMaxLevelXOffset = uiXOffset;
        MiscDynamicParams.fCascadeInd = static_cast<float>(iCascadeIndex);
        UpdateConstantBuffer(FrameAttribs.pd3dDeviceContext, m_pcbMiscParams, &MiscDynamicParams, sizeof(MiscDynamicParams));
        //cbuffer cbMiscDynamicParams : register( b4 )
        FrameAttribs.pd3dDeviceContext->PSSetConstantBuffers(4, 1, &m_pcbMiscParams.p);

        if( iStep == 2 )
        {
            // At the initial pass, the shader gathers 8 depths which will be used for
            // PCF filtering at the sample location and its next neighbor along the slice 
            // and outputs min/max depths

            // Texture2D<float4> g_tex2DSliceUVDirAndOrigin   : register( t6 );
            FrameAttribs.pd3dDeviceContext->PSSetShaderResources( 6, 1, &m_ptex2DSliceUVDirAndOriginSRV.p );
            // Texture2DArray<float> g_tex2DLiSpaceCloudTransparency : register( t14 );
            FrameAttribs.pd3dDeviceContext->PSSetShaderResources( 14, 1, &FrameAttribs.ptex2DLiSpCloudTransparencySRV );
        }
        else
        {
            // At the subsequent passes, the shader loads two min/max values from the next finer level 
            // to compute next level of the binary tree

            //Texture2D<float> g_tex2DLiSpCldDensityEpipolarScan  : register( t15 );
            FrameAttribs.pd3dDeviceContext->PSSetShaderResources( 15, 1, &m_ptex2DCldDensEpipolarScanSRV[ (uiParity+1)%2 ].p );
        }

        UINT uiCurrLevelWidth = m_PostProcessingAttribs.m_uiMinMaxShadowMapResolution / iStep;
        RenderQuad( FrameAttribs.pd3dDeviceContext, 
                    (iStep>2) ? m_ComputeCldDensEpiScanLevelTech : m_InitializeCldDensEpipolarScanTech,
                    uiCurrLevelWidth, iTexHeight, 
                    uiXOffset, 0 );

        // All the data must reside in 0-th texture, so copy current level, if necessary, from 1-st texture
        if( uiParity == 1 )
        {
            D3D11_BOX SrcBox;
            SrcBox.left = uiXOffset;
            SrcBox.right = uiXOffset + uiCurrLevelWidth;
            SrcBox.top = 0;
            SrcBox.bottom = iTexHeight;
            SrcBox.front = 0;
            SrcBox.back = 1;
            FrameAttribs.pd3dDeviceContext->CopySubresourceRegion(presCldDensEpipolarScan0, 0, uiXOffset, 0, 0,
                                                                  presCldDensEpipolarScan1, 0, &SrcBox);
        }

        uiPrevXOffset = uiXOffset;
        uiXOffset += uiCurrLevelWidth;
    }

    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: DoRayMarching(SFrameAttribs &FrameAttribs, 
                                            UINT uiMaxStepsAlongRay, 
                                            const SShadowMapAttribs &SMAttribs, 
                                            int iCascadeIndex)
{
    CRenderTechnique &DoRayMarchTech = m_DoRayMarchTech[m_PostProcessingAttribs.m_bUse1DMinMaxTree ? 1 : 0];
    if( !DoRayMarchTech.IsValid()  )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("CASCADE_PROCESSING_MODE", m_PostProcessingAttribs.m_uiCascadeProcessingMode);
        Macros.Finalize();

        DoRayMarchTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        DoRayMarchTech.CreatePixelShaderFromFile( m_strEffectPath, m_PostProcessingAttribs.m_bUse1DMinMaxTree ? "RayMarchMinMaxOptPS" : "RayMarchPS", Macros );
        DoRayMarchTech.SetVS( m_pGenerateScreenSizeQuadVS );
        // Sample locations for which ray marching should be performed will be marked in stencil with 2
        DoRayMarchTech.SetDS( m_pNoDepth_StEqual_KeepStencilDS, 2 );
        DoRayMarchTech.SetRS( m_pSolidFillNoCullRS );
        DoRayMarchTech.SetBS( m_pAdditiveBlendBS );
    }

    //float BlackColor[] = {0,0,0,0};
    // NOTE: this is for debug purposes only:
    //FrameAttribs.pd3dDeviceContext->ClearRenderTargetView(m_ptex2DInitialScatteredLightRTV, BlackColor);
    ID3D11RenderTargetView *pRTVs[] = {m_ptex2DInitialScatteredLightRTV};
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(_countof(pRTVs), pRTVs, m_ptex2DEpipolarImageDSV);

    SMiscDynamicParams MiscDynamicParams = {NULL};
    MiscDynamicParams.fMaxStepsAlongRay = static_cast<float>( uiMaxStepsAlongRay );
    MiscDynamicParams.fCascadeInd = static_cast<float>(iCascadeIndex);
    UpdateConstantBuffer(FrameAttribs.pd3dDeviceContext, m_pcbMiscParams, &MiscDynamicParams, sizeof(MiscDynamicParams));
    //cbuffer cbMiscDynamicParams : register( b4 )
    FrameAttribs.pd3dDeviceContext->PSSetConstantBuffers(4, 1, &m_pcbMiscParams.p);

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        m_ptex2DCameraSpaceZSRV,            // Texture2D<float>  g_tex2DCamSpaceZ              : register( t0 );
        m_ptex2DCoordinateTextureSRV,       // Texture2D<float2> g_tex2DCoordinates            : register( t1 );
        m_ptex2DEpipolarCamSpaceZSRV,       //Texture2D<float> g_tex2DEpipolarCamSpaceZ        : register( t2 );
        FrameAttribs.ptex2DShadowMapSRV,    // Texture2D<float>  g_tex2DLightSpaceDepthMap     : register( t3 );
        m_ptex2DMinMaxShadowMapSRV[0],      // Texture2D<float2> g_tex2DMinMaxLightSpaceDepth  : register( t4 );
        m_ptex2DOccludedNetDensityToAtmTopSRV, // Texture2D<float2> g_tex2DOccludedNetDensityToAtmTop : register( t5 );
        m_ptex2DSliceUVDirAndOriginSRV,      // Texture2D<float4> g_tex2DSliceUVDirAndOrigin    : register( t6 );
        m_ptex3DSingleScatteringSRV,         // Texture3D<float3> g_tex3DSingleSctrLUT          : register( t7 );
        m_ptex3DHighOrderScatteringSRV,      // Texture3D<float3> g_tex3DHighOrderSctrLUT       : register( t8 );
        m_ptex3DMultipleScatteringSRV,       // Texture3D<float3> g_tex3DMultipleSctrLUT        : register( t9 );
        nullptr,                             // t10
        FrameAttribs.ptex2DScrSpaceCloudTransparencySRV,// Texture2D<float>  g_tex2DScrSpaceCloudTransparency : register( t11 );
        FrameAttribs.ptex2DScrSpaceCloudMinMaxDistSRV,  // Texture2D<float2> g_tex2DScrSpaceCloudMinMaxDist   : register( t12 );
        nullptr,
        FrameAttribs.ptex2DLiSpCloudTransparencySRV,     // Texture2DArray<float> g_tex2DLiSpaceCloudTransparency : register( t14 );
        m_ptex2DCldDensEpipolarScanSRV[0],               // Texture2D<float> g_tex2DLiSpCldDensityEpipolarScan    : register( t15 );
        m_ptex2DEpipolarCloudTranspSRV                   // Texture2D<float> g_tex2DEpipolarCloudTransparency     : register( t16 );
    };
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);
    
    int iNumInst = 0;
    if( m_PostProcessingAttribs.m_bEnableLightShafts )
    {
        switch(m_PostProcessingAttribs.m_uiCascadeProcessingMode)
        {
            case CASCADE_PROCESSING_MODE_SINGLE_PASS:
            case CASCADE_PROCESSING_MODE_MULTI_PASS: 
                iNumInst = 1; 
                break;
            case CASCADE_PROCESSING_MODE_MULTI_PASS_INST: 
                iNumInst = m_PostProcessingAttribs.m_iNumCascades - m_PostProcessingAttribs.m_iFirstCascade; 
                break;
        }
    }
    else
    {
        iNumInst = 1;
    }

    // Depth stencil view now contains 2 for these pixels, for which ray marchings is to be performed
    // Depth stencil state is configured to pass only these pixels and discard the rest
    RenderQuad( FrameAttribs.pd3dDeviceContext,
                DoRayMarchTech,
                m_PostProcessingAttribs.m_uiMaxSamplesInSlice, m_PostProcessingAttribs.m_uiNumEpipolarSlices,
                0,0,
                iNumInst);
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: InterpolateInsctrIrradiance(SFrameAttribs &FrameAttribs)
{
    if( !m_InterpolateIrradianceTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();
               
        m_InterpolateIrradianceTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_InterpolateIrradianceTech.CreatePixelShaderFromFile( m_strEffectPath, "InterpolateIrradiancePS", Macros );
        m_InterpolateIrradianceTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_InterpolateIrradianceTech.SetDS( m_pDisableDepthTestDS );
        m_InterpolateIrradianceTech.SetRS( m_pSolidFillNoCullRS );
        m_InterpolateIrradianceTech.SetBS( m_pDefaultBS );
    }
    ID3D11RenderTargetView *pRTVs[] = {m_ptex2DEpipolarInscatteringRTV};
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(_countof(pRTVs), pRTVs, m_ptex2DEpipolarImageDSV);
    
    ID3D11ShaderResourceView *pSRVs[] = 
    {
        m_ptex2DInitialScatteredLightSRV,   // Texture2D<uint2>  g_tex2DInitialInsctrIrradiance: register( t5 );
        m_ptex2DInterpolationSourcesSRV     // Texture2D<float3> g_tex2DInterpolationSource    : register( t6 );
    };
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(5, _countof(pSRVs), pSRVs);
    RenderQuad( FrameAttribs.pd3dDeviceContext,
                m_InterpolateIrradianceTech,
                m_PostProcessingAttribs.m_uiMaxSamplesInSlice, m_PostProcessingAttribs.m_uiNumEpipolarSlices );
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: CreateLowResLuminanceTexture(ID3D11Device *pDevice)
{
    D3D11_TEXTURE2D_DESC LowResLuminanceTexDesc = 
    {
        1 << (sm_iLowResLuminanceMips-1),   //UINT Width;
        1 << (sm_iLowResLuminanceMips-1),   //UINT Height;
        sm_iLowResLuminanceMips,            //UINT MipLevels;
        1,                                  //UINT ArraySize;
        DXGI_FORMAT_R16_FLOAT,              //DXGI_FORMAT Format;
        {1,0},                              //DXGI_SAMPLE_DESC SampleDesc;
        D3D11_USAGE_DEFAULT,                //D3D11_USAGE Usage;
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,           //UINT BindFlags;
        0,                                  //UINT CPUAccessFlags;
        D3D11_RESOURCE_MISC_GENERATE_MIPS                                  //UINT MiscFlags;
    };

    CComPtr<ID3D11Texture2D> ptex2DLowResLuminance;
    // Create 2-D texture, shader resource and target view buffers on the device
    HRESULT hr;
    V( pDevice->CreateTexture2D( &LowResLuminanceTexDesc, NULL, &ptex2DLowResLuminance) );
    V( pDevice->CreateShaderResourceView( ptex2DLowResLuminance, NULL, &m_ptex2DLowResLuminanceSRV) );
    V( pDevice->CreateRenderTargetView( ptex2DLowResLuminance, NULL, &m_ptex2DLowResLuminanceRTV) );

    // Create 2-D texture, shader resource and target view buffers on the device
    LowResLuminanceTexDesc.Width = 1;
    LowResLuminanceTexDesc.Height = 1;
    LowResLuminanceTexDesc.MipLevels = 1;
    LowResLuminanceTexDesc.MiscFlags = 0;
    int Zero = 0;
    D3D11_SUBRESOURCE_DATA InitData = {&Zero, 2, 0};
    CComPtr<ID3D11Texture2D> ptex2DAverageLuminance;
    V( pDevice->CreateTexture2D( &LowResLuminanceTexDesc, &InitData, &ptex2DAverageLuminance) );
    V( pDevice->CreateShaderResourceView( ptex2DAverageLuminance, NULL, &m_ptex2DAverageLuminanceSRV) );
    V( pDevice->CreateRenderTargetView( ptex2DAverageLuminance, NULL, &m_ptex2DAverageLuminanceRTV) );
}

void CLightSctrPostProcess :: UnwarpEpipolarScattering(SFrameAttribs &FrameAttribs, bool bRenderLuminance)
{
    if( !m_UnwarpEpipolarSctrImgTech.IsValid()  )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("PERFORM_TONE_MAPPING", true);
        Macros.AddShaderMacro("AUTO_EXPOSURE", m_PostProcessingAttribs.m_bAutoExposure);
        Macros.AddShaderMacro("TONE_MAPPING_MODE", m_PostProcessingAttribs.m_uiToneMappingMode);
        Macros.AddShaderMacro("CORRECT_INSCATTERING_AT_DEPTH_BREAKS", m_PostProcessingAttribs.m_bCorrectScatteringAtDepthBreaks);
        Macros.Finalize();
        
        m_UnwarpEpipolarSctrImgTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_UnwarpEpipolarSctrImgTech.CreatePixelShaderFromFile( m_strEffectPath, "ApplyInscatteredRadiancePS", Macros );
        m_UnwarpEpipolarSctrImgTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_UnwarpEpipolarSctrImgTech.SetDS( m_pDisableDepthTestIncrStencilDS );
        m_UnwarpEpipolarSctrImgTech.SetRS( m_pSolidFillNoCullRS );
        m_UnwarpEpipolarSctrImgTech.SetBS( m_pDefaultBS );
    }

    if( !m_UnwarpAndRenderLuminanceTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("PERFORM_TONE_MAPPING", false);
        Macros.AddShaderMacro("CORRECT_INSCATTERING_AT_DEPTH_BREAKS", false);
        Macros.Finalize();
        
        m_UnwarpAndRenderLuminanceTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_UnwarpAndRenderLuminanceTech.CreatePixelShaderFromFile( m_strEffectPath, "ApplyInscatteredRadiancePS", Macros );
        m_UnwarpAndRenderLuminanceTech.SetVS( m_pGenerateScreenSizeQuadVS );
        // Do not use stencil
        m_UnwarpAndRenderLuminanceTech.SetDS( m_pDisableDepthTestDS );
        m_UnwarpAndRenderLuminanceTech.SetRS( m_pSolidFillNoCullRS );
        m_UnwarpAndRenderLuminanceTech.SetBS( m_pDefaultBS );
    }

    CRenderTechnique &Tech = bRenderLuminance ? m_UnwarpAndRenderLuminanceTech : m_UnwarpEpipolarSctrImgTech;

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        m_ptex2DCameraSpaceZSRV,                // Texture2D<float>  g_tex2DCamSpaceZ              : register( t0 );
        FrameAttribs.ptex2DSrcColorBufferSRV,   // Texture2D<float4> g_tex2DColorBuffer            : register( t1 );
        m_ptex2DEpipolarCamSpaceZSRV,           // Texture2D<float>  g_tex2DEpipolarCamSpaceZ      : register( t2 );
        m_ptex2DEpipolarInscatteringSRV,        // Texture2D<float3> g_tex2DScatteredColor         : register( t3 );
        m_ptex2DSliceEndpointsSRV,              // Texture2D<float4> g_tex2DSliceEndPoints         : register( t4 );
        m_ptex2DOccludedNetDensityToAtmTopSRV,  // Texture2D<float2> g_tex2DOccludedNetDensityToAtmTop : register( t5 );
        m_ptex2DEpipolarExtinctionSRV,          // Texture2D<float3> g_tex2DEpipolarExtinction     : register( t6 );
        nullptr,                                // t7
        nullptr,                                // t8
        nullptr,                                // t9
        m_ptex2DAverageLuminanceSRV,            // Texture2D<float>  g_tex2DAverageLuminance       : register( t10 );
        FrameAttribs.ptex2DScrSpaceCloudTransparencySRV, // Texture2D<float>  g_tex2DScrSpaceCloudTransparency : register( t11 );
        FrameAttribs.ptex2DScrSpaceCloudMinMaxDistSRV,   // Texture2D<float2> g_tex2DScrSpaceCloudMinMaxDist      : register( t12 );
        FrameAttribs.ptex2DScrSpaceCloudColorSRV,        // Texture2D<float4> g_tex2DScrSpaceCloudColor        : register( t13 );
        nullptr,                                         // t14
        nullptr,                                         // t15
        m_ptex2DEpipolarCloudTranspSRV                   // Texture2D<float> g_tex2DEpipolarCloudTransparency : register( t16 );
    };
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

    // Unwarp inscattering image and apply it to attenuated backgorund
    RenderQuad( FrameAttribs.pd3dDeviceContext, Tech );
}

void CLightSctrPostProcess :: UpdateAverageLuminance(SFrameAttribs &FrameAttribs)
{
    if( !m_UpdateAverageLuminanceTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro( "LIGHT_ADAPTATION", m_PostProcessingAttribs.m_bLightAdaptation );
        Macros.AddShaderMacro("LOW_RES_LUMINANCE_MIPS", sm_iLowResLuminanceMips);
        Macros.Finalize();
        
        m_UpdateAverageLuminanceTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_UpdateAverageLuminanceTech.CreatePixelShaderFromFile( m_strEffectPath, "UpdateAverageLuminancePS", Macros );
        m_UpdateAverageLuminanceTech.SetVS( m_pGenerateScreenSizeQuadVS );
        m_UpdateAverageLuminanceTech.SetDS( m_pDisableDepthTestDS );
        m_UpdateAverageLuminanceTech.SetRS( m_pSolidFillNoCullRS );
        m_UpdateAverageLuminanceTech.SetBS( m_pAlphaBlendBS );
    }

    ID3D11RenderTargetView *pRTVs[] = {m_ptex2DAverageLuminanceRTV};
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(_countof(pRTVs), pRTVs, nullptr);

    ID3D11ShaderResourceView *pSRVs[] = {m_ptex2DLowResLuminanceSRV};
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);

    SMiscDynamicParams MiscDynamicParams = {NULL};
    MiscDynamicParams.fElapsedTime = (float)FrameAttribs.dElapsedTime;
    UpdateConstantBuffer(FrameAttribs.pd3dDeviceContext, m_pcbMiscParams, &MiscDynamicParams, sizeof(MiscDynamicParams));
    //cbuffer cbMiscDynamicParams : register( b4 )
    FrameAttribs.pd3dDeviceContext->PSSetConstantBuffers(4, 1, &m_pcbMiscParams.p);

    // Update average luminance
    RenderQuad( FrameAttribs.pd3dDeviceContext, m_UpdateAverageLuminanceTech, 1, 1 );
}

void CLightSctrPostProcess :: FixInscatteringAtDepthBreaks(SFrameAttribs &FrameAttribs, 
                                                           UINT uiMaxStepsAlongRay, 
                                                           const SShadowMapAttribs &SMAttribs,
                                                           bool bRenderLuminance)
{
    bool bApplBG = true;
    auto &FixInsctrAtDepthBreaksTech = m_FixInsctrAtDepthBreaksTech[(bApplBG ? 1 : 0) + (bRenderLuminance ? 2 : 0)];
    if( !FixInsctrAtDepthBreaksTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.AddShaderMacro("CASCADE_PROCESSING_MODE", CASCADE_PROCESSING_MODE_SINGLE_PASS);
        Macros.AddShaderMacro("PERFORM_TONE_MAPPING", !bRenderLuminance);
        Macros.AddShaderMacro("AUTO_EXPOSURE", m_PostProcessingAttribs.m_bAutoExposure);
        Macros.AddShaderMacro("TONE_MAPPING_MODE", m_PostProcessingAttribs.m_uiToneMappingMode);
        Macros.Finalize();
        FixInsctrAtDepthBreaksTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        FixInsctrAtDepthBreaksTech.CreatePixelShaderFromFile( m_strEffectPath, bApplBG ? "FixAndApplyInscatteredRadiancePS" : "FixInscatteredRadiancePS", Macros );
        FixInsctrAtDepthBreaksTech.SetVS( m_pGenerateScreenSizeQuadVS );
        // If rendering luminance only, disable depth and stencil tests to render all pixels
        FixInsctrAtDepthBreaksTech.SetDS( bRenderLuminance ? m_pDisableDepthTestDS : m_pNoDepth_StEqual_KeepStencilDS, 0 );
        FixInsctrAtDepthBreaksTech.SetRS( m_pSolidFillNoCullRS );
        // If rendering luminance only, use default blend state to overwrite old luminance values
        FixInsctrAtDepthBreaksTech.SetBS( bRenderLuminance ? m_pDefaultBS : m_pAdditiveBlendBS );
    }

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        m_ptex2DCameraSpaceZSRV,                // Texture2D<float>  g_tex2DCamSpaceZ              : register( t0 );
        FrameAttribs.ptex2DSrcColorBufferSRV,   // Texture2D<float4> g_tex2DColorBuffer            : register( t1 );
        m_ptex2DEpipolarCamSpaceZSRV,           // Texture2D<float> g_tex2DEpipolarCamSpaceZ       : register( t2 );
        FrameAttribs.ptex2DShadowMapSRV,        // Texture2D<float>  g_tex2DLightSpaceDepthMap     : register( t3 );
        m_ptex2DMinMaxShadowMapSRV[0],          // Texture2D<float2> g_tex2DMinMaxLightSpaceDepth  : register( t4 );
        m_ptex2DOccludedNetDensityToAtmTopSRV,  // Texture2D<float2> g_tex2DOccludedNetDensityToAtmTop : register( t5 );
        m_ptex2DSliceUVDirAndOriginSRV,         // Texture2D<float4> g_tex2DSliceUVDirAndOrigin    : register( t6 );
        m_ptex3DSingleScatteringSRV,            // Texture3D<float3> g_tex3DSingleSctrLUT          : register( t7 );
        m_ptex3DHighOrderScatteringSRV,         // Texture3D<float3> g_tex3DHighOrderSctrLUT       : register( t8 );
        m_ptex3DMultipleScatteringSRV,          // Texture3D<float3> g_tex3DMultipleSctrLUT        : register( t9  );
        m_ptex2DAverageLuminanceSRV,            // Texture2D<float>  g_tex2DAverageLuminance       : register( t10 );
        FrameAttribs.ptex2DScrSpaceCloudTransparencySRV,// Texture2D<float>  g_tex2DScrSpaceCloudTransparency : register( t11 );
        FrameAttribs.ptex2DScrSpaceCloudMinMaxDistSRV,  // Texture2D<float2> g_tex2DScrSpaceCloudMinMaxDist   : register( t12 );
        FrameAttribs.ptex2DScrSpaceCloudColorSRV,       // Texture2D<float4> g_tex2DScrSpaceCloudColor        : register( t13 );
        FrameAttribs.ptex2DLiSpCloudTransparencySRV     // Texture2DArray<float> g_tex2DLiSpaceCloudTransparency : register( t14 );
    };
    FrameAttribs.pd3dDeviceContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);
    
    SMiscDynamicParams MiscDynamicParams = {NULL};
    MiscDynamicParams.fMaxStepsAlongRay = static_cast<float>( uiMaxStepsAlongRay );
    MiscDynamicParams.fCascadeInd = static_cast<float>(m_PostProcessingAttribs.m_iFirstCascade);
    UpdateConstantBuffer(FrameAttribs.pd3dDeviceContext, m_pcbMiscParams, &MiscDynamicParams, sizeof(MiscDynamicParams));
    //cbuffer cbMiscDynamicParams : register( b4 )
    FrameAttribs.pd3dDeviceContext->PSSetConstantBuffers(4, 1, &m_pcbMiscParams.p);
        
    RenderQuad( FrameAttribs.pd3dDeviceContext, FixInsctrAtDepthBreaksTech);
}

void CLightSctrPostProcess :: RenderSampleLocations(SFrameAttribs &FrameAttribs)
{
    if( !m_RenderSampleLocationsTech.IsValid() )
    {
        CD3DShaderMacroHelper Macros;
        DefineMacros(Macros);
        Macros.Finalize();
        m_RenderSampleLocationsTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_RenderSampleLocationsTech.CreateVGPShadersFromFile( m_strEffectPath, "PassThroughVS", "RenderSamplePositionsGS", "RenderSampleLocationsPS", Macros );
        m_RenderSampleLocationsTech.SetDS( m_pDisableDepthTestDS );
        m_RenderSampleLocationsTech.SetRS( m_pSolidFillNoCullRS );
        D3D11_BLEND_DESC OverBlendStateDesc;
        ZeroMemory(&OverBlendStateDesc, sizeof(OverBlendStateDesc));
        OverBlendStateDesc.IndependentBlendEnable = FALSE;
        OverBlendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        OverBlendStateDesc.RenderTarget[0].BlendEnable    = TRUE;
        OverBlendStateDesc.RenderTarget[0].BlendOp        = D3D11_BLEND_OP_ADD;
        OverBlendStateDesc.RenderTarget[0].BlendOpAlpha   = D3D11_BLEND_OP_ADD;
        OverBlendStateDesc.RenderTarget[0].DestBlend      = D3D11_BLEND_INV_SRC_ALPHA;
        OverBlendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
        OverBlendStateDesc.RenderTarget[0].SrcBlend       = D3D11_BLEND_SRC_ALPHA;
        OverBlendStateDesc.RenderTarget[0].SrcBlendAlpha  = D3D11_BLEND_ONE;
        CComPtr<ID3D11BlendState> pOverBS;
        HRESULT hr;
        V( FrameAttribs.pd3dDevice->CreateBlendState( &OverBlendStateDesc, &pOverBS) );
        m_RenderSampleLocationsTech.SetBS( pOverBS );
    }

    ID3D11ShaderResourceView *pSRVs[] = 
    {
        NULL,
        m_ptex2DCoordinateTextureSRV,               // Texture2D<float2> g_tex2DCoordinates            : register( t1 );
        NULL,                                       // t2
        NULL,                                       // t3
        NULL,                                       // t4
        NULL,                                       // t5
        m_ptex2DInterpolationSourcesSRV,            // Texture2D<uint2>  g_tex2DInterpolationSource    : register( t6 );
    };
    FrameAttribs.pd3dDeviceContext->GSSetShaderResources(0, _countof(pSRVs), pSRVs);

    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(1, &FrameAttribs.pDstRTV, NULL);
    FrameAttribs.pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_POINTLIST);
    UINT offset[1] = {0};
    UINT stride[1] = {0};
    ID3D11Buffer *ppBuffers[1] = {0};
    // Set the device's first and only vertex buffer with zero stride and offset
    FrameAttribs.pd3dDeviceContext->IASetVertexBuffers(0,1,ppBuffers,stride,offset);
    // There is no input-layout object and the primitive topology is triangle strip
    FrameAttribs.pd3dDeviceContext->IASetInputLayout(NULL);
    m_RenderSampleLocationsTech.Apply();
    FrameAttribs.pd3dDeviceContext->Draw(m_PostProcessingAttribs.m_uiMaxSamplesInSlice * m_PostProcessingAttribs.m_uiNumEpipolarSlices,0);
    UnbindResources( FrameAttribs.pd3dDeviceContext );
}

void CLightSctrPostProcess :: PerformPostProcessing(SFrameAttribs &FrameAttribs,
                                                    SPostProcessingAttribs &PPAttribs)
{
    HRESULT hr;

    if( GetAsyncKeyState(VK_F8) )
    {
        m_ReconstrCamSpaceZTech.Release();
        m_RendedSliceEndpointsTech.Release();
        m_RendedCoordTexTech.Release();
        m_RefineSampleLocationsTech.Release();
        m_RenderCoarseUnshadowedInsctrTech.Release();
        m_MarkRayMarchingSamplesInStencilTech.Release();
        m_RenderSliceUVDirInSMTech.Release();
        m_InitializeMinMaxShadowMapTech.Release();
        m_ComputeMinMaxSMLevelTech.Release();
        m_InitializeCldDensEpipolarScanTech.Release();
        m_ComputeCldDensEpiScanLevelTech.Release();
        for(size_t i=0; i<_countof(m_DoRayMarchTech); ++i)
            m_DoRayMarchTech[i].Release();
        m_InterpolateIrradianceTech.Release();
        m_UnwarpEpipolarSctrImgTech.Release();
        m_UnwarpAndRenderLuminanceTech.Release();
        m_UpdateAverageLuminanceTech.Release();
        for(size_t i=0; i<_countof(m_FixInsctrAtDepthBreaksTech); ++i)
            m_FixInsctrAtDepthBreaksTech[i].Release();
        m_RenderSampleLocationsTech.Release();
        m_RenderSunTech.Release();
        m_PrecomputeSingleSctrTech.Release();
        m_ComputeSctrRadianceTech.Release();
        m_ComputeScatteringOrderTech.Release();
        m_AddScatteringOrderTech.Release();
    }
    bool bUseCombinedMinMaxTexture = PPAttribs.m_uiCascadeProcessingMode == CASCADE_PROCESSING_MODE_SINGLE_PASS ||
                                     PPAttribs.m_uiCascadeProcessingMode == CASCADE_PROCESSING_MODE_MULTI_PASS_INST ||
                                     PPAttribs.m_bCorrectScatteringAtDepthBreaks || 
                                     PPAttribs.m_uiLightSctrTechnique == LIGHT_SCTR_TECHNIQUE_BRUTE_FORCE;

    if( PPAttribs.m_uiNumEpipolarSlices != m_PostProcessingAttribs.m_uiNumEpipolarSlices || 
        PPAttribs.m_uiMaxSamplesInSlice != m_PostProcessingAttribs.m_uiMaxSamplesInSlice ||
        PPAttribs.m_bOptimizeSampleLocations != m_PostProcessingAttribs.m_bOptimizeSampleLocations )
        m_RendedSliceEndpointsTech.Release();
    
    if( PPAttribs.m_uiMaxSamplesInSlice != m_PostProcessingAttribs.m_uiMaxSamplesInSlice )
        m_RendedCoordTexTech.Release();

    if( PPAttribs.m_uiMaxSamplesInSlice != m_PostProcessingAttribs.m_uiMaxSamplesInSlice ||
        PPAttribs.m_uiInitialSampleStepInSlice != m_PostProcessingAttribs.m_uiInitialSampleStepInSlice ||
        PPAttribs.m_uiRefinementCriterion != m_PostProcessingAttribs.m_uiRefinementCriterion ||
        PPAttribs.m_bAutoExposure != m_PostProcessingAttribs.m_bAutoExposure ||
        PPAttribs.m_bEnableClouds != m_PostProcessingAttribs.m_bEnableClouds )
        m_RefineSampleLocationsTech.Release();

    if( PPAttribs.m_bUse1DMinMaxTree != m_PostProcessingAttribs.m_bUse1DMinMaxTree ||
        bUseCombinedMinMaxTexture != m_bUseCombinedMinMaxTexture ||
        PPAttribs.m_uiNumEpipolarSlices != m_PostProcessingAttribs.m_uiNumEpipolarSlices ||
        PPAttribs.m_bIs32BitMinMaxMipMap != m_PostProcessingAttribs.m_bIs32BitMinMaxMipMap )
    {
        m_InitializeMinMaxShadowMapTech.Release();
    }

    if( bUseCombinedMinMaxTexture != m_bUseCombinedMinMaxTexture ||
        PPAttribs.m_uiNumEpipolarSlices != m_PostProcessingAttribs.m_uiNumEpipolarSlices )
    {
        m_InitializeCldDensEpipolarScanTech.Release();
    }
        
    if( PPAttribs.m_bUse1DMinMaxTree != m_PostProcessingAttribs.m_bUse1DMinMaxTree ||
        PPAttribs.m_uiCascadeProcessingMode != m_PostProcessingAttribs.m_uiCascadeProcessingMode ||
        bUseCombinedMinMaxTexture != m_bUseCombinedMinMaxTexture ||
        PPAttribs.m_bEnableLightShafts != m_PostProcessingAttribs.m_bEnableLightShafts ||
        PPAttribs.m_uiMultipleScatteringMode != m_PostProcessingAttribs.m_uiMultipleScatteringMode ||
        PPAttribs.m_uiSingleScatteringMode != m_PostProcessingAttribs.m_uiSingleScatteringMode ||
        PPAttribs.m_bEnableClouds != m_PostProcessingAttribs.m_bEnableClouds ||
        PPAttribs.m_uiShaftsFromCloudsMode != m_PostProcessingAttribs.m_uiShaftsFromCloudsMode )
    {
        for(int i=0; i<_countof(m_DoRayMarchTech); ++i)
            m_DoRayMarchTech[i].Release();
    }

    if( PPAttribs.m_uiNumEpipolarSlices != m_PostProcessingAttribs.m_uiNumEpipolarSlices ||
        PPAttribs.m_uiMaxSamplesInSlice != m_PostProcessingAttribs.m_uiMaxSamplesInSlice )
    {
        m_UnwarpEpipolarSctrImgTech.Release();
        m_UnwarpAndRenderLuminanceTech.Release();
    }

    if( PPAttribs.m_bAutoExposure != m_PostProcessingAttribs.m_bAutoExposure || 
        PPAttribs.m_uiToneMappingMode != m_PostProcessingAttribs.m_uiToneMappingMode ||
        PPAttribs.m_bCorrectScatteringAtDepthBreaks != m_PostProcessingAttribs.m_bCorrectScatteringAtDepthBreaks  ||
        PPAttribs.m_bEnableClouds != m_PostProcessingAttribs.m_bEnableClouds )
    {
        m_UnwarpEpipolarSctrImgTech.Release();
    }

    if( PPAttribs.m_bLightAdaptation != m_PostProcessingAttribs.m_bLightAdaptation )
    {
        m_UpdateAverageLuminanceTech.Release();
    }

    if( PPAttribs.m_uiCascadeProcessingMode != m_PostProcessingAttribs.m_uiCascadeProcessingMode ||
        bUseCombinedMinMaxTexture != m_bUseCombinedMinMaxTexture ||
        PPAttribs.m_bEnableLightShafts != m_PostProcessingAttribs.m_bEnableLightShafts ||
        PPAttribs.m_uiMultipleScatteringMode != m_PostProcessingAttribs.m_uiMultipleScatteringMode  ||
        PPAttribs.m_uiSingleScatteringMode != m_PostProcessingAttribs.m_uiSingleScatteringMode ||
        PPAttribs.m_bAutoExposure != m_PostProcessingAttribs.m_bAutoExposure || 
        PPAttribs.m_uiToneMappingMode != m_PostProcessingAttribs.m_uiToneMappingMode  ||
        PPAttribs.m_bEnableClouds != m_PostProcessingAttribs.m_bEnableClouds ||
        PPAttribs.m_uiShaftsFromCloudsMode != m_PostProcessingAttribs.m_uiShaftsFromCloudsMode )
    {
        for(size_t i=0; i<_countof(m_FixInsctrAtDepthBreaksTech); ++i)
            m_FixInsctrAtDepthBreaksTech[i].Release();
    }
    
    if( PPAttribs.m_uiMaxSamplesInSlice != m_PostProcessingAttribs.m_uiMaxSamplesInSlice || 
        PPAttribs.m_uiNumEpipolarSlices != m_PostProcessingAttribs.m_uiNumEpipolarSlices )
    {
        m_ptex2DCoordinateTextureRTV.Release();
        m_ptex2DCoordinateTextureSRV.Release();
    }
    
    if( PPAttribs.m_uiMinMaxShadowMapResolution != m_PostProcessingAttribs.m_uiMinMaxShadowMapResolution || 
        PPAttribs.m_uiNumEpipolarSlices != m_PostProcessingAttribs.m_uiNumEpipolarSlices ||
        PPAttribs.m_bUse1DMinMaxTree != m_PostProcessingAttribs.m_bUse1DMinMaxTree ||
        PPAttribs.m_bIs32BitMinMaxMipMap != m_PostProcessingAttribs.m_bIs32BitMinMaxMipMap ||
        bUseCombinedMinMaxTexture != m_bUseCombinedMinMaxTexture ||
        bUseCombinedMinMaxTexture && 
            (PPAttribs.m_iFirstCascade != m_PostProcessingAttribs.m_iFirstCascade || 
             PPAttribs.m_iNumCascades  != m_PostProcessingAttribs.m_iNumCascades) )
    {
        for(int i=0; i < _countof(m_ptex2DMinMaxShadowMapSRV); ++i)
            m_ptex2DMinMaxShadowMapSRV[i].Release();
        for(int i=0; i < _countof(m_ptex2DMinMaxShadowMapRTV); ++i)
            m_ptex2DMinMaxShadowMapRTV[i].Release();
    }

    if( PPAttribs.m_uiMinMaxShadowMapResolution != m_PostProcessingAttribs.m_uiMinMaxShadowMapResolution || 
        PPAttribs.m_uiNumEpipolarSlices != m_PostProcessingAttribs.m_uiNumEpipolarSlices ||
        bUseCombinedMinMaxTexture != m_bUseCombinedMinMaxTexture ||
        bUseCombinedMinMaxTexture && 
            (PPAttribs.m_iFirstCascade != m_PostProcessingAttribs.m_iFirstCascade || 
             PPAttribs.m_iNumCascades  != m_PostProcessingAttribs.m_iNumCascades) )
    {
        for(int i=0; i < _countof(m_ptex2DCldDensEpipolarScanSRV); ++i)
            m_ptex2DCldDensEpipolarScanSRV[i].Release();
        for(int i=0; i < _countof(m_ptex2DCldDensEpipolarScanRTV); ++i)
            m_ptex2DCldDensEpipolarScanRTV[i].Release();
    }

    if( PPAttribs.m_iNumCascades != m_PostProcessingAttribs.m_iNumCascades )
    {
        m_ptex2DSliceUVDirAndOriginSRV.Release();
        m_ptex2DSliceUVDirAndOriginRTV.Release();
    }

    if( PPAttribs.m_uiCascadeProcessingMode != m_PostProcessingAttribs.m_uiCascadeProcessingMode )
    {
        m_ComputeMinMaxSMLevelTech.Release();
    }
    
    if( PPAttribs.m_uiExtinctionEvalMode != m_PostProcessingAttribs.m_uiExtinctionEvalMode )
    {
        m_ptex2DEpipolarExtinctionRTV.Release();
        m_ptex2DEpipolarExtinctionSRV.Release();
        m_UnwarpEpipolarSctrImgTech.Release();
        m_UnwarpAndRenderLuminanceTech.Release();
        m_RenderCoarseUnshadowedInsctrTech.Release();
    }

    if( PPAttribs.m_uiSingleScatteringMode != m_PostProcessingAttribs.m_uiSingleScatteringMode ||
        PPAttribs.m_uiMultipleScatteringMode != m_PostProcessingAttribs.m_uiMultipleScatteringMode  ||
        PPAttribs.m_bEnableClouds != m_PostProcessingAttribs.m_bEnableClouds )
        m_RenderCoarseUnshadowedInsctrTech.Release();

    bool bRecomputeSctrCoeffs = m_PostProcessingAttribs.m_bUseCustomSctrCoeffs != PPAttribs.m_bUseCustomSctrCoeffs ||
                                m_PostProcessingAttribs.m_fAerosolDensityScale != PPAttribs.m_fAerosolDensityScale ||
                                m_PostProcessingAttribs.m_fAerosolAbsorbtionScale != PPAttribs.m_fAerosolAbsorbtionScale ||
                                PPAttribs.m_bUseCustomSctrCoeffs && 
                                    ( m_PostProcessingAttribs.m_f4CustomRlghBeta != PPAttribs.m_f4CustomRlghBeta ||
                                      m_PostProcessingAttribs.m_f4CustomMieBeta  != PPAttribs.m_f4CustomMieBeta );

    m_PostProcessingAttribs = PPAttribs;
    m_bUseCombinedMinMaxTexture = bUseCombinedMinMaxTexture;

    if( bRecomputeSctrCoeffs )
    {
        m_ptex2DOccludedNetDensityToAtmTopSRV.Release();
        m_ptex2DOccludedNetDensityToAtmTopRTV.Release();
        m_ptex3DSingleScatteringSRV.Release();
        m_ptex2DAmbientSkyLightRTV.Release();
        m_ptex2DAmbientSkyLightSRV.Release();
        ComputeScatteringCoefficients(FrameAttribs.pd3dDeviceContext);
    }

    if( !m_ptex2DCoordinateTextureRTV || !m_ptex2DCoordinateTextureSRV )
    {
        V( CreateTextures(FrameAttribs.pd3dDevice) );
    }

    if( !m_ptex2DMinMaxShadowMapSRV[0] && m_PostProcessingAttribs.m_bUse1DMinMaxTree )
    {
        V( CreateMinMaxShadowMap(FrameAttribs.pd3dDevice) );
    }

    if( !m_ptex2DCldDensEpipolarScanSRV[0] && m_PostProcessingAttribs.m_bEnableClouds && m_PostProcessingAttribs.m_bUse1DMinMaxTree )
    {
        V( CreateCldDensEpipolarScanTex(FrameAttribs.pd3dDevice) );
    }

    SLightAttribs &LightAttribs = *FrameAttribs.pLightAttribs;

    // Note that in fact the outermost visible screen pixels do not lie exactly on the boundary (+1 or -1), but are biased by
    // 0.5 screen pixel size inwards. Using these adjusted boundaries improves precision and results in
    // smaller number of pixels which require inscattering correction
    assert( LightAttribs.bIsLightOnScreen == (abs(FrameAttribs.pLightAttribs->f4LightScreenPos.x) <= 1.f - 1.f/(float)m_uiBackBufferWidth && 
                                              abs(FrameAttribs.pLightAttribs->f4LightScreenPos.y) <= 1.f - 1.f/(float)m_uiBackBufferHeight) );

    const auto &SMAttribs = FrameAttribs.pLightAttribs->ShadowAttribs;

    //UpdateConstantBuffer(FrameAttribs.pd3dDeviceContext, m_pcbLightAttribs, &LightAttribs, sizeof(LightAttribs));
    UpdateConstantBuffer(FrameAttribs.pd3dDeviceContext, m_pcbPostProcessingAttribs, &m_PostProcessingAttribs, sizeof(m_PostProcessingAttribs));
    
    // Set constant buffers that will be used by all pixel shaders and compute shader
    ID3D11Buffer *pCBs[] = {m_pcbPostProcessingAttribs, m_pcbMediaAttribs, FrameAttribs.pcbCameraAttribs, FrameAttribs.pcbLightAttribs};
    FrameAttribs.pd3dDeviceContext->VSSetConstantBuffers(0, _countof(pCBs), pCBs);
    FrameAttribs.pd3dDeviceContext->GSSetConstantBuffers(0, _countof(pCBs), pCBs);
    FrameAttribs.pd3dDeviceContext->PSSetConstantBuffers(0, _countof(pCBs), pCBs);
    FrameAttribs.pd3dDeviceContext->CSSetConstantBuffers(0, _countof(pCBs), pCBs);
        

    ID3D11SamplerState *pSamplers[] = { m_psamLinearClamp, m_psamLinearBorder0, m_psamComparison, m_psamPointClamp };
    FrameAttribs.pd3dDeviceContext->PSSetSamplers(0, _countof(pSamplers), pSamplers);
    FrameAttribs.pd3dDeviceContext->CSSetSamplers(0, 1, pSamplers);

    D3D11_VIEWPORT OrigViewPort;
    UINT iNumOldViewports = 1;
    FrameAttribs.pd3dDeviceContext->RSGetViewports(&iNumOldViewports, &OrigViewPort);

    if( !m_ptex2DOccludedNetDensityToAtmTopSRV )
    {
        CreatePrecomputedOpticalDepthTexture(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
    }
    
    if( (m_PostProcessingAttribs.m_uiMultipleScatteringMode > MULTIPLE_SCTR_MODE_NONE ||
         PPAttribs.m_uiSingleScatteringMode == SINGLE_SCTR_MODE_LUT) &&
        !m_ptex3DSingleScatteringSRV )
    {
        CreatePrecomputedScatteringLUT(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
    }

    if( m_PostProcessingAttribs.m_bAutoExposure && !m_ptex2DLowResLuminanceRTV )
    {
        CreateLowResLuminanceTexture(FrameAttribs.pd3dDevice);
    }

    float Zero[4]={0,0,0,0};
    FrameAttribs.pd3dDeviceContext->ClearRenderTargetView(FrameAttribs.pDstRTV, Zero);

    RenderSun(FrameAttribs);

    ReconstructCameraSpaceZ(FrameAttribs);

    if( m_PostProcessingAttribs.m_uiLightSctrTechnique == LIGHT_SCTR_TECHNIQUE_EPIPOLAR_SAMPLING )
    {
        
        RenderSliceEndpoints(FrameAttribs);

        // Render coordinate texture and camera space z for epipolar location
        RenderCoordinateTexture(FrameAttribs);

        if( m_PostProcessingAttribs.m_uiRefinementCriterion == REFINEMENT_CRITERION_INSCTR_DIFF || 
            m_PostProcessingAttribs.m_uiExtinctionEvalMode == EXTINCTION_EVAL_MODE_EPIPOLAR )
        {
            RenderCoarseUnshadowedInsctr(FrameAttribs);
        }

        // Refine initial ray marching samples
        RefineSampleLocations(FrameAttribs);

        // Mark all ray marching samples in stencil
        MarkRayMarchingSamples( FrameAttribs );

        if( m_PostProcessingAttribs.m_bEnableLightShafts && m_PostProcessingAttribs.m_bUse1DMinMaxTree )
        {
            RenderSliceUVDirAndOrig(FrameAttribs);
        }

        FrameAttribs.pd3dDeviceContext->ClearRenderTargetView(m_ptex2DInitialScatteredLightRTV, Zero);
        int iLastCascade = (m_PostProcessingAttribs.m_bEnableLightShafts && m_PostProcessingAttribs.m_uiCascadeProcessingMode == CASCADE_PROCESSING_MODE_MULTI_PASS) ? m_PostProcessingAttribs.m_iNumCascades - 1 : m_PostProcessingAttribs.m_iFirstCascade;
        for(int iCascadeInd = m_PostProcessingAttribs.m_iFirstCascade; iCascadeInd <= iLastCascade; ++iCascadeInd)
        {
            // Build min/max mip map
            if( m_PostProcessingAttribs.m_bEnableLightShafts && m_PostProcessingAttribs.m_bUse1DMinMaxTree )
            {
                Build1DMinMaxMipMap(FrameAttribs, iCascadeInd);
                if( m_PostProcessingAttribs.m_bEnableClouds && m_PostProcessingAttribs.m_uiShaftsFromCloudsMode == SHAFTS_FROM_CLOUDS_TRANSPARENCY_MAP )
                {
                    RenderCldDensEpipolarScan(FrameAttribs, iCascadeInd);
                }
            }

            // Perform ray marching for selected samples
            DoRayMarching(FrameAttribs, m_PostProcessingAttribs.m_uiShadowMapResolution, SMAttribs, iCascadeInd);
        }

        // Interpolate ray marching samples onto the rest of samples
        InterpolateInsctrIrradiance(FrameAttribs);

        const UINT uiMaxStepsAlongRayAtDepthBreak0 = min(m_PostProcessingAttribs.m_uiShadowMapResolution/4, 256);
        const UINT uiMaxStepsAlongRayAtDepthBreak1 = min(m_PostProcessingAttribs.m_uiShadowMapResolution/8, 128);
        
        ID3D11DepthStencilView *pDSV = m_ptex2DScreenSizeDSV;
        ID3D11RenderTargetView *pRTV[] = { FrameAttribs.pDstRTV };
        FrameAttribs.pd3dDeviceContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0, 0);
        
        if( m_PostProcessingAttribs.m_bAutoExposure )
        {
            // Render scene luminance to low-resolution texture
            FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(1, &m_ptex2DLowResLuminanceRTV.p, nullptr);
            UnwarpEpipolarScattering(FrameAttribs, true);
            FrameAttribs.pd3dDeviceContext->GenerateMips(m_ptex2DLowResLuminanceSRV);

            UpdateAverageLuminance(FrameAttribs);
        }
        // Transform inscattering irradiance from epipolar coordinates back to rectangular
        FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(1, pRTV, pDSV);
        UnwarpEpipolarScattering(FrameAttribs, false);
    
        // Correct inscattering for pixels, for which no suitable interpolation sources were found
        if( m_PostProcessingAttribs.m_bCorrectScatteringAtDepthBreaks )
        {
            FixInscatteringAtDepthBreaks(FrameAttribs, uiMaxStepsAlongRayAtDepthBreak0, SMAttribs, false);
        }

        if( m_PostProcessingAttribs.m_bShowSampling )
        {
            RenderSampleLocations(FrameAttribs);
        }
    }
    else if(m_PostProcessingAttribs.m_uiLightSctrTechnique == LIGHT_SCTR_TECHNIQUE_BRUTE_FORCE )
    {
        if( m_PostProcessingAttribs.m_bAutoExposure )
        {
            // Render scene luminance to low-resolution texture
            FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(1, &m_ptex2DLowResLuminanceRTV.p, nullptr);
            FixInscatteringAtDepthBreaks(FrameAttribs, m_PostProcessingAttribs.m_uiShadowMapResolution, SMAttribs, true);
            FrameAttribs.pd3dDeviceContext->GenerateMips(m_ptex2DLowResLuminanceSRV);

            UpdateAverageLuminance(FrameAttribs);
        }

        FrameAttribs.pd3dDeviceContext->ClearDepthStencilView(m_ptex2DScreenSizeDSV, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 0, 0);
        FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(1, &FrameAttribs.pDstRTV, m_ptex2DScreenSizeDSV);
        FixInscatteringAtDepthBreaks(FrameAttribs, m_PostProcessingAttribs.m_uiShadowMapResolution, SMAttribs, false);
    }
}

HRESULT CLightSctrPostProcess :: CreateTextures(ID3D11Device* pd3dDevice)
{
    HRESULT hr;

    D3D11_TEXTURE2D_DESC CoordinateTexDesc = 
    {
        m_PostProcessingAttribs.m_uiMaxSamplesInSlice,         //UINT Width;
        m_PostProcessingAttribs.m_uiNumEpipolarSlices,         //UINT Height;
        1,                                                     //UINT MipLevels;
        1,                                                     //UINT ArraySize;
        DXGI_FORMAT_R32G32_FLOAT,                              //DXGI_FORMAT Format;
        {1,0},                                                 //DXGI_SAMPLE_DESC SampleDesc;
        D3D11_USAGE_DEFAULT,                                   //D3D11_USAGE Usage;
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, //UINT BindFlags;
        0,                                                     //UINT CPUAccessFlags;
        0,                                                     //UINT MiscFlags;
    };

    {
        CComPtr<ID3D11Texture2D> ptex2DCoordinateTexture;
        // Create 2-D texture, shader resource and target view buffers on the device
        V_RETURN( pd3dDevice->CreateTexture2D( &CoordinateTexDesc, NULL, &ptex2DCoordinateTexture) );
        V_RETURN( pd3dDevice->CreateShaderResourceView( ptex2DCoordinateTexture, NULL, &m_ptex2DCoordinateTextureSRV)  );
        V_RETURN( pd3dDevice->CreateRenderTargetView( ptex2DCoordinateTexture, NULL, &m_ptex2DCoordinateTextureRTV)  );
    }
    
    {
        m_ptex2DSliceEndpointsSRV.Release();
        m_ptex2DSliceEndpointsRTV.Release();
        D3D11_TEXTURE2D_DESC InterpolationSourceTexDesc = CoordinateTexDesc;
        InterpolationSourceTexDesc.Width = m_PostProcessingAttribs.m_uiNumEpipolarSlices;
        InterpolationSourceTexDesc.Height = 1;
        InterpolationSourceTexDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        InterpolationSourceTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        CComPtr<ID3D11Texture2D> ptex2DSliceEndpoints;
        // Create 2-D texture, shader resource and target view buffers on the device
        V_RETURN( pd3dDevice->CreateTexture2D( &InterpolationSourceTexDesc, NULL, &ptex2DSliceEndpoints) );
        V_RETURN( pd3dDevice->CreateShaderResourceView( ptex2DSliceEndpoints, NULL, &m_ptex2DSliceEndpointsSRV)  );
        V_RETURN( pd3dDevice->CreateRenderTargetView( ptex2DSliceEndpoints, NULL, &m_ptex2DSliceEndpointsRTV)  );
    }

    {
        m_ptex2DInterpolationSourcesSRV.Release();
        m_ptex2DInterpolationSourcesUAV.Release();
        D3D11_TEXTURE2D_DESC InterpolationSourceTexDesc = CoordinateTexDesc;
        InterpolationSourceTexDesc.Format = DXGI_FORMAT_R16G16_UINT;
        InterpolationSourceTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        CComPtr<ID3D11Texture2D> ptex2DInterpolationSource;
        // Create 2-D texture, shader resource and target view buffers on the device
        V_RETURN( pd3dDevice->CreateTexture2D( &InterpolationSourceTexDesc, NULL, &ptex2DInterpolationSource) );
        V_RETURN( pd3dDevice->CreateShaderResourceView( ptex2DInterpolationSource, NULL, &m_ptex2DInterpolationSourcesSRV)  );
        V_RETURN( pd3dDevice->CreateUnorderedAccessView( ptex2DInterpolationSource, NULL, &m_ptex2DInterpolationSourcesUAV)  );
    }

    {
        m_ptex2DEpipolarCamSpaceZSRV.Release();
        m_ptex2DEpipolarCamSpaceZRTV.Release();
        D3D11_TEXTURE2D_DESC EpipolarCamSpaceZTexDesc = CoordinateTexDesc;
        EpipolarCamSpaceZTexDesc.Format = DXGI_FORMAT_R32_FLOAT;
        EpipolarCamSpaceZTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        CComPtr<ID3D11Texture2D> ptex2DEpipolarCamSpace;
        // Create 2-D texture, shader resource and target view buffers on the device
        V_RETURN( pd3dDevice->CreateTexture2D( &EpipolarCamSpaceZTexDesc, NULL, &ptex2DEpipolarCamSpace) );
        V_RETURN( pd3dDevice->CreateShaderResourceView( ptex2DEpipolarCamSpace, NULL, &m_ptex2DEpipolarCamSpaceZSRV)  );
        V_RETURN( pd3dDevice->CreateRenderTargetView( ptex2DEpipolarCamSpace, NULL, &m_ptex2DEpipolarCamSpaceZRTV)  );
    }

    {
        m_ptex2DEpipolarCloudTranspSRV.Release();
        m_ptex2DEpipolarCloudTranspRTV.Release();
        D3D11_TEXTURE2D_DESC EpipolarCloudTranspTexDesc = CoordinateTexDesc;
        EpipolarCloudTranspTexDesc.Format = DXGI_FORMAT_R8_UNORM;
        EpipolarCloudTranspTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        CComPtr<ID3D11Texture2D> ptex2DEpipolarCloudTransp;
        // Create 2-D texture, shader resource and target view buffers on the device
        V_RETURN( pd3dDevice->CreateTexture2D( &EpipolarCloudTranspTexDesc, NULL, &ptex2DEpipolarCloudTransp) );
        V_RETURN( pd3dDevice->CreateShaderResourceView( ptex2DEpipolarCloudTransp, NULL, &m_ptex2DEpipolarCloudTranspSRV)  );
        V_RETURN( pd3dDevice->CreateRenderTargetView( ptex2DEpipolarCloudTransp, NULL, &m_ptex2DEpipolarCloudTranspRTV)  );
    }

    {
        m_ptex2DEpipolarInscatteringSRV.Release();
        m_ptex2DEpipolarInscatteringRTV.Release();
        D3D11_TEXTURE2D_DESC ScatteredLightTexDesc = CoordinateTexDesc;
        // R8G8B8A8_UNORM texture does not provide sufficient precision which causes 
        // interpolation artifacts especially noticeable in low intensity regions
        ScatteredLightTexDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        CComPtr<ID3D11Texture2D> ptex2DEpipolarInscattering;
        // Create 2-D texture, shader resource and target view buffers on the device
        V_RETURN( pd3dDevice->CreateTexture2D( &ScatteredLightTexDesc, NULL, &ptex2DEpipolarInscattering) );
        V_RETURN( pd3dDevice->CreateShaderResourceView( ptex2DEpipolarInscattering, NULL, &m_ptex2DEpipolarInscatteringSRV)  );
        V_RETURN( pd3dDevice->CreateRenderTargetView( ptex2DEpipolarInscattering, NULL, &m_ptex2DEpipolarInscatteringRTV)  );

        m_ptex2DInitialScatteredLightSRV.Release();
        m_ptex2DInitialScatteredLightRTV.Release();
        CComPtr<ID3D11Texture2D> ptex2DInitialScatteredLight;
        // Create 2-D texture, shader resource and target view buffers on the device
        V_RETURN( pd3dDevice->CreateTexture2D( &ScatteredLightTexDesc, NULL, &ptex2DInitialScatteredLight) );
        V_RETURN( pd3dDevice->CreateShaderResourceView( ptex2DInitialScatteredLight, NULL, &m_ptex2DInitialScatteredLightSRV)  );
        V_RETURN( pd3dDevice->CreateRenderTargetView( ptex2DInitialScatteredLight, NULL, &m_ptex2DInitialScatteredLightRTV)  );

        // Release extinction texture so that it is re-created when first needed
        m_ptex2DEpipolarExtinctionSRV.Release();
        m_ptex2DEpipolarExtinctionRTV.Release();
    }

    {
        m_ptex2DEpipolarImageDSV.Release();
        D3D11_TEXTURE2D_DESC EpipolarDeptTexDesc = CoordinateTexDesc;
        EpipolarDeptTexDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        EpipolarDeptTexDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        CComPtr<ID3D11Texture2D> ptex2DEpipolarImage;
        V_RETURN( pd3dDevice->CreateTexture2D( &EpipolarDeptTexDesc, NULL, &ptex2DEpipolarImage) );
        V_RETURN( pd3dDevice->CreateDepthStencilView( ptex2DEpipolarImage, NULL, &m_ptex2DEpipolarImageDSV) );
    }

    {
        m_ptex2DSliceUVDirAndOriginSRV.Release();
        m_ptex2DSliceUVDirAndOriginRTV.Release();
    }
        
    return S_OK;
}

HRESULT CLightSctrPostProcess :: CreateMinMaxShadowMap(ID3D11Device* pd3dDevice)
{
    D3D11_TEXTURE2D_DESC MinMaxShadowMapTexDesc = 
    {
        // Min/max shadow map does not contain finest resolution level of the shadow map
        m_PostProcessingAttribs.m_uiMinMaxShadowMapResolution, //UINT Width;
        m_PostProcessingAttribs.m_uiNumEpipolarSlices,         //UINT Height;
        1,                                                     //UINT MipLevels;
        1,                                                     //UINT ArraySize;
        m_PostProcessingAttribs.m_bIs32BitMinMaxMipMap ? DXGI_FORMAT_R32G32_FLOAT : DXGI_FORMAT_R16G16_UNORM, //DXGI_FORMAT Format;
        {1,0},                                                 //DXGI_SAMPLE_DESC SampleDesc;
        D3D11_USAGE_DEFAULT,                                   //D3D11_USAGE Usage;
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, //UINT BindFlags;
        0,                                                     //UINT CPUAccessFlags;
        0,                                                     //UINT MiscFlags;
    };

    if( m_bUseCombinedMinMaxTexture )
    {
        MinMaxShadowMapTexDesc.Height *= (m_PostProcessingAttribs.m_iNumCascades - m_PostProcessingAttribs.m_iFirstCascade);
    }
    
    HRESULT hr;
    for(int i=0; i < 2; ++i)
    {
        m_ptex2DMinMaxShadowMapSRV[i].Release();
        m_ptex2DMinMaxShadowMapRTV[i].Release();

        CComPtr<ID3D11Texture2D> ptex2DMinMaxShadowMap;
        // Create 2-D texture, shader resource and target view buffers on the device
        V_RETURN( pd3dDevice->CreateTexture2D( &MinMaxShadowMapTexDesc, NULL, &ptex2DMinMaxShadowMap) );
        V_RETURN( pd3dDevice->CreateShaderResourceView( ptex2DMinMaxShadowMap, NULL, &m_ptex2DMinMaxShadowMapSRV[i])  );
        V_RETURN( pd3dDevice->CreateRenderTargetView( ptex2DMinMaxShadowMap, NULL, &m_ptex2DMinMaxShadowMapRTV[i])  );
    }

    return S_OK;
}

HRESULT CLightSctrPostProcess :: CreateCldDensEpipolarScanTex(ID3D11Device* pd3dDevice)
{
    // Cloud density epipolar scan texture must be consistent with 1D min/max binary tree
    // This means that for each sample from 1D min/max tree there must be corresponding sample
    // in cloud density epipolar scan.
    D3D11_TEXTURE2D_DESC CloudDensityEpipolarScanTexDesc = 
    {
        // Finest resolution level of the light space density is not stored
        m_PostProcessingAttribs.m_uiMinMaxShadowMapResolution, //UINT Width;
        m_PostProcessingAttribs.m_uiNumEpipolarSlices,         //UINT Height;
        1,                                                     //UINT MipLevels;
        1,                                                     //UINT ArraySize;
        DXGI_FORMAT_R8_UNORM,                                 //DXGI_FORMAT Format;
        {1,0},                                                 //DXGI_SAMPLE_DESC SampleDesc;
        D3D11_USAGE_DEFAULT,                                   //D3D11_USAGE Usage;
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, //UINT BindFlags;
        0,                                                     //UINT CPUAccessFlags;
        0,                                                     //UINT MiscFlags;
    };

    if( m_bUseCombinedMinMaxTexture )
    {
        CloudDensityEpipolarScanTexDesc.Height *= (m_PostProcessingAttribs.m_iNumCascades - m_PostProcessingAttribs.m_iFirstCascade);
    }
    
    HRESULT hr;
    for(int i=0; i < 2; ++i)
    {
        m_ptex2DCldDensEpipolarScanSRV[i].Release();
        m_ptex2DCldDensEpipolarScanRTV[i].Release();

        CComPtr<ID3D11Texture2D> ptex2DCldDensEpipolarScan;
        // Create 2-D texture, shader resource and target view buffers on the device
        V_RETURN( pd3dDevice->CreateTexture2D( &CloudDensityEpipolarScanTexDesc, NULL, &ptex2DCldDensEpipolarScan) );
        V_RETURN( pd3dDevice->CreateShaderResourceView( ptex2DCldDensEpipolarScan, NULL, &m_ptex2DCldDensEpipolarScanSRV[i])  );
        V_RETURN( pd3dDevice->CreateRenderTargetView( ptex2DCldDensEpipolarScan, NULL, &m_ptex2DCldDensEpipolarScanRTV[i])  );
    }

    return S_OK;
}



D3DXVECTOR2 exp(const D3DXVECTOR2 &fX){ return D3DXVECTOR2(exp(fX.x), exp(fX.y)); }
D3DXVECTOR3 exp(const D3DXVECTOR3 &fX){ return D3DXVECTOR3(exp(fX.x), exp(fX.y), exp(fX.z)); }
D3DXVECTOR2 operator * (const D3DXVECTOR2 &fX, const D3DXVECTOR2 &fY){ return D3DXVECTOR2(fX.x*fY.x, fX.y*fY.y); }
D3DXVECTOR3 operator * (const D3DXVECTOR3 &fX, const D3DXVECTOR3 &fY){ return D3DXVECTOR3(fX.x*fY.x, fX.y*fY.y, fX.z*fY.z); }
D3DXVECTOR2 operator / (const D3DXVECTOR2 &fX, const D3DXVECTOR2 &fY){ return D3DXVECTOR2(fX.x/fY.x, fX.y/fY.y); }

// fCosChi = Pi/2
D3DXVECTOR2 ChapmanOrtho(const D3DXVECTOR2 &f2x)
{
    static const float fConst = static_cast<float>( sqrt(D3DX_PI / 2) );
    D3DXVECTOR2 f2SqrtX = D3DXVECTOR2( sqrt(f2x.x), sqrt(f2x.y) );
    return fConst * ( D3DXVECTOR2(1.f,1.f) / (2.f * f2SqrtX) + f2SqrtX );
}

// |fCosChi| < Pi/2
D3DXVECTOR2 f2ChapmanRising(const D3DXVECTOR2 &f2X, float fCosChi)
{
    D3DXVECTOR2 f2ChOrtho = ChapmanOrtho(f2X);
    return f2ChOrtho / ((f2ChOrtho-D3DXVECTOR2(1,1))*fCosChi + D3DXVECTOR2(1,1));
}

D3DXVECTOR2 GetDensityIntegralFromChapmanFunc(float fHeightAboveSurface,
                                         const D3DXVECTOR3 &f3EarthCentreToPointDir,
                                         const D3DXVECTOR3 &f3RayDir,
                                         const SAirScatteringAttribs &SctrMediaAttribs)
{
    // Note: there is no intersection test with the Earth. However,
    // optical depth through the Earth is large, which effectively
    // occludes the light
    float fCosChi = D3DXVec3Dot(&f3EarthCentreToPointDir, &f3RayDir);
    D3DXVECTOR2 f2x = (fHeightAboveSurface + SctrMediaAttribs.fEarthRadius) * D3DXVECTOR2(1.f / SctrMediaAttribs.f2ParticleScaleHeight.x, 1.f / SctrMediaAttribs.f2ParticleScaleHeight.y);
    D3DXVECTOR2 f2VerticalAirMass = SctrMediaAttribs.f2ParticleScaleHeight * exp(-D3DXVECTOR2(fHeightAboveSurface,fHeightAboveSurface)/SctrMediaAttribs.f2ParticleScaleHeight);
    if( fCosChi >= 0.f )
    {
        return f2VerticalAirMass * f2ChapmanRising(f2x, fCosChi);
    }
    else
    {
        float fSinChi = sqrt(1.f - fCosChi*fCosChi);
        float fh0 = (fHeightAboveSurface + SctrMediaAttribs.fEarthRadius) * fSinChi - SctrMediaAttribs.fEarthRadius;
        D3DXVECTOR2 f2VerticalAirMass0 = SctrMediaAttribs.f2ParticleScaleHeight * exp(-D3DXVECTOR2(fh0,fh0)/SctrMediaAttribs.f2ParticleScaleHeight);
        D3DXVECTOR2 f2x0 = D3DXVECTOR2(fh0 + SctrMediaAttribs.fEarthRadius,fh0 + SctrMediaAttribs.fEarthRadius)/SctrMediaAttribs.f2ParticleScaleHeight;
        D3DXVECTOR2 f2ChOrtho_x0 = ChapmanOrtho(f2x0);
        D3DXVECTOR2 f2Ch = f2ChapmanRising(f2x, -fCosChi);
        return f2VerticalAirMass0 * (2.f * f2ChOrtho_x0) - f2VerticalAirMass*f2Ch;
    }
}

void CLightSctrPostProcess :: ComputeSunColor( const D3DXVECTOR3 &vDirectionOnSun,
                                               const D3DXVECTOR4 &f4ExtraterrestrialSunColor,
                                               D3DXVECTOR4 &f4SunColorAtGround,
                                               D3DXVECTOR4 &f4AmbientLight)
{

    // Compute the ambient light values
    float zenithFactor = min( max(vDirectionOnSun.y, 0.0f), 1.0f);
    f4AmbientLight.x = zenithFactor*0.15f;
    f4AmbientLight.y = zenithFactor*0.1f;
    f4AmbientLight.z = max(0.005f, zenithFactor*0.25f);
    f4AmbientLight.w = 0.0f;

    D3DXVECTOR2 f2NetParticleDensityToAtmTop = GetDensityIntegralFromChapmanFunc(0, D3DXVECTOR3(0,1,0), vDirectionOnSun, m_MediaParams);


    D3DXVECTOR3 f3RlghExtCoeff;
    D3DXVec3Maximize(&f3RlghExtCoeff, (D3DXVECTOR3*)&m_MediaParams.f4RayleighExtinctionCoeff, &D3DXVECTOR3(1e-8f,1e-8f,1e-8f));
    D3DXVECTOR3 f3RlghOpticalDepth = f3RlghExtCoeff * f2NetParticleDensityToAtmTop.x;
    D3DXVECTOR3 f3MieExtCoeff;
    D3DXVec3Maximize(&f3MieExtCoeff, (D3DXVECTOR3*)&m_MediaParams.f4MieExtinctionCoeff, &D3DXVECTOR3(1e-8f,1e-8f,1e-8f));
    D3DXVECTOR3 f3MieOpticalDepth  = f3MieExtCoeff * f2NetParticleDensityToAtmTop.y;
    D3DXVECTOR3 f3TotalExtinction = exp( -(f3RlghOpticalDepth + f3MieOpticalDepth ) );
    const float fEarthReflectance = 0.1f;// See [BN08]
    (D3DXVECTOR3&)f4SunColorAtGround = ((D3DXVECTOR3&)f4ExtraterrestrialSunColor) * f3TotalExtinction * fEarthReflectance;
}

void CLightSctrPostProcess :: ComputeScatteringCoefficients(ID3D11DeviceContext *pDeviceCtx)
{
    // For details, see "A practical Analytic Model for Daylight" by Preetham & Hoffman, p.23

    // Wave lengths
    // [BN08] follows [REK04] and gives the following values for Rayleigh scattering coefficients:
    // RayleighBetha(lambda = (680nm, 550nm, 440nm) ) = (5.8, 13.5, 33.1)e-6
    static const double dWaveLengths[] = 
    {
        680e-9,     // red
        550e-9,     // green
        440e-9      // blue
    }; 
        
    // Calculate angular and total scattering coefficients for Rayleigh scattering:
    {
        D3DXVECTOR4 &f4AngularRayleighSctrCoeff = m_MediaParams.f4AngularRayleighSctrCoeff;
        D3DXVECTOR4 &f4TotalRayleighSctrCoeff = m_MediaParams.f4TotalRayleighSctrCoeff;
        D3DXVECTOR4 &f4RayleighExtinctionCoeff = m_MediaParams.f4RayleighExtinctionCoeff;
    
        double n = 1.0003;    // - Refractive index of air in the visible spectrum
        double N = 2.545e+25; // - Number of molecules per unit volume
        double Pn = 0.035;    // - Depolarization factor for air which exoresses corrections 
                              //   due to anisotropy of air molecules

        double dRayleighConst = 8.0*D3DX_PI*D3DX_PI*D3DX_PI * (n*n - 1.0) * (n*n - 1.0) / (3.0 * N) * (6.0 + 3.0*Pn) / (6.0 - 7.0*Pn);
        for(int WaveNum = 0; WaveNum < 3; WaveNum++)
        {
            double dSctrCoeff;
            if( m_PostProcessingAttribs.m_bUseCustomSctrCoeffs )
                dSctrCoeff = f4TotalRayleighSctrCoeff[WaveNum] = m_PostProcessingAttribs.m_f4CustomRlghBeta[WaveNum];
            else
            {
                double Lambda2 = dWaveLengths[WaveNum] * dWaveLengths[WaveNum];
                double Lambda4 = Lambda2 * Lambda2;
                dSctrCoeff = dRayleighConst / Lambda4;
                // Total Rayleigh scattering coefficient is the integral of angular scattering coefficient in all directions
                f4TotalRayleighSctrCoeff[WaveNum] = static_cast<float>( dSctrCoeff );
            }
            // Angular scattering coefficient is essentially volumetric scattering coefficient multiplied by the
            // normalized phase function
            // p(Theta) = 3/(16*Pi) * (1 + cos^2(Theta))
            // f4AngularRayleighSctrCoeff contains all the terms exepting 1 + cos^2(Theta):
            f4AngularRayleighSctrCoeff[WaveNum] = static_cast<float>( 3.0 / (16.0*D3DX_PI) * dSctrCoeff );
            // f4AngularRayleighSctrCoeff[WaveNum] = f4TotalRayleighSctrCoeff[WaveNum] * p(Theta)
        }
        // Air molecules do not absorb light, so extinction coefficient is only caused by out-scattering
        f4RayleighExtinctionCoeff = f4TotalRayleighSctrCoeff;
    }

    // Calculate angular and total scattering coefficients for Mie scattering:
    {
        D3DXVECTOR4 &f4AngularMieSctrCoeff = m_MediaParams.f4AngularMieSctrCoeff;
        D3DXVECTOR4 &f4TotalMieSctrCoeff = m_MediaParams.f4TotalMieSctrCoeff;
        D3DXVECTOR4 &f4MieExtinctionCoeff = m_MediaParams.f4MieExtinctionCoeff;
        
        if( m_PostProcessingAttribs.m_bUseCustomSctrCoeffs )
        {
            f4TotalMieSctrCoeff = m_PostProcessingAttribs.m_f4CustomMieBeta * m_PostProcessingAttribs.m_fAerosolDensityScale;
        }
        else
        {
            const bool bUsePreethamMethod = false;
            if( bUsePreethamMethod )
            {
                // Values for K came from the table 2 in the "A practical Analytic Model 
                // for Daylight" by Preetham & Hoffman, p.28
                double K[] = 
                { 
                    0.68455,                //  K[650nm]
                    0.678781,               //  K[570nm]
                    (0.668532+0.669765)/2.0 // (K[470nm]+K[480nm])/2
                };

                assert( m_MediaParams.fTurbidity >= 1.f );
        
                // Beta is an Angstrom's turbidity coefficient and is approximated by:
                //float beta = 0.04608365822050f * m_fTurbidity - 0.04586025928522f; ???????

                double c = (0.6544*m_MediaParams.fTurbidity - 0.6510)*1E-16; // concentration factor
                const double v = 4; // Junge's exponent
        
                double dTotalMieBetaTerm = 0.434 * c * D3DX_PI * pow(2.0*D3DX_PI, v-2);

                for(int WaveNum = 0; WaveNum < 3; WaveNum++)
                {
                    double Lambdav_minus_2 = pow( dWaveLengths[WaveNum], v-2);
                    double dTotalMieSctrCoeff = dTotalMieBetaTerm * K[WaveNum] / Lambdav_minus_2;
                    f4TotalMieSctrCoeff[WaveNum]   = static_cast<float>( dTotalMieSctrCoeff );
                }
            
                //AtmScatteringAttribs.f4AngularMieSctrCoeff *= 0.02f;
                //AtmScatteringAttribs.f4TotalMieSctrCoeff *= 0.02f;
            }
            else
            {
                // [BN08] uses the following value (independent of wavelength) for Mie scattering coefficient: 2e-5
                // For g=0.76 and MieBetha=2e-5 [BN08] was able to reproduce the same luminance as given by the 
                // reference CIE sky light model 
                const float fMieBethaBN08 = 2e-5f * m_PostProcessingAttribs.m_fAerosolDensityScale;
                m_MediaParams.f4TotalMieSctrCoeff = D3DXVECTOR4(fMieBethaBN08, fMieBethaBN08, fMieBethaBN08, 0);
            }
        }

        for(int WaveNum = 0; WaveNum < 3; WaveNum++)
        {
            // Normalized to unity Cornette-Shanks phase function has the following form:
            // F(theta) = 1/(4*PI) * 3*(1-g^2) / (2*(2+g^2)) * (1+cos^2(theta)) / (1 + g^2 - 2g*cos(theta))^(3/2)
            // The angular scattering coefficient is the volumetric scattering coefficient multiplied by the phase 
            // function. 1/(4*PI) is baked into the f4AngularMieSctrCoeff, the other terms are baked into f4CS_g
            f4AngularMieSctrCoeff[WaveNum] = f4TotalMieSctrCoeff[WaveNum]  / static_cast<float>(4.0 * D3DX_PI);
            // [BN08] also uses slight absorption factor which is 10% of scattering
            f4MieExtinctionCoeff[WaveNum] = f4TotalMieSctrCoeff[WaveNum] * (1.f + m_PostProcessingAttribs.m_fAerosolAbsorbtionScale);
        }
    }
    
    {
        // For g=0.76 and MieBetha=2e-5 [BN08] was able to reproduce the same luminance as is given by the 
        // reference CIE sky light model 
        // Cornette phase function (see Nishita et al. 93):
        // F(theta) = 1/(4*PI) * 3*(1-g^2) / (2*(2+g^2)) * (1+cos^2(theta)) / (1 + g^2 - 2g*cos(theta))^(3/2)
        // 1/(4*PI) is baked into the f4AngularMieSctrCoeff
        D3DXVECTOR4 &f4CS_g = m_MediaParams.f4CS_g;
        float f_g = m_MediaParams.m_fAerosolPhaseFuncG;
        f4CS_g.x = 3*(1.f - f_g*f_g) / ( 2*(2.f + f_g*f_g) );
        f4CS_g.y = 1.f + f_g*f_g;
        f4CS_g.z = -2.f*f_g;
        f4CS_g.w = 1.f;
    }

    m_MediaParams.f4TotalExtinctionCoeff = m_MediaParams.f4RayleighExtinctionCoeff + m_MediaParams.f4MieExtinctionCoeff;

    if( pDeviceCtx && m_pcbMediaAttribs )
    {
        pDeviceCtx->UpdateSubresource(m_pcbMediaAttribs, 0, NULL, &m_MediaParams, 0, 0);
    }
}

void CLightSctrPostProcess :: RenderSun(SFrameAttribs &FrameAttribs)
{
    if( FrameAttribs.pLightAttribs->f4LightScreenPos.w <= 0 )
        return;

    if( !m_RenderSunTech.IsValid() )
    {
        m_RenderSunTech.SetDeviceAndContext(FrameAttribs.pd3dDevice, FrameAttribs.pd3dDeviceContext);
        m_RenderSunTech.CreateVGPShadersFromFile( m_strEffectPath, "SunVS", nullptr, "SunPS", nullptr );
        m_RenderSunTech.SetDS( m_pEnableDepthCmpEqDS );
        m_RenderSunTech.SetRS( m_pSolidFillNoCullRS );
        m_RenderSunTech.SetBS( m_pDefaultBS );
    }

    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(1, &FrameAttribs.ptex2DSrcColorBufferRTV, FrameAttribs.ptex2DSrcDepthBufferDSV);
    RenderQuad( FrameAttribs.pd3dDeviceContext, m_RenderSunTech);
    
    FrameAttribs.pd3dDeviceContext->OMSetRenderTargets(0, NULL, NULL);
}

void CLightSctrPostProcess :: CreateAmbientSkyLightTexture(ID3D11Device *pDevice, ID3D11DeviceContext *pContext)
{
    CRenderTechnique PrecomputeAmbientSkyLightTech;
    PrecomputeAmbientSkyLightTech.SetDeviceAndContext(pDevice, pContext);
    CD3DShaderMacroHelper Macros;
    Macros.AddShaderMacro( "NUM_RANDOM_SPHERE_SAMPLES", sm_iNumRandomSamplesOnSphere );
    Macros.Finalize();
    PrecomputeAmbientSkyLightTech.CreatePixelShaderFromFile( m_strEffectPath, "PrecomputeAmbientSkyLightPS", Macros );
    PrecomputeAmbientSkyLightTech.SetVS( m_pGenerateScreenSizeQuadVS );
    PrecomputeAmbientSkyLightTech.SetDS( m_pDisableDepthTestDS, 0 );
    PrecomputeAmbientSkyLightTech.SetRS( m_pSolidFillNoCullRS );
    PrecomputeAmbientSkyLightTech.SetBS( m_pDefaultBS );

    D3D11_TEXTURE2D_DESC AmbientSkyLightTexDesc = 
    {
        sm_iAmbientSkyLightTexDim,          //UINT Width;
        1,                                  //UINT Height;
        1,                                  //UINT MipLevels;
        1,                                  //UINT ArraySize;
        DXGI_FORMAT_R16G16B16A16_FLOAT,     //DXGI_FORMAT Format;
        {1,0},                              //DXGI_SAMPLE_DESC SampleDesc;
        D3D11_USAGE_DEFAULT,                //D3D11_USAGE Usage;
        D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,           //UINT BindFlags;
        0,                                  //UINT CPUAccessFlags;
        0,                                  //UINT MiscFlags;
    };

    m_ptex2DAmbientSkyLightSRV.Release();
    m_ptex2DAmbientSkyLightRTV.Release();
    CComPtr<ID3D11Texture2D> ptex2DAmbientSkyLight;
    // Create 2-D texture, shader resource and target view buffers on the device
    HRESULT hr;
    V( pDevice->CreateTexture2D( &AmbientSkyLightTexDesc, NULL, &ptex2DAmbientSkyLight) );
    V( pDevice->CreateShaderResourceView( ptex2DAmbientSkyLight, NULL, &m_ptex2DAmbientSkyLightSRV)  );
    V( pDevice->CreateRenderTargetView( ptex2DAmbientSkyLight, NULL, &m_ptex2DAmbientSkyLightRTV)  );
    
    CComPtr<ID3D11RenderTargetView> pOrigRTV;
    CComPtr<ID3D11DepthStencilView> pOrigDSV;
    pContext->OMGetRenderTargets(1, &pOrigRTV, &pOrigDSV);

    D3D11_VIEWPORT OrigViewPort;
    UINT iNumOldViewports = 1;
    pContext->RSGetViewports(&iNumOldViewports, &OrigViewPort);

    ID3D11Buffer *pCBs[] = {m_pcbPostProcessingAttribs, m_pcbMediaAttribs};
    pContext->VSSetConstantBuffers(0, _countof(pCBs), pCBs);
    pContext->PSSetConstantBuffers(0, _countof(pCBs), pCBs);

    ID3D11SamplerState *pSamplers[] = { m_psamLinearClamp, m_psamLinearBorder0, m_psamComparison, m_psamPointClamp };
    pContext->PSSetSamplers(0, _countof(pSamplers), pSamplers);

    if( !m_ptex2DOccludedNetDensityToAtmTopSRV )
    {
        CreatePrecomputedOpticalDepthTexture(pDevice, pContext);
    }
    
    if( !m_ptex3DSingleScatteringSRV )
    {
        CreatePrecomputedScatteringLUT(pDevice, pContext);
    }

    ID3D11ShaderResourceView *pSRVs[2] = {m_ptex3DMultipleScatteringSRV, m_ptex2DSphereRandomSamplingSRV};
    pContext->PSSetShaderResources(0, _countof(pSRVs), pSRVs);            

    pContext->OMSetRenderTargets(1, &m_ptex2DAmbientSkyLightRTV.p, nullptr);
    RenderQuad( pContext, PrecomputeAmbientSkyLightTech, sm_iAmbientSkyLightTexDim, 1);

    pContext->RSSetViewports(iNumOldViewports, &OrigViewPort);
    pContext->OMSetRenderTargets(1, &pOrigRTV.p, pOrigDSV);
}

ID3D11ShaderResourceView* CLightSctrPostProcess :: GetAmbientSkyLightSRV(ID3D11Device *pDevice, ID3D11DeviceContext *pContext)
{
    if( !m_ptex2DAmbientSkyLightSRV )
    {
        CreateAmbientSkyLightTexture(pDevice, pContext);
    }

    return m_ptex2DAmbientSkyLightSRV;
}
