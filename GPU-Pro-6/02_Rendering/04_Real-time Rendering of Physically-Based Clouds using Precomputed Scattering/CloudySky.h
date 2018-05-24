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

#include <d3dx9math.h>

#include "Structures.fxh"
// Temporary:
#undef float2
#undef float3
#undef float4

#include <stdio.h>

#include "CPUT_DX11.h"
#include "CPUTMaterial.h"

#include <D3D11.h> // for D3D11_BUFFER_DESC
#include <xnamath.h> // for XMFLOAT

#include <time.h> // for srand(time)

#include "EarthHemisphere.h"
#include "ElevationDataSource.h"

struct FrameConstantBuffer
{
    XMFLOAT4 Eye;
    XMFLOAT4 LookAt;
    XMFLOAT4 Up;
    XMFLOAT4 LightDirection;  

	XMMATRIX  worldMatrix;
    XMMATRIX  viewMatrix;	
	XMMATRIX  projectionMatrix;
};

static const CPUTControlID ID_MAIN_PANEL = 10;
static const CPUTControlID ID_SCATTERING_ATTRIBS_PANEL = 20;
static const CPUTControlID ID_ADDITIONAL_ATTRIBS_PANEL = 30;
static const CPUTControlID ID_TONE_MAPPING_ATTRIBS_PANEL = 40;
static const CPUTControlID ID_HELP_TEXT_PANEL = 50;
static const CPUTControlID ID_IGNORE_CONTROL_ID = -1;
static const CPUTControlID CONTROL_PANEL_IDS[] = {ID_MAIN_PANEL, ID_SCATTERING_ATTRIBS_PANEL, ID_ADDITIONAL_ATTRIBS_PANEL, ID_TONE_MAPPING_ATTRIBS_PANEL};

enum CONTROL_IDS
{
    ID_SELECT_PANEL_COMBO = 100,
    ID_FULLSCREEN_BUTTON,
    ID_ENABLE_VSYNC,
    ID_ENABLE_LIGHT_SCATTERING,
    ID_ENABLE_LIGHT_SHAFTS,
    ID_ENABLE_CLOUDS,
    ID_SHAFTS_FROM_CLOUDS_MODE,
    ID_ANIMATE_SUN,
    ID_LIGHT_SCTR_TECHNIQUE,
    ID_NUM_INTEGRATION_STEPS,
    ID_NUM_EPIPOLAR_SLICES,
    ID_NUM_SAMPLES_IN_EPIPOLAR_SLICE,
    ID_INITIAL_SAMPLE_STEP_IN_EPIPOLAR_SLICE,
    ID_EPIPOLE_SAMPLING_DENSITY_FACTOR,
    ID_REFINEMENT_THRESHOLD,
    ID_SHOW_SAMPLING,
    ID_MIN_MAX_SHADOW_MAP_OPTIMIZATION,
    ID_OPTIMIZE_SAMPLE_LOCATIONS,
    ID_CORRECT_SCATTERING_AT_DEPTH_BREAKS,
    ID_SCATTERING_SCALE,
    ID_MIDDLE_GRAY,
    ID_WHITE_POINT,
    ID_LUM_SATURATION,
    ID_AUTO_EXPOSURE,
    ID_TONE_MAPPING_MODE,
    ID_LIGHT_ADAPTATION,
    ID_SHOW_DEPTH_BREAKS,   
    ID_SHOW_LIGHTING_ONLY_CHECK,
    ID_SHADOW_MAP_RESOLUTION,
    ID_CLOUD_DENSITY_MAP_RESOLUTION,
    ID_CLOUD_DOWNSCALE_FACTOR_DROPDOWN,
    ID_CLOUDINESS_SLIDER,
    ID_CLOUD_ALTITUDE_SLIDER,
    ID_CLOUD_THICKNESS_SLIDER,
    ID_CLOUD_SPEED_SLIDER,
    ID_NUM_PARTICLE_RINGS_SLIDER,
    ID_PARTICLE_RING_DIM_SLIDER,
    ID_MAX_PARTICLE_LAYERS_SLIDER,
    ID_CLOUD_TYPE_DROPDOWN,
    ID_VOLUMERTIC_BLENDING_CHECK,
    ID_PARTICLE_RENDERING_DISTANCE_SLIDER,
    ID_USE_CUSTOM_SCTR_COEFFS_CHECK,
    ID_RLGH_COLOR_BTN,
    ID_MIE_COLOR_BTN,
    ID_SINGLE_SCTR_MODE_DROPDOWN,
    ID_MULTIPLE_SCTR_MODE_DROPDOWN,
    ID_NUM_CASCADES_DROPDOWN,
    ID_SHOW_CASCADES_CHECK,
    ID_SMOOTH_SHADOWS_CHECK,
    ID_BEST_CASCADE_SEARCH_CHECK,
    ID_CASCADE_PARTITIONING_SLIDER,
    ID_CASCADE_PROCESSING_MODE_DROPDOWN,
    ID_FIRST_CASCADE_TO_RAY_MARCH_DROPDOWN,
    ID_REFINEMENT_CRITERION_DROPDOWN,
    ID_EXTINCTION_EVAL_MODE_DROPDOWN,
    ID_MIN_MAX_MIP_FORMAT_DROPDOWN,
    ID_AEROSOL_DENSITY_SCALE_SLIDER,
    ID_AEROSOL_ABSORBTION_SCALE_SLIDER,
    ID_TEXTLINES = 1000
};



// DirectX 11 Sample
//-----------------------------------------------------------------------------
class CCloudySkySample:public CPUT_DX11
{
public:
    CCloudySkySample();
    virtual ~CCloudySkySample();

    // Event handling
    virtual CPUTEventHandledCode HandleKeyboardEvent(CPUTKey key);
    virtual CPUTEventHandledCode HandleMouseEvent(int x, int y, int wheel, CPUTMouseState state);
    virtual void                 HandleCallbackEvent( CPUTEventID Event, CPUTControlID ControlID, CPUTControl* pControl );

    // 'callback' handlers for rendering events.  Derived from CPUT_DX11
    virtual void Create();
    virtual void Render(double deltaSeconds);
    virtual void Update(double deltaSeconds);
    virtual void ResizeWindow(UINT width, UINT height);
    
    void Shutdown();

    HRESULT ParseConfigurationFile( LPCWSTR ConfigFilePath );

private:

    HRESULT CreateShadowMap(ID3D11Device* pd3dDevice);
    void ReleaseShadowMap();

    HRESULT CreateCloudDensityMap(ID3D11Device* pd3dDevice);
    void ReleaseCloudDensityMap();

    HRESULT CreateTmpBackBuffAndDepthBuff(ID3D11Device* pd3dDevice);
    void ReleaseTmpBackBuffAndDepthBuff();
    
    void RenderShadowMap(ID3D11DeviceContext *pContext, 
                         SLightAttribs &LightAttribs,
                         float fTime);

    void Destroy();

    float GetSceneExtent();

    class CLightSctrPostProcess *m_pLightSctrPP;
    class CCloudsController *m_pCloudsController;

    CComPtr<ID3D11Buffer> m_pcbCameraAttribs;

    UINT m_uiShadowMapResolution, m_uiCloudDensityMapResolution;
    float m_fCascadePartitioningFactor;
    bool m_bEnableLightScattering;
    bool m_bAnimateSun;
    
    static const int m_iMinEpipolarSlices = 32;
    static const int m_iMaxEpipolarSlices = 2048;
    static const int m_iMinSamplesInEpipolarSlice = 32;
    static const int m_iMaxSamplesInEpipolarSlice = 2048;
    static const int m_iMaxEpipoleSamplingDensityFactor = 32;
    static const int m_iMinInitialSamplesInEpipolarSlice = 8;
    SPostProcessingAttribs m_PPAttribs;
    SGlobalCloudAttribs m_CloudAttribs;

    float m_fScatteringScale;

    std::vector< CComPtr<ID3D11DepthStencilView> > m_pShadowMapDSVs;
    CComPtr<ID3D11ShaderResourceView> m_pShadowMapSRV; 

    std::vector< CComPtr<ID3D11RenderTargetView> > m_pLiSpCloudTransparencyRTVs, m_pLiSpCloudMinMaxDepthRTVs;
    CComPtr<ID3D11ShaderResourceView> m_pLiSpCloudTransparencySRV, m_pLiSpCloudMinMaxDepthSRV; 

    CPUTRenderTargetColor*  m_pOffscreenRenderTarget;
    CPUTRenderTargetDepth*  m_pOffscreenDepth;

    CPUTCamera*           m_pDirectionalLightCamera;
    CPUTCamera*           m_pDirLightOrienationCamera;
    CPUTCameraController* mpCameraController;
    CPUTCameraController* m_pLightController;

    CPUTTimerWin          m_Timer;
    float                 m_fElapsedTime;
    float                 m_fCloudTime;
    float                 m_fCloudTimeScale;

    D3DXVECTOR4 m_f4LightColor;
    
    int m_iGUIMode;
    
    SRenderingParams m_TerrainRenderParams;
	std::wstring m_strRawDEMDataFile;
	std::wstring m_strMtrlMaskFile;
    std::wstring m_strTileTexPaths[CEarthHemsiphere::NUM_TILE_TEXTURES];
    std::wstring m_strNormalMapTexPaths[CEarthHemsiphere::NUM_TILE_TEXTURES];

	std::auto_ptr<CElevationDataSource> m_pElevDataSource;

    CEarthHemsiphere m_EarthHemisphere;

    D3DXMATRIX  m_CameraViewMatrix;
	D3DXVECTOR3 m_CameraPos;

    CComPtr<ID3D11Buffer> m_pcbLightAttribs;
    UINT m_uiBackBufferWidth, m_uiBackBufferHeight;

    CPUTDropdown* m_pSelectPanelDropDowns[_countof(CONTROL_PANEL_IDS)];
    UINT m_uiSelectedPanelInd;

    float m_fMinElevation, m_fMaxElevation;
private:
    CCloudySkySample(const CCloudySkySample&);
    const CCloudySkySample& operator = (const CCloudySkySample&);
};
