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

#ifndef _STRCUTURES_FXH_
#define _STRCUTURES_FXH_

#define PI 3.1415928f

#ifdef __cplusplus

#   define float2 D3DXVECTOR2
#   define float3 D3DXVECTOR3
#   define float4 D3DXVECTOR4
#   define uint UINT
#   define SEMANTIC( S )

#else

#   define BOOL bool // Do not use bool, because sizeof(bool)==1 !
#   define SEMANTIC( S ) : S

#endif

#ifdef __cplusplus
#   define CHECK_STRUCT_ALIGNMENT(s) static_assert( sizeof(s) % 16 == 0, "sizeof("#s") is not multiple of 16" );
#else
#   define CHECK_STRUCT_ALIGNMENT(s)
#endif



#define MAX_CASCADES 8
struct SCascadeAttribs
{
	float4 f4LightSpaceScale;
	float4 f4LightSpaceScaledBias;
    float4 f4StartEndZ;
    float4 f4LightProjSpaceFilterRadius;
};
#ifdef __cplusplus
static_assert( (sizeof(SCascadeAttribs) % 16) == 0, "sizeof(SCascadeAttribs) is not multiple of 16" );
#endif

struct SShadowMapAttribs
{
    // 0
#ifdef __cplusplus
    D3DXMATRIX mWorldToLightViewT; // Matrices in HLSL are COLUMN-major while D3DXMATRIX is ROW major
#else
    matrix mWorldToLightView;  // Transform from view space to light projection space
#endif
    // 16
    SCascadeAttribs Cascades[MAX_CASCADES];

#ifdef __cplusplus
    float fCascadeCamSpaceZEnd[MAX_CASCADES];
    D3DXMATRIX mWorldToShadowMapUVDepthT[MAX_CASCADES];
#else
	float4 f4CascadeCamSpaceZEnd[MAX_CASCADES/4];
    matrix mWorldToShadowMapUVDepth[MAX_CASCADES];
#endif

    // Do not use bool, because sizeof(bool)==1 !
	BOOL bVisualizeCascades;

    float3 f3Padding;
};
#ifdef __cplusplus
static_assert( (sizeof(SShadowMapAttribs) % 16) == 0, "sizeof(SShadowMapAttribs) is not multiple of 16" );
#endif


struct SLightAttribs
{
    float4 f4DirOnLight;
    float4 f4AmbientLight;
    float4 f4LightScreenPos;
    float4 f4ExtraterrestrialSunColor;

    BOOL bIsLightOnScreen;
    float fMaxShadowCamSpaceZ;
    float2 f2Dummy;

    SShadowMapAttribs ShadowAttribs;
};
CHECK_STRUCT_ALIGNMENT(SLightAttribs);

struct SCameraAttribs
{
    float4 f4CameraPos;            ///< Camera world position
    float fNearPlaneZ; 
    float fFarPlaneZ; // fNearPlaneZ < fFarPlaneZ
    float2 f2Dummy;
    
    float3 f3ViewDir;
    float fDummy;

    float4 f4ViewFrustumPlanes[6];

#ifdef __cplusplus
    D3DXMATRIX WorldViewProjT;
    D3DXMATRIX mViewT;
    D3DXMATRIX mProjT;
    D3DXMATRIX mViewProjInvT;
#else
    matrix WorldViewProj;
    matrix mView;
    matrix mProj;
    matrix mViewProjInv;
#endif
};
CHECK_STRUCT_ALIGNMENT(SCameraAttribs);

#define LIGHT_SCTR_TECHNIQUE_EPIPOLAR_SAMPLING 0
#define LIGHT_SCTR_TECHNIQUE_BRUTE_FORCE 1

#define CASCADE_PROCESSING_MODE_SINGLE_PASS 0
#define CASCADE_PROCESSING_MODE_MULTI_PASS 1
#define CASCADE_PROCESSING_MODE_MULTI_PASS_INST 2

#define REFINEMENT_CRITERION_DEPTH_DIFF 0
#define REFINEMENT_CRITERION_INSCTR_DIFF 1

// Extinction evaluation mode used when attenuating background
#define EXTINCTION_EVAL_MODE_PER_PIXEL 0// Evaluate extinction for each pixel using analytic formula 
                                        // by Eric Bruneton
#define EXTINCTION_EVAL_MODE_EPIPOLAR 1 // Render extinction in epipolar space and perform
                                        // bilateral filtering in the same manner as for
                                        // inscattering

#define SINGLE_SCTR_MODE_NONE 0
#define SINGLE_SCTR_MODE_INTEGRATION 1
#define SINGLE_SCTR_MODE_LUT 2

#define MULTIPLE_SCTR_MODE_NONE 0
#define MULTIPLE_SCTR_MODE_UNOCCLUDED 1
#define MULTIPLE_SCTR_MODE_OCCLUDED 2

#define TONE_MAPPING_MODE_EXP 0
#define TONE_MAPPING_MODE_REINHARD 1
#define TONE_MAPPING_MODE_REINHARD_MOD 2
#define TONE_MAPPING_MODE_UNCHARTED2 3
#define TONE_MAPPING_FILMIC_ALU 4
#define TONE_MAPPING_LOGARITHMIC 5
#define TONE_MAPPING_ADAPTIVE_LOG 6

#define SHAFTS_FROM_CLOUDS_NONE 0
#define SHAFTS_FROM_CLOUDS_SHADOW_MAP 1
#define SHAFTS_FROM_CLOUDS_TRANSPARENCY_MAP 2

struct SPostProcessingAttribs
{
    uint m_uiNumEpipolarSlices;
    uint m_uiMaxSamplesInSlice;
    uint m_uiInitialSampleStepInSlice;
    uint m_uiEpipoleSamplingDensityFactor;

    float m_fRefinementThreshold;
    // do not use bool, because sizeof(bool)==1 and as a result bool variables
    // will be incorrectly mapped on GPU constant buffer
    BOOL m_bShowSampling; 
    BOOL m_bCorrectScatteringAtDepthBreaks; 
    BOOL m_bShowDepthBreaks; 

    BOOL m_bShowLightingOnly;
    BOOL m_bOptimizeSampleLocations;
    BOOL m_bEnableLightShafts;
    uint m_uiInstrIntegralSteps;
    
    float2 m_f2ShadowMapTexelSize;
    uint m_uiShadowMapResolution;
    uint m_uiMinMaxShadowMapResolution;

    BOOL m_bUse1DMinMaxTree;
    float m_fMaxShadowMapStep;
    float m_fMiddleGray;
    uint m_uiLightSctrTechnique;

    int m_iNumCascades;
    int m_iFirstCascade;
    float m_fNumCascades;
    float m_fFirstCascade;

    uint m_uiCascadeProcessingMode;
    uint m_uiRefinementCriterion;
    BOOL m_bIs32BitMinMaxMipMap;
    uint m_uiMultipleScatteringMode;

    uint m_uiSingleScatteringMode;
    BOOL m_bAutoExposure;
    uint m_uiToneMappingMode;
    BOOL m_bLightAdaptation;
    
    float m_fWhitePoint;
    float m_fLuminanceSaturation;
    BOOL m_bEnableClouds;
    float m_fCloudAltitiude;

    uint m_uiShaftsFromCloudsMode;
    uint m_uiLiSpCldDensResolution;
    float m_fLiSpCldDensResolution;
    float fDummy;
    
    uint m_uiExtinctionEvalMode;
    BOOL m_bUseCustomSctrCoeffs;
    float m_fAerosolDensityScale;
    float m_fAerosolAbsorbtionScale;

    float4 m_f4CustomRlghBeta;
    float4 m_f4CustomMieBeta;

#ifdef __cplusplus
    SPostProcessingAttribs() : 
        m_uiNumEpipolarSlices(512),
        m_uiMaxSamplesInSlice(256),
        m_uiInitialSampleStepInSlice(16),
        // Note that sampling near the epipole is very cheap since only a few steps
        // required to perform ray marching
        m_uiEpipoleSamplingDensityFactor(2),
        m_fRefinementThreshold(0.03f),
        m_bShowSampling(FALSE),
        m_bCorrectScatteringAtDepthBreaks(FALSE),
        m_bShowDepthBreaks(FALSE),
        m_bShowLightingOnly(FALSE),
        m_bOptimizeSampleLocations(TRUE),
        m_bEnableLightShafts(TRUE),
        m_uiInstrIntegralSteps(30),
        m_bUse1DMinMaxTree(TRUE),
        m_fMaxShadowMapStep(16.f),
        m_f2ShadowMapTexelSize(0,0),
        m_uiMinMaxShadowMapResolution(0),
        m_fMiddleGray(0.18f),
        m_uiLightSctrTechnique(LIGHT_SCTR_TECHNIQUE_EPIPOLAR_SAMPLING),
        m_iNumCascades(0),
        m_iFirstCascade(1),
        m_fNumCascades(0),
        m_fFirstCascade(1),
        m_uiCascadeProcessingMode(CASCADE_PROCESSING_MODE_SINGLE_PASS),
        m_uiRefinementCriterion(REFINEMENT_CRITERION_INSCTR_DIFF),
        m_bIs32BitMinMaxMipMap(FALSE),
        m_uiMultipleScatteringMode(MULTIPLE_SCTR_MODE_UNOCCLUDED),
        m_uiSingleScatteringMode(SINGLE_SCTR_MODE_INTEGRATION),
        m_bAutoExposure(TRUE),
        m_uiToneMappingMode(TONE_MAPPING_MODE_UNCHARTED2),
        m_bLightAdaptation(TRUE),
        m_fWhitePoint(3.f),
        m_fLuminanceSaturation(1.f),
        m_bEnableClouds(TRUE),
        m_fCloudAltitiude(1000),
        m_uiShaftsFromCloudsMode(SHAFTS_FROM_CLOUDS_TRANSPARENCY_MAP),
        m_uiLiSpCldDensResolution(256),
        m_fLiSpCldDensResolution(256),
        m_uiExtinctionEvalMode(EXTINCTION_EVAL_MODE_EPIPOLAR),
        m_bUseCustomSctrCoeffs(FALSE),
        m_fAerosolDensityScale(1.f),
        m_fAerosolAbsorbtionScale(0.1f),
        m_f4CustomRlghBeta( 5.8e-6f, 13.5e-6f, 33.1e-6f, 0.f ),
        m_f4CustomMieBeta(2.0e-5f, 2.0e-5f, 2.0e-5f, 0.f)
        {}
#endif
};
CHECK_STRUCT_ALIGNMENT(SPostProcessingAttribs);

struct SAirScatteringAttribs
{
    // Angular Rayleigh scattering coefficient contains all the terms exepting 1 + cos^2(Theta):
    // Pi^2 * (n^2-1)^2 / (2*N) * (6+3*Pn)/(6-7*Pn)
    float4 f4AngularRayleighSctrCoeff;
    // Total Rayleigh scattering coefficient is the integral of angular scattering coefficient in all directions
    // and is the following:
    // 8 * Pi^3 * (n^2-1)^2 / (3*N) * (6+3*Pn)/(6-7*Pn)
    float4 f4TotalRayleighSctrCoeff;
    float4 f4RayleighExtinctionCoeff;

    // Note that angular scattering coefficient is essentially a phase function multiplied by the
    // total scattering coefficient
    float4 f4AngularMieSctrCoeff;
    float4 f4TotalMieSctrCoeff;
    float4 f4MieExtinctionCoeff;

    float4 f4TotalExtinctionCoeff;
    // Cornette-Shanks phase function (see Nishita et al. 93) normalized to unity has the following form:
    // F(theta) = 1/(4*PI) * 3*(1-g^2) / (2*(2+g^2)) * (1+cos^2(theta)) / (1 + g^2 - 2g*cos(theta))^(3/2)
    float4 f4CS_g; // x == 3*(1-g^2) / (2*(2+g^2))
                   // y == 1 + g^2
                   // z == -2*g

    float fEarthRadius;
    float fAtmTopHeight;
    float2 f2ParticleScaleHeight;
    
    float fTurbidity;
    float fAtmTopRadius;
    float m_fAerosolPhaseFuncG;
    float m_fDummy;


#ifdef __cplusplus
    SAirScatteringAttribs():        
        f2ParticleScaleHeight(7994.f, 1200.f),
        // Air molecules and aerosols are assumed to be distributed
        // between 6360 km and 6420 km
        fEarthRadius(6360000.f),
        fAtmTopHeight(80000.f),
        fTurbidity(1.02f),
        m_fAerosolPhaseFuncG(0.76f)
    {
        fAtmTopRadius = fEarthRadius + fAtmTopHeight;
    }
#endif
};

CHECK_STRUCT_ALIGNMENT(SAirScatteringAttribs);

struct SMiscDynamicParams
{
    float fMaxStepsAlongRay;   // Maximum number of steps during ray tracing
    float fCascadeInd;
    float2 f2WQ; // Used when pre-computing inscattering look-up table

    uint uiDepthSlice;
    float fElapsedTime;
    float2 f2Dummy;

#ifdef __cplusplus
    uint uiSrcMinMaxLevelXOffset;
    uint uiSrcMinMaxLevelYOffset;
    uint uiDstMinMaxLevelXOffset;
    uint uiDstMinMaxLevelYOffset;
#else
    uint4 ui4SrcDstMinMaxLevelOffset;
#endif
};
CHECK_STRUCT_ALIGNMENT(SMiscDynamicParams);

struct SGlobalCloudAttribs
{
    uint uiInnerRingDim;
    uint uiRingExtension;
    uint uiRingDimension;
    uint uiNumRings;

    uint uiMaxLayers;
    uint uiNumCells;
    uint uiMaxParticles;
    uint uiDownscaleFactor;

    float fCloudDensityThreshold;
    float fCloudThickness;
    float fCloudAltitude;
    float fParticleCutOffDist;

    float fTime;
    float fCloudVolumeDensity;
    float2 f2LiSpCloudDensityDim;

    uint uiBackBufferWidth;
    uint uiBackBufferHeight;
    uint uiDownscaledBackBufferWidth;
    uint uiDownscaledBackBufferHeight;

    float fBackBufferWidth;
    float fBackBufferHeight;
    float fDownscaledBackBufferWidth;
    float fDownscaledBackBufferHeight;

    float fTileTexWidth;
    float fTileTexHeight;
    uint uiLiSpFirstListIndTexDim;
    uint uiNumCascades;

    float4 f4Parameter;

    float fScatteringCoeff;
    float fAttenuationCoeff;
    uint uiNumParticleLayers;
    uint uiDensityGenerationMethod;
    
    BOOL bVolumetricBlending;
	uint uiParameter;
	uint uiDensityBufferScale;
    float fReferenceParticleRadius;

    float4 f4TilingFrustumPlanes[6];

#ifdef __cplusplus
    D3DXMATRIX mParticleTilingT; // Matrices in HLSL are COLUMN-major while D3DXMATRIX is ROW major
#else
    matrix mParticleTiling;  // Transform from view space to light projection space
#endif

#ifdef __cplusplus
    SGlobalCloudAttribs() : 
        uiInnerRingDim(128),
        uiRingExtension(4),
        uiRingDimension(uiRingExtension + uiInnerRingDim + uiRingExtension),
        uiNumRings(5),
        uiMaxLayers(4),
        uiNumCells(0),
        uiMaxParticles(0),
        uiDownscaleFactor(2),
        fCloudDensityThreshold(0.35f),
        fTime(0),
        fCloudThickness(700.f),
        fCloudAltitude(3000.f),
        fParticleCutOffDist(2e+5f),
        fCloudVolumeDensity(5e-3f),
        f4Parameter(0,0,0,0),
        f2LiSpCloudDensityDim(512,512),
        uiBackBufferWidth(1024),
        uiBackBufferHeight(768),
        uiDownscaledBackBufferWidth(uiBackBufferWidth/uiDownscaleFactor),
        uiDownscaledBackBufferHeight(uiBackBufferHeight/uiDownscaleFactor),
        fBackBufferWidth((float)uiBackBufferWidth),
        fBackBufferHeight((float)uiBackBufferHeight),
        fDownscaledBackBufferWidth((float)uiDownscaledBackBufferWidth),
        fDownscaledBackBufferHeight((float)uiDownscaledBackBufferHeight),
        fTileTexWidth(32),
        fTileTexHeight(32),
        uiLiSpFirstListIndTexDim(128),
        uiNumCascades(0),
        fScatteringCoeff(0.07f), // Typical scattering coefficient lies in the range 0.01 - 0.1 m^-1
        fAttenuationCoeff(fScatteringCoeff),
        uiNumParticleLayers(1),
        uiDensityGenerationMethod(0),
        bVolumetricBlending(TRUE),
		uiDensityBufferScale(2),
		fReferenceParticleRadius(200.f)
    {
    }
#endif
};
CHECK_STRUCT_ALIGNMENT(SGlobalCloudAttribs);

struct SCloudCellAttribs
{
    float3 f3Center;
    float fSize;

    float3 f3Normal;
    uint uiNumActiveLayers;

    float3 f3Tangent;
    float fDensity;

    float3 f3Bitangent;
    float fMorphFadeout;

    uint uiPackedLocation;
};

struct SParticleAttribs
{
    float3 f3Pos;
    float fSize;
    float fRndAzimuthBias;
    float fDensity;
};

struct SCloudParticleLighting
{
    float4 f4SunLight;
	float2 f2SunLightAttenuation; // x == Direct Sun Light Attenuation
								  // y == Indirect Sun Light Attenuation
    float4 f4AmbientLight;
};

struct SParticleListKnot
{
	uint uiParticleID;
	int iNextKnotInd;
};

static uint PackParticleIJRing(uint i, uint j, uint ring, uint layer)
{
    return i | (j<<12) | (ring<<24) | (layer<<28);
}

#ifdef __cplusplus
static void UnPackParticleIJRing(uint ID, uint &i, uint &j, uint &ring, uint &layer) 
#else
static void UnPackParticleIJRing(uint ID, out uint i, out uint j, out uint ring, out uint layer) 
#endif
{
    i=(ID)&((1<<12)-1); 
    j=((ID)>>12)&((1<<12)-1); 
    ring = ((ID)>>24) & ((1<<4)-1);
    layer = (ID)>>28;
}

static uint GetNumActiveLayers(int iMaxLayers, int iRing)
{
    return iMaxLayers;//(iMaxLayers + (1<<iRing)-1) >> iRing;
}

struct SParticleLayer
{
    float2 f2MinMaxDist;
    float fOpticalMass;
    float3 f3Color;
#ifdef __cplusplus
    SParticleLayer():
        f2MinMaxDist(+FLT_MAX, +FLT_MAX),
        fOpticalMass(0),
        f3Color(0,0,0)
    {}
#endif
};

struct SParticleIdAndDist
{
    uint uiID;
    float fDistToCamera;
};

#endif //_STRCUTURES_FXH_