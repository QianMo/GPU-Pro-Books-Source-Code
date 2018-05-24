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
#include "Common.fxh"
#include "CloudsCommon.fxh"

#ifndef DENSITY_GENERATION_METHOD
#   define DENSITY_GENERATION_METHOD 0
#endif

Texture3D<float>       g_tex3DNoise                 : register( t0 );
Texture3D<float>       g_tex3DSingleScattering      : register( t0 );
Texture3D<float>       g_tex3DMultipleScattering    : register( t1 );
Texture3D<float>       g_tex3DPrevSctrOrder         : register( t0 );
Texture3D<float>       g_tex3DGatheredScattering    : register( t0 );
Texture3D<float>       g_tex3DSingleScatteringLUT    : register( t0 );
Texture3D<float>       g_tex3DMultipleScatteringLUT  : register( t1 );

cbuffer cbPostProcessingAttribs : register( b0 )
{
    SGlobalCloudAttribs g_GlobalCloudAttribs;
};

SamplerState samLinearWrap : register( s1 );
SamplerState samPointWrap : register( s2 );

#define NUM_INTEGRATION_STEPS 64

struct SScreenSizeQuadVSOutput
{
    float4 m_f4Pos : SV_Position;
    float2 m_f2PosPS : PosPS; // Position in projection space [-1,1]x[-1,1]
};

// Vertex shader for generating screen-size quad
SScreenSizeQuadVSOutput ScreenSizeQuadVS(in uint VertexId : SV_VertexID)
{
    float4 MinMaxUV = float4(-1, -1, 1, 1);
    
    SScreenSizeQuadVSOutput Verts[4] = 
    {
        {float4(MinMaxUV.xy, 1.0, 1.0), MinMaxUV.xy}, 
        {float4(MinMaxUV.xw, 1.0, 1.0), MinMaxUV.xw},
        {float4(MinMaxUV.zy, 1.0, 1.0), MinMaxUV.zy},
        {float4(MinMaxUV.zw, 1.0, 1.0), MinMaxUV.zw}
    };

    return Verts[VertexId];
}

float GetRandomDensity(in float3 f3Pos, float fStartFreq, int iNumOctaves = 3, float fAmplitudeScale = 0.6)
{
    float fNoise = 0;
    float fAmplitude = 1;
    float fFreq = fStartFreq;
    for(int i=0; i < iNumOctaves; ++i)
    {
        fNoise += (g_tex3DNoise.SampleLevel(samLinearWrap, f3Pos*fFreq, 0) - 0.5) * fAmplitude;
        fFreq *= 1.7;
        fAmplitude *= fAmplitudeScale;
    }
    return fNoise;
}

float GetPyroSphereDensity(float3 f3CurrPos)
{
    float fDistToCenter = length(f3CurrPos);
    float3 f3NormalizedPos = f3CurrPos / fDistToCenter;
    float fNoise = GetRandomDensity(f3NormalizedPos, 0.15, 4, 0.8);
    float fDensity = fDistToCenter + 0.35*fNoise < 0.8 ? 1 : 0;
    return fDensity;
}

float GetMetabolDensity(in float r)
{
    float r2 = r*r;
    float r4 = r2*r2;
    float r6 = r4*r2;
    return saturate(-4.0/9.0 * r6 + 17.0/9.0 * r4 - 22.0/9.0 * r2 + 1);
}

float ComputeDensity(float3 f3CurrPos)
{
	float fDistToCenter = length(f3CurrPos);
    float fMetabolDensity = GetMetabolDensity(fDistToCenter);
	float fDensity = 0.f;
#if DENSITY_GENERATION_METHOD == 0
    fDensity = saturate( 1.0*saturate(fMetabolDensity) + 1*pow(fMetabolDensity,0.5)*(GetRandomDensity(f3CurrPos + 0.5, 0.15, 4, 0.7 )) );
#elif DENSITY_GENERATION_METHOD == 1
    fDensity = 1.0*saturate(fMetabolDensity) + 1.0*pow(fMetabolDensity,0.5)*(GetRandomDensity(f3CurrPos, 0.1,4,0.8)) > 0.1 ? 1 : 0;
    //fDensity = saturate(fMetabolDensity) - 2*pow(fMetabolDensity,0.5)*GetRandomDensity(f3CurrPos, 0.2, 4, 0.7) > 0.05 ? 1 : 0;
#elif DENSITY_GENERATION_METHOD == 2
    fDensity = GetPyroSphereDensity(f3CurrPos);
#endif
	return fDensity;
}

// This shader computes level 0 of the maximum density mip map
float2 PrecomputeOpticalDepthPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    float3 f3NormalizedStartPos, f3RayDir;
    OpticalDepthLUTCoordsToWorldParams( float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy), f3NormalizedStartPos, f3RayDir );
    
    // Intersect the view ray with the unit sphere:
    float2 f2RayIsecs;
    // f3NormalizedStartPos  is located exactly on the surface; slightly move start pos inside the sphere
    // to avoid precision issues
    GetRaySphereIntersection(f3NormalizedStartPos + f3RayDir*1e-4, f3RayDir, 0, 1.f, f2RayIsecs);
    
    if( f2RayIsecs.x > f2RayIsecs.y )
        return 0;

    float3 f3EndPos = f3NormalizedStartPos + f3RayDir * f2RayIsecs.y;
    float fNumSteps = NUM_INTEGRATION_STEPS;
    float3 f3Step = (f3EndPos - f3NormalizedStartPos) / fNumSteps;
    float fTotalDensity = 0;
    for(float fStepNum=0.5; fStepNum < fNumSteps; ++fStepNum)
    {
        float3 f3CurrPos = f3NormalizedStartPos + f3Step * fStepNum;
        float fDensity = ComputeDensity(f3CurrPos);
        fTotalDensity += fDensity;
    }
    return fTotalDensity / fNumSteps;
}


float PrecomputeSingleSctrPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    float4 f4LUTCoords = float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy);

    float3 f3EntryPointUSSpace, f3ViewRayUSSpace, f3LightDirUSSpace;
    ParticleScatteringLUTToWorldParams(f4LUTCoords, f3EntryPointUSSpace, f3ViewRayUSSpace, f3LightDirUSSpace, false);

    // Intersect view ray with the unit sphere:
    float2 f2RayIsecs;
    // f3NormalizedStartPos  is located exactly on the surface; slightly move the start pos inside the sphere
    // to avoid precision issues
    float3 f3BiasedEntryPoint = f3EntryPointUSSpace + f3ViewRayUSSpace*1e-4;
    GetRaySphereIntersection(f3BiasedEntryPoint, f3ViewRayUSSpace, 0, 1.f, f2RayIsecs);
    if( f2RayIsecs.y < f2RayIsecs.x )
        return 0;
    float3 f3EndPos = f3BiasedEntryPoint + f3ViewRayUSSpace * f2RayIsecs.y;

    float fNumSteps = NUM_INTEGRATION_STEPS;
    float3 f3Step = (f3EndPos - f3EntryPointUSSpace) / fNumSteps;
    float fStepLen = length(f3Step);
    float fCloudMassToCamera = 0;
    float fParticleRadius = g_GlobalCloudAttribs.fReferenceParticleRadius;
    float fInscattering = 0;
    for(float fStepNum=0.5; fStepNum < fNumSteps; ++fStepNum)
    {
        float3 f3CurrPos = f3EntryPointUSSpace + f3Step * fStepNum;
        float fCloudMassToLight = 0;
        GetRaySphereIntersection(f3CurrPos, f3LightDirUSSpace, 0, 1.f, f2RayIsecs);
        if( f2RayIsecs.y > f2RayIsecs.x )
        {
			// Since we are using the light direction (not direction on light), we have to use 
			// the first intersection point:
            fCloudMassToLight = abs(f2RayIsecs.x) * fParticleRadius;
        }

        float fTotalLightAttenuation = exp( -g_GlobalCloudAttribs.fAttenuationCoeff * (fCloudMassToLight + fCloudMassToCamera) );
        fInscattering += fTotalLightAttenuation * g_GlobalCloudAttribs.fScatteringCoeff;
        fCloudMassToCamera += fStepLen * fParticleRadius;
    }

    return fInscattering * fStepLen * fParticleRadius;
}


float GatherScatteringPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    float4 f4LUTCoords = float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy);

    float3 f3PosUSSpace, f3ViewRayUSSpace, f3LightDirUSSpace;
    ParticleScatteringLUTToWorldParams(f4LUTCoords, f3PosUSSpace, f3ViewRayUSSpace, f3LightDirUSSpace, false);

    float3 f3LocalX, f3LocalY, f3LocalZ;
    ConstructLocalFrameXYZ(-normalize(f3PosUSSpace), f3LightDirUSSpace, f3LocalX, f3LocalY, f3LocalZ);

    float fGatheredScattering = 0;
    float fTotalSolidAngle = 0;
    const float fNumZenithAngles = VOL_SCATTERING_IN_PARTICLE_LUT_DIM.z;
    const float fNumAzimuthAngles = VOL_SCATTERING_IN_PARTICLE_LUT_DIM.y;
    const float fZenithSpan = PI;
    const float fAzimuthSpan = 2*PI;
    for(float ZenithAngleNum = 0.5; ZenithAngleNum < fNumZenithAngles; ++ZenithAngleNum)
        for(float AzimuthAngleNum = 0.5; AzimuthAngleNum < fNumAzimuthAngles; ++AzimuthAngleNum)
        {
            float ZenithAngle = ZenithAngleNum/fNumZenithAngles * fZenithSpan;
            float AzimuthAngle = (AzimuthAngleNum/fNumAzimuthAngles - 0.5) * fAzimuthSpan;
            float3 f3CurrDir = GetDirectionInLocalFrameXYZ(f3LocalX, f3LocalY, f3LocalZ, ZenithAngle, AzimuthAngle);
            float4 f4CurrDirLUTCoords = WorldParamsToParticleScatteringLUT(f3PosUSSpace, f3CurrDir, f3LightDirUSSpace, false);
            float fCurrDirScattering = 0;
            SAMPLE_4D_LUT(g_tex3DPrevSctrOrder, VOL_SCATTERING_IN_PARTICLE_LUT_DIM, f4CurrDirLUTCoords, 0, fCurrDirScattering);
            if( g_GlobalCloudAttribs.f4Parameter.w == 1 )
            {
                fCurrDirScattering *= HGPhaseFunc( dot(-f3CurrDir, f3LightDirUSSpace) );
            }
            fCurrDirScattering *= HGPhaseFunc( dot(f3CurrDir, f3ViewRayUSSpace), 0.7 );

            float fdZenithAngle = fZenithSpan / fNumZenithAngles;
            float fdAzimuthAngle = fAzimuthSpan / fNumAzimuthAngles * sin(ZenithAngle);
            float fDiffSolidAngle = fdZenithAngle * fdAzimuthAngle;
            fTotalSolidAngle += fDiffSolidAngle;
            fGatheredScattering += fCurrDirScattering * fDiffSolidAngle;
        }
    
    // Total solid angle should be 4*PI. Renormalize to fix discretization issues
    fGatheredScattering *= 4*PI / fTotalSolidAngle;

    return fGatheredScattering;
}


float ComputeScatteringOrderPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    float4 f4StartPointLUTCoords = float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.xy);

    float3 f3PosUSSpace, f3ViewRayUSSpace, f3LightDirUSSpace;
    ParticleScatteringLUTToWorldParams(f4StartPointLUTCoords, f3PosUSSpace, f3ViewRayUSSpace, f3LightDirUSSpace, false);

    // Intersect view ray with the unit sphere:
    float2 f2RayIsecs;
    // f3NormalizedStartPos  is located exactly on the surface; slightly move start pos inside the sphere
    // to avoid precision issues
    float3 f3BiasedPos = f3PosUSSpace + f3ViewRayUSSpace*1e-4;
    GetRaySphereIntersection(f3BiasedPos, f3ViewRayUSSpace, 0, 1.f, f2RayIsecs);
    if( f2RayIsecs.y < f2RayIsecs.x )
        return 0;

    float3 f3EndPos = f3BiasedPos + f3ViewRayUSSpace * f2RayIsecs.y;
    float fNumSteps = max(VOL_SCATTERING_IN_PARTICLE_LUT_DIM.w*2, NUM_INTEGRATION_STEPS)*2;
    float3 f3Step = (f3EndPos - f3PosUSSpace) / fNumSteps;
    float fStepLen = length(f3Step);
    float fCloudMassToCamera = 0;
    float fParticleRadius = g_GlobalCloudAttribs.fReferenceParticleRadius;
    float fInscattering = 0;

    float fPrevGatheredSctr = 0;
    SAMPLE_4D_LUT(g_tex3DGatheredScattering, VOL_SCATTERING_IN_PARTICLE_LUT_DIM, f4StartPointLUTCoords, 0, fPrevGatheredSctr);
	// Light attenuation == 1
    for(float fStepNum=1; fStepNum <= fNumSteps; ++fStepNum)
    {
        float3 f3CurrPos = f3PosUSSpace + f3Step * fStepNum;

        fCloudMassToCamera += fStepLen * fParticleRadius;
        float fAttenuationToCamera = exp( -g_GlobalCloudAttribs.fAttenuationCoeff * fCloudMassToCamera );

        float4 f4CurrDirLUTCoords = WorldParamsToParticleScatteringLUT(f3CurrPos, f3ViewRayUSSpace, f3LightDirUSSpace, false);
        float fGatheredScattering = 0;
        SAMPLE_4D_LUT(g_tex3DGatheredScattering, VOL_SCATTERING_IN_PARTICLE_LUT_DIM, f4CurrDirLUTCoords, 0, fGatheredScattering);
        fGatheredScattering *= fAttenuationToCamera;

        fInscattering += (fGatheredScattering + fPrevGatheredSctr) /2;
        fPrevGatheredSctr = fGatheredScattering;
    }

    return fInscattering * fStepLen * fParticleRadius * g_GlobalCloudAttribs.fScatteringCoeff;
}


float AccumulateMultipleScattering(SScreenSizeQuadVSOutput In) : SV_Target
{
    float3 f3LUTCoords = float3(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.x);
    float fMultipleSctr = g_tex3DPrevSctrOrder.SampleLevel(samPointWrap, f3LUTCoords, 0);
    return fMultipleSctr;
}

void RenderScatteringLUTSlicePS(SScreenSizeQuadVSOutput In,
                                out float fSingleSctr : SV_Target0,
                                out float fMultipleSctr : SV_Target1)
{
    // Scattering on the surface of the sphere is stored in the last 4D-slice
    float4 f4LUTCoords = float4(ProjToUV(In.m_f2PosPS), g_GlobalCloudAttribs.f4Parameter.x, 1 - 0.5/VOL_SCATTERING_IN_PARTICLE_LUT_DIM.w);
    // We only need directions into the sphere
    f4LUTCoords.z = (f4LUTCoords.z - 0.5/SRF_SCATTERING_IN_PARTICLE_LUT_DIM.z)/2.f + 0.5/VOL_SCATTERING_IN_PARTICLE_LUT_DIM.z;

    SAMPLE_4D_LUT(g_tex3DSingleScattering,   VOL_SCATTERING_IN_PARTICLE_LUT_DIM, f4LUTCoords, 0, fSingleSctr);
    SAMPLE_4D_LUT(g_tex3DMultipleScattering, VOL_SCATTERING_IN_PARTICLE_LUT_DIM, f4LUTCoords, 0, fMultipleSctr);
}
