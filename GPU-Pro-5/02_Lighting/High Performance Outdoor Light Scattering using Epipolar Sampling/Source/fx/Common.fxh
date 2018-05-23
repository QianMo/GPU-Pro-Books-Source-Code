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

#include "Structures.fxh"

#define FLT_MAX 3.402823466e+38f

#define RGB_TO_LUMINANCE float3(0.212671, 0.715160, 0.072169)

// Using static definitions instead of constant buffer variables is 
// more efficient because the compiler is able to optimize the code 
// more aggressively

#ifndef NUM_EPIPOLAR_SLICES
#   define NUM_EPIPOLAR_SLICES 1024
#endif

#ifndef MAX_SAMPLES_IN_SLICE
#   define MAX_SAMPLES_IN_SLICE 512
#endif

#ifndef SCREEN_RESLOUTION
#   define SCREEN_RESLOUTION float2(1024,768)
#endif

#define MIN_MAX_DATA_FORMAT float2

#ifndef CASCADE_PROCESSING_MODE
#   define CASCADE_PROCESSING_MODE CASCADE_PROCESSING_MODE_SINGLE_PASS
#endif

#ifndef USE_COMBINED_MIN_MAX_TEXTURE
#   define USE_COMBINED_MIN_MAX_TEXTURE 1
#endif

#ifndef EXTINCTION_EVAL_MODE
#   define EXTINCTION_EVAL_MODE EXTINCTION_EVAL_MODE_EPIPOLAR
#endif

#ifndef AUTO_EXPOSURE
#   define AUTO_EXPOSURE 1
#endif

cbuffer cbPostProcessingAttribs : register( b0 )
{
    SPostProcessingAttribs g_PPAttribs;
};

cbuffer cbParticipatingMediaScatteringParams : register( b1 )
{
    SAirScatteringAttribs g_MediaParams;
}

// Frame parameters
cbuffer cbCameraAttribs : register( b2 )
{
    SCameraAttribs g_CameraAttribs;
}

cbuffer cbLightParams : register( b3 )
{
    SLightAttribs g_LightAttribs;
}

cbuffer cbMiscDynamicParams : register( b4 )
{
    SMiscDynamicParams g_MiscParams;
}

SamplerState samLinearClamp : register( s0 )
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

Texture2D<float>  g_tex2DDepthBuffer            : register( t0 );
Texture2D<float>  g_tex2DCamSpaceZ              : register( t0 );
Texture2D<float4> g_tex2DSliceEndPoints         : register( t4 );
Texture2D<float2> g_tex2DCoordinates            : register( t1 );
Texture2D<float>  g_tex2DEpipolarCamSpaceZ      : register( t2 );
Texture2D<uint2>  g_tex2DInterpolationSource    : register( t6 );
Texture2DArray<float> g_tex2DLightSpaceDepthMap : register( t3 );
Texture2D<float4> g_tex2DSliceUVDirAndOrigin    : register( t6 );
Texture2D<MIN_MAX_DATA_FORMAT> g_tex2DMinMaxLightSpaceDepth  : register( t4 );
Texture2D<float3> g_tex2DInitialInsctrIrradiance: register( t5 );
Texture2D<float4> g_tex2DColorBuffer            : register( t1 );
Texture2D<float3> g_tex2DScatteredColor         : register( t3 );
Texture2D<float2> g_tex2DOccludedNetDensityToAtmTop : register( t5 );
Texture2D<float3> g_tex2DEpipolarExtinction     : register( t6 );
Texture3D<float3> g_tex3DSingleSctrLUT          : register( t7 );
Texture3D<float3> g_tex3DHighOrderSctrLUT       : register( t8 );
Texture3D<float3> g_tex3DMultipleSctrLUT        : register( t9 );
Texture2D<float3> g_tex2DSphereRandomSampling   : register( t1 );
Texture3D<float3> g_tex3DPreviousSctrOrder      : register( t0 );
Texture3D<float3> g_tex3DPointwiseSctrRadiance  : register( t0 );
Texture2D<float>  g_tex2DAverageLuminance       : register( t10 );
Texture2D<float>  g_tex2DLowResLuminance        : register( t0 );


float3 ProjSpaceXYZToWorldSpace(in float3 f3PosPS)
{
    // We need to compute depth before applying view-proj inverse matrix
    float fDepth = g_CameraAttribs.mProj[2][2] + g_CameraAttribs.mProj[3][2] / f3PosPS.z;
    float4 ReconstructedPosWS = mul( float4(f3PosPS.xy,fDepth,1), g_CameraAttribs.mViewProjInv );
    ReconstructedPosWS /= ReconstructedPosWS.w;
    return ReconstructedPosWS.xyz;
}

void GetRaySphereIntersection(in float3 f3RayOrigin,
                              in float3 f3RayDirection,
                              in float3 f3SphereCenter,
                              in float fSphereRadius,
                              out float2 f2Intersections)
{
    // http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
    f3RayOrigin -= f3SphereCenter;
    float A = dot(f3RayDirection, f3RayDirection);
    float B = 2 * dot(f3RayOrigin, f3RayDirection);
    float C = dot(f3RayOrigin,f3RayOrigin) - fSphereRadius*fSphereRadius;
    float D = B*B - 4*A*C;
    // If discriminant is negative, there are no real roots hence the ray misses the
    // sphere
    if( D<0 )
    {
        f2Intersections = -1;
    }
    else
    {
        D = sqrt(D);
        f2Intersections = float2(-B - D, -B + D) / (2*A); // A must be positive here!!
    }
}

void GetRaySphereIntersection2(in float3 f3RayOrigin,
                               in float3 f3RayDirection,
                               in float3 f3SphereCenter,
                               in float2 f2SphereRadius,
                               out float4 f4Intersections)
{
    // http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
    f3RayOrigin -= f3SphereCenter;
    float A = dot(f3RayDirection, f3RayDirection);
    float B = 2 * dot(f3RayOrigin, f3RayDirection);
    float2 C = dot(f3RayOrigin,f3RayOrigin) - f2SphereRadius*f2SphereRadius;
    float2 D = B*B - 4*A*C;
    // If discriminant is negative, there are no real roots hence the ray misses the
    // sphere
    float2 f2RealRootMask = (D.xy >= 0);
    D = sqrt( max(D,0) );
    f4Intersections =   f2RealRootMask.xxyy * float4(-B - D.x, -B + D.x, -B - D.y, -B + D.y) / (2*A) + 
                      (1-f2RealRootMask.xxyy) * float4(-1,-1,-1,-1);
}

float GetAverageSceneLuminance()
{
#if AUTO_EXPOSURE
    float fAveLogLum = g_tex2DAverageLuminance.Load( int3(0,0,0) );
#else
    float fAveLogLum =  0.1;
#endif
    fAveLogLum = max(0.05, fAveLogLum); // Average luminance is an approximation to the key of the scene
    return fAveLogLum;
}
