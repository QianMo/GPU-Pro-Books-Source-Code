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

#include "Common.fxh"

#ifndef OPTIMIZE_SAMPLE_LOCATIONS
#   define OPTIMIZE_SAMPLE_LOCATIONS 1
#endif

#ifndef CORRECT_INSCATTERING_AT_DEPTH_BREAKS
#   define CORRECT_INSCATTERING_AT_DEPTH_BREAKS 0
#endif

//#define SHADOW_MAP_DEPTH_BIAS 1e-4

#ifndef TRAPEZOIDAL_INTEGRATION
#   define TRAPEZOIDAL_INTEGRATION 1
#endif

#ifndef ENABLE_LIGHT_SHAFTS
#   define ENABLE_LIGHT_SHAFTS 1
#endif

#ifndef IS_32BIT_MIN_MAX_MAP
#   define IS_32BIT_MIN_MAX_MAP 0
#endif

#ifndef SINGLE_SCATTERING_MODE
#   define SINGLE_SCATTERING_MODE SINGLE_SCTR_MODE_LUT
#endif

#ifndef MULTIPLE_SCATTERING_MODE
#   define MULTIPLE_SCATTERING_MODE MULTIPLE_SCTR_MODE_OCCLUDED
#endif

#ifndef PRECOMPUTED_SCTR_LUT_DIM
#   define PRECOMPUTED_SCTR_LUT_DIM float4(32,128,32,16)
#endif

#ifndef NUM_RANDOM_SPHERE_SAMPLES
#   define NUM_RANDOM_SPHERE_SAMPLES 128
#endif

#ifndef PERFORM_TONE_MAPPING
#   define PERFORM_TONE_MAPPING 1
#endif

#ifndef LOW_RES_LUMINANCE_MIPS
#   define LOW_RES_LUMINANCE_MIPS 7
#endif

#ifndef TONE_MAPPING_MODE
#   define TONE_MAPPING_MODE TONE_MAPPING_MODE_REINHARD_MOD
#endif

#ifndef LIGHT_ADAPTATION
#   define LIGHT_ADAPTATION 1
#endif

#ifndef SHAFTS_FROM_CLOUDS_MODE
#   define SHAFTS_FROM_CLOUDS_MODE SHAFTS_FROM_CLOUDS_TRANSPARENCY_MAP
#endif

#define INVALID_EPIPOLAR_LINE float4(-1000,-1000, -100, -100)

//--------------------------------------------------------------------------------------
// Texture samplers
//--------------------------------------------------------------------------------------

SamplerState samLinearBorder0 : register( s1 )
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Border;
    AddressV = Border;
    BorderColor = float4(0.0, 0.0, 0.0, 0.0);
};

SamplerComparisonState samComparison : register( s2 )
{
    Filter = COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    AddressU = Border;
    AddressV = Border;
    ComparisonFunc = GREATER;
    BorderColor = float4(0.0, 0.0, 0.0, 0.0);
};

SamplerState samPointClamp : register( s3 );

//--------------------------------------------------------------------------------------
// Depth stencil states
//--------------------------------------------------------------------------------------

// Depth stencil state disabling depth test
DepthStencilState DSS_NoDepthTest
{
    DepthEnable = false;
    DepthWriteMask = ZERO;
};

DepthStencilState DSS_NoDepthTestIncrStencil
{
    DepthEnable = false;
    DepthWriteMask = ZERO;
    STENCILENABLE = true;
    FRONTFACESTENCILFUNC = ALWAYS;
    BACKFACESTENCILFUNC = ALWAYS;
    FRONTFACESTENCILPASS = INCR;
    BACKFACESTENCILPASS = INCR;
};

DepthStencilState DSS_NoDepth_StEqual_IncrStencil
{
    DepthEnable = false;
    DepthWriteMask = ZERO;
    STENCILENABLE = true;
    FRONTFACESTENCILFUNC = EQUAL;
    BACKFACESTENCILFUNC = EQUAL;
    FRONTFACESTENCILPASS = INCR;
    BACKFACESTENCILPASS = INCR;
    FRONTFACESTENCILFAIL = KEEP;
    BACKFACESTENCILFAIL = KEEP;
};

DepthStencilState DSS_NoDepth_StEqual_KeepStencil
{
    DepthEnable = false;
    DepthWriteMask = ZERO;
    STENCILENABLE = true;
    FRONTFACESTENCILFUNC = EQUAL;
    BACKFACESTENCILFUNC = EQUAL;
    FRONTFACESTENCILPASS = KEEP;
    BACKFACESTENCILPASS = KEEP;
    FRONTFACESTENCILFAIL = KEEP;
    BACKFACESTENCILFAIL = KEEP;
};

//--------------------------------------------------------------------------------------
// Rasterizer states
//--------------------------------------------------------------------------------------

// Rasterizer state for solid fill mode with no culling
RasterizerState RS_SolidFill_NoCull
{
    FILLMODE = Solid;
    CullMode = NONE;
};


// Blend state disabling blending
BlendState NoBlending
{
    BlendEnable[0] = FALSE;
    BlendEnable[1] = FALSE;
    BlendEnable[2] = FALSE;
};

float GetCamSpaceZ(in float2 ScreenSpaceUV)
{
    return g_tex2DCamSpaceZ.SampleLevel(samLinearClamp, ScreenSpaceUV, 0);
}

 
float3 Uncharted2Tonemap(float3 x)
{
    // http://www.gdcvault.com/play/1012459/Uncharted_2__HDR_Lighting
    // http://filmicgames.com/archives/75 - the coefficients are from here
    float A = 0.15; // Shoulder Strength
    float B = 0.50; // Linear Strength
    float C = 0.10; // Linear Angle
    float D = 0.20; // Toe Strength
    float E = 0.02; // Toe Numerator
    float F = 0.30; // Toe Denominator
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F; // E/F = Toe Angle
}

float3 ToneMap(in float3 f3Color)
{
    float fAveLogLum = GetAverageSceneLuminance();
    
    //const float middleGray = 1.03 - 2 / (2 + log10(fAveLogLum+1));
    const float middleGray = g_PPAttribs.m_fMiddleGray;
    // Compute scale factor such that average luminance maps to middle gray
    float fLumScale = middleGray / fAveLogLum;
    
    f3Color = max(f3Color, 0);
    float fInitialPixelLum = max(dot(RGB_TO_LUMINANCE, f3Color), 1e-10);
    float fScaledPixelLum = fInitialPixelLum * fLumScale;
    float3 f3ScaledColor = f3Color * fLumScale;

    float whitePoint = g_PPAttribs.m_fWhitePoint;

#if TONE_MAPPING_MODE == TONE_MAPPING_MODE_EXP
    
    float  fToneMappedLum = 1.0 - exp( -fScaledPixelLum );
    return fToneMappedLum * pow(f3Color / fInitialPixelLum, g_PPAttribs.m_fLuminanceSaturation);

#elif TONE_MAPPING_MODE == TONE_MAPPING_MODE_REINHARD || TONE_MAPPING_MODE == TONE_MAPPING_MODE_REINHARD_MOD

    // http://www.cs.utah.edu/~reinhard/cdrom/tonemap.pdf
    // http://imdoingitwrong.wordpress.com/2010/08/19/why-reinhard-desaturates-my-blacks-3/
    // http://content.gpwiki.org/index.php/D3DBook:High-Dynamic_Range_Rendering

    float  L_xy = fScaledPixelLum;
#   if TONE_MAPPING_MODE == TONE_MAPPING_MODE_REINHARD
        float  fToneMappedLum = L_xy / (1 + L_xy);
#   else
	    float  fToneMappedLum = L_xy * (1 + L_xy / (whitePoint*whitePoint)) / (1 + L_xy);
#   endif
	return fToneMappedLum * pow(f3Color / fInitialPixelLum, g_PPAttribs.m_fLuminanceSaturation);

#elif TONE_MAPPING_MODE == TONE_MAPPING_MODE_UNCHARTED2

    // http://filmicgames.com/archives/75
    float ExposureBias = 2.0f;
    float3 curr = Uncharted2Tonemap(ExposureBias*f3ScaledColor);
    float3 whiteScale = 1.0f/Uncharted2Tonemap(whitePoint);
    return curr*whiteScale;

#elif TONE_MAPPING_MODE == TONE_MAPPING_FILMIC_ALU

    // http://www.gdcvault.com/play/1012459/Uncharted_2__HDR_Lighting
    float3 f3ToneMappedColor = max(0, f3ScaledColor - 0.004f);
    f3ToneMappedColor = (f3ToneMappedColor * (6.2f * f3ToneMappedColor + 0.5f)) / 
                        (f3ToneMappedColor * (6.2f * f3ToneMappedColor + 1.7f)+ 0.06f);
    // result has 1/2.2 gamma baked in
    return pow(f3ToneMappedColor, 2.2f);

#elif TONE_MAPPING_MODE == TONE_MAPPING_LOGARITHMIC
    
    // http://www.mpi-inf.mpg.de/resources/tmo/logmap/logmap.pdf
    float fToneMappedLum = log10(1 + fScaledPixelLum) / log10(1 + whitePoint);
	return fToneMappedLum * pow(f3Color / fInitialPixelLum, g_PPAttribs.m_fLuminanceSaturation);

#elif TONE_MAPPING_MODE == TONE_MAPPING_ADAPTIVE_LOG

    // http://www.mpi-inf.mpg.de/resources/tmo/logmap/logmap.pdf
    float Bias = 0.85;
    float fToneMappedLum = 
        1 / log10(1 + whitePoint) *
        log(1 + fScaledPixelLum) / log( 2 + 8 * pow( fScaledPixelLum / whitePoint, log(Bias) / log(0.5f)) );
	return fToneMappedLum * pow(f3Color / fInitialPixelLum, g_PPAttribs.m_fLuminanceSaturation);

#endif
}

float4 UpdateAverageLuminancePS() : SV_Target
{
#if LIGHT_ADAPTATION
    const float fAdaptationRate = 1.f;
    float fNewLuminanceWeight = 1 - exp( - fAdaptationRate * g_MiscParams.fElapsedTime );
#else
    float fNewLuminanceWeight = 1;
#endif
    return float4( exp( g_tex2DLowResLuminance.Load( int3(0,0,LOW_RES_LUMINANCE_MIPS-1) ) ), 0, 0, fNewLuminanceWeight );
}

float3 ProjSpaceXYToWorldSpace(in float2 f2PosPS)
{
    // We can sample camera space z texture using bilinear filtering
    float fCamSpaceZ = g_tex2DCamSpaceZ.SampleLevel(samLinearClamp, ProjToUV(f2PosPS), 0);
    return ProjSpaceXYZToWorldSpace(float3(f2PosPS, fCamSpaceZ));
}

float3 WorldSpaceToShadowMapUV(in float3 f3PosWS, in matrix mWorldToShadowMapUVDepth)
{
    float4 f4ShadowMapUVDepth = mul( float4(f3PosWS, 1), mWorldToShadowMapUVDepth );
    // Shadow map projection matrix is orthographic, so we do not need to divide by w
    //f4ShadowMapUVDepth.xyz /= f4ShadowMapUVDepth.w;
    
    // Applying depth bias results in light leaking through the opaque objects when looking directly
    // at the light source
    return f4ShadowMapUVDepth.xyz;
}

struct SScreenSizeQuadVSOutput
{
    float4 m_f4Pos : SV_Position;
    float2 m_f2PosPS : PosPS; // Position in projection space [-1,1]x[-1,1]
    float m_fInstID : InstanceID;
};

SScreenSizeQuadVSOutput GenerateScreenSizeQuadVS(in uint VertexId : SV_VertexID,
                                                 in uint InstID : SV_InstanceID)
{
    float4 MinMaxUV = float4(-1, -1, 1, 1);
    
    SScreenSizeQuadVSOutput Verts[4] = 
    {
        {float4(MinMaxUV.xy, 1.0, 1.0), MinMaxUV.xy, InstID}, 
        {float4(MinMaxUV.xw, 1.0, 1.0), MinMaxUV.xw, InstID},
        {float4(MinMaxUV.zy, 1.0, 1.0), MinMaxUV.zy, InstID},
        {float4(MinMaxUV.zw, 1.0, 1.0), MinMaxUV.zw, InstID}
    };

    return Verts[VertexId];
}

float ReconstructCameraSpaceZPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    float fDepth = g_tex2DDepthBuffer.Load( uint3(In.m_f4Pos.xy,0) );
    float fCamSpaceZ = g_CameraAttribs.mProj[3][2]/(fDepth - g_CameraAttribs.mProj[2][2]);
    return fCamSpaceZ;
};

technique11 ReconstructCameraSpaceZ
{
    pass
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_NoDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, ReconstructCameraSpaceZPS() ) );
    }
}

const float4 GetOutermostScreenPixelCoords()
{
    // The outermost visible screen pixels centers do not lie exactly on the boundary (+1 or -1), but are biased by
    // 0.5 screen pixel size inwards
    //
    //                                        2.0
    //    |<---------------------------------------------------------------------->|
    //
    //       2.0/Res
    //    |<--------->|
    //    |     X     |      X     |     X     |    ...    |     X     |     X     |
    //   -1     |                                                            |    +1
    //          |                                                            |
    //          |                                                            |
    //      -1 + 1.0/Res                                                  +1 - 1.0/Res
    //
    // Using shader macro is much more efficient than using constant buffer variable
    // because the compiler is able to optimize the code more aggressively
    // return float4(-1,-1,1,1) + float4(1, 1, -1, -1)/g_PPAttribs.m_f2ScreenResolution.xyxy;
    return float4(-1,-1,1,1) + float4(1, 1, -1, -1) / SCREEN_RESLOUTION.xyxy;
}

// When checking if a point is inside the screen, we must test against 
// the biased screen boundaries 
bool IsValidScreenLocation(in float2 f2XY)
{
    const float SAFETY_EPSILON = 0.2f;
    return all( abs(f2XY) <= 1.f - (1.f - SAFETY_EPSILON) / SCREEN_RESLOUTION.xy );
}

// This function computes entry point of the epipolar line given its exit point
//                  
//    g_LightAttribs.f4LightScreenPos
//       *
//        \
//         \  f2EntryPoint
//        __\/___
//       |   \   |
//       |    \  |
//       |_____\_|
//           | |
//           | f2ExitPoint
//           |
//        Exit boundary
float2 GetEpipolarLineEntryPoint(float2 f2ExitPoint)
{
    float2 f2EntryPoint;

    //if( IsValidScreenLocation(g_LightAttribs.f4LightScreenPos.xy) )
    if( g_LightAttribs.bIsLightOnScreen )
    {
        // If light source is on the screen, its location is entry point for each epipolar line
        f2EntryPoint = g_LightAttribs.f4LightScreenPos.xy;
    }
    else
    {
        // If light source is outside the screen, we need to compute intersection of the ray with
        // the screen boundaries
        
        // Compute direction from the light source to the exit point
        // Note that exit point must be located on shrinked screen boundary
        float2 f2RayDir = f2ExitPoint.xy - g_LightAttribs.f4LightScreenPos.xy;
        float fDistToExitBoundary = length(f2RayDir);
        f2RayDir /= fDistToExitBoundary;
        // Compute signed distances along the ray from the light position to all four boundaries
        // The distances are computed as follows using vector instructions:
        // float fDistToLeftBoundary   = abs(f2RayDir.x) > 1e-5 ? (-1 - g_LightAttribs.f4LightScreenPos.x) / f2RayDir.x : -FLT_MAX;
        // float fDistToBottomBoundary = abs(f2RayDir.y) > 1e-5 ? (-1 - g_LightAttribs.f4LightScreenPos.y) / f2RayDir.y : -FLT_MAX;
        // float fDistToRightBoundary  = abs(f2RayDir.x) > 1e-5 ? ( 1 - g_LightAttribs.f4LightScreenPos.x) / f2RayDir.x : -FLT_MAX;
        // float fDistToTopBoundary    = abs(f2RayDir.y) > 1e-5 ? ( 1 - g_LightAttribs.f4LightScreenPos.y) / f2RayDir.y : -FLT_MAX;
        
        // Note that in fact the outermost visible screen pixels do not lie exactly on the boundary (+1 or -1), but are biased by
        // 0.5 screen pixel size inwards. Using these adjusted boundaries improves precision and results in
        // smaller number of pixels which require inscattering correction
        float4 f4Boundaries = GetOutermostScreenPixelCoords();
        bool4 b4IsCorrectIntersectionFlag = abs(f2RayDir.xyxy) > 1e-5;
        float4 f4DistToBoundaries = (f4Boundaries - g_LightAttribs.f4LightScreenPos.xyxy) / (f2RayDir.xyxy + !b4IsCorrectIntersectionFlag);
        // Addition of !b4IsCorrectIntersectionFlag is required to prevent divison by zero
        // Note that such incorrect lanes will be masked out anyway

        // We now need to find first intersection BEFORE the intersection with the exit boundary
        // This means that we need to find maximum intersection distance which is less than fDistToBoundary
        // We thus need to skip all boundaries, distance to which is greater than the distance to exit boundary
        // Using -FLT_MAX as the distance to these boundaries will result in skipping them:
        b4IsCorrectIntersectionFlag = b4IsCorrectIntersectionFlag && ( f4DistToBoundaries < (fDistToExitBoundary - 1e-4) );
        f4DistToBoundaries = b4IsCorrectIntersectionFlag * f4DistToBoundaries + 
                            !b4IsCorrectIntersectionFlag * float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

        float fFirstIntersecDist = 0;
        fFirstIntersecDist = max(fFirstIntersecDist, f4DistToBoundaries.x);
        fFirstIntersecDist = max(fFirstIntersecDist, f4DistToBoundaries.y);
        fFirstIntersecDist = max(fFirstIntersecDist, f4DistToBoundaries.z);
        fFirstIntersecDist = max(fFirstIntersecDist, f4DistToBoundaries.w);
        
        // The code above is equivalent to the following lines:
        // fFirstIntersecDist = fDistToLeftBoundary   < fDistToBoundary-1e-4 ? max(fFirstIntersecDist, fDistToLeftBoundary)   : fFirstIntersecDist;
        // fFirstIntersecDist = fDistToBottomBoundary < fDistToBoundary-1e-4 ? max(fFirstIntersecDist, fDistToBottomBoundary) : fFirstIntersecDist;
        // fFirstIntersecDist = fDistToRightBoundary  < fDistToBoundary-1e-4 ? max(fFirstIntersecDist, fDistToRightBoundary)  : fFirstIntersecDist;
        // fFirstIntersecDist = fDistToTopBoundary    < fDistToBoundary-1e-4 ? max(fFirstIntersecDist, fDistToTopBoundary)    : fFirstIntersecDist;

        // Now we can compute entry point:
        f2EntryPoint = g_LightAttribs.f4LightScreenPos.xy + f2RayDir * fFirstIntersecDist;

        // For invalid rays, coordinates are outside [-1,1]x[-1,1] area
        // and such rays will be discarded
        //
        //       g_LightAttribs.f4LightScreenPos
        //             *
        //              \|
        //               \-f2EntryPoint
        //               |\
        //               | \  f2ExitPoint 
        //               |__\/___
        //               |       |
        //               |       |
        //               |_______|
        //
    }

    return f2EntryPoint;
}

float4 GenerateSliceEndpointsPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    float2 f2UV = ProjToUV(In.m_f2PosPS);

    // Note that due to the rasterization rules, UV coordinates are biased by 0.5 texel size.
    //
    //      0.5     1.5     2.5     3.5
    //   |   X   |   X   |   X   |   X   |     ....       
    //   0       1       2       3       4   f2UV * TexDim
    //   X - locations where rasterization happens
    //
    // We need to remove this offset. Also clamp to [0,1] to fix fp32 precision issues
    float fEpipolarSlice = saturate(f2UV.x - 0.5f / (float)NUM_EPIPOLAR_SLICES);

    // fEpipolarSlice now lies in the range [0, 1 - 1/NUM_EPIPOLAR_SLICES]
    // 0 defines location in exacatly left top corner, 1 - 1/NUM_EPIPOLAR_SLICES defines
    // position on the top boundary next to the top left corner
    uint uiBoundary = clamp(floor( fEpipolarSlice * 4 ), 0, 3);
    float fPosOnBoundary = frac( fEpipolarSlice * 4 );

    bool4 b4BoundaryFlags = bool4( uiBoundary.xxxx == uint4(0,1,2,3) );

    // Note that in fact the outermost visible screen pixels do not lie exactly on the boundary (+1 or -1), but are biased by
    // 0.5 screen pixel size inwards. Using these adjusted boundaries improves precision and results in
    // samller number of pixels which require inscattering correction
    float4 f4OutermostScreenPixelCoords = GetOutermostScreenPixelCoords();// xyzw = (left, bottom, right, top)

    // Check if there can definitely be no correct intersection with the boundary:
    //  
    //  Light.x <= LeftBnd    Light.y <= BottomBnd     Light.x >= RightBnd     Light.y >= TopBnd    
    //                                                                                 *             
    //          ____                 ____                    ____                   __/_             
    //        .|    |               |    |                  |    |  .*             |    |            
    //      .' |____|               |____|                  |____|.'               |____|            
    //     *                           \                                                               
    //                                  *                                                  
    //     Left Boundary       Bottom Boundary           Right Boundary          Top Boundary 
    //
    bool4 b4IsInvalidBoundary = bool4( (g_LightAttribs.f4LightScreenPos.xyxy - f4OutermostScreenPixelCoords.xyzw) * float4(1,1,-1,-1) <= 0 );
    if( dot(b4IsInvalidBoundary, b4BoundaryFlags) )
        return INVALID_EPIPOLAR_LINE;
    // Additinal check above is required to eliminate false epipolar lines which can appear is shown below.
    // The reason is that we have to use some safety delta when performing check in IsValidScreenLocation() 
    // function. If we do not do this, we will miss valid entry points due to precision issues.
    // As a result there could appear false entry points which fall into the safety region, but in fact lie
    // outside the screen boundary:
    //
    //   LeftBnd-Delta LeftBnd           
    //                      false epipolar line
    //          |        |  /
    //          |        | /          
    //          |        |/         X - false entry point
    //          |        *
    //          |       /|
    //          |------X-|-----------  BottomBnd
    //          |     /  |
    //          |    /   |
    //          |___/____|___________ BottomBnd-Delta
    //          
    //          


    //             <------
    //   +1   0,1___________0.75
    //          |     3     |
    //        | |           | A
    //        | |0         2| |
    //        V |           | |
    //   -1     |_____1_____|
    //       0.25  ------>  0.5
    //
    //         -1          +1
    //

    //                                   Left             Bottom           Right              Top   
    float4 f4BoundaryXPos = float4(               0, fPosOnBoundary,                1, 1-fPosOnBoundary);
    float4 f4BoundaryYPos = float4( 1-fPosOnBoundary,              0,  fPosOnBoundary,                1);
    // Select the right coordinates for the boundary
    float2 f2ExitPointPosOnBnd = float2( dot(f4BoundaryXPos, b4BoundaryFlags), dot(f4BoundaryYPos, b4BoundaryFlags) );
    float2 f2ExitPoint = lerp(f4OutermostScreenPixelCoords.xy, f4OutermostScreenPixelCoords.zw, f2ExitPointPosOnBnd);
    // GetEpipolarLineEntryPoint() gets exit point on SHRINKED boundary
    float2 f2EntryPoint = GetEpipolarLineEntryPoint(f2ExitPoint);
    
#if OPTIMIZE_SAMPLE_LOCATIONS
    // If epipolar slice is not invisible, advance its exit point if necessary
    if( IsValidScreenLocation(f2EntryPoint) )
    {
        // Compute length of the epipolar line in screen pixels:
        float fEpipolarSliceScreenLen = length( (f2ExitPoint - f2EntryPoint) * SCREEN_RESLOUTION.xy / 2 );
        // If epipolar line is too short, update epipolar line exit point to provide 1:1 texel to screen pixel correspondence:
        f2ExitPoint = f2EntryPoint + (f2ExitPoint - f2EntryPoint) * max((float)MAX_SAMPLES_IN_SLICE / fEpipolarSliceScreenLen, 1);
    }
#endif

    return float4(f2EntryPoint, f2ExitPoint);
}


void GenerateCoordinateTexturePS(SScreenSizeQuadVSOutput In, 
                                 out float2 f2XY : SV_Target0,
                                 out float fCamSpaceZ : SV_Target1
#if ENABLE_CLOUDS
                               , out float fEpipolarCldTransp : SV_Target2
#endif
                                 )

{
    float4 f4SliceEndPoints = g_tex2DSliceEndPoints.Load( int3(In.m_f4Pos.y,0,0) );
    
    // If slice entry point is outside [-1,1]x[-1,1] area, the slice is completely invisible
    // and we can skip it from further processing.
    // Note that slice exit point can lie outside the screen, if sample locations are optimized
    if( !IsValidScreenLocation(f4SliceEndPoints.xy) )
    {
        // Discard invalid slices
        // Such slices will not be marked in the stencil and as a result will always be skipped
        discard;
    }

    float2 f2UV = ProjToUV(In.m_f2PosPS);

    // Note that due to the rasterization rules, UV coordinates are biased by 0.5 texel size.
    //
    //      0.5     1.5     2.5     3.5
    //   |   X   |   X   |   X   |   X   |     ....       
    //   0       1       2       3       4   f2UV * f2TexDim
    //   X - locations where rasterization happens
    //
    // We need remove this offset:
    float fSamplePosOnEpipolarLine = f2UV.x - 0.5f / (float)MAX_SAMPLES_IN_SLICE;
    // fSamplePosOnEpipolarLine is now in the range [0, 1 - 1/MAX_SAMPLES_IN_SLICE]
    // We need to rescale it to be in [0, 1]
    fSamplePosOnEpipolarLine *= (float)MAX_SAMPLES_IN_SLICE / ((float)MAX_SAMPLES_IN_SLICE-1.f);
    fSamplePosOnEpipolarLine = saturate(fSamplePosOnEpipolarLine);

    // Compute interpolated position between entry and exit points:
    f2XY = lerp(f4SliceEndPoints.xy, f4SliceEndPoints.zw, fSamplePosOnEpipolarLine);
    if( !IsValidScreenLocation(f2XY) )
    {
        // Discard pixels that fall behind the screen
        // This can happen if slice exit point was optimized
        discard;
    }

    // Compute camera space z for current location
    float2 f2ScrSpaceUV = ProjToUV(f2XY);
    fCamSpaceZ = GetCamSpaceZ( f2ScrSpaceUV );

#if ENABLE_CLOUDS
    // Sampling cloud transparency texture with linear filtering yields worse results
    // Suppose epipolar sample is located at the boundary of opaque cloud:
    //
    //    *************  cloud
    //    *************
    //           X
    //     .   .   .  .
    //
    // Linear filtering would return 0.5, which is wrong. The sample should be considered
    // as either completely transparent or completely opaque
    fEpipolarCldTransp = g_tex2DScrSpaceCloudTransparency.SampleLevel( samPointClamp, f2ScrSpaceUV, 0);
#endif
};


technique11 GenerateCoordinateTexture
{
    pass
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        // Increase stencil value for all valid rays
        SetDepthStencilState( DSS_NoDepthTestIncrStencil, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, GenerateCoordinateTexturePS() ) );
    }
}

static const float4 g_f4IncorrectSliceUVDirAndStart = float4(-10000, -10000, 0, 0);
float4 RenderSliceUVDirInShadowMapTexturePS(SScreenSizeQuadVSOutput In) : SV_Target
{
    uint uiSliceInd = In.m_f4Pos.x;
    // Load epipolar slice endpoints
    float4 f4SliceEndpoints = g_tex2DSliceEndPoints.Load(  uint3(uiSliceInd,0,0) );
    // All correct entry points are completely inside the [-1+1/W,1-1/W]x[-1+1/H,1-1/H] area
    if( !IsValidScreenLocation(f4SliceEndpoints.xy) )
        return g_f4IncorrectSliceUVDirAndStart;

    uint uiCascadeInd = In.m_f4Pos.y;
    matrix mWorldToShadowMapUVDepth = g_LightAttribs.ShadowAttribs.mWorldToShadowMapUVDepth[uiCascadeInd];

    // Reconstruct slice exit point position in world space
    float3 f3SliceExitWS = ProjSpaceXYZToWorldSpace( float3(f4SliceEndpoints.zw, g_LightAttribs.ShadowAttribs.Cascades[uiCascadeInd].f4StartEndZ.y) );
    // Transform it to the shadow map UV
    float2 f2SliceExitUV = WorldSpaceToShadowMapUV(f3SliceExitWS, mWorldToShadowMapUVDepth).xy;
    
    // Compute camera position in shadow map UV space
    float2 f2SliceOriginUV = WorldSpaceToShadowMapUV(g_CameraAttribs.f4CameraPos.xyz, mWorldToShadowMapUVDepth).xy;

    // Compute slice direction in shadow map UV space
    float2 f2SliceDir = f2SliceExitUV - f2SliceOriginUV;
    f2SliceDir /= max(abs(f2SliceDir.x), abs(f2SliceDir.y));
    
    float4 f4BoundaryMinMaxXYXY = float4(0,0,1,1) + float4(0.5, 0.5, -0.5, -0.5)*g_PPAttribs.m_f2ShadowMapTexelSize.xyxy;
    if( any( (f2SliceOriginUV.xyxy - f4BoundaryMinMaxXYXY) * float4( 1, 1, -1, -1) < 0 ) )
    {
        // If slice origin in UV coordinates falls beyond [0,1]x[0,1] region, we have
        // to continue the ray and intersect it with this rectangle
        //                  
        //    f2SliceOriginUV
        //       *
        //        \
        //         \  New f2SliceOriginUV
        //    1   __\/___
        //       |       |
        //       |       |
        //    0  |_______|
        //       0       1
        //           
        
        // First, compute signed distances from the slice origin to all four boundaries
        bool4 b4IsValidIsecFlag = abs(f2SliceDir.xyxy) > 1e-6;
        float4 f4DistToBoundaries = (f4BoundaryMinMaxXYXY - f2SliceOriginUV.xyxy) / (f2SliceDir.xyxy + !b4IsValidIsecFlag);

        //We consider only intersections in the direction of the ray
        b4IsValidIsecFlag = b4IsValidIsecFlag && (f4DistToBoundaries>0);
        // Compute the second intersection coordinate
        float4 f4IsecYXYX = f2SliceOriginUV.yxyx + f4DistToBoundaries * f2SliceDir.yxyx;
        
        // Select only these coordinates that fall onto the boundary
        b4IsValidIsecFlag = b4IsValidIsecFlag && (f4IsecYXYX >= f4BoundaryMinMaxXYXY.yxyx) && (f4IsecYXYX <= f4BoundaryMinMaxXYXY.wzwz);
        // Replace distances to all incorrect boundaries with the large value
        f4DistToBoundaries = b4IsValidIsecFlag * f4DistToBoundaries + 
                            !b4IsValidIsecFlag * float4(+FLT_MAX, +FLT_MAX, +FLT_MAX, +FLT_MAX);
        // Select the closest valid intersection
        float2 f2MinDist = min(f4DistToBoundaries.xy, f4DistToBoundaries.zw);
        float fMinDist = min(f2MinDist.x, f2MinDist.y);
        
        // Update origin
        f2SliceOriginUV = f2SliceOriginUV + fMinDist * f2SliceDir;
    }
    
    f2SliceDir *= g_PPAttribs.m_f2ShadowMapTexelSize;

    return float4(f2SliceDir, f2SliceOriginUV);
}

technique11 RenderSliceUVDirInShadowMapTexture
{
    pass p0
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        // Only interpolation samples will not be discarded and increase the stencil value
        SetDepthStencilState( DSS_NoDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, RenderSliceUVDirInShadowMapTexturePS() ) );
    }
}

// Note that min/max shadow map does not contain finest resolution level
// The first level it contains corresponds to step == 2
MIN_MAX_DATA_FORMAT InitializeMinMaxShadowMapPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    uint uiSliceInd;
    float fCascadeInd;
#if USE_COMBINED_MIN_MAX_TEXTURE
    fCascadeInd = floor(In.m_f4Pos.y / NUM_EPIPOLAR_SLICES);
    uiSliceInd = In.m_f4Pos.y - fCascadeInd * NUM_EPIPOLAR_SLICES;
    fCascadeInd += g_PPAttribs.m_fFirstCascade;
#else
    uiSliceInd = In.m_f4Pos.y;
    fCascadeInd = g_MiscParams.fCascadeInd;
#endif
    // Load slice direction in shadow map
    float4 f4SliceUVDirAndOrigin = g_tex2DSliceUVDirAndOrigin.Load( uint3(uiSliceInd, fCascadeInd, 0) );
    // Calculate current sample position on the ray
    float2 f2CurrUV = f4SliceUVDirAndOrigin.zw + f4SliceUVDirAndOrigin.xy * floor(In.m_f4Pos.x) * 2.f;
    
    float4 f4MinDepth = 1;
    float4 f4MaxDepth = 0;
    // Gather 8 depths which will be used for PCF filtering for this sample and its immediate neighbor 
    // along the epipolar slice
    // Note that if the sample is located outside the shadow map, Gather() will return 0 as 
    // specified by the samLinearBorder0. As a result volumes outside the shadow map will always be lit
    for( float i=0; i<=1; ++i )
    {
        float4 f4Depths = g_tex2DLightSpaceDepthMap.Gather(samLinearBorder0, float3(f2CurrUV + i * f4SliceUVDirAndOrigin.xy, fCascadeInd) );
        f4MinDepth = min(f4MinDepth, f4Depths);
        f4MaxDepth = max(f4MaxDepth, f4Depths);
    }

    f4MinDepth.xy = min(f4MinDepth.xy, f4MinDepth.zw);
    f4MinDepth.x = min(f4MinDepth.x, f4MinDepth.y);

    f4MaxDepth.xy = max(f4MaxDepth.xy, f4MaxDepth.zw);
    f4MaxDepth.x = max(f4MaxDepth.x, f4MaxDepth.y);
#if !IS_32BIT_MIN_MAX_MAP
    const float R16_UNORM_PRECISION = 1.f / (float)(1<<16);
    f4MinDepth.x = floor(f4MinDepth.x/R16_UNORM_PRECISION)*R16_UNORM_PRECISION;
    f4MaxDepth.x =  ceil(f4MaxDepth.x/R16_UNORM_PRECISION)*R16_UNORM_PRECISION;
#endif
    return float2(f4MinDepth.x, f4MaxDepth.x);
}

// 1D min max mip map is arranged as follows:
//
//    g_MiscParams.ui4SrcDstMinMaxLevelOffset.x
//     |
//     |      g_MiscParams.ui4SrcDstMinMaxLevelOffset.z
//     |_______|____ __
//     |       |    |  |
//     |       |    |  |
//     |       |    |  |
//     |       |    |  |
//     |_______|____|__|
//     |<----->|<-->|
//         |     |
//         |    uiMinMaxShadowMapResolution/
//      uiMinMaxShadowMapResolution/2
//                         
MIN_MAX_DATA_FORMAT ComputeMinMaxShadowMapLevelPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    uint2 uiDstSampleInd = uint2(In.m_f4Pos.xy);
    uint2 uiSrcSample0Ind = uint2(g_MiscParams.ui4SrcDstMinMaxLevelOffset.x + (uiDstSampleInd.x - g_MiscParams.ui4SrcDstMinMaxLevelOffset.z)*2, uiDstSampleInd.y);
    uint2 uiSrcSample1Ind = uiSrcSample0Ind + uint2(1,0);
    MIN_MAX_DATA_FORMAT fnMinMaxDepth0 = g_tex2DMinMaxLightSpaceDepth.Load( uint3(uiSrcSample0Ind,0) );
    MIN_MAX_DATA_FORMAT fnMinMaxDepth1 = g_tex2DMinMaxLightSpaceDepth.Load( uint3(uiSrcSample1Ind,0) );

    float2 f2MinMaxDepth;
    f2MinMaxDepth.x = min(fnMinMaxDepth0.x, fnMinMaxDepth1.x);
    f2MinMaxDepth.y = max(fnMinMaxDepth0.y, fnMinMaxDepth1.y);
    return f2MinMaxDepth;
}

technique11 BuildMinMaxMipMap
{
    pass PInitializeMinMaxShadowMap
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        // Only interpolation samples will not be discarded and increase the stencil value
        SetDepthStencilState( DSS_NoDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, InitializeMinMaxShadowMapPS() ) );
    }

    pass PComputeMinMaxShadowMapLevel
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        // Only interpolation samples will not be discarded and increase the stencil value
        SetDepthStencilState( DSS_NoDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, ComputeMinMaxShadowMapLevelPS() ) );
    }
}



// Note that like min/max shadow map, cloud density scan texture does not contain 
// the finest resolution level. The first level it contains corresponds to step == 2
float InitCldDensEpipolarScanPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    uint uiSliceInd;
    float fCascadeInd;
#if USE_COMBINED_MIN_MAX_TEXTURE
    fCascadeInd = floor(In.m_f4Pos.y / NUM_EPIPOLAR_SLICES);
    uiSliceInd = In.m_f4Pos.y - fCascadeInd * NUM_EPIPOLAR_SLICES;
    fCascadeInd += g_PPAttribs.m_fFirstCascade;
#else
    uiSliceInd = In.m_f4Pos.y;
    fCascadeInd = g_MiscParams.fCascadeInd;
#endif
    // Load slice direction in shadow map
    float4 f4SliceUVDirAndOrigin = g_tex2DSliceUVDirAndOrigin.Load( uint3(uiSliceInd, fCascadeInd, 0) );
    // Calculate current sample position on the ray
    float2 f2CurrUV = f4SliceUVDirAndOrigin.zw + f4SliceUVDirAndOrigin.xy * floor(In.m_f4Pos.x) * 2.f;
    
    float fTransparency = 0;
    for( float i=0; i<=1; ++i )
    {
        fTransparency += g_tex2DLiSpaceCloudTransparency.SampleLevel(samLinearClamp, float3(f2CurrUV + i * f4SliceUVDirAndOrigin.xy, fCascadeInd), 0 );
    }
    fTransparency /= 2.f;
    return fTransparency;
}

// Cloud density epipolar scan texture is arrange in the same way as 1D min/max tree
float ComputeCldDensEpipolarScanLevelPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    uint2 uiDstSampleInd = uint2(In.m_f4Pos.xy);
    uint2 uiSrcSample0Ind = uint2(g_MiscParams.ui4SrcDstMinMaxLevelOffset.x + (uiDstSampleInd.x - g_MiscParams.ui4SrcDstMinMaxLevelOffset.z)*2, uiDstSampleInd.y);
    uint2 uiSrcSample1Ind = uiSrcSample0Ind + uint2(1,0);
    float fTransp0 = g_tex2DLiSpCldDensityEpipolarScan.Load( uint3(uiSrcSample0Ind,0) );
    float fTransp1 = g_tex2DLiSpCldDensityEpipolarScan.Load( uint3(uiSrcSample1Ind,0) );
    return (fTransp0 + fTransp1)/2.f;
}


void MarkRayMarchingSamplesInStencilPS(SScreenSizeQuadVSOutput In)
{
    uint2 ui2InterpolationSources = g_tex2DInterpolationSource.Load( uint3(In.m_f4Pos.xy,0) );
    // Ray marching samples are interpolated from themselves, so it is easy to detect them:
    if( ui2InterpolationSources.x != ui2InterpolationSources.y )
          discard;
}

technique11 MarkRayMarchingSamplesInStencil
{
    pass
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        // Only interpolation samples will not be discarded and increase the stencil value
        SetDepthStencilState( DSS_NoDepth_StEqual_IncrStencil, 1 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, MarkRayMarchingSamplesInStencilPS() ) );
    }
}

float3 InterpolateIrradiancePS(SScreenSizeQuadVSOutput In) : SV_Target
{
    uint uiSampleInd = In.m_f4Pos.x;
    uint uiSliceInd = In.m_f4Pos.y;
    // Get interpolation sources
    uint2 ui2InterpolationSources = g_tex2DInterpolationSource.Load( uint3(uiSampleInd, uiSliceInd, 0) );
    float fInterpolationPos = float(uiSampleInd - ui2InterpolationSources.x) / float( max(ui2InterpolationSources.y - ui2InterpolationSources.x,1) );

    float3 f3SrcInsctr0 = g_tex2DInitialInsctrIrradiance.Load( uint3(ui2InterpolationSources.x, uiSliceInd, 0) );
    float3 f3SrcInsctr1 = g_tex2DInitialInsctrIrradiance.Load( uint3(ui2InterpolationSources.y, uiSliceInd, 0));

    // Ray marching samples are interpolated from themselves
    return lerp(f3SrcInsctr0, f3SrcInsctr1, fInterpolationPos);
}

technique11 InterpolateIrradiance
{
    pass
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_NoDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, InterpolateIrradiancePS() ) );
    }
}

void UnwarpEpipolarInsctrImage( SScreenSizeQuadVSOutput In, 
                                in float fCamSpaceZ,
                                in float fScrSpaceCldTransp,
                                out float3 f3Inscattering,
                                out float3 f3Extinction )
{
    // Compute direction of the ray going from the light through the pixel
    float2 f2RayDir = normalize( In.m_f2PosPS - g_LightAttribs.f4LightScreenPos.xy );

    // Find, which boundary the ray intersects. For this, we will 
    // find which two of four half spaces the f2RayDir belongs to
    // Each of four half spaces is produced by the line connecting one of four
    // screen corners and the current pixel:
    //    ________________        _______'________           ________________           
    //   |'            . '|      |      '         |         |                |          
    //   | '       . '    |      |     '          |      .  |                |          
    //   |  '  . '        |      |    '           |        '|.        hs1    |          
    //   |   *.           |      |   *     hs0    |         |  '*.           |          
    //   |  '   ' .       |      |  '             |         |      ' .       |          
    //   | '        ' .   |      | '              |         |          ' .   |          
    //   |'____________ '_|      |'_______________|         | ____________ '_.          
    //                           '                                             '
    //                           ________________  .        '________________  
    //                           |             . '|         |'               | 
    //                           |   hs2   . '    |         | '              | 
    //                           |     . '        |         |  '             | 
    //                           | . *            |         |   *            | 
    //                         . '                |         |    '           | 
    //                           |                |         | hs3 '          | 
    //                           |________________|         |______'_________| 
    //                                                              '
    // The equations for the half spaces are the following:
    //bool hs0 = (In.m_f2PosPS.x - (-1)) * f2RayDir.y < f2RayDir.x * (In.m_f2PosPS.y - (-1));
    //bool hs1 = (In.m_f2PosPS.x -  (1)) * f2RayDir.y < f2RayDir.x * (In.m_f2PosPS.y - (-1));
    //bool hs2 = (In.m_f2PosPS.x -  (1)) * f2RayDir.y < f2RayDir.x * (In.m_f2PosPS.y -  (1));
    //bool hs3 = (In.m_f2PosPS.x - (-1)) * f2RayDir.y < f2RayDir.x * (In.m_f2PosPS.y -  (1));
    // Note that in fact the outermost visible screen pixels do not lie exactly on the boundary (+1 or -1), but are biased by
    // 0.5 screen pixel size inwards. Using these adjusted boundaries improves precision and results in
    // smaller number of pixels which require inscattering correction
    float4 f4Boundaries = GetOutermostScreenPixelCoords();//left, bottom, right, top
    float4 f4HalfSpaceEquationTerms = (In.m_f2PosPS.xxyy - f4Boundaries.xzyw/*float4(-1,1,-1,1)*/) * f2RayDir.yyxx;
    bool4 b4HalfSpaceFlags = f4HalfSpaceEquationTerms.xyyx < f4HalfSpaceEquationTerms.zzww;

    // Now compute mask indicating which of four sectors the f2RayDir belongs to and consiquently
    // which border the ray intersects:
    //    ________________ 
    //   |'            . '|         0 : hs3 && !hs0
    //   | '   3   . '    |         1 : hs0 && !hs1
    //   |  '  . '        |         2 : hs1 && !hs2
    //   |0  *.       2   |         3 : hs2 && !hs3
    //   |  '   ' .       |
    //   | '   1    ' .   |
    //   |'____________ '_|
    //
    bool4 b4SectorFlags = b4HalfSpaceFlags.wxyz && !b4HalfSpaceFlags.xyzw;
    // Note that b4SectorFlags now contains true (1) for the exit boundary and false (0) for 3 other

    // Compute distances to boundaries according to following lines:
    //float fDistToLeftBoundary   = abs(f2RayDir.x) > 1e-5 ? ( -1 - g_LightAttribs.f4LightScreenPos.x) / f2RayDir.x : -FLT_MAX;
    //float fDistToBottomBoundary = abs(f2RayDir.y) > 1e-5 ? ( -1 - g_LightAttribs.f4LightScreenPos.y) / f2RayDir.y : -FLT_MAX;
    //float fDistToRightBoundary  = abs(f2RayDir.x) > 1e-5 ? (  1 - g_LightAttribs.f4LightScreenPos.x) / f2RayDir.x : -FLT_MAX;
    //float fDistToTopBoundary    = abs(f2RayDir.y) > 1e-5 ? (  1 - g_LightAttribs.f4LightScreenPos.y) / f2RayDir.y : -FLT_MAX;
    float4 f4DistToBoundaries = ( f4Boundaries - g_LightAttribs.f4LightScreenPos.xyxy ) / (f2RayDir.xyxy + float4( abs(f2RayDir.xyxy)<1e-6 ) );
    // Select distance to the exit boundary:
    float fDistToExitBoundary = dot( b4SectorFlags, f4DistToBoundaries );
    // Compute exit point on the boundary:
    float2 f2ExitPoint = g_LightAttribs.f4LightScreenPos.xy + f2RayDir * fDistToExitBoundary;

    // Compute epipolar slice for each boundary:
    //if( LeftBoundary )
    //    fEpipolarSlice = 0.0  - (LeftBoudaryIntersecPoint.y   -   1 )/2 /4;
    //else if( BottomBoundary )
    //    fEpipolarSlice = 0.25 + (BottomBoudaryIntersecPoint.x - (-1))/2 /4;
    //else if( RightBoundary )
    //    fEpipolarSlice = 0.5  + (RightBoudaryIntersecPoint.y  - (-1))/2 /4;
    //else if( TopBoundary )
    //    fEpipolarSlice = 0.75 - (TopBoudaryIntersecPoint.x      - 1 )/2 /4;
    float4 f4EpipolarSlice = float4(0, 0.25, 0.5, 0.75) + 
        saturate( (f2ExitPoint.yxyx - f4Boundaries.wxyz)*float4(-1, +1, +1, -1) / (f4Boundaries.wzwz - f4Boundaries.yxyx) ) / 4.0;
    // Select the right value:
    float fEpipolarSlice = dot(b4SectorFlags, f4EpipolarSlice);

    // Now find two closest epipolar slices, from which we will interpolate
    // First, find index of the slice which precedes our slice
    // Note that 0 <= fEpipolarSlice <= 1, and both 0 and 1 refer to the first slice
    float fPrecedingSliceInd = min( floor(fEpipolarSlice*NUM_EPIPOLAR_SLICES), NUM_EPIPOLAR_SLICES-1 );

    // Compute EXACT texture coordinates of preceding and succeeding slices and their weights
    // Note that slice 0 is stored in the first texel which has exact texture coordinate 0.5/NUM_EPIPOLAR_SLICES
    // (search for "fEpipolarSlice = saturate(f2UV.x - 0.5f / (float)NUM_EPIPOLAR_SLICES)"):
    float fSrcSliceV[2];
    // Compute V coordinate to refer exactly the center of the slice row
    fSrcSliceV[0] = fPrecedingSliceInd/NUM_EPIPOLAR_SLICES + 0.5f/(float)NUM_EPIPOLAR_SLICES;
    // Use frac() to wrap around to the first slice from the next-to-last slice:
    fSrcSliceV[1] = frac( fSrcSliceV[0] + 1.f/(float)NUM_EPIPOLAR_SLICES );
        
    // Compute slice weights
    float fSliceWeights[2];
    fSliceWeights[1] = (fEpipolarSlice*NUM_EPIPOLAR_SLICES) - fPrecedingSliceInd;
    fSliceWeights[0] = 1 - fSliceWeights[1];

    f3Inscattering = 0;
    f3Extinction = 0;
    float fTotalWeight = 0;
    [unroll]
    for(int i=0; i<2; ++i)
    {
        // Load epipolar line endpoints
        float4 f4SliceEndpoints = g_tex2DSliceEndPoints.SampleLevel( samLinearClamp, float2(fSrcSliceV[i], 0.5), 0 );

        // Compute line direction on the screen
        float2 f2SliceDir = f4SliceEndpoints.zw - f4SliceEndpoints.xy;
        float fSliceLenSqr = dot(f2SliceDir, f2SliceDir);
        
        // Project current pixel onto the epipolar line
        float fSamplePosOnLine = dot((In.m_f2PosPS - f4SliceEndpoints.xy), f2SliceDir) / max(fSliceLenSqr, 1e-8);
        // Compute index of the slice on the line
        // Note that the first sample on the line (fSamplePosOnLine==0) is exactly the Entry Point, while 
        // the last sample (fSamplePosOnLine==1) is exactly the Exit Point
        // (search for "fSamplePosOnEpipolarLine *= (float)MAX_SAMPLES_IN_SLICE / ((float)MAX_SAMPLES_IN_SLICE-1.f)")
        float fSampleInd = fSamplePosOnLine * (float)(MAX_SAMPLES_IN_SLICE-1);
       
        // We have to manually perform bilateral filtering of the scattered radiance texture to
        // eliminate artifacts at depth discontinuities

        float fPrecedingSampleInd = floor(fSampleInd);
        // Get bilinear filtering weight
        float fUWeight = fSampleInd - fPrecedingSampleInd;
        // Get texture coordinate of the left source texel. Again, offset by 0.5 is essential
        // to align with the texel center
        float fPrecedingSampleU = (fPrecedingSampleInd + 0.5) / (float)(MAX_SAMPLES_IN_SLICE);
    
        float2 f2SctrColorUV = float2(fPrecedingSampleU, fSrcSliceV[i]);

        // Gather 4 camera space z values
        // Note that we need to bias f2SctrColorUV by 0.5 texel size to refer the location between all four texels and
        // get the required values for sure
        // The values in float4, which Gather() returns are arranged as follows:
        //   _______ _______
        //  |       |       |
        //  |   x   |   y   |
        //  |_______o_______|  o gather location
        //  |       |       |
        //  |   *w  |   z   |  * f2SctrColorUV
        //  |_______|_______|
        //  |<----->|
        //     1/f2ScatteredColorTexDim.x
        
        // x == g_tex2DEpipolarCamSpaceZ.SampleLevel(samPointClamp, f2SctrColorUV, 0, int2(0,1))
        // y == g_tex2DEpipolarCamSpaceZ.SampleLevel(samPointClamp, f2SctrColorUV, 0, int2(1,1))
        // z == g_tex2DEpipolarCamSpaceZ.SampleLevel(samPointClamp, f2SctrColorUV, 0, int2(1,0))
        // w == g_tex2DEpipolarCamSpaceZ.SampleLevel(samPointClamp, f2SctrColorUV, 0, int2(0,0))

        const float2 f2ScatteredColorTexDim = float2(MAX_SAMPLES_IN_SLICE, NUM_EPIPOLAR_SLICES);
        float2 f2SrcLocationsCamSpaceZ = g_tex2DEpipolarCamSpaceZ.Gather(samLinearClamp, f2SctrColorUV + float2(0.5, 0.5) / f2ScatteredColorTexDim.xy).wz;
        
        // Compute depth weights in a way that if the difference is less than the threshold, the weight is 1 and
        // the weights fade out to 0 as the difference becomes larger than the threshold:
        float2 f2MaxZ = max( f2SrcLocationsCamSpaceZ, max(fCamSpaceZ,1) );
        float2 f2DepthWeights = saturate( g_PPAttribs.m_fRefinementThreshold / max( abs(fCamSpaceZ-f2SrcLocationsCamSpaceZ)/f2MaxZ, g_PPAttribs.m_fRefinementThreshold ) );
        // Note that if the sample is located outside the [-1,1]x[-1,1] area, the sample is invalid and fCurrCamSpaceZ == fInvalidCoordinate
        // Depth weight computed for such sample will be zero
        f2DepthWeights = pow(f2DepthWeights, 4);

        float2 f2CldTransparencyWeights = 1;
#if ENABLE_CLOUDS
        // If clouds are enabled, also take cloud transparency into account
        // Performance note: this adds ~5-10% to the unwarping stage time
        float2 f2SrcLocationsCldTransp = g_tex2DEpipolarCloudTransparency.Gather(samLinearClamp, f2SctrColorUV + float2(0.5, 0.5) / f2ScatteredColorTexDim.xy).wz;
        float2 f2MaxTransp = max(f2SrcLocationsCldTransp, max(fScrSpaceCldTransp, 0.001));
        f2CldTransparencyWeights = saturate( g_PPAttribs.m_fRefinementThreshold / max( abs(fScrSpaceCldTransp-f2SrcLocationsCldTransp)/f2MaxTransp, g_PPAttribs.m_fRefinementThreshold ) );
        f2CldTransparencyWeights = pow(f2CldTransparencyWeights, 4);
        f2CldTransparencyWeights = max(f2CldTransparencyWeights, 0.01);
#endif

        // Multiply bilinear weights with the depth weights:
        float2 f2BilateralUWeights = float2(1-fUWeight, fUWeight) * f2DepthWeights * f2CldTransparencyWeights * fSliceWeights[i];
        // If the sample projection is behind [0,1], we have to discard this slice
        // We however must take into account the fact that if at least one sample from the two 
        // bilinear sources is correct, the sample can still be properly computed
        //        
        //            -1       0       1                  N-2     N-1      N              Sample index
        // |   X   |   X   |   X   |   X   |  ......   |   X   |   X   |   X   |   X   |
        //         1-1/(N-1)   0    1/(N-1)                        1   1+1/(N-1)          fSamplePosOnLine   
        //             |                                                   |
        //             |<-------------------Clamp range------------------->|                   
        //
        f2BilateralUWeights *= (abs(fSamplePosOnLine - 0.5) < 0.5 + 1.f / (MAX_SAMPLES_IN_SLICE-1));
        // We now need to compute the following weighted summ:
        //f3FilteredSliceCol = 
        //    f2BilateralUWeights.x * g_tex2DScatteredColor.SampleLevel(samPoint, f2SctrColorUV, 0, int2(0,0)) +
        //    f2BilateralUWeights.y * g_tex2DScatteredColor.SampleLevel(samPoint, f2SctrColorUV, 0, int2(1,0));

        // We will use hardware to perform bilinear filtering and get this value using single bilinear fetch:

        // Offset:                  (x=1,y=0)                (x=1,y=0)               (x=0,y=0)
        float fSubpixelUOffset = f2BilateralUWeights.y / max(f2BilateralUWeights.x + f2BilateralUWeights.y, 0.001);
        fSubpixelUOffset /= f2ScatteredColorTexDim.x;
        
        float3 f3FilteredSliceInsctr = 
            (f2BilateralUWeights.x + f2BilateralUWeights.y) * 
                g_tex2DScatteredColor.SampleLevel(samLinearClamp, f2SctrColorUV + float2(fSubpixelUOffset, 0), 0);
        f3Inscattering += f3FilteredSliceInsctr;

#if EXTINCTION_EVAL_MODE == EXTINCTION_EVAL_MODE_EPIPOLAR
        float3 f3FilteredSliceExtinction = 
            (f2BilateralUWeights.x + f2BilateralUWeights.y) * 
                g_tex2DEpipolarExtinction.SampleLevel(samLinearClamp, f2SctrColorUV + float2(fSubpixelUOffset, 0), 0);
        f3Extinction += f3FilteredSliceExtinction;
#endif

        // Update total weight
        fTotalWeight += dot(f2BilateralUWeights, 1);
    }

#if CORRECT_INSCATTERING_AT_DEPTH_BREAKS
    if( fTotalWeight < 1e-2 )
    {
        // Discarded pixels will keep 0 value in stencil and will be later
        // processed to correct scattering
        discard;
    }
#endif
    
    f3Inscattering /= fTotalWeight;
    f3Extinction /= fTotalWeight;
}

float2 GetDensityIntegralAnalytic(float r, float mu, float d);
float3 GetExtinction(in float3 f3StartPos, in float3 f3EndPos);
float3 GetExtinction(in float3 f3StartPos, in float3 f3EyeDir, in float fRayLength);
float3 GetExtinctionUnverified(in float3 f3StartPos, in float3 f3EndPos, in float3 f3ViewDir, in float3 f3EarthCentre);

float3 ApplyInscatteredRadiancePS(SScreenSizeQuadVSOutput In) : SV_Target
{
    float2 f2UV = ProjToUV(In.m_f2PosPS);
    float fCamSpaceZ = GetCamSpaceZ( f2UV );
    float fCloudTransparency = 1;

#if ENABLE_CLOUDS
    fCloudTransparency    = g_tex2DScrSpaceCloudTransparency.SampleLevel( samLinearClamp, f2UV, 0);
    float3 f3CloudsColor  = g_tex2DScrSpaceCloudColor.SampleLevel( samLinearClamp, f2UV, 0).rgb;
    float fDistToCloud   = g_tex2DScrSpaceCloudMinMaxDist.SampleLevel( samLinearClamp, f2UV, 0).x;
#endif

    float3 f3Inscttering, f3Extinction;
    UnwarpEpipolarInsctrImage(In, fCamSpaceZ, fCloudTransparency, f3Inscttering, f3Extinction);

    float3 f3BackgroundColor = 0;
    [branch]
    if( !g_PPAttribs.m_bShowLightingOnly )
    {
        f3BackgroundColor = g_tex2DColorBuffer.SampleLevel( samPointClamp, f2UV, 0).rgb;
        // fFarPlaneZ is pre-multiplied with 0.999999f
        f3BackgroundColor *= (fCamSpaceZ > g_CameraAttribs.fFarPlaneZ) ? g_LightAttribs.f4ExtraterrestrialSunColor.rgb : 1;

#if EXTINCTION_EVAL_MODE == EXTINCTION_EVAL_MODE_PER_PIXEL
        float3 f3ReconstructedPosWS = ProjSpaceXYZToWorldSpace(float3(In.m_f2PosPS.xy, fCamSpaceZ));
        f3Extinction = GetExtinction(g_CameraAttribs.f4CameraPos.xyz, f3ReconstructedPosWS);
#endif
        f3BackgroundColor *= f3Extinction;
    }

#if ENABLE_CLOUDS
    float3 f3CloudExtinction = 1;
    [branch]
    if( fDistToCloud < +FLT_MAX*0.9 )
    {
        float3 f3ReconstructedPosWS = ProjSpaceXYZToWorldSpace(float3(In.m_f2PosPS.xy, fCamSpaceZ));
        float3 f3ViewDir = normalize(f3ReconstructedPosWS - g_CameraAttribs.f4CameraPos.xyz);
        f3CloudExtinction = GetExtinction(g_CameraAttribs.f4CameraPos.xyz, f3ViewDir, fDistToCloud);
    }
    f3CloudsColor *= f3CloudExtinction;
    f3BackgroundColor = f3BackgroundColor*fCloudTransparency + f3CloudsColor;
#endif

#if PERFORM_TONE_MAPPING
    return ToneMap(f3BackgroundColor + f3Inscttering);
#else
    const float DELTA = 0.00001;
    return log( max(DELTA, dot(f3BackgroundColor + f3Inscttering, RGB_TO_LUMINANCE)) );
#endif
}

technique11 ApplyInscatteredRadiance
{
    pass PUnwarpInsctr
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_NoDepthTestIncrStencil, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, ApplyInscatteredRadiancePS() ) );
    }
}


struct PassThroughVS_Output
{
    uint uiVertexID : VERTEX_ID;
};

PassThroughVS_Output PassThroughVS(uint VertexID : SV_VertexID)
{
    PassThroughVS_Output Out = {VertexID};
    return Out;
}



struct SRenderSamplePositionsGS_Output
{
    float4 f4PosPS : SV_Position;
    float3 f3Color : COLOR;
    float2 f2PosXY : XY;
    float4 f4QuadCenterAndSize : QUAD_CENTER_SIZE;
};
[maxvertexcount(4)]
void RenderSamplePositionsGS(point PassThroughVS_Output In[1], 
                             inout TriangleStream<SRenderSamplePositionsGS_Output> triStream )
{
    uint2 CoordTexDim;
    g_tex2DCoordinates.GetDimensions(CoordTexDim.x, CoordTexDim.y);
    uint2 TexelIJ = uint2( In[0].uiVertexID%CoordTexDim.x, In[0].uiVertexID/CoordTexDim.x );
    float2 f2QuadCenterPos = g_tex2DCoordinates.Load(int3(TexelIJ,0));

    uint2 ui2InterpolationSources = g_tex2DInterpolationSource.Load( uint3(TexelIJ,0) );
    bool bIsInterpolation = ui2InterpolationSources.x != ui2InterpolationSources.y;

    float2 f2QuadSize = (bIsInterpolation ? 1.f : 4.f) / SCREEN_RESLOUTION.xy;
    float4 MinMaxUV = float4(f2QuadCenterPos.x-f2QuadSize.x, f2QuadCenterPos.y - f2QuadSize.y, f2QuadCenterPos.x+f2QuadSize.x, f2QuadCenterPos.y + f2QuadSize.y);
    
    float3 f3Color = bIsInterpolation ? float3(0.5,0,0) : float3(1,0,0);
    float4 Verts[4] = 
    {
        float4(MinMaxUV.xy, 1.0, 1.0), 
        float4(MinMaxUV.xw, 1.0, 1.0),
        float4(MinMaxUV.zy, 1.0, 1.0),
        float4(MinMaxUV.zw, 1.0, 1.0)
    };

    for(int i=0; i<4; i++)
    {
        SRenderSamplePositionsGS_Output Out;
        Out.f4PosPS = Verts[i];
        Out.f2PosXY = Out.f4PosPS.xy;
        Out.f3Color = f3Color;
        Out.f4QuadCenterAndSize = float4(f2QuadCenterPos, f2QuadSize);
        triStream.Append( Out );
    }
}

float4 RenderSampleLocationsPS(SRenderSamplePositionsGS_Output In) : SV_Target
{
    return float4(In.f3Color, 1 - pow( length( (In.f2PosXY - In.f4QuadCenterAndSize.xy) / In.f4QuadCenterAndSize.zw),4) );
}

BlendState OverBS
{
    BlendEnable[0] = TRUE;
    RenderTargetWriteMask[0] = 0x0F;
    BlendOp = ADD;
    SrcBlend = SRC_ALPHA;
    DestBlend = INV_SRC_ALPHA;
    SrcBlendAlpha = ZERO;
    DestBlendAlpha = INV_SRC_ALPHA;
};

technique11 RenderSampleLocations
{
    pass
    {
        SetBlendState( OverBS, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_NoDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, PassThroughVS() ) );
        SetGeometryShader( CompileShader(gs_4_0, RenderSamplePositionsGS() ) );
        SetPixelShader( CompileShader(ps_5_0, RenderSampleLocationsPS() ) );
    }
}


float4 WorldParams2InsctrLUTCoords(float fHeight,
                                   float fCosViewZenithAngle,
                                   float fCosSunZenithAngle,
                                   float fCosSunViewAngle,
                                   in float4 f4PrevUVWQ = -1);

float3 LookUpPrecomputedScattering(float3 f3StartPoint, 
                                   float3 f3ViewDir, 
                                   float3 f3EarthCentre,
                                   float3 f3DirOnLight,
                                   in Texture3D<float3> tex3DScatteringLUT,
                                   inout float4 f4UVWQ)
{
    float3 f3EarthCentreToPointDir = f3StartPoint - f3EarthCentre;
    float fDistToEarthCentre = length(f3EarthCentreToPointDir);
    f3EarthCentreToPointDir /= fDistToEarthCentre;
    float fHeightAboveSurface = fDistToEarthCentre - EARTH_RADIUS;
    float fCosViewZenithAngle = dot( f3EarthCentreToPointDir, f3ViewDir    );
    float fCosSunZenithAngle  = dot( f3EarthCentreToPointDir, f3DirOnLight );
    float fCosSunViewAngle    = dot( f3ViewDir,               f3DirOnLight );

    // Provide previous look-up coordinates
    f4UVWQ = WorldParams2InsctrLUTCoords(fHeightAboveSurface, fCosViewZenithAngle,
                                         fCosSunZenithAngle, fCosSunViewAngle, 
                                         f4UVWQ);

    float3 f3UVW0; 
    f3UVW0.xy = f4UVWQ.xy;
    float fQ0Slice = floor(f4UVWQ.w * PRECOMPUTED_SCTR_LUT_DIM.w - 0.5);
    fQ0Slice = clamp(fQ0Slice, 0, PRECOMPUTED_SCTR_LUT_DIM.w-1);
    float fQWeight = (f4UVWQ.w * PRECOMPUTED_SCTR_LUT_DIM.w - 0.5) - fQ0Slice;
    fQWeight = max(fQWeight, 0);
    float2 f2SliceMinMaxZ = float2(fQ0Slice, fQ0Slice+1)/PRECOMPUTED_SCTR_LUT_DIM.w + float2(0.5,-0.5) / (PRECOMPUTED_SCTR_LUT_DIM.z*PRECOMPUTED_SCTR_LUT_DIM.w);
    f3UVW0.z =  (fQ0Slice + f4UVWQ.z) / PRECOMPUTED_SCTR_LUT_DIM.w;
    f3UVW0.z = clamp(f3UVW0.z, f2SliceMinMaxZ.x, f2SliceMinMaxZ.y);
    
    float fQ1Slice = min(fQ0Slice+1, PRECOMPUTED_SCTR_LUT_DIM.w-1);
    float fNextSliceOffset = (fQ1Slice - fQ0Slice) / PRECOMPUTED_SCTR_LUT_DIM.w;
    float3 f3UVW1 = f3UVW0 + float3(0,0,fNextSliceOffset);
    float3 f3Insctr0 = tex3DScatteringLUT.SampleLevel(samLinearClamp, f3UVW0, 0);
    float3 f3Insctr1 = tex3DScatteringLUT.SampleLevel(samLinearClamp, f3UVW1, 0);
    float3 f3Inscattering = lerp(f3Insctr0, f3Insctr1, fQWeight);

    return f3Inscattering;
}

float2 GetNetParticleDensity(in float fHeightAboveSurface,
                             in float fCosZenithAngle)
{
    float fRelativeHeightAboveSurface = fHeightAboveSurface / ATM_TOP_HEIGHT;
    return g_tex2DOccludedNetDensityToAtmTop.SampleLevel(samLinearClamp, float2(fRelativeHeightAboveSurface, fCosZenithAngle*0.5+0.5), 0).xy;
}

float2 GetNetParticleDensity(in float3 f3Pos,
                             in float3 f3EarthCentre,
                             in float3 f3RayDir)
{
    float3 f3EarthCentreToPointDir = f3Pos - f3EarthCentre;
    float fDistToEarthCentre = length(f3EarthCentreToPointDir);
    f3EarthCentreToPointDir /= fDistToEarthCentre;
    float fHeightAboveSurface = fDistToEarthCentre - EARTH_RADIUS;
    float fCosZenithAngle = dot( f3EarthCentreToPointDir, f3RayDir );
    return GetNetParticleDensity(fHeightAboveSurface, fCosZenithAngle);
}

void ApplyPhaseFunctions(inout float3 f3RayleighInscattering,
                         inout float3 f3MieInscattering,
                         in float cosTheta)
{
    f3RayleighInscattering *= g_MediaParams.f4AngularRayleighSctrCoeff.rgb * (1.0 + cosTheta*cosTheta);
    
    // Apply Cornette-Shanks phase function (see Nishita et al. 93):
    // F(theta) = 1/(4*PI) * 3*(1-g^2) / (2*(2+g^2)) * (1+cos^2(theta)) / (1 + g^2 - 2g*cos(theta))^(3/2)
    // f4CS_g = ( 3*(1-g^2) / (2*(2+g^2)), 1+g^2, -2g, 1 )
    float fDenom = rsqrt( dot(g_MediaParams.f4CS_g.yz, float2(1.f, cosTheta)) ); // 1 / (1 + g^2 - 2g*cos(theta))^(1/2)
    float fCornettePhaseFunc = g_MediaParams.f4CS_g.x * (fDenom*fDenom*fDenom) * (1 + cosTheta*cosTheta);
    f3MieInscattering *= g_MediaParams.f4AngularMieSctrCoeff.rgb * fCornettePhaseFunc;
}

// This function computes atmospheric properties in the given point
void GetAtmosphereProperties(in float3 f3Pos,
                             in float3 f3EarthCentre,
                             in float3 f3DirOnLight,
                             out float2 f2ParticleDensity,
                             out float2 f2NetParticleDensityToAtmTop)
{
    // Calculate the point height above the SPHERICAL Earth surface:
    float3 f3EarthCentreToPointDir = f3Pos - f3EarthCentre;
    float fDistToEarthCentre = length(f3EarthCentreToPointDir);
    f3EarthCentreToPointDir /= fDistToEarthCentre;
    float fHeightAboveSurface = fDistToEarthCentre - EARTH_RADIUS;

    f2ParticleDensity = exp( -fHeightAboveSurface / PARTICLE_SCALE_HEIGHT );

    // Get net particle density from the integration point to the top of the atmosphere:
    float fCosSunZenithAngleForCurrPoint = dot( f3EarthCentreToPointDir, f3DirOnLight );
    f2NetParticleDensityToAtmTop = GetNetParticleDensity(fHeightAboveSurface, fCosSunZenithAngleForCurrPoint);
}

// This function computes differential inscattering for the given particle densities 
// (without applying phase functions)
void ComputePointDiffInsctr(in float2 f2ParticleDensityInCurrPoint,
                            in float2 f2NetParticleDensityFromCam,
                            in float2 f2NetParticleDensityToAtmTop,
                            out float3 f3DRlghInsctr,
                            out float3 f3DMieInsctr)
{
    // Compute total particle density from the top of the atmosphere through the integraion point to camera
    float2 f2TotalParticleDensity = f2NetParticleDensityFromCam + f2NetParticleDensityToAtmTop;
        
    // Get optical depth
    float3 f3TotalRlghOpticalDepth = g_MediaParams.f4RayleighExtinctionCoeff.rgb * f2TotalParticleDensity.x;
    float3 f3TotalMieOpticalDepth  = g_MediaParams.f4MieExtinctionCoeff.rgb      * f2TotalParticleDensity.y;
        
    // And total extinction for the current integration point:
    float3 f3TotalExtinction = exp( -(f3TotalRlghOpticalDepth + f3TotalMieOpticalDepth) );

    f3DRlghInsctr = f2ParticleDensityInCurrPoint.x * f3TotalExtinction;
    f3DMieInsctr  = f2ParticleDensityInCurrPoint.y * f3TotalExtinction; 
}

void ComputeInsctrIntegral(in float3 f3RayStart,
                           in float3 f3RayEnd,
                           in float3 f3EarthCentre,
                           in float3 f3DirOnLight,
                           inout float2 f2NetParticleDensityFromCam,
                           inout float3 f3RayleighInscattering,
                           inout float3 f3MieInscattering,
                           uniform const float fNumSteps,
                           in float fCloudTransparency,
                           in float fDistToCloud)
{
    float3 f3Step = (f3RayEnd - f3RayStart) / fNumSteps;
    float fStepLen = length(f3Step);

#if TRAPEZOIDAL_INTEGRATION
    // For trapezoidal integration we need to compute some variables for the starting point of the ray
    float2 f2PrevParticleDensity = 0;
    float2 f2NetParticleDensityToAtmTop = 0;
    GetAtmosphereProperties(f3RayStart, f3EarthCentre, f3DirOnLight, f2PrevParticleDensity, f2NetParticleDensityToAtmTop);

    float3 f3PrevDiffRInsctr = 0, f3PrevDiffMInsctr = 0;
    ComputePointDiffInsctr(f2PrevParticleDensity, f2NetParticleDensityFromCam, f2NetParticleDensityToAtmTop, f3PrevDiffRInsctr, f3PrevDiffMInsctr);
#endif

#if ENABLE_CLOUDS
    fDistToCloud -= length(f3RayStart - g_CameraAttribs.f4CameraPos.xyz);
#endif

#if TRAPEZOIDAL_INTEGRATION
    // With trapezoidal integration, we will evaluate the function at the end of each section and 
    // compute area of a trapezoid
    for(float fStepNum = 1.f; fStepNum <= fNumSteps; fStepNum += 1.f)
#else
    // With stair-step integration, we will evaluate the function at the middle of each section and 
    // compute area of a rectangle
    for(float fStepNum = 0.5f; fStepNum < fNumSteps; fStepNum += 1.f)
#endif
    {
        float3 f3CurrPos = f3RayStart + f3Step * fStepNum;
        float2 f2ParticleDensity, f2NetParticleDensityToAtmTop;
        GetAtmosphereProperties(f3CurrPos, f3EarthCentre, f3DirOnLight, f2ParticleDensity, f2NetParticleDensityToAtmTop);

        // Accumulate net particle density from the camera to the integration point:
#if TRAPEZOIDAL_INTEGRATION
        f2NetParticleDensityFromCam += (f2PrevParticleDensity + f2ParticleDensity) * (fStepLen / 2.f);
        f2PrevParticleDensity = f2ParticleDensity;
#else
        f2NetParticleDensityFromCam += f2ParticleDensity * fStepLen;
#endif

        float3 f3DRlghInsctr, f3DMieInsctr;
        ComputePointDiffInsctr(f2ParticleDensity, f2NetParticleDensityFromCam, f2NetParticleDensityToAtmTop, f3DRlghInsctr, f3DMieInsctr);

        float fTransparency = 1;
#if ENABLE_CLOUDS
        fTransparency = lerp(1, fCloudTransparency, saturate( (fStepLen - fDistToCloud) / fStepLen) );
#endif

#if TRAPEZOIDAL_INTEGRATION
        f3RayleighInscattering += (f3DRlghInsctr + f3PrevDiffRInsctr) * (fStepLen / 2.f) * fTransparency;
        f3MieInscattering      += (f3DMieInsctr  + f3PrevDiffMInsctr) * (fStepLen / 2.f) * fTransparency;

        f3PrevDiffRInsctr = f3DRlghInsctr;
        f3PrevDiffMInsctr = f3DMieInsctr;
#else
        f3RayleighInscattering += f3DRlghInsctr * fStepLen * fTransparency;
        f3MieInscattering      += f3DMieInsctr * fStepLen * fTransparency;
#endif
#if ENABLE_CLOUDS
        fDistToCloud -= fStepLen;
#endif
    }
}

void IntegrateUnshadowedInscattering(in float3 f3RayStart, 
                                     in float3 f3RayEnd,
                                     in float3 f3ViewDir,
                                     in float3 f3EarthCentre,
                                     in float3 f3DirOnLight,
                                     uniform const float fNumSteps,
                                     out float3 f3Inscattering,
                                     out float3 f3Extinction,
                                     in float fCloudTransparency,
                                     in float fDistToCloud)
{
    float2 f2NetParticleDensityFromCam = 0;
    float3 f3RayleighInscattering = 0;
    float3 f3MieInscattering = 0;
    ComputeInsctrIntegral( f3RayStart,
                           f3RayEnd,
                           f3EarthCentre,
                           f3DirOnLight,
                           f2NetParticleDensityFromCam,
                           f3RayleighInscattering,
                           f3MieInscattering,
                           fNumSteps,
                           fCloudTransparency,
                           fDistToCloud );

    float3 f3TotalRlghOpticalDepth = g_MediaParams.f4RayleighExtinctionCoeff.rgb * f2NetParticleDensityFromCam.x;
    float3 f3TotalMieOpticalDepth  = g_MediaParams.f4MieExtinctionCoeff.rgb      * f2NetParticleDensityFromCam.y;
    f3Extinction = exp( -(f3TotalRlghOpticalDepth + f3TotalMieOpticalDepth) );

    // Apply phase function
    // Note that cosTheta = dot(DirOnCamera, LightDir) = dot(ViewDir, DirOnLight) because
    // DirOnCamera = -ViewDir and LightDir = -DirOnLight
    float cosTheta = dot(f3ViewDir, f3DirOnLight);
    ApplyPhaseFunctions(f3RayleighInscattering, f3MieInscattering, cosTheta);

    f3Inscattering = f3RayleighInscattering + f3MieInscattering;
}

void ComputeUnshadowedInscattering(float2 f2SampleLocation, 
                                   float fCamSpaceZ,
                                   float fCloudTransparency,
                                   float fDistToCloud,
                                   uniform const float fNumSteps,
                                   out float3 f3Inscattering,
                                   out float3 f3Extinction)
{
    f3Inscattering = 0;
    f3Extinction = 1;
    float3 f3RayTermination = ProjSpaceXYZToWorldSpace( float3(f2SampleLocation, fCamSpaceZ) );
    float3 f3CameraPos = g_CameraAttribs.f4CameraPos.xyz;
    float3 f3ViewDir = f3RayTermination - f3CameraPos;
    float fRayLength = length(f3ViewDir);
    f3ViewDir /= fRayLength;

    float3 f3EarthCentre =  - float3(0,1,0) * EARTH_RADIUS;
    float2 f2RayAtmTopIsecs;
    GetRaySphereIntersection( f3CameraPos, f3ViewDir, f3EarthCentre, 
                              ATM_TOP_RADIUS, 
                              f2RayAtmTopIsecs);
    if( f2RayAtmTopIsecs.y <= 0 )
        return;

    float fDistToRayStart = max(0, f2RayAtmTopIsecs.x);
    float3 f3RayStart = f3CameraPos + f3ViewDir * fDistToRayStart;
    if( fCamSpaceZ > g_CameraAttribs.fFarPlaneZ ) // fFarPlaneZ is pre-multiplied with 0.999999f
        fRayLength = +FLT_MAX;
    fRayLength = min(fRayLength, f2RayAtmTopIsecs.y);
    float3 f3RayEnd = f3CameraPos + f3ViewDir * fRayLength;
            
#if SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_INTEGRATION
    IntegrateUnshadowedInscattering(f3RayStart, 
                                    f3RayEnd,
                                    f3ViewDir,
                                    f3EarthCentre,
                                    g_LightAttribs.f4DirOnLight.xyz,
                                    fNumSteps,
                                    f3Inscattering,
                                    f3Extinction,
                                    fCloudTransparency,
                                    fDistToCloud);
#endif

#if SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_LUT || MULTIPLE_SCATTERING_MODE > MULTIPLE_SCTR_MODE_NONE

#if MULTIPLE_SCATTERING_MODE > MULTIPLE_SCTR_MODE_NONE
    #if SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_LUT
        Texture3D<float3> tex3DSctrLUT = g_tex3DMultipleSctrLUT;
    #elif SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_NONE || SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_INTEGRATION
        Texture3D<float3> tex3DSctrLUT = g_tex3DHighOrderSctrLUT;
    #endif
#else
    Texture3D<float3> tex3DSctrLUT = g_tex3DSingleSctrLUT;
#endif

    f3Extinction = GetExtinctionUnverified(f3RayStart, f3RayEnd, f3ViewDir, f3EarthCentre);

    // To avoid artifacts, we must be consistent when performing look-ups into the scattering texture, i.e.
    // we must assure that if the first look-up is above (below) horizon, then the second look-up
    // is also above (below) horizon. 
    float4 f4UVWQ = -1;
    // Provide previous look-up coordinates to the function to assure that look-ups are consistent
    float3 f3InsctrFromRayStart = LookUpPrecomputedScattering(f3RayStart, f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, tex3DSctrLUT, f4UVWQ);
    float3 f3InsctrFromRayEnd   = LookUpPrecomputedScattering(f3RayEnd,   f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, tex3DSctrLUT, f4UVWQ);

#if ENABLE_CLOUDS
    if( fDistToCloud < fRayLength )
    {
        float3 f3CloudStart = f3CameraPos + f3ViewDir * fDistToCloud;
        float3 f3InsctrFromCloud = LookUpPrecomputedScattering(f3CloudStart, f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, tex3DSctrLUT, f4UVWQ);
        float3 f3ExtinctionToCloud = GetExtinctionUnverified(f3RayStart, f3CloudStart, f3ViewDir, f3EarthCentre);

        //f3Inscattering += f3InsctrFromRayStart - f3ExtinctionToCloud * f3InsctrFromCloud + 
        //    (f3InsctrFromCloud * f3ExtinctionToCloud - f3Extinction * f3InsctrFromRayEnd) * fCloudTransparency;

        f3Inscattering += (fCloudTransparency - 1) * f3ExtinctionToCloud * f3InsctrFromCloud;
        f3InsctrFromRayEnd *= fCloudTransparency;
    }
#endif
    
    f3Inscattering +=  f3InsctrFromRayStart - f3Extinction * f3InsctrFromRayEnd;
    
#endif


}

// This function calculates inscattered light integral over the ray from the camera to 
// the specified world space position using ray marching
float3 ComputeShadowedInscattering( in const float2 f2RayMarchingSampleLocation,
                                    in       float fRayEndCamSpaceZ,
                                    in       float fCascadeInd,
                                    in const float fScrSpaceCloudTransparency,
                                    in const float fScrSpaceDistToCloud,
                                    in uniform const bool bUse1DMinMaxMipMap = false,
                                    in const uint uiEpipolarSliceInd = 0 )
{   
    float3 f3CameraPos = g_CameraAttribs.f4CameraPos.xyz;
    uint uiCascadeInd = fCascadeInd;
    
    // Compute the ray termination point, full ray length and view direction
    float3 f3RayTermination = ProjSpaceXYZToWorldSpace( float3(f2RayMarchingSampleLocation, fRayEndCamSpaceZ) );
    float3 f3FullRay = f3RayTermination - f3CameraPos;
    float fFullRayLength = length(f3FullRay);
    float3 f3ViewDir = f3FullRay / fFullRayLength;

    const float3 f3EarthCentre = float3(0, -EARTH_RADIUS, 0);

    // Intersect the ray with the top of the atmosphere and the Earth:
    float4 f4Isecs;
    GetRaySphereIntersection2(f3CameraPos, f3ViewDir, f3EarthCentre, 
                              float2(ATM_TOP_RADIUS, EARTH_RADIUS), f4Isecs);
    float2 f2RayAtmTopIsecs = f4Isecs.xy; 
    float2 f2RayEarthIsecs  = f4Isecs.zw;
    
    if( f2RayAtmTopIsecs.y <= 0 )
    {
        //                                                          view dir
        //                                                        /
        //             d<0                                       /
        //               *--------->                            *
        //            .      .                             .   /  . 
        //  .  '                    '  .         .  '         /\         '  .
        //                                                   /  f2rayatmtopisecs.y < 0
        //
        // the camera is outside the atmosphere and the ray either does not intersect the
        // top of it or the intersection point is behind the camera. In either
        // case there is no inscattering
        return 0;
    }

    // Restrict the camera position to the top of the atmosphere
    float fDistToAtmosphere = max(f2RayAtmTopIsecs.x, 0);
    float3 f3RestrainedCameraPos = f3CameraPos + fDistToAtmosphere * f3ViewDir;

    // Limit the ray length by the distance to the top of the atmosphere if the ray does not hit terrain
    float fOrigRayLength = fFullRayLength;
    if( fRayEndCamSpaceZ > g_CameraAttribs.fFarPlaneZ ) // fFarPlaneZ is pre-multiplied with 0.999999f
        fFullRayLength = +FLT_MAX;
    // Limit the ray length by the distance to the point where the ray exits the atmosphere
    fFullRayLength = min(fFullRayLength, f2RayAtmTopIsecs.y);

    // If there is an intersection with the Earth surface, limit the tracing distance to the intersection
    if( f2RayEarthIsecs.x > 0 )
    {
        fFullRayLength = min(fFullRayLength, f2RayEarthIsecs.x);
    }

#if ENABLE_CLOUDS
    float2 f2CloudLayerIsecs;
    GetRaySphereIntersection(f3CameraPos, f3ViewDir, f3EarthCentre, 
                             EARTH_RADIUS + g_PPAttribs.m_fCloudAltitiude, f2CloudLayerIsecs);
    float fDistToCloudLayer = f2CloudLayerIsecs.x > 0 ? f2CloudLayerIsecs.x : (f2CloudLayerIsecs.y > 0 ? f2CloudLayerIsecs.y : +FLT_MAX);
#endif

    fRayEndCamSpaceZ *= fFullRayLength / fOrigRayLength; 
    
    float3 f3RayleighInscattering = 0;
    float3 f3MieInscattering = 0;
    float2 f2ParticleNetDensityFromCam = 0;
    float3 f3RayEnd = 0, f3RayStart = 0;
    
    // Note that cosTheta = dot(DirOnCamera, LightDir) = dot(ViewDir, DirOnLight) because
    // DirOnCamera = -ViewDir and LightDir = -DirOnLight
    float cosTheta = dot(f3ViewDir, g_LightAttribs.f4DirOnLight.xyz);
    
    float fCascadeEndCamSpaceZ = 0;
    float fTotalLitLength = 0, fTotalMarchedLength = 0; // Required for multiple scattering
    float fDistToFirstLitSection = -1; // Used only in when SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_LUT

#if CASCADE_PROCESSING_MODE == CASCADE_PROCESSING_MODE_SINGLE_PASS
    for(; uiCascadeInd < (uint)g_PPAttribs.m_iNumCascades; ++uiCascadeInd, ++fCascadeInd)
#else
    for(int i=0; i<1; ++i)
#endif
    {
        float2 f2CascadeStartEndCamSpaceZ = g_LightAttribs.ShadowAttribs.Cascades[uiCascadeInd].f4StartEndZ.xy;
        float fCascadeStartCamSpaceZ = f2CascadeStartEndCamSpaceZ.x;//(uiCascadeInd > (uint)g_PPAttribs.m_iFirstCascade) ? f2CascadeStartEndCamSpaceZ.x : 0;
        fCascadeEndCamSpaceZ = f2CascadeStartEndCamSpaceZ.y;
        
        // Check if the ray terminates before it enters current cascade 
        if( fRayEndCamSpaceZ < fCascadeStartCamSpaceZ )
        {
#if CASCADE_PROCESSING_MODE == CASCADE_PROCESSING_MODE_SINGLE_PASS
            break;
#else
            return 0;
#endif
        }

        // Truncate the ray against the far and near planes of the current cascade:
        float fRayEndRatio = min( fRayEndCamSpaceZ, fCascadeEndCamSpaceZ ) / fRayEndCamSpaceZ;
        float fRayStartRatio = fCascadeStartCamSpaceZ / fRayEndCamSpaceZ;
        float fDistToRayStart = fFullRayLength * fRayStartRatio;
        float fDistToRayEnd   = fFullRayLength * fRayEndRatio;

        // If the camera is outside the atmosphere and the ray intersects the top of it,
        // we must start integration from the first intersection point.
        // If the camera is in the atmosphere, first intersection point is always behind the camera 
        // and thus is negative
        //                               
        //                      
        //                     
        //                   *                                              /
        //              .   /  .                                       .   /  . 
        //    .  '         /\         '  .                   .  '         /\         '  .
        //                /  f2RayAtmTopIsecs.x > 0                      /  f2RayAtmTopIsecs.y > 0
        //                                                              *
        //                 f2RayAtmTopIsecs.y > 0                         f2RayAtmTopIsecs.x < 0
        //                /                                              /
        //
        fDistToRayStart = max(fDistToRayStart, f2RayAtmTopIsecs.x);
        fDistToRayEnd   = max(fDistToRayEnd,   f2RayAtmTopIsecs.x);
        
        // To properly compute scattering from the space, we must 
        // set up ray end position before extiting the loop
        f3RayEnd   = f3CameraPos + f3ViewDir * fDistToRayEnd;
        f3RayStart = f3CameraPos + f3ViewDir * fDistToRayStart;

#if CASCADE_PROCESSING_MODE != CASCADE_PROCESSING_MODE_SINGLE_PASS
        float r = length(f3RestrainedCameraPos - f3EarthCentre);
        float fCosZenithAngle = dot(f3RestrainedCameraPos-f3EarthCentre, f3ViewDir) / r;
        float fDist = max(fDistToRayStart - fDistToAtmosphere, 0);
        f2ParticleNetDensityFromCam = GetDensityIntegralAnalytic(r, fCosZenithAngle, fDist);
#endif

        float fRayLength = fDistToRayEnd - fDistToRayStart;
        if( fRayLength <= 10 )
        {
#if CASCADE_PROCESSING_MODE == CASCADE_PROCESSING_MODE_SINGLE_PASS
            continue;
#else
            if( (int)uiCascadeInd == g_PPAttribs.m_iNumCascades-1 )
                // We need to process remaining part of the ray
                break;
            else
                return 0;
#endif
        }

        // We trace the ray in the light projection space, not in the world space
        // Compute shadow map UV coordinates of the ray end point and its depth in the light space
        matrix mWorldToShadowMapUVDepth = g_LightAttribs.ShadowAttribs.mWorldToShadowMapUVDepth[uiCascadeInd];
        float3 f3StartUVAndDepthInLightSpace = WorldSpaceToShadowMapUV(f3RayStart, mWorldToShadowMapUVDepth);
        //f3StartUVAndDepthInLightSpace.z -= SHADOW_MAP_DEPTH_BIAS;
        float3 f3EndUVAndDepthInLightSpace = WorldSpaceToShadowMapUV(f3RayEnd, mWorldToShadowMapUVDepth);
        //f3EndUVAndDepthInLightSpace.z -= SHADOW_MAP_DEPTH_BIAS;

        // Calculate normalized trace direction in the light projection space and its length
        float3 f3ShadowMapTraceDir = f3EndUVAndDepthInLightSpace.xyz - f3StartUVAndDepthInLightSpace.xyz;
        // If the ray is directed exactly at the light source, trace length will be zero
        // Clamp to a very small positive value to avoid division by zero
        float fTraceLenInShadowMapUVSpace = max( length( f3ShadowMapTraceDir.xy ), 1e-7 );
        // Note that f3ShadowMapTraceDir.xy can be exactly zero
        f3ShadowMapTraceDir /= fTraceLenInShadowMapUVSpace;
    
        float fShadowMapUVStepLen = 0;
        float2 f2SliceOriginUV = 0;
        float2 f2SliceDirUV = 0;
        uint uiMinMaxTexYInd = 0;
        if( bUse1DMinMaxMipMap )
        {
            // Get UV direction for this slice
            float4 f4SliceUVDirAndOrigin = g_tex2DSliceUVDirAndOrigin.Load( uint3(uiEpipolarSliceInd,uiCascadeInd,0) );
            f2SliceDirUV = f4SliceUVDirAndOrigin.xy;
            //if( all(f4SliceUVDirAndOrigin == g_f4IncorrectSliceUVDirAndStart) )
            //{
            //    return float3(0,0,0);
            //}
            //return float3(f4SliceUVDirAndOrigin.xy,0);
            // Scale with the shadow map texel size
            fShadowMapUVStepLen = length(f2SliceDirUV);
            f2SliceOriginUV = f4SliceUVDirAndOrigin.zw;
         
#if USE_COMBINED_MIN_MAX_TEXTURE
            uiMinMaxTexYInd = uiEpipolarSliceInd + (uiCascadeInd - g_PPAttribs.m_iFirstCascade) * g_PPAttribs.m_uiNumEpipolarSlices;
#else
            uiMinMaxTexYInd = uiEpipolarSliceInd;
#endif

        }
        else
        {
            //Calculate length of the trace step in light projection space
            float fMaxTraceDirDim = max( abs(f3ShadowMapTraceDir.x), abs(f3ShadowMapTraceDir.y) );
            fShadowMapUVStepLen = (fMaxTraceDirDim > 0) ? (g_PPAttribs.m_f2ShadowMapTexelSize.x / fMaxTraceDirDim) : 0;
            // Take into account maximum number of steps specified by the g_MiscParams.fMaxStepsAlongRay
            fShadowMapUVStepLen = max(fTraceLenInShadowMapUVSpace/g_MiscParams.fMaxStepsAlongRay, fShadowMapUVStepLen);
        }
    
        // Calcualte ray step length in world space
        float fRayStepLengthWS = fRayLength * (fShadowMapUVStepLen / fTraceLenInShadowMapUVSpace);
        // Note that fTraceLenInShadowMapUVSpace can be very small when looking directly at sun
        // Since fShadowMapUVStepLen is at least one shadow map texel in size, 
        // fShadowMapUVStepLen / fTraceLenInShadowMapUVSpace >> 1 in this case and as a result
        // fRayStepLengthWS >> fRayLength

#if ENABLE_CLOUDS
        float fDistToCloudFromRayStart = fScrSpaceDistToCloud - fDistToRayStart;
        // Compute altitude of the ray start point
        float fPrevAltitude = length(f3RayStart - f3EarthCentre) - EARTH_RADIUS;
        float fDistToCloudLayerFromRayStart = fDistToCloudLayer - fDistToRayStart;
#endif
        // March the ray
        float fDistanceMarchedInCascade = 0;
        float3 f3CurrShadowMapUVAndDepthInLightSpace = f3StartUVAndDepthInLightSpace.xyz;

        // The following variables are used only if 1D min map optimization is enabled
        uint uiMinLevel = 0;
        // It is essential to round initial sample pos to the closest integer
        float fFractionalSamplePos = length(f3StartUVAndDepthInLightSpace.xy - f2SliceOriginUV.xy)/fShadowMapUVStepLen;
        float fRoundedSamplePos = ceil( fFractionalSamplePos );
        uint uiCurrSamplePos = fRoundedSamplePos;
        uint uiCurrTreeLevel = 0;
        // Note that min/max shadow map does not contain finest resolution level
        // The first level it contains corresponds to step == 2
        int iLevelDataOffset = -int(g_PPAttribs.m_uiMinMaxShadowMapResolution);
        float fStepScale = 1.f;
        if( bUse1DMinMaxMipMap )
        {
            // If 1D min/max trees are used, we must sample the shadow map (cloud transparency
            // map) at the same locations as when 1D min/max binary tree (cloud density epipolar 
            // scan) was constructed. We align with the sample locations on the very first step of ray 
            // marching by tweaking fStepScale
            // 
            //
            //                          fRoundedSamplePos
            //                                 |
            //           fFractionalSamplePos  |
            //                      |          |
            //     0             1  |          2             3     Sampling locations 
            //     X-------------X--|----------X-------------X
            //                      |          |
            //                      |--------->|
            //                       Adjsutment     
            fStepScale = fRoundedSamplePos - fFractionalSamplePos;
            
            // This adjustment is not important for 1D min/max tree, because conservative
            // bounds are computed. 
            // To the contrary, for sampling cloud density, it is crucially important to 
            // avoid ringing artifacts
        }

        float fMaxStepScale = g_PPAttribs.m_fMaxShadowMapStep;
#if SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_INTEGRATION || (ENABLE_CLOUDS && SHAFTS_FROM_CLOUDS_MODE == SHAFTS_FROM_CLOUDS_TRANSPARENCY_MAP)
        // In order for the numerical integration to be accurate enough, it is necessary to make 
        // at least 10 steps along the ray. To assure this, limit the maximum world step by 
        // 1/10 of the ray length.
        // To avoid aliasing artifacts due to unstable sampling along the view ray, do this for
        // each cascade separately
        float fMaxAllowedWorldStepLen = fRayLength/10;
        fMaxStepScale = min(fMaxStepScale, fMaxAllowedWorldStepLen/fRayStepLengthWS);
        
        // Make sure that the world step length is not greater than the maximum allowable length
        if( fRayStepLengthWS > fMaxAllowedWorldStepLen )
        {
            fRayStepLengthWS = fMaxAllowedWorldStepLen;
            // Recalculate shadow map UV step len
            fShadowMapUVStepLen = fTraceLenInShadowMapUVSpace * fRayStepLengthWS / fRayLength;
            // Disable 1D min/max optimization. Note that fMaxStepScale < 1 anyway since 
            // fRayStepLengthWS > fMaxAllowedWorldStepLen. Thus there is no real need to
            // make the max shadow map step negative. We do this just for clarity
            fMaxStepScale = -1;
        }
#endif

        // Scale trace direction in light projection space to calculate the step in shadow map
        float3 f3ShadowMapUVAndDepthStep = f3ShadowMapTraceDir * fShadowMapUVStepLen;
        
#if ENABLE_CLOUDS && SHAFTS_FROM_CLOUDS_MODE == SHAFTS_FROM_CLOUDS_TRANSPARENCY_MAP
        // Compute cloud transparency texture sampling LOD
        float fUVDeltaLen = length(f3ShadowMapUVAndDepthStep.xy * g_PPAttribs.m_fLiSpCldDensResolution.xx);
        fUVDeltaLen = max(fUVDeltaLen, 1e-10);
        float fCloudTextureLOD = log2( fUVDeltaLen );
#endif

        [loop]
        while( fDistanceMarchedInCascade < fRayLength )
        {
            // Clamp depth to a very small positive value to avoid z-fighting at camera location
            float fCurrDepthInLightSpace = max(f3CurrShadowMapUVAndDepthInLightSpace.z, 1e-7);
            float IsInLight = 0;

            if( bUse1DMinMaxMipMap )
            {
                // If the step scale can be doubled without exceeding the maximum allowed scale and 
                // the sample is located at the appropriate position, advance to the next coarser level
                // We must skip alignment step (when fStepScale < 1)
                if( fStepScale >= 1 && 2*fStepScale < fMaxStepScale && ((uiCurrSamplePos & ((2<<uiCurrTreeLevel)-1)) == 0) )
                {
                    iLevelDataOffset += g_PPAttribs.m_uiMinMaxShadowMapResolution >> uiCurrTreeLevel;
                    uiCurrTreeLevel++;
                    fStepScale *= 2.f;
                }

                while(uiCurrTreeLevel > uiMinLevel)
                {
                    // Compute light space depths at the ends of the current ray section

                    // What we need here is actually depth which is divided by the camera view space z
                    // Thus depth can be correctly interpolated in screen space:
                    // http://www.comp.nus.edu.sg/~lowkl/publications/lowk_persp_interp_techrep.pdf
                    // A subtle moment here is that we need to be sure that we can skip fStepScale samples 
                    // starting from 0 up to fStepScale-1. We do not need to do any checks against the sample fStepScale away:
                    //
                    //     --------------->
                    //
                    //          *
                    //               *         *
                    //     *              *     
                    //     0    1    2    3
                    //
                    //     |------------------>|
                    //        fStepScale = 4
                    float fNextLightSpaceDepth = f3CurrShadowMapUVAndDepthInLightSpace.z + f3ShadowMapUVAndDepthStep.z * (fStepScale-1);
                    float2 f2StartEndDepthOnRaySection = float2(f3CurrShadowMapUVAndDepthInLightSpace.z, fNextLightSpaceDepth);
                    f2StartEndDepthOnRaySection = f2StartEndDepthOnRaySection;//max(f2StartEndDepthOnRaySection, 1e-7);

                    // Load 1D min/max depths
                    float2 f2CurrMinMaxDepth = g_tex2DMinMaxLightSpaceDepth.Load( uint3( (uiCurrSamplePos>>uiCurrTreeLevel) + iLevelDataOffset, uiMinMaxTexYInd, 0) );
                
                    // Since we use complimentary depth buffer, the relations are reversed
                    IsInLight = all( f2StartEndDepthOnRaySection >= f2CurrMinMaxDepth.yy );
                    bool bIsInShadow = all( f2StartEndDepthOnRaySection < f2CurrMinMaxDepth.xx );

                    bool bRefineStep = false;
#if ENABLE_CLOUDS && SHAFTS_FROM_CLOUDS_MODE == SHAFTS_FROM_CLOUDS_TRANSPARENCY_MAP
                    // We need to step through the intersection with the cloud layer at the fine step
                    // Also, refine step at the end of the ray
                    // Performance note: this additional condition adds ~10% to the ray marching time
                    // It can be traded for slight artifacts on the clouds and near the ground
                    bRefineStep = 
                        fDistanceMarchedInCascade < fDistToCloudLayerFromRayStart + fRayStepLengthWS * fStepScale  &&
                        fDistanceMarchedInCascade > fDistToCloudLayerFromRayStart - fRayStepLengthWS * fStepScale || 
                        fDistanceMarchedInCascade > fRayLength - fRayStepLengthWS * fStepScale ;
#endif

                    if( (IsInLight || bIsInShadow) && !bRefineStep )
                        // If the ray section is fully lit or shadowed, we can break the loop
                        break;
                    // If the ray section is neither fully lit, nor shadowed, we have to go to the finer level
                    uiCurrTreeLevel--;
                    iLevelDataOffset -= (int)(g_PPAttribs.m_uiMinMaxShadowMapResolution >> uiCurrTreeLevel);
                    fStepScale /= 2.f;
                };

                // If we are at the finest level, sample the shadow map with PCF
                [branch]
                if( uiCurrTreeLevel <= uiMinLevel )
                {
                    IsInLight = g_tex2DLightSpaceDepthMap.SampleCmpLevelZero( samComparison, float3(f3CurrShadowMapUVAndDepthInLightSpace.xy,fCascadeInd), fCurrDepthInLightSpace  ).x;
                }
            }
            else
            {
                IsInLight = g_tex2DLightSpaceDepthMap.SampleCmpLevelZero( samComparison, float3(f3CurrShadowMapUVAndDepthInLightSpace.xy,fCascadeInd), fCurrDepthInLightSpace ).x;
            }

            float fRemainingDist = max(fRayLength - fDistanceMarchedInCascade, 0);
            float fIntegrationStep = min(fRayStepLengthWS * fStepScale, fRemainingDist);
            float fIntegrationDist = fDistanceMarchedInCascade + fIntegrationStep/2;

            float fTransparency = 1;
            float fLiSpCloudTransparency = 1;
#if ENABLE_CLOUDS
            // It is very important to smooth transparency to avoid banding artifacts
            //
            //                   1         fTransparency          Transparency  
            //
            //          |----------------|----------------|
            //          |            fDistToCloud                 StartDist = fDistanceMarchedInCascade
            //          |
            //       StartDist                          
            //          |<------------------------------->|
            //                    fIntegrationStep
            //fTransparency = (fDistanceMarchedInCascade + 1*fIntegrationStep < fDistToCloudFromRayStart) ? 1 : fScrSpaceCloudTransparency;//
            fTransparency = lerp(1, fScrSpaceCloudTransparency, saturate( (fDistanceMarchedInCascade + fIntegrationStep - fDistToCloudFromRayStart) / fIntegrationStep) );

#   if SHAFTS_FROM_CLOUDS_MODE == SHAFTS_FROM_CLOUDS_TRANSPARENCY_MAP
                // Compute altitude of the end point of this ray section (not middle point)
                float3 f3SectionEnd = f3RayStart + f3ViewDir * (fDistanceMarchedInCascade + fIntegrationStep);
                float fAltitude = length(f3SectionEnd - f3EarthCentre) - EARTH_RADIUS;
                if( IsInLight > 0 )
                {
                    // Compute which part of the current section is under cloud
                    float fMaxHeight = max(fPrevAltitude, fAltitude);
                    float fMinHeight = min(fPrevAltitude, fAltitude);
                    float fPartUnderCloud = saturate( (g_PPAttribs.m_fCloudAltitiude - fMinHeight) / max((fMaxHeight - fMinHeight), 1e-10) );
                    if( fPartUnderCloud > 1e-3 )
                    {
                        // Sample cloud transparency
                        float fCloudTransparency = 1;
                        if(  bUse1DMinMaxMipMap && fStepScale > 1 )
                        {
                            fCloudTransparency = g_tex2DLiSpCldDensityEpipolarScan.Load( uint3( (uiCurrSamplePos>>uiCurrTreeLevel) + iLevelDataOffset, uiMinMaxTexYInd, 0) );
                        }
                        else
                        {
                            fCloudTransparency = g_tex2DLiSpaceCloudTransparency.SampleLevel( samLinearClamp, float3(f3CurrShadowMapUVAndDepthInLightSpace.xy,fCascadeInd), fCloudTextureLOD).x;
                        }
                        // Compute cloud light space transparency accounting for covered fraction
                        fLiSpCloudTransparency = lerp(1, fCloudTransparency, fPartUnderCloud);
                    }
                }
                fPrevAltitude = fAltitude;
#   endif
#endif

#if SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_INTEGRATION
            float3 f3CurrPos = f3RayStart + f3ViewDir * fIntegrationDist;

            // Calculate integration point height above the SPHERICAL Earth surface:
            float3 f3EarthCentreToPointDir = f3CurrPos - f3EarthCentre;
            float fDistToEarthCentre = length(f3EarthCentreToPointDir);
            f3EarthCentreToPointDir /= fDistToEarthCentre;
            float fHeightAboveSurface = fDistToEarthCentre - EARTH_RADIUS;

            float2 f2ParticleDensity = exp( -fHeightAboveSurface / PARTICLE_SCALE_HEIGHT );

            // Do not use this branch as it only degrades performance
            //if( IsInLight == 0)
            //    continue;

            // Get net particle density from the integration point to the top of the atmosphere:
            float fCosSunZenithAngle = dot( f3EarthCentreToPointDir, g_LightAttribs.f4DirOnLight.xyz );
            float2 f2NetParticleDensityToAtmTop = GetNetParticleDensity(fHeightAboveSurface, fCosSunZenithAngle);
        
            // Compute total particle density from the top of the atmosphere through the integraion point to camera
            float2 f2TotalParticleDensity = f2ParticleNetDensityFromCam + f2NetParticleDensityToAtmTop;
        
            // Update net particle density from the camera to the integration point:
            f2ParticleNetDensityFromCam += f2ParticleDensity * fIntegrationStep;

            // Get optical depth
            float3 f3TotalRlghOpticalDepth = g_MediaParams.f4RayleighExtinctionCoeff.rgb * f2TotalParticleDensity.x;
            float3 f3TotalMieOpticalDepth  = g_MediaParams.f4MieExtinctionCoeff.rgb      * f2TotalParticleDensity.y;
        
            // And total extinction for the current integration point:
            float3 f3TotalExtinction = exp( -(f3TotalRlghOpticalDepth + f3TotalMieOpticalDepth) );

            f2ParticleDensity *= fIntegrationStep * IsInLight * fTransparency * fLiSpCloudTransparency;
            f3RayleighInscattering += f2ParticleDensity.x * f3TotalExtinction;
            f3MieInscattering      += f2ParticleDensity.y * f3TotalExtinction;
#endif

#if MULTIPLE_SCATTERING_MODE == MULTIPLE_SCTR_MODE_OCCLUDED || SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_LUT
            // Store the distance where the ray first enters the light
            fDistToFirstLitSection = (fDistToFirstLitSection < 0 && IsInLight > 0) ? fTotalMarchedLength : fDistToFirstLitSection;
#endif
            f3CurrShadowMapUVAndDepthInLightSpace += f3ShadowMapUVAndDepthStep * fStepScale;
            uiCurrSamplePos += (fStepScale >= 1) ? (1 << uiCurrTreeLevel) : 0; // int -> float conversions are slow
            fDistanceMarchedInCascade += fRayStepLengthWS * fStepScale;

#if MULTIPLE_SCATTERING_MODE == MULTIPLE_SCTR_MODE_OCCLUDED || SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_LUT
            fTotalLitLength += fIntegrationStep * IsInLight * fTransparency * fLiSpCloudTransparency;
            fTotalMarchedLength += fIntegrationStep;
#endif
            // The very first step is alignment step and fStepScale < 1
            fStepScale = (fStepScale >= 1) ? fStepScale :  1;
        }
    }

#if MULTIPLE_SCATTERING_MODE == MULTIPLE_SCTR_MODE_OCCLUDED || SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_LUT
    // If the whole ray is in shadow, set the distance to the first lit section to the
    // total marched distance
    if( fDistToFirstLitSection < 0 )
        fDistToFirstLitSection = fTotalMarchedLength;
#endif

    float3 f3RemainingRayStart = 0;
    float fRemainingLength = 0;
    if( 
#if CASCADE_PROCESSING_MODE != CASCADE_PROCESSING_MODE_SINGLE_PASS
        (int)uiCascadeInd == g_PPAttribs.m_iNumCascades-1 && 
#endif
        fRayEndCamSpaceZ > fCascadeEndCamSpaceZ 
       )
    {
        f3RemainingRayStart = f3RayEnd;
        f3RayEnd = f3CameraPos + fFullRayLength * f3ViewDir;
        fRemainingLength = length(f3RayEnd - f3RemainingRayStart);
#if SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_INTEGRATION
        // Do not allow integration step to become less than 50 km
        // Maximum possible view ray length is 2023 km (from the top of the
        // atmosphere touching the Earth and then again to the top of the 
        // atmosphere).
        // For such ray, 41 integration step will be performed
        // Also assure that at least 20 steps are always performed
        float fMinStep = 50000.f;
        float fMumSteps = max(20, ceil(fRemainingLength/fMinStep) );
        ComputeInsctrIntegral(f3RemainingRayStart,
                              f3RayEnd,
                              f3EarthCentre,
                              g_LightAttribs.f4DirOnLight.xyz,
                              f2ParticleNetDensityFromCam,
                              f3RayleighInscattering,
                              f3MieInscattering,
                              fMumSteps,
                              fScrSpaceCloudTransparency,
                              fScrSpaceDistToCloud);
#endif
    }

    float3 f3InsctrIntegral = 0;

#if SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_INTEGRATION
    // Apply phase functions
    // Note that cosTheta = dot(DirOnCamera, LightDir) = dot(ViewDir, DirOnLight) because
    // DirOnCamera = -ViewDir and LightDir = -DirOnLight
    ApplyPhaseFunctions(f3RayleighInscattering, f3MieInscattering, cosTheta);

    f3InsctrIntegral = f3RayleighInscattering + f3MieInscattering;
#endif

#if CASCADE_PROCESSING_MODE == CASCADE_PROCESSING_MODE_SINGLE_PASS
    // Note that the first cascade used for ray marching must contain camera within it
    // otherwise this expression might fail
    f3RayStart = f3RestrainedCameraPos;
#endif

#if SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_LUT || MULTIPLE_SCATTERING_MODE == MULTIPLE_SCTR_MODE_OCCLUDED

#if MULTIPLE_SCATTERING_MODE == MULTIPLE_SCTR_MODE_OCCLUDED
    #if SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_LUT
        Texture3D<float3> tex3DSctrLUT = g_tex3DMultipleSctrLUT;
    #elif SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_NONE || SINGLE_SCATTERING_MODE == SINGLE_SCTR_MODE_INTEGRATION
        Texture3D<float3> tex3DSctrLUT = g_tex3DHighOrderSctrLUT;
    #endif
#else
    Texture3D<float3> tex3DSctrLUT = g_tex3DSingleSctrLUT;
#endif

    float3 f3MultipleScattering = 0;
    if( fTotalLitLength > 0 )
    {    
        float3 f3LitSectionStart = f3RayStart + fDistToFirstLitSection * f3ViewDir;
        float3 f3LitSectionEnd = f3LitSectionStart + fTotalLitLength * f3ViewDir;

        float3 f3ExtinctionToStart = GetExtinctionUnverified(f3RestrainedCameraPos, f3LitSectionStart, f3ViewDir, f3EarthCentre);
        float4 f4UVWQ = -1;
        f3MultipleScattering = f3ExtinctionToStart * LookUpPrecomputedScattering(f3LitSectionStart, f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, tex3DSctrLUT, f4UVWQ); 
        
        float3 f3ExtinctionToEnd = GetExtinctionUnverified(f3RestrainedCameraPos, f3LitSectionEnd, f3ViewDir,  f3EarthCentre);
        // To avoid artifacts, we must be consistent when performing look-ups into the scattering texture, i.e.
        // we must assure that if the first look-up is above (below) horizon, then the second look-up
        // is also above (below) horizon.
        // We provide previous look-up coordinates to the function so that it is able to figure out where the first look-up
        // was performed
        f3MultipleScattering -= f3ExtinctionToEnd * LookUpPrecomputedScattering(f3LitSectionEnd, f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, tex3DSctrLUT, f4UVWQ);
        
        f3InsctrIntegral += max(f3MultipleScattering, 0);
    }

    // Add contribution from the reminder of the ray behind the largest cascade
    if( fRemainingLength > 0 )
    {
        float3 f3ExtinctionFromRemainderStart = GetExtinctionUnverified(f3RestrainedCameraPos, f3RemainingRayStart, f3ViewDir, f3EarthCentre);
        float4 f4UVWQ = -1;
        
        float3 f3InsctrFromRemainderStart = LookUpPrecomputedScattering(f3RemainingRayStart, f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, tex3DSctrLUT, f4UVWQ);
        
        float3 f3ExtinctionFromRayEnd = GetExtinctionUnverified(f3RestrainedCameraPos, f3RayEnd, f3ViewDir, f3EarthCentre);
        float3 f3InsctrFromRayEnd = LookUpPrecomputedScattering(f3RayEnd, f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, tex3DSctrLUT, f4UVWQ);

        float3 f3RemainingInsctr = 0;
#       if ENABLE_CLOUDS
            if( fScrSpaceDistToCloud < fFullRayLength )
            {
                float fDistToRemainderStart = length(f3RemainingRayStart - f3CameraPos);
                if( fScrSpaceDistToCloud < fDistToRemainderStart )
                {
                    f3InsctrFromRemainderStart *= fScrSpaceCloudTransparency;
                    f3InsctrFromRayEnd         *= fScrSpaceCloudTransparency;
                }
                else
                {
                    float3 f3CloudStart = f3CameraPos + f3ViewDir * fScrSpaceDistToCloud;
                    float3 f3InsctrFromCloud = LookUpPrecomputedScattering(f3CloudStart, f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, tex3DSctrLUT, f4UVWQ);
                    float3 f3ExtinctionToCloud = GetExtinctionUnverified(f3RestrainedCameraPos, f3CloudStart, f3ViewDir, f3EarthCentre);

                    //f3Inscattering += f3InsctrFromRayStart - f3ExtinctionToCloud * f3InsctrFromCloud + 
                    //    (f3InsctrFromCloud * f3ExtinctionToCloud - f3Extinction * f3InsctrFromRayEnd) * fCloudTransparency;

                    f3RemainingInsctr += (fScrSpaceCloudTransparency - 1) * f3ExtinctionToCloud * f3InsctrFromCloud;
                    f3InsctrFromRayEnd *= fScrSpaceCloudTransparency;
                }
            }
#       endif
        f3RemainingInsctr += f3ExtinctionFromRemainderStart * f3InsctrFromRemainderStart - f3ExtinctionFromRayEnd * f3InsctrFromRayEnd;
        f3InsctrIntegral += max(f3RemainingInsctr, 0);
    }
#endif

#if MULTIPLE_SCATTERING_MODE == MULTIPLE_SCTR_MODE_UNOCCLUDED
    {
        float4 f4UVWQ = -1;
        float3 f3ExtinctionToRayStart = GetExtinctionUnverified(f3RestrainedCameraPos, f3RayStart, f3ViewDir, f3EarthCentre);
        float3 f3HigherOrderSctrFromRayStart = LookUpPrecomputedScattering(f3RayStart, f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, g_tex3DHighOrderSctrLUT, f4UVWQ); 
         
        float3 f3ExtinctionToRayEnd = GetExtinctionUnverified(f3RestrainedCameraPos, f3RayEnd, f3ViewDir, f3EarthCentre);
        // We provide previous look-up coordinates to the function so that it is able to figure out where the first look-up
        // was performed
        float3 f3HigherOrderSctrFromRayEnd = LookUpPrecomputedScattering(f3RayEnd, f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, g_tex3DHighOrderSctrLUT, f4UVWQ); 

        float3 f3HighOrderScattering = 0;
#       if ENABLE_CLOUDS
            float fDistToRayEnd = dot(f3RayEnd - f3CameraPos, f3ViewDir);
            if( fScrSpaceDistToCloud < fDistToRayEnd )
            {
                float fDistToRayStart = dot(f3RayStart - f3CameraPos, f3ViewDir);
                if( fScrSpaceDistToCloud < fDistToRayStart )
                {
                    f3HigherOrderSctrFromRayStart *= fScrSpaceCloudTransparency;
                    f3HigherOrderSctrFromRayEnd   *= fScrSpaceCloudTransparency;
                }
                else
                {
                    float3 f3CloudStart = f3CameraPos + f3ViewDir * fScrSpaceDistToCloud;
                    float3 f3HighOrderSctrFromCloud = LookUpPrecomputedScattering(f3CloudStart, f3ViewDir, f3EarthCentre, g_LightAttribs.f4DirOnLight.xyz, g_tex3DHighOrderSctrLUT, f4UVWQ);
                    // TODO: so this potentially duplicate computation:
                    float3 f3ExtinctionToCloud = GetExtinctionUnverified(f3RestrainedCameraPos, f3CloudStart, f3ViewDir, f3EarthCentre);

                    //f3HighOrderScattering = f3ExtinctionToRayStart * f3HigherOrderSctrFromRayStart - f3ExtinctionToCloud * f3HighOrderSctrFromCloud + 
                    //    (f3HighOrderSctrFromCloud * f3ExtinctionToCloud - f3ExtinctionToRayEnd * f3HigherOrderSctrFromRayEnd) * fCloudTransparency;

                    f3HighOrderScattering += (fScrSpaceCloudTransparency - 1) * f3ExtinctionToCloud * f3HighOrderSctrFromCloud;
                    f3HigherOrderSctrFromRayEnd *= fScrSpaceCloudTransparency;
                }
            }
#       endif
        f3HighOrderScattering += f3ExtinctionToRayStart * f3HigherOrderSctrFromRayStart - f3ExtinctionToRayEnd * f3HigherOrderSctrFromRayEnd;

        f3InsctrIntegral += max(f3HighOrderScattering,0);
    }
#endif

    return f3InsctrIntegral * g_LightAttribs.f4ExtraterrestrialSunColor.rgb;
}

float3 RayMarchMinMaxOptPS(SScreenSizeQuadVSOutput In) : SV_TARGET
{
    uint2 ui2SamplePosSliceInd = uint2(In.m_f4Pos.xy);
    float2 f2SampleLocation = g_tex2DCoordinates.Load( uint3(ui2SamplePosSliceInd, 0) );
    float fRayEndCamSpaceZ = g_tex2DEpipolarCamSpaceZ.Load( uint3(ui2SamplePosSliceInd, 0) );

    [branch]
    if( any(abs(f2SampleLocation) > 1+1e-3) )
        return 0;

    float fCascade = g_MiscParams.fCascadeInd + In.m_fInstID;
    float fCloudTransparency = 0, fDistToCloud = 0;
#if ENABLE_CLOUDS
    float2 f2UV = ProjToUV(f2SampleLocation);
    fCloudTransparency = g_tex2DEpipolarCloudTransparency.Load( uint3(ui2SamplePosSliceInd, 0) );
    // Use point sampling to be consistent with cloud tranparency
    fDistToCloud       = g_tex2DScrSpaceCloudMinMaxDist.SampleLevel( samPointClamp, f2UV, 0).x;
#endif

#if ENABLE_LIGHT_SHAFTS
    return ComputeShadowedInscattering(f2SampleLocation, 
                                 fRayEndCamSpaceZ,
                                 fCascade,
                                 fCloudTransparency,
                                 fDistToCloud,
                                 true,  // Use min/max optimization
                                 ui2SamplePosSliceInd.y);
#else
    float3 f3Inscattering, f3Extinction;
    ComputeUnshadowedInscattering(f2SampleLocation, fRayEndCamSpaceZ, fCloudTransparency, fDistToCloud, g_PPAttribs.m_uiInstrIntegralSteps, f3Inscattering, f3Extinction);
    f3Inscattering *= g_LightAttribs.f4ExtraterrestrialSunColor.rgb;
    return f3Inscattering;
#endif
}

float3 RayMarchPS(SScreenSizeQuadVSOutput In) : SV_TARGET
{
    float2 f2SampleLocation = g_tex2DCoordinates.Load( uint3(In.m_f4Pos.xy, 0) );
    float fRayEndCamSpaceZ = g_tex2DEpipolarCamSpaceZ.Load( uint3(In.m_f4Pos.xy, 0) );

    [branch]
    if( any(abs(f2SampleLocation) > 1+1e-3) )
        return 0;

    float fCloudTransparency = 0, fDistToCloud = 0;
#if ENABLE_CLOUDS
    float2 f2UV = ProjToUV(f2SampleLocation);
    fCloudTransparency = g_tex2DEpipolarCloudTransparency.Load( uint3(In.m_f4Pos.xy, 0) );
    // Use point sampling to be consistent with cloud tranparency
    fDistToCloud       = g_tex2DScrSpaceCloudMinMaxDist.SampleLevel( samPointClamp, f2UV, 0).x;
#endif

#if ENABLE_LIGHT_SHAFTS
    float fCascade = g_MiscParams.fCascadeInd + In.m_fInstID;
    return ComputeShadowedInscattering(f2SampleLocation, 
                                 fRayEndCamSpaceZ,
                                 fCascade,
                                 fCloudTransparency,
                                 fDistToCloud,
                                 false, // Do not use min/max optimization
                                 0 // Ignored
                                 );
#else
    float3 f3Inscattering, f3Extinction;
    ComputeUnshadowedInscattering(f2SampleLocation, fRayEndCamSpaceZ, fCloudTransparency, fDistToCloud, g_PPAttribs.m_uiInstrIntegralSteps, f3Inscattering, f3Extinction);
    f3Inscattering *= g_LightAttribs.f4ExtraterrestrialSunColor.rgb;
    return f3Inscattering;
#endif
}

technique10 DoRayMarch
{
    pass P0
    {
        // Skip all samples which are not marked in the stencil as ray marching
        SetDepthStencilState( DSS_NoDepth_StEqual_IncrStencil, 2 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader(NULL);
        SetPixelShader( CompileShader( ps_5_0, RayMarchPS() ) );
    }

    pass P1
    {
        // Skip all samples which are not marked in the stencil as ray marching
        SetDepthStencilState( DSS_NoDepth_StEqual_IncrStencil, 2 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader(NULL);
        SetPixelShader( CompileShader( ps_5_0, RayMarchMinMaxOptPS() ) );
    }
}

float3 FixInscatteredRadiancePS(SScreenSizeQuadVSOutput In) : SV_Target
{
    if( g_PPAttribs.m_bShowDepthBreaks )
        return float3(0,1,0);

    float2 f2UV = ProjToUV(In.m_f2PosPS.xy);
    float fCascade = g_MiscParams.fCascadeInd + In.m_fInstID;
    float fRayEndCamSpaceZ = g_tex2DCamSpaceZ.SampleLevel( samLinearClamp, f2UV, 0 );

    float fCloudTransparency = 0, fDistToCloud = 0;
#if ENABLE_CLOUDS
    fCloudTransparency = g_tex2DScrSpaceCloudTransparency.SampleLevel( samLinearClamp, f2UV, 0);
    fDistToCloud       = g_tex2DScrSpaceCloudMinMaxDist.SampleLevel( samLinearClamp, f2UV, 0).x;
#endif

#if ENABLE_LIGHT_SHAFTS
    return ComputeShadowedInscattering(In.m_f2PosPS.xy, 
                              fRayEndCamSpaceZ,
                              fCascade,
                              fCloudTransparency,
                              fDistToCloud,
                              false, // We cannot use min/max optimization at depth breaks
                              0 // Ignored
                              );
#else
    float3 f3Inscattering, f3Extinction;
    ComputeUnshadowedInscattering(In.m_f2PosPS.xy, fRayEndCamSpaceZ, fCloudTransparency, fDistToCloud, g_PPAttribs.m_uiInstrIntegralSteps, f3Inscattering, f3Extinction);
    f3Inscattering *= g_LightAttribs.f4ExtraterrestrialSunColor.rgb;
    return f3Inscattering;
#endif

}

float3 FixAndApplyInscatteredRadiancePS(SScreenSizeQuadVSOutput In) : SV_Target
{
    if( g_PPAttribs.m_bShowDepthBreaks )
        return float3(0,1,0);
    float2 f2UV = ProjToUV(In.m_f2PosPS);
    float fCamSpaceZ = GetCamSpaceZ( f2UV );
    float3 f3BackgroundColor = 0;
    [branch]
    if( !g_PPAttribs.m_bShowLightingOnly )
    {
        f3BackgroundColor = g_tex2DColorBuffer.Load(int3(In.m_f4Pos.xy,0)).rgb;
        f3BackgroundColor *= (fCamSpaceZ > g_CameraAttribs.fFarPlaneZ) ? g_LightAttribs.f4ExtraterrestrialSunColor.rgb : 1;
        float3 f3ReconstructedPosWS = ProjSpaceXYZToWorldSpace(float3(In.m_f2PosPS.xy, fCamSpaceZ));
        float3 f3Extinction = GetExtinction(g_CameraAttribs.f4CameraPos.xyz, f3ReconstructedPosWS);
        f3BackgroundColor *= f3Extinction.rgb;
    }
    
    float fCascade = g_MiscParams.fCascadeInd + In.m_fInstID;

    float fCloudTransparency = 0, fDistToCloud = 0;
#if ENABLE_CLOUDS
    float3 f3CloudsColor = g_tex2DScrSpaceCloudColor.SampleLevel( samLinearClamp, f2UV, 0).rgb;
    fCloudTransparency = g_tex2DScrSpaceCloudTransparency.SampleLevel( samLinearClamp, f2UV, 0);
    fDistToCloud       = g_tex2DScrSpaceCloudMinMaxDist.SampleLevel( samLinearClamp, f2UV, 0).x;

    float3 f3CloudExtinction = 1;
    [branch]
    if( fDistToCloud < +FLT_MAX*0.9 )
    {
        float3 f3ReconstructedPosWS = ProjSpaceXYZToWorldSpace(float3(In.m_f2PosPS.xy, fCamSpaceZ));
        float3 f3ViewDir = normalize(f3ReconstructedPosWS - g_CameraAttribs.f4CameraPos.xyz);
        f3CloudExtinction = GetExtinction(g_CameraAttribs.f4CameraPos.xyz, f3ViewDir, fDistToCloud);
    }
    f3CloudsColor *= f3CloudExtinction;
    f3BackgroundColor = f3BackgroundColor*fCloudTransparency + f3CloudsColor;
#endif

#if ENABLE_LIGHT_SHAFTS
    float3 f3InsctrColor = 
        ComputeShadowedInscattering(In.m_f2PosPS.xy, 
                              fCamSpaceZ,
                              fCascade,
                              fCloudTransparency,
                              fDistToCloud,
                              false, // We cannot use min/max optimization at depth breaks
                              0 // Ignored
                              );
#else
    float3 f3InsctrColor, f3Extinction;
    ComputeUnshadowedInscattering(In.m_f2PosPS.xy, fCamSpaceZ, fCloudTransparency, fDistToCloud, g_PPAttribs.m_uiInstrIntegralSteps, f3InsctrColor, f3Extinction);
    f3InsctrColor *= g_LightAttribs.f4ExtraterrestrialSunColor.rgb;
#endif

#if PERFORM_TONE_MAPPING
    return ToneMap(f3BackgroundColor + f3InsctrColor);
#else
    const float DELTA = 0.00001;
    return log( max(DELTA, dot(f3BackgroundColor + f3InsctrColor, RGB_TO_LUMINANCE)) );
#endif
}

technique11 FixInscatteredRadiance
{
    pass PRenderScatteringOnly
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_NoDepth_StEqual_IncrStencil, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, FixAndApplyInscatteredRadiancePS() ) );
    }
}

void RenderCoarseUnshadowedInsctrPS(SScreenSizeQuadVSOutput In,
                                    out float3 f3Inscattering : SV_Target0
#if EXTINCTION_EVAL_MODE == EXTINCTION_EVAL_MODE_EPIPOLAR
                                  , out float3 f3Extinction   : SV_Target1
#endif
                                  ) 
{
    // Compute unshadowed inscattering from the camera to the ray end point using few steps
    float fCamSpaceZ =  g_tex2DEpipolarCamSpaceZ.Load( uint3(In.m_f4Pos.xy, 0) );
    float2 f2SampleLocation = g_tex2DCoordinates.Load( uint3(In.m_f4Pos.xy, 0) );
#if EXTINCTION_EVAL_MODE != EXTINCTION_EVAL_MODE_EPIPOLAR
    float3 f3Extinction = 1;
#endif

    float fCloudTransparency = 0, fDistToCloud = +FLT_MAX;
#if ENABLE_CLOUDS
    float2 f2UV = ProjToUV(f2SampleLocation);
    fCloudTransparency = g_tex2DEpipolarCloudTransparency.Load( uint3(In.m_f4Pos.xy, 0) );
    fDistToCloud       = g_tex2DScrSpaceCloudMinMaxDist.SampleLevel( samPointClamp, f2UV, 0).x;
#endif
    ComputeUnshadowedInscattering(f2SampleLocation, fCamSpaceZ, fCloudTransparency, fDistToCloud,
                                  7, // Use hard-coded constant here so that compiler can optimize the code
                                     // more efficiently
                                  f3Inscattering, f3Extinction);
    f3Inscattering *= g_LightAttribs.f4ExtraterrestrialSunColor.rgb;
}

technique11 RenderCoarseUnshadowedInsctr
{
    pass PRenderScatteringOnly
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_NoDepth_StEqual_KeepStencil, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, RenderCoarseUnshadowedInsctrPS() ) );
    }
}

float2 IntegrateParticleDensity(in float3 f3Start, 
                                in float3 f3End,
                                in float3 f3EarthCentre,
                                float fNumSteps )
{
    float3 f3Step = (f3End - f3Start) / fNumSteps;
    float fStepLen = length(f3Step);
        
    float fStartHeightAboveSurface = abs( length(f3Start - f3EarthCentre) - g_MediaParams.fEarthRadius );
    float2 f2PrevParticleDensity = exp( -fStartHeightAboveSurface / g_MediaParams.f2ParticleScaleHeight );

    float2 f2ParticleNetDensity = 0;
    for(float fStepNum = 1; fStepNum <= fNumSteps; fStepNum += 1.f)
    {
        float3 f3CurrPos = f3Start + f3Step * fStepNum;
        float fHeightAboveSurface = abs( length(f3CurrPos - f3EarthCentre) - g_MediaParams.fEarthRadius );
        float2 f2ParticleDensity = exp( -fHeightAboveSurface / g_MediaParams.f2ParticleScaleHeight );
        f2ParticleNetDensity += (f2ParticleDensity + f2PrevParticleDensity) * fStepLen / 2.f;
        f2PrevParticleDensity = f2ParticleDensity;
    }
    return f2ParticleNetDensity;
}


float2 IntegrateParticleDensityAlongRay(in float3 f3Pos, 
                                        in float3 f3RayDir,
                                        float3 f3EarthCentre, 
                                        uniform const float fNumSteps,
                                        uniform const bool bOccludeByEarth)
{
    if( bOccludeByEarth )
    {
        // If the ray intersects the Earth, return huge optical depth
        float2 f2RayEarthIsecs; 
        GetRaySphereIntersection(f3Pos, f3RayDir, f3EarthCentre, g_MediaParams.fEarthRadius, f2RayEarthIsecs);
        if( f2RayEarthIsecs.x > 0 )
            return 1e+20;
    }

    // Get intersection with the top of the atmosphere (the start point must always be under the top of it)
    //      
    //                     /
    //                .   /  . 
    //      .  '         /\         '  .
    //                  /  f2RayAtmTopIsecs.y > 0
    //                 *
    //                   f2RayAtmTopIsecs.x < 0
    //                  /
    //      
    float2 f2RayAtmTopIsecs;
    GetRaySphereIntersection(f3Pos, f3RayDir, f3EarthCentre, g_MediaParams.fAtmTopRadius, f2RayAtmTopIsecs);
    float fIntegrationDist = f2RayAtmTopIsecs.y;

    float3 f3RayEnd = f3Pos + f3RayDir * fIntegrationDist;

    return IntegrateParticleDensity(f3Pos, f3RayEnd, f3EarthCentre, fNumSteps);
}

float2 PrecomputeNetDensityToAtmTopPS( SScreenSizeQuadVSOutput In ) : SV_Target0
{
    float2 f2UV = ProjToUV(In.m_f2PosPS);
    // Do not allow start point be at the Earth surface and on the top of the atmosphere
    float fStartHeight = clamp( lerp(0, g_MediaParams.fAtmTopHeight, f2UV.x), 10, g_MediaParams.fAtmTopHeight-10 );

    float fCosTheta = -In.m_f2PosPS.y;
    float fSinTheta = sqrt( saturate(1 - fCosTheta*fCosTheta) );
    float3 f3RayStart = float3(0, 0, fStartHeight);
    float3 f3RayDir = float3(fSinTheta, 0, fCosTheta);
    
    float3 f3EarthCentre = float3(0,0,-g_MediaParams.fEarthRadius);

    const float fNumSteps = 200;
    return IntegrateParticleDensityAlongRay(f3RayStart, f3RayDir, f3EarthCentre, fNumSteps, true);
}


technique11 PrecomputeNetDensityToAtmTopTech
{
    pass P0
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_NoDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, PrecomputeNetDensityToAtmTopPS() ) );
    }
}

// This function for analytical evaluation of particle density integral is 
// provided by Eric Bruneton
// http://www-evasion.inrialpes.fr/Membres/Eric.Bruneton/
//
// optical depth for ray (r,mu) of length d, using analytic formula
// (mu=cos(view zenith angle)), intersections with ground ignored
float2 GetDensityIntegralAnalytic(float r, float mu, float d) 
{
    float2 f2A = sqrt( (0.5/PARTICLE_SCALE_HEIGHT.xy) * r );
    float4 f4A01 = f2A.xxyy * float2(mu, mu + d / r).xyxy;
    float4 f4A01s = sign(f4A01);
    float4 f4A01sq = f4A01*f4A01;
    
    float2 f2X;
    f2X.x = f4A01s.y > f4A01s.x ? exp(f4A01sq.x) : 0.0;
    f2X.y = f4A01s.w > f4A01s.z ? exp(f4A01sq.z) : 0.0;
    
    float4 f4Y = f4A01s / (2.3193*abs(f4A01) + sqrt(1.52*f4A01sq + 4.0)) * float3(1.0, exp(-d/PARTICLE_SCALE_HEIGHT.xy*(d/(2.0*r)+mu))).xyxz;

    return sqrt((6.2831*PARTICLE_SCALE_HEIGHT)*r) * exp((EARTH_RADIUS-r)/PARTICLE_SCALE_HEIGHT.xy) * (f2X + float2( dot(f4Y.xy, float2(1.0, -1.0)), dot(f4Y.zw, float2(1.0, -1.0)) ));
}

float3 GetExtinctionUnverified(in float3 f3StartPos, in float3 f3EndPos, float3 f3EyeDir, float3 f3EarthCentre)
{
#if 0
    float2 f2ParticleDensity = IntegrateParticleDensity(f3StartPos, f3EndPos, f3EarthCentre, 20);
#else
    float r = length(f3StartPos-f3EarthCentre);
    float fCosZenithAngle = dot(f3StartPos-f3EarthCentre, f3EyeDir) / r;
    float2 f2ParticleDensity = GetDensityIntegralAnalytic(r, fCosZenithAngle, length(f3StartPos - f3EndPos));
#endif

    // Get optical depth
    float3 f3TotalRlghOpticalDepth = g_MediaParams.f4RayleighExtinctionCoeff.rgb * f2ParticleDensity.x;
    float3 f3TotalMieOpticalDepth  = g_MediaParams.f4MieExtinctionCoeff.rgb * f2ParticleDensity.y;
        
    // Compute extinction
    float3 f3Extinction = exp( -(f3TotalRlghOpticalDepth + f3TotalMieOpticalDepth) );
    return f3Extinction;
}

float3 GetExtinction(in float3 f3StartPos, in float3 f3EyeDir, in float fRayLength)
{
    float3 f3EarthCentre = /*g_CameraAttribs.f4CameraPos.xyz*float3(1,0,1)*/ - float3(0,1,0) * EARTH_RADIUS;

    float2 f2RayAtmTopIsecs; 
    // Compute intersections of the view ray with the atmosphere
    GetRaySphereIntersection(f3StartPos, f3EyeDir, f3EarthCentre, ATM_TOP_RADIUS, f2RayAtmTopIsecs);
    // If the ray misses the atmosphere, there is no extinction
    if( f2RayAtmTopIsecs.y < 0 )return 1;

    // Do not let the start and end point be outside the atmosphere
    float3 f3EndPos = f3StartPos + f3EyeDir * min(f2RayAtmTopIsecs.y, fRayLength);
    f3StartPos += f3EyeDir * max(f2RayAtmTopIsecs.x, 0);

    return GetExtinctionUnverified(f3StartPos, f3EndPos, f3EyeDir, f3EarthCentre);
}

float3 GetExtinction(in float3 f3StartPos, in float3 f3EndPos)
{
    float3 f3EyeDir = f3EndPos - f3StartPos;
    float fRayLength = length(f3EyeDir);
    f3EyeDir /= fRayLength;

    return GetExtinction(f3StartPos, f3EyeDir, fRayLength);
}

float GetCosHorizonAnlge(float fHeight)
{
    // Due to numeric precision issues, fHeight might sometimes be slightly negative
    fHeight = max(fHeight, 0);
    return -sqrt(fHeight * (2*EARTH_RADIUS + fHeight) ) / (EARTH_RADIUS + fHeight);
}

float ZenithAngle2TexCoord(float fCosZenithAngle, float fHeight, in float fTexDim, float power, float fPrevTexCoord)
{
    fCosZenithAngle = fCosZenithAngle;
    float fTexCoord;
    float fCosHorzAngle = GetCosHorizonAnlge(fHeight);
    // When performing look-ups into the scattering texture, it is very important that all the look-ups are consistent
    // wrt to the horizon. This means that if the first look-up is above (below) horizon, then the second look-up
    // should also be above (below) horizon. 
    // We use previous texture coordinate, if it is provided, to find out if previous look-up was above or below
    // horizon. If texture coordinate is negative, then this is the first look-up
    bool bIsAboveHorizon = fPrevTexCoord >= 0.5;
    bool bIsBelowHorizon = 0 <= fPrevTexCoord && fPrevTexCoord < 0.5;
    if(  bIsAboveHorizon || 
        !bIsBelowHorizon && (fCosZenithAngle > fCosHorzAngle) )
    {
        // Scale to [0,1]
        fTexCoord = saturate( (fCosZenithAngle - fCosHorzAngle) / (1 - fCosHorzAngle) );
        fTexCoord = pow(fTexCoord, power);
        // Now remap texture coordinate to the upper half of the texture.
        // To avoid filtering across discontinuity at 0.5, we must map
        // the texture coordinate to [0.5 + 0.5/fTexDim, 1 - 0.5/fTexDim]
        //
        //      0.5   1.5               D/2+0.5        D-0.5  texture coordinate x dimension
        //       |     |                   |            |
        //    |  X  |  X  | .... |  X  ||  X  | .... |  X  |  
        //       0     1          D/2-1   D/2          D-1    texel index
        //
        fTexCoord = 0.5f + 0.5f / fTexDim + fTexCoord * (fTexDim/2 - 1) / fTexDim;
    }
    else
    {
        fTexCoord = saturate( (fCosHorzAngle - fCosZenithAngle) / (fCosHorzAngle - (-1)) );
        fTexCoord = pow(fTexCoord, power);
        // Now remap texture coordinate to the lower half of the texture.
        // To avoid filtering across discontinuity at 0.5, we must map
        // the texture coordinate to [0.5, 0.5 - 0.5/fTexDim]
        //
        //      0.5   1.5        D/2-0.5             texture coordinate x dimension
        //       |     |            |       
        //    |  X  |  X  | .... |  X  ||  X  | .... 
        //       0     1          D/2-1   D/2        texel index
        //
        fTexCoord = 0.5f / fTexDim + fTexCoord * (fTexDim/2 - 1) / fTexDim;
    }    

    return fTexCoord;
}

float TexCoord2ZenithAngle(float fTexCoord, float fHeight, in float fTexDim, float power)
{
    float fCosZenithAngle;

    float fCosHorzAngle = GetCosHorizonAnlge(fHeight);
    if( fTexCoord > 0.5 )
    {
        // Remap to [0,1] from the upper half of the texture [0.5 + 0.5/fTexDim, 1 - 0.5/fTexDim]
        fTexCoord = saturate( (fTexCoord - (0.5f + 0.5f / fTexDim)) * fTexDim / (fTexDim/2 - 1) );
        fTexCoord = pow(fTexCoord, 1/power);
        // Assure that the ray does NOT hit Earth
        fCosZenithAngle = max( (fCosHorzAngle + fTexCoord * (1 - fCosHorzAngle)), fCosHorzAngle + 1e-4);
    }
    else
    {
        // Remap to [0,1] from the lower half of the texture [0.5, 0.5 - 0.5/fTexDim]
        fTexCoord = saturate((fTexCoord - 0.5f / fTexDim) * fTexDim / (fTexDim/2 - 1));
        fTexCoord = pow(fTexCoord, 1/power);
        // Assure that the ray DOES hit Earth
        fCosZenithAngle = min( (fCosHorzAngle - fTexCoord * (fCosHorzAngle - (-1))), fCosHorzAngle - 1e-4);
    }
    return fCosZenithAngle;
}

static const float SafetyHeightMargin = 16.f;
#define NON_LINEAR_PARAMETERIZATION 1
static const float HeightPower = 0.5f;
static const float ViewZenithPower = 0.2;
static const float SunViewPower = 1.5f;

void InsctrLUTCoords2WorldParams(in float4 f4UVWQ,
                                 out float fHeight,
                                 out float fCosViewZenithAngle,
                                 out float fCosSunZenithAngle,
                                 out float fCosSunViewAngle)
{
#if NON_LINEAR_PARAMETERIZATION
    // Rescale to exactly 0,1 range
    f4UVWQ.xzw = saturate((f4UVWQ* PRECOMPUTED_SCTR_LUT_DIM - 0.5) / (PRECOMPUTED_SCTR_LUT_DIM-1)).xzw;

    f4UVWQ.x = pow( f4UVWQ.x, 1/HeightPower );
    // Allowable height range is limited to [SafetyHeightMargin, AtmTopHeight - SafetyHeightMargin] to
    // avoid numeric issues at the Earth surface and the top of the atmosphere
    fHeight = f4UVWQ.x * (g_MediaParams.fAtmTopHeight - 2*SafetyHeightMargin) + SafetyHeightMargin;

    fCosViewZenithAngle = TexCoord2ZenithAngle(f4UVWQ.y, fHeight, PRECOMPUTED_SCTR_LUT_DIM.y, ViewZenithPower);
    
    // Use Eric Bruneton's formula for cosine of the sun-zenith angle
    fCosSunZenithAngle = tan((2.0 * f4UVWQ.z - 1.0 + 0.26) * 1.1) / tan(1.26 * 1.1);

    f4UVWQ.w = sign(f4UVWQ.w - 0.5) * pow( abs((f4UVWQ.w - 0.5)*2), 1/SunViewPower)/2 + 0.5;
    fCosSunViewAngle = cos(f4UVWQ.w*PI);
#else
    // Rescale to exactly 0,1 range
    f4UVWQ = (f4UVWQ * PRECOMPUTED_SCTR_LUT_DIM - 0.5) / (PRECOMPUTED_SCTR_LUT_DIM-1);

    // Allowable height range is limited to [SafetyHeightMargin, AtmTopHeight - SafetyHeightMargin] to
    // avoid numeric issues at the Earth surface and the top of the atmosphere
    fHeight = f4UVWQ.x * (g_MediaParams.fAtmTopHeight - 2*SafetyHeightMargin) + SafetyHeightMargin;

    fCosViewZenithAngle = f4UVWQ.y * 2 - 1;
    fCosSunZenithAngle  = f4UVWQ.z * 2 - 1;
    fCosSunViewAngle    = f4UVWQ.w * 2 - 1;
#endif

    fCosViewZenithAngle = clamp(fCosViewZenithAngle, -1, +1);
    fCosSunZenithAngle  = clamp(fCosSunZenithAngle,  -1, +1);
    // Compute allowable range for the cosine of the sun view angle for the given
    // view zenith and sun zenith angles
    float D = (1.0 - fCosViewZenithAngle * fCosViewZenithAngle) * (1.0 - fCosSunZenithAngle  * fCosSunZenithAngle);
    
    // !!!!  IMPORTANT NOTE regarding NVIDIA hardware !!!!

    // There is a very weird issue on NVIDIA hardware with clamp(), saturate() and min()/max() 
    // functions. No matter what function is used, fCosViewZenithAngle and fCosSunZenithAngle
    // can slightly fall outside [-1,+1] range causing D to be negative
    // Using saturate(D), max(D, 0) and even D>0?D:0 does not work!
    // The only way to avoid taking the square root of negative value and obtaining NaN is 
    // to use max() with small positive value:
    D = sqrt( max(D, 1e-20) );
    
    // The issue was reproduceable on NV GTX 680, driver version 9.18.13.2723 (9/12/2013).
    // The problem does not arise on Intel hardware

    float2 f2MinMaxCosSunViewAngle = fCosViewZenithAngle*fCosSunZenithAngle + float2(-D, +D);
    // Clamp to allowable range
    fCosSunViewAngle    = clamp(fCosSunViewAngle, f2MinMaxCosSunViewAngle.x, f2MinMaxCosSunViewAngle.y);
}

float4 WorldParams2InsctrLUTCoords(float fHeight,
                                   float fCosViewZenithAngle,
                                   float fCosSunZenithAngle,
                                   float fCosSunViewAngle,
                                   in float4 f4RefUVWQ)
{
    float4 f4UVWQ;

    // Limit allowable height range to [SafetyHeightMargin, AtmTopHeight - SafetyHeightMargin] to
    // avoid numeric issues at the Earth surface and the top of the atmosphere
    // (ray/Earth and ray/top of the atmosphere intersection tests are unstable when fHeight == 0 and
    // fHeight == AtmTopHeight respectively)
    fHeight = clamp(fHeight, SafetyHeightMargin, g_MediaParams.fAtmTopHeight - SafetyHeightMargin);
    f4UVWQ.x = saturate( (fHeight - SafetyHeightMargin) / (g_MediaParams.fAtmTopHeight - 2*SafetyHeightMargin) );

#if NON_LINEAR_PARAMETERIZATION
    f4UVWQ.x = pow(f4UVWQ.x, HeightPower);

    f4UVWQ.y = ZenithAngle2TexCoord(fCosViewZenithAngle, fHeight, PRECOMPUTED_SCTR_LUT_DIM.y, ViewZenithPower, f4RefUVWQ.y);
    
    // Use Eric Bruneton's formula for cosine of the sun-zenith angle
    f4UVWQ.z = (atan(max(fCosSunZenithAngle, -0.1975) * tan(1.26 * 1.1)) / 1.1 + (1.0 - 0.26)) * 0.5;

    fCosSunViewAngle = clamp(fCosSunViewAngle, -1, +1);
    f4UVWQ.w = acos(fCosSunViewAngle) / PI;
    f4UVWQ.w = sign(f4UVWQ.w - 0.5) * pow( abs((f4UVWQ.w - 0.5)/0.5), SunViewPower)/2 + 0.5;
    
    f4UVWQ.xzw = ((f4UVWQ * (PRECOMPUTED_SCTR_LUT_DIM-1) + 0.5) / PRECOMPUTED_SCTR_LUT_DIM).xzw;
#else
    f4UVWQ.y = (fCosViewZenithAngle+1.f) / 2.f;
    f4UVWQ.z = (fCosSunZenithAngle +1.f) / 2.f;
    f4UVWQ.w = (fCosSunViewAngle   +1.f) / 2.f;

    f4UVWQ = (f4UVWQ * (PRECOMPUTED_SCTR_LUT_DIM-1) + 0.5) / PRECOMPUTED_SCTR_LUT_DIM;
#endif

    return f4UVWQ;
}

float3 ComputeViewDir(in float fCosViewZenithAngle)
{
    return float3(sqrt(saturate(1 - fCosViewZenithAngle*fCosViewZenithAngle)), fCosViewZenithAngle, 0);
}

float3 ComputeLightDir(in float3 f3ViewDir, in float fCosSunZenithAngle, in float fCosSunViewAngle)
{
    float3 f3DirOnLight;
    f3DirOnLight.x = (f3ViewDir.x > 0) ? (fCosSunViewAngle - fCosSunZenithAngle * f3ViewDir.y) / f3ViewDir.x : 0;
    f3DirOnLight.y = fCosSunZenithAngle;
    f3DirOnLight.z = sqrt( saturate(1 - dot(f3DirOnLight.xy, f3DirOnLight.xy)) );
    // Do not normalize f3DirOnLight! Even if its length is not exactly 1 (which can 
    // happen because of fp precision issues), all the dot products will still be as 
    // specified, which is essentially important. If we normalize the vector, all the 
    // dot products will deviate, resulting in wrong pre-computation.
    // Since fCosSunViewAngle is clamped to allowable range, f3DirOnLight should always
    // be normalized. However, due to some issues on NVidia hardware sometimes
    // it may not be as that (see IMPORTANT NOTE regarding NVIDIA hardware)
    //f3DirOnLight = normalize(f3DirOnLight);
    return f3DirOnLight;
}

// This shader pre-computes the radiance of single scattering at a given point in given
// direction.
float3 PrecomputeSingleScatteringPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    // Get attributes for the current point
    float2 f2UV = ProjToUV(In.m_f2PosPS);
    float fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle;
    InsctrLUTCoords2WorldParams(float4(f2UV, g_MiscParams.f2WQ), fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle );
    float3 f3EarthCentre =  - float3(0,1,0) * EARTH_RADIUS;
    float3 f3RayStart = float3(0, fHeight, 0);
    float3 f3ViewDir = ComputeViewDir(fCosViewZenithAngle);
    float3 f3DirOnLight = ComputeLightDir(f3ViewDir, fCosSunZenithAngle, fCosSunViewAngle);
  
    // Intersect view ray with the top of the atmosphere and the Earth
    float4 f4Isecs;
    GetRaySphereIntersection2( f3RayStart, f3ViewDir, f3EarthCentre, 
                               float2(EARTH_RADIUS, ATM_TOP_RADIUS), 
                               f4Isecs);
    float2 f2RayEarthIsecs  = f4Isecs.xy;
    float2 f2RayAtmTopIsecs = f4Isecs.zw;

    if(f2RayAtmTopIsecs.y <= 0)
        return 0; // This is just a sanity check and should never happen
                  // as the start point is always under the top of the 
                  // atmosphere (look at InsctrLUTCoords2WorldParams())

    // Set the ray length to the distance to the top of the atmosphere
    float fRayLength = f2RayAtmTopIsecs.y;
    // If ray hits Earth, limit the length by the distance to the surface
    if(f2RayEarthIsecs.x > 0)
        fRayLength = min(fRayLength, f2RayEarthIsecs.x);
    
    float3 f3RayEnd = f3RayStart + f3ViewDir * fRayLength;

    float fCloudTransparency = 1;
    float fDitToCloud = +FLT_MAX;
    // Integrate single-scattering
    float3 f3Inscattering, f3Extinction;
    IntegrateUnshadowedInscattering(f3RayStart, 
                                    f3RayEnd,
                                    f3ViewDir,
                                    f3EarthCentre,
                                    f3DirOnLight.xyz,
                                    100,
                                    f3Inscattering,
                                    f3Extinction,
                                    fCloudTransparency,
                                    fDitToCloud);

    return f3Inscattering;
}

// This shader pre-computes the radiance of light scattered at a given point in given
// direction. It multiplies the previous order in-scattered light with the phase function 
// for each type of particles and integrates the result over the whole set of directions,
// see eq. (7) in [Bruneton and Neyret 08].
float3 ComputeSctrRadiancePS(SScreenSizeQuadVSOutput In) : SV_Target
{
    // Get attributes for the current point
    float2 f2UV = ProjToUV(In.m_f2PosPS);
    float fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle;
    InsctrLUTCoords2WorldParams( float4(f2UV, g_MiscParams.f2WQ), fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle );
    float3 f3EarthCentre =  - float3(0,1,0) * EARTH_RADIUS;
    float3 f3RayStart = float3(0, fHeight, 0);
    float3 f3ViewDir = ComputeViewDir(fCosViewZenithAngle);
    float3 f3DirOnLight = ComputeLightDir(f3ViewDir, fCosSunZenithAngle, fCosSunViewAngle);
    
    // Compute particle density scale factor
    float2 f2ParticleDensity = exp( -fHeight / PARTICLE_SCALE_HEIGHT );
    
    float3 f3SctrRadiance = 0;
    // Go through a number of samples randomly distributed over the sphere
    for(int iSample = 0; iSample < NUM_RANDOM_SPHERE_SAMPLES; ++iSample)
    {
        // Get random direction
        float3 f3RandomDir = normalize( g_tex2DSphereRandomSampling.Load(int3(iSample,0,0)) );
        // Get the previous order in-scattered light when looking in direction f3RandomDir (the light thus goes in direction -f3RandomDir)
        float4 f4UVWQ = -1;
        float3 f3PrevOrderSctr = LookUpPrecomputedScattering(f3RayStart, f3RandomDir, f3EarthCentre, f3DirOnLight.xyz, g_tex3DPreviousSctrOrder, f4UVWQ); 
        
        // Apply phase functions for each type of particles
        // Note that total scattering coefficients are baked into the angular scattering coeffs
        float3 f3DRlghInsctr = f2ParticleDensity.x * f3PrevOrderSctr;
        float3 f3DMieInsctr  = f2ParticleDensity.y * f3PrevOrderSctr;
        float fCosTheta = dot(f3ViewDir, f3RandomDir);
        ApplyPhaseFunctions(f3DRlghInsctr, f3DMieInsctr, fCosTheta);

        f3SctrRadiance += f3DRlghInsctr + f3DMieInsctr;
    }
    // Since we tested N random samples, each sample covered 4*Pi / N solid angle
    // Note that our phase function is normalized to 1 over the sphere. For instance,
    // uniform phase function would be p(theta) = 1 / (4*Pi).
    // Notice that for uniform intensity I if we get N samples, we must obtain exactly I after
    // numeric integration
    return f3SctrRadiance * 4*PI / NUM_RANDOM_SPHERE_SAMPLES;
}

// This shader computes in-scattering order for a given point and direction. It performs integration of the 
// light scattered at particular point along the ray, see eq. (11) in [Bruneton and Neyret 08].
float3 ComputeScatteringOrderPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    // Get attributes for the current point
    float2 f2UV = ProjToUV(In.m_f2PosPS);
    float fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle;
    InsctrLUTCoords2WorldParams(float4(f2UV, g_MiscParams.f2WQ), fHeight, fCosViewZenithAngle, fCosSunZenithAngle, fCosSunViewAngle );
    float3 f3EarthCentre =  - float3(0,1,0) * EARTH_RADIUS;
    float3 f3RayStart = float3(0, fHeight, 0);
    float3 f3ViewDir = ComputeViewDir(fCosViewZenithAngle);
    float3 f3DirOnLight = ComputeLightDir(f3ViewDir, fCosSunZenithAngle, fCosSunViewAngle);
    
    // Intersect the ray with the atmosphere and Earth
    float4 f4Isecs;
    GetRaySphereIntersection2( f3RayStart, f3ViewDir, f3EarthCentre, 
                               float2(EARTH_RADIUS, ATM_TOP_RADIUS), 
                               f4Isecs);
    float2 f2RayEarthIsecs  = f4Isecs.xy;
    float2 f2RayAtmTopIsecs = f4Isecs.zw;

    if(f2RayAtmTopIsecs.y <= 0)
        return 0; // This is just a sanity check and should never happen
                  // as the start point is always under the top of the 
                  // atmosphere (look at InsctrLUTCoords2WorldParams())

    float fRayLength = f2RayAtmTopIsecs.y;
    if(f2RayEarthIsecs.x > 0)
        fRayLength = min(fRayLength, f2RayEarthIsecs.x);
    
    float3 f3RayEnd = f3RayStart + f3ViewDir * fRayLength;

    const float fNumSamples = 64;
    float fStepLen = fRayLength / fNumSamples;

    float4 f4UVWQ = -1;
    float3 f3PrevSctrRadiance = LookUpPrecomputedScattering(f3RayStart, f3ViewDir, f3EarthCentre, f3DirOnLight.xyz, g_tex3DPointwiseSctrRadiance, f4UVWQ); 
    float2 f2PrevParticleDensity = exp( -fHeight / PARTICLE_SCALE_HEIGHT );

    float2 f2NetParticleDensityFromCam = 0;
    float3 f3Inscattering = 0;

    for(float fSample=1; fSample <= fNumSamples; ++fSample)
    {
        float3 f3Pos = lerp(f3RayStart, f3RayEnd, fSample/fNumSamples);

        float fCurrHeight = length(f3Pos - f3EarthCentre) - EARTH_RADIUS;
        float2 f2ParticleDensity = exp( -fCurrHeight / PARTICLE_SCALE_HEIGHT );

        f2NetParticleDensityFromCam += (f2PrevParticleDensity + f2ParticleDensity) * (fStepLen / 2.f);
        f2PrevParticleDensity = f2ParticleDensity;
        
        // Get optical depth
        float3 f3RlghOpticalDepth = g_MediaParams.f4RayleighExtinctionCoeff.rgb * f2NetParticleDensityFromCam.x;
        float3 f3MieOpticalDepth  = g_MediaParams.f4MieExtinctionCoeff.rgb      * f2NetParticleDensityFromCam.y;
        
        // Compute extinction from the camera for the current integration point:
        float3 f3ExtinctionFromCam = exp( -(f3RlghOpticalDepth + f3MieOpticalDepth) );

        // Get attenuated scattered light radiance in the current point
        float4 f4UVWQ = -1;
        float3 f3SctrRadiance = f3ExtinctionFromCam * LookUpPrecomputedScattering(f3Pos, f3ViewDir, f3EarthCentre, f3DirOnLight.xyz, g_tex3DPointwiseSctrRadiance, f4UVWQ); 
        // Update in-scattering integral
        f3Inscattering += (f3SctrRadiance +  f3PrevSctrRadiance) * (fStepLen/2.f);
        f3PrevSctrRadiance = f3SctrRadiance;
    }

    return f3Inscattering;
}

float3 AddScatteringOrderPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    // Accumulate in-scattering using alpha-blending
    return g_tex3DPreviousSctrOrder.Load( uint4(In.m_f4Pos.xy, g_MiscParams.uiDepthSlice, 0) );
}

technique11 PrecomputeScatteringTech
{
    pass P0
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_NoDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, PrecomputeSingleScatteringPS() ) );
    }

    pass P1
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_NoDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, ComputeSctrRadiancePS() ) );
    }

    pass P2
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetDepthStencilState( DSS_NoDepthTest, 0 );

        SetVertexShader( CompileShader(vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, ComputeScatteringOrderPS() ) );
    }
}

float3 PrecomputeAmbientSkyLightPS(SScreenSizeQuadVSOutput In) : SV_Target
{
    float fU = ProjToUV(In.m_f2PosPS).x;
    float3 f3RayStart = float3(0,20,0);
    float3 f3EarthCentre =  -float3(0,1,0) * EARTH_RADIUS;
    float fCosZenithAngle = clamp(fU * 2 - 1, -1, +1);
    float3 f3DirOnLight = float3(sqrt(saturate(1 - fCosZenithAngle*fCosZenithAngle)), fCosZenithAngle, 0);
    float3 f3SkyLight = 0;
    // Go through a number of random directions on the sphere
    for(int iSample = 0; iSample < NUM_RANDOM_SPHERE_SAMPLES; ++iSample)
    {
        // Get random direction
        float3 f3RandomDir = normalize( g_tex2DSphereRandomSampling.Load(int3(iSample,0,0)) );
        // Reflect directions from the lower hemisphere
        f3RandomDir.y = abs(f3RandomDir.y);
        // Get multiple scattered light radiance when looking in direction f3RandomDir (the light thus goes in direction -f3RandomDir)
        float4 f4UVWQ = -1;
        float3 f3Sctr = LookUpPrecomputedScattering(f3RayStart, f3RandomDir, f3EarthCentre, f3DirOnLight.xyz, g_tex3DPreviousSctrOrder, f4UVWQ); 
        // Accumulate ambient irradiance through the horizontal plane
        f3SkyLight += f3Sctr * dot(f3RandomDir, float3(0,1,0));
    }
    // Each sample covers 2 * PI / NUM_RANDOM_SPHERE_SAMPLES solid angle (integration is performed over
    // upper hemisphere)
    return f3SkyLight * 2 * PI / NUM_RANDOM_SPHERE_SAMPLES;
}

struct SSunVSOutput
{
    float4 m_f4Pos : SV_Position;
    float2 m_f2PosPS : PosPS; // Position in projection space [-1,1]x[-1,1]
};

static const float fSunAngularRadius =  32.f/2.f / 60.f * ((2.f * PI)/180); // Sun angular DIAMETER is 32 arc minutes
static const float fTanSunAngularRadius = tan(fSunAngularRadius);

SSunVSOutput SunVS(in uint VertexId : SV_VertexID)
{
    float2 fCotanHalfFOV = float2( g_CameraAttribs.mProj[0][0], g_CameraAttribs.mProj[1][1] );
    float2 f2SunScreenPos = g_LightAttribs.f4LightScreenPos.xy;
    float2 f2SunScreenSize = fTanSunAngularRadius * fCotanHalfFOV;
    float4 MinMaxUV = f2SunScreenPos.xyxy + float4(-1,-1,1,1) * f2SunScreenSize.xyxy;
 
    SSunVSOutput Verts[4] = 
    {
        {float4(MinMaxUV.xy, 0.0, 1.0), MinMaxUV.xy}, 
        {float4(MinMaxUV.xw, 0.0, 1.0), MinMaxUV.xw},
        {float4(MinMaxUV.zy, 0.0, 1.0), MinMaxUV.zy},
        {float4(MinMaxUV.zw, 0.0, 1.0), MinMaxUV.zw}
    };

    return Verts[VertexId];
}

float3 SunPS(SSunVSOutput In) : SV_Target
{
    float2 fCotanHalfFOV = float2( g_CameraAttribs.mProj[0][0], g_CameraAttribs.mProj[1][1] );
    float2 f2SunScreenSize = fTanSunAngularRadius * fCotanHalfFOV;
    float2 f2dXY = (In.m_f2PosPS - g_LightAttribs.f4LightScreenPos.xy) / f2SunScreenSize;
    return sqrt(saturate(1 - dot(f2dXY, f2dXY)));
}

technique11 RenderSunTech
{
    pass P0
    {
        SetBlendState( NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );
        SetRasterizerState( RS_SolidFill_NoCull );

        SetVertexShader( CompileShader(vs_5_0, SunVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader(ps_5_0, SunPS() ) );
    }
}
