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

#include "TerrainStructs.fxh"
#include "..\fx\Structures.fxh"
#include "..\fx\Common.fxh"

// Texturing modes
#define TM_HEIGHT_BASED 0             // Simple height-based texturing mode using 1D look-up table
#define TM_MATERIAL_MASK 1
#define TM_MATERIAL_MASK_NM 2

#ifndef TEXTURING_MODE
#   define TEXTURING_MODE TM_MATERIAL_MASK_NM
#endif

#ifndef NUM_TILE_TEXTURES
#	define NUM_TILE_TEXTURES 5
#endif

#ifndef NUM_SHADOW_CASCADES
#   define NUM_SHADOW_CASCADES 4
#endif

#ifndef ENABLE_CLOUDS
#   define ENABLE_CLOUDS 1
#endif

static const float g_fEarthReflectance = 0.4f;

cbuffer cbTerrainAttribs : register( b0 )
{
    STerrainAttribs g_TerrainAttribs;
}

cbuffer cbNMGenerationAttribs : register( b0 )
{
    SNMGenerationAttribs g_NMGenerationAttribs;
}

SamplerState samPointClamp : register( s0 )
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

SamplerState samLinearMirror : register( s1 )
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Mirror;
    AddressV = Mirror;
};

SamplerState samLinearWrap : register( s2 )
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

SamplerComparisonState samComparison : register (s3)
{
    Filter = COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};

Texture2D<float> g_tex2DElevationMap  : register( t0 );
// Normal map stores only x,y components. z component is calculated as sqrt(1 - x^2 - y^2)
Texture2D g_tex2DNormalMap            : register( t1 );
Texture2D<float4> g_tex2DMtrlMap      : register( t2 );
Texture2DArray<float4> g_tex2DShadowMap : register (t3);
Texture2DArray<float> g_tex2DLiSpCloudTransparency : register (t4);
Texture2D<float4> g_tex2DTileTextures[NUM_TILE_TEXTURES]   : register( t5 );   // Material texture
Texture2D<float3> g_tex2DTileNormalMaps[NUM_TILE_TEXTURES] : register( t10 );   // Material texture
//Texture2D<float3> g_tex2DElevationColor: register( t4 );
Texture2D<float3> g_tex2DAmbientSkylight            : register( t1 ); // Used in VS

#ifndef SMOOTH_SHADOWS
#   define SMOOTH_SHADOWS 1
#endif


float2 ComputeReceiverPlaneDepthBias(float3 ShadowUVDepthDX, float3 ShadowUVDepthDY)
{    
    // Compute (dDepth/dU, dDepth/dV):
    //  
    //  | dDepth/dU |    | dX/dU    dX/dV |T  | dDepth/dX |     | dU/dX    dU/dY |-1T | dDepth/dX |
    //                 =                                     =                                      =
    //  | dDepth/dV |    | dY/dU    dY/dV |   | dDepth/dY |     | dV/dX    dV/dY |    | dDepth/dY |
    //
    //  | A B |-1   | D  -B |                      | A B |-1T   | D  -C |                                   
    //            =           / det                           =           / det                    
    //  | C D |     |-C   A |                      | C D |      |-B   A |
    //
    //  | dDepth/dU |           | dV/dY   -dV/dX |  | dDepth/dX |
    //                 = 1/det                                       
    //  | dDepth/dV |           |-dU/dY    dU/dX |  | dDepth/dY |

    float2 biasUV;
    //               dV/dY       V      dDepth/dX    D       dV/dX       V     dDepth/dY     D
    biasUV.x =   ShadowUVDepthDY.y * ShadowUVDepthDX.z - ShadowUVDepthDX.y * ShadowUVDepthDY.z;
    //               dU/dY       U      dDepth/dX    D       dU/dX       U     dDepth/dY     D
    biasUV.y = - ShadowUVDepthDY.x * ShadowUVDepthDX.z + ShadowUVDepthDX.x * ShadowUVDepthDY.z;

    float Det = (ShadowUVDepthDX.x * ShadowUVDepthDY.y) - (ShadowUVDepthDX.y * ShadowUVDepthDY.x);
	biasUV /= sign(Det) * max( abs(Det), 1e-20 );
    //biasUV = abs(Det) > 1e-7 ? biasUV / abs(Det) : 0;// sign(Det) * max( abs(Det), 1e-10 );
    return biasUV;
}

float ComputeShadowAmount(in float3 f3PosInLightViewSpace, in float fCameraSpaceZ, out float3 f3ShadowMapUVDepth, out float Cascade)
{
    float3 f3PosInCascadeProjSpace = 0, f3CascadeLightSpaceScale = 0;
    FindCascade( f3PosInLightViewSpace.xyz, fCameraSpaceZ, f3PosInCascadeProjSpace, f3CascadeLightSpaceScale, Cascade);
    if( Cascade == NUM_SHADOW_CASCADES )
        return 1;

    f3ShadowMapUVDepth.xy = float2(0.5, 0.5) + float2(0.5, -0.5) * f3PosInCascadeProjSpace.xy;
    f3ShadowMapUVDepth.z = f3PosInCascadeProjSpace.z;
        
    float3 f3ddXShadowMapUVDepth = ddx(f3PosInLightViewSpace) * f3CascadeLightSpaceScale * float3(0.5,-0.5,1);
    float3 f3ddYShadowMapUVDepth = ddy(f3PosInLightViewSpace) * f3CascadeLightSpaceScale * float3(0.5,-0.5,1);

    float2 f2DepthSlopeScaledBias = ComputeReceiverPlaneDepthBias(f3ddXShadowMapUVDepth, f3ddYShadowMapUVDepth);
    float2 ShadowMapDim; float Elems;
    g_tex2DShadowMap.GetDimensions(ShadowMapDim.x, ShadowMapDim.y, Elems);
    f2DepthSlopeScaledBias /= ShadowMapDim.xy;

    float fractionalSamplingError = dot( float2(1.f, 1.f), abs(f2DepthSlopeScaledBias.xy) );
    f3ShadowMapUVDepth.z += fractionalSamplingError;
    
    float fLightAmount = g_tex2DShadowMap.SampleCmp(samComparison, float3(f3ShadowMapUVDepth.xy, Cascade), float(f3ShadowMapUVDepth.z)).x;

#if SMOOTH_SHADOWS
        int2 Offsets[] = 
        {
            int2(-1,-1),
            int2(+1,-1),
            int2(-1,+1),
            int2(+1,+1),
        };
        [unroll]
        for(int i=0; i<4; ++i)
        {
            float fDepthBias = dot(Offsets[i].xy, f2DepthSlopeScaledBias.xy);
            fLightAmount += g_tex2DShadowMap.SampleCmp(samComparison, float3(f3ShadowMapUVDepth.xy, Cascade), f3ShadowMapUVDepth.z + fDepthBias, Offsets[i]).x;
        }
        fLightAmount /= 5;
#endif
    return fLightAmount;
}

void CombineMaterials(in float4 MtrlWeights,
                      in float2 f2TileUV,
                      out float3 SurfaceColor,
                      out float3 SurfaceNormalTS)
{
    SurfaceNormalTS = 0;
    // Normalize weights and compute base material weight
    MtrlWeights /= max( dot(MtrlWeights, float4(1,1,1,1)) , 1 );
    float BaseMaterialWeight = saturate(1 - dot(MtrlWeights, float4(1,1,1,1)));
    
    // The mask is already sharp

    ////Sharpen the mask
    //float2 TmpMin2 = min(MtrlWeights.rg, MtrlWeights.ba);
    //float Min = min(TmpMin2.r, TmpMin2.g);
    //Min = min(Min, BaseMaterialWeight);
    //float p = 4;
    //BaseMaterialWeight = pow(BaseMaterialWeight-Min, p);
    //MtrlWeights = pow(MtrlWeights-Min, p);
    //float NormalizationFactor = dot(MtrlWeights, float4(1,1,1,1)) + BaseMaterialWeight;
    //MtrlWeights /= NormalizationFactor;
    //BaseMaterialWeight /= NormalizationFactor;

	// Get diffuse color of the base material
    float4 BaseMaterialDiffuse = g_tex2DTileTextures[0].Sample(samLinearWrap, f2TileUV.xy / g_TerrainAttribs.m_fBaseMtrlTilingScale);
    float4x4 MaterialColors = (float4x4)0;

    // Get tangent space normal of the base material
#if TEXTURING_MODE == TM_MATERIAL_MASK_NM
    float3 BaseMaterialNormal = g_tex2DTileNormalMaps[0].Sample(samLinearWrap, f2TileUV.xy / g_TerrainAttribs.m_fBaseMtrlTilingScale);
    float4x3 MaterialNormals = (float4x3)0;
#endif

    float4 f4TilingScale = g_TerrainAttribs.m_f4TilingScale;
    float fTilingScale[5] = {0, f4TilingScale.x, f4TilingScale.y, f4TilingScale.z, f4TilingScale.w};
    // Load material colors and normals
    [unroll]for(int iTileTex = 1; iTileTex < NUM_TILE_TEXTURES; iTileTex++)
	{
        const float fThresholdWeight = 3.f/256.f;
        MaterialColors[iTileTex-1] = 
			MtrlWeights[iTileTex-1] > fThresholdWeight ? 
				g_tex2DTileTextures[iTileTex].Sample(samLinearWrap, f2TileUV.xy  / fTilingScale[iTileTex]) : 0.f;
#if TEXTURING_MODE == TM_MATERIAL_MASK_NM
        MaterialNormals[iTileTex-1] = 
			MtrlWeights[iTileTex-1] > fThresholdWeight ? 
				g_tex2DTileNormalMaps[iTileTex].Sample(samLinearWrap, f2TileUV.xy / fTilingScale[iTileTex]) : 0.f;
#endif
	}
    // Blend materials and normals using the weights
    SurfaceColor = BaseMaterialDiffuse.rgb * BaseMaterialWeight + mul(MtrlWeights, MaterialColors).rgb;

#if TEXTURING_MODE == TM_MATERIAL_MASK_NM
    SurfaceNormalTS = BaseMaterialNormal * BaseMaterialWeight + mul(MtrlWeights, MaterialNormals);
    SurfaceNormalTS = normalize(SurfaceNormalTS*2-1);
#endif
}

RasterizerState RS_SolidFill;//Set by the app; can be biased or not
//{
//    FILLMODE = Solid;
//    CullMode = Back;
//    FrontCounterClockwise = true;
//};

RasterizerState RS_SolidFill_NoCull
{
    FILLMODE = Solid;
    CullMode = None;
};

RasterizerState RS_Wireframe_NoCull
{
    FILLMODE = Wireframe;
    CullMode = None;
};

BlendState BS_DisableBlending
{
    BlendEnable[0] = FALSE;
    BlendEnable[1] = FALSE;
    BlendEnable[2] = FALSE;
};

DepthStencilState DSS_EnableDepthTest
{
    DepthEnable = TRUE;
    DepthWriteMask = ALL;
    DEPTHFUNC = GREATER;
};

DepthStencilState DSS_DisableDepthTest
{
    DepthEnable = FALSE;
    DepthWriteMask = ZERO;
};

struct SHemisphereVSOutput
{
    float4 f4PosPS : SV_Position;
    float2 TileTexUV  : TileTextureUV;
    float3 f3Normal : Normal;
    float3 f3PosInLightViewSpace : POS_IN_LIGHT_VIEW_SPACE;
    float fCameraSpaceZ : CAMERA_SPACE_Z;
    float2 f2MaskUV0 : MASK_UV0;
    float3 f3Tangent : TANGENT;
    float3 f3Bitangent : BITANGENT;
    float3 f3SunLightExtinction : EXTINCTION;
    float3 f3AmbientSkyLight : AMBIENT_SKY_LIGHT;
};

SHemisphereVSOutput HemisphereVS(in float3 f3PosWS : WORLD_POS,
                                 in float2 f2MaskUV0 : MASK0_UV)
{
    SHemisphereVSOutput Out;
    Out.TileTexUV = f3PosWS.xz;

    Out.f4PosPS = mul( float4(f3PosWS,1), g_CameraAttribs.WorldViewProj);
    
    float4 ShadowMapSpacePos = mul( float4(f3PosWS,1), g_LightAttribs.ShadowAttribs.mWorldToLightView);
    Out.f3PosInLightViewSpace = ShadowMapSpacePos.xyz / ShadowMapSpacePos.w;
    Out.fCameraSpaceZ = Out.f4PosPS.w;
    Out.f2MaskUV0 = f2MaskUV0;
    float3 f3Normal = normalize(f3PosWS - float3(0, -g_TerrainAttribs.m_fEarthRadius, 0));
    Out.f3Normal = f3Normal;
    Out.f3Tangent = normalize( cross(f3Normal, float3(0,0,1)) );
    Out.f3Bitangent = normalize( cross(Out.f3Tangent, f3Normal) );

    GetSunLightExtinctionAndSkyLight(f3PosWS, Out.f3SunLightExtinction, Out.f3AmbientSkyLight, g_tex2DOccludedNetDensityToAtmTop, g_tex2DAmbientSkylight);

    return Out;
}

float4 HemisphereZOnlyVS(in float3 f3PosWS : WORLD_POS) : SV_Position
{
    float4 f4PosPS = mul( float4(f3PosWS,1), g_CameraAttribs.WorldViewProj);
    return f4PosPS;
}

float3 HemispherePS(SHemisphereVSOutput In) : SV_Target
{
    float3 EarthNormal = normalize(In.f3Normal);
    float3 EarthTangent = normalize(In.f3Tangent);
    float3 EarthBitangent = normalize(In.f3Bitangent);
    float3 f3TerrainNormal;
    f3TerrainNormal.xz = g_tex2DNormalMap.Sample(samLinearMirror, In.f2MaskUV0.xy).xy;
    // Since UVs are mirrored, we have to adjust normal coords accordingly:
    float2 f2XZSign = sign( 0.5 - frac(In.f2MaskUV0.xy/2) );
    f3TerrainNormal.xz *= f2XZSign;

    f3TerrainNormal.y = sqrt( saturate(1 - dot(f3TerrainNormal.xz,f3TerrainNormal.xz)) );
    //float3 Tangent   = normalize(float3(1,0,In.HeightMapGradients.x));
    //float3 Bitangent = normalize(float3(0,1,In.HeightMapGradients.y));
    f3TerrainNormal = normalize( mul(f3TerrainNormal, float3x3(EarthTangent, EarthNormal, EarthBitangent)) );
    
    float4 MtrlWeights = g_tex2DMtrlMap.Sample(samLinearMirror, In.f2MaskUV0.xy);
    float3 SurfaceColor, SurfaceNormalTS;
    CombineMaterials(MtrlWeights, In.TileTexUV, SurfaceColor.xyz, SurfaceNormalTS);
    
    float3 f3TerrainTangent = normalize( cross(f3TerrainNormal, float3(0,0,1)) );
    float3 f3TerrainBitangent = normalize( cross(f3TerrainTangent, f3TerrainNormal) );
    float3 f3Normal = normalize( mul(SurfaceNormalTS.xzy, float3x3(f3TerrainTangent, f3TerrainNormal, f3TerrainBitangent)) );

    // Attenuate extraterrestrial sun color with the extinction factor
    float3 f3SunLight = g_LightAttribs.f4ExtraterrestrialSunColor.rgb * In.f3SunLightExtinction;
    // Ambient sky light is not pre-multiplied with the sun intensity
    float3 f3AmbientSkyLight = g_LightAttribs.f4ExtraterrestrialSunColor.rgb * In.f3AmbientSkyLight;
    // Account for occlusion by the ground plane
    f3AmbientSkyLight *= saturate((1 + dot(EarthNormal, f3Normal))/2.f);

    // We need to divide diffuse color by PI to get the reflectance value
    float3 SurfaceReflectance = SurfaceColor * g_fEarthReflectance / PI;

    float Cascade;
    float3 f3ShadowMapUVDepth = 0;
    float fLightAmount = ComputeShadowAmount(In.f3PosInLightViewSpace.xyz, In.fCameraSpaceZ, f3ShadowMapUVDepth, Cascade);
    float DiffuseIllumination = max(0, dot(f3Normal, g_LightAttribs.f4DirOnLight.xyz));
    
#if ENABLE_CLOUDS
    float fTransparency = 1.f;
    if( Cascade < NUM_SHADOW_CASCADES )
    {
        fTransparency = g_tex2DLiSpCloudTransparency.SampleLevel(samLinearClamp, float3(f3ShadowMapUVDepth.xy, Cascade), 0);
    }
    fLightAmount *= fTransparency;
#endif

    float fMaxShadowCamSpaceZ = g_LightAttribs.fMaxShadowCamSpaceZ;
    float fShadowStrength = saturate((fMaxShadowCamSpaceZ - In.fCameraSpaceZ) / (fMaxShadowCamSpaceZ*0.1));
    fLightAmount = lerp(1, fLightAmount, fShadowStrength);
    
    float3 f3CascadeColor = 0;
    if( g_LightAttribs.ShadowAttribs.bVisualizeCascades )
    {
        f3CascadeColor = (Cascade < NUM_SHADOW_CASCADES ? g_TerrainAttribs.f4CascadeColors[Cascade].rgb : float3(1,1,1)) / 8 ;
    }
    
    float3 f3FinalColor = f3CascadeColor +  SurfaceReflectance * (fLightAmount*DiffuseIllumination*f3SunLight + f3AmbientSkyLight);

    return f3FinalColor;
}

technique11 RenderHemisphereTech
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_5_0, HemisphereVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_5_0, HemispherePS() ) );
    }
}


struct SScreenSizeQuadVSOutput
{
    float4 m_f4Pos : SV_Position;
    float2 m_f2PosPS : PosPS; // Position in projection space [-1,1]x[-1,1]
};

SScreenSizeQuadVSOutput GenerateScreenSizeQuadVS(in uint VertexId : SV_VertexID)
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

float3 ComputeNormal(float2 f2ElevMapUV,
                     float fSampleSpacingInterval,
                     float fMIPLevel)
{
#   define GET_ELEV(Offset) g_tex2DElevationMap.SampleLevel( samPointClamp, f2ElevMapUV, fMIPLevel, Offset)

#if 1
    float Height00 = GET_ELEV( int2( -1, -1) );
    float Height10 = GET_ELEV( int2(  0, -1) );
    float Height20 = GET_ELEV( int2( +1, -1) );

    float Height01 = GET_ELEV( int2( -1, 0) );
    //float Height11 = GET_ELEV( int2(  0, 0) );
    float Height21 = GET_ELEV( int2( +1, 0) );

    float Height02 = GET_ELEV( int2( -1, +1) );
    float Height12 = GET_ELEV( int2(  0, +1) );
    float Height22 = GET_ELEV( int2( +1, +1) );

    float3 Grad;
    Grad.x = (Height00+Height01+Height02) - (Height20+Height21+Height22);
    Grad.y = (Height00+Height10+Height20) - (Height02+Height12+Height22);
    Grad.z = fSampleSpacingInterval * 6.f;
    //Grad.x = (3*Height00+10*Height01+3*Height02) - (3*Height20+10*Height21+3*Height22);
    //Grad.y = (3*Height00+10*Height10+3*Height20) - (3*Height02+10*Height12+3*Height22);
    //Grad.z = fSampleSpacingInterval * 32.f;
#else
    float Height1 = GET_ELEV( int2( 1, 0) );
    float Height2 = GET_ELEV( int2(-1, 0) );
    float Height3 = GET_ELEV( int2( 0, 1) );
    float Height4 = GET_ELEV( int2( 0,-1) );
       
    float3 Grad;
    Grad.x = Height2 - Height1;
    Grad.y = Height4 - Height3;
    Grad.z = fSampleSpacingInterval * 2.f;
#endif
    Grad.xy *= HEIGHT_MAP_SCALE*g_NMGenerationAttribs.m_fElevationScale;
    float3 Normal = normalize( Grad );

    return Normal;
}

float2 GenerateNormalMapPS(SScreenSizeQuadVSOutput In) : SV_TARGET
{
	float2 f2UV = float2(0.5,0.5) + float2(0.5,-0.5) * In.m_f2PosPS.xy;
    float3 Normal = ComputeNormal( f2UV, g_NMGenerationAttribs.m_fSampleSpacingInterval*exp2(g_NMGenerationAttribs.m_fMIPLevel), g_NMGenerationAttribs.m_fMIPLevel );
    // Only xy components are stored. z component is calculated in the shader
    return Normal.xy;
}

technique11 RenderNormalMapTech
{
    pass
    {
        SetDepthStencilState( DSS_DisableDepthTest, 0 );
        SetRasterizerState( RS_SolidFill_NoCull );
        SetBlendState( BS_DisableBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF );

        SetVertexShader( CompileShader( vs_5_0, GenerateScreenSizeQuadVS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_5_0, GenerateNormalMapPS() ) );
    }
}
