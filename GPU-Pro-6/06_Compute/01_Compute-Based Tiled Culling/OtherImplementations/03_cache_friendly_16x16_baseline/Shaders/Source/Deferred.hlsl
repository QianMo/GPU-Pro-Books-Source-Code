//
// Copyright 2014 ADVANCED MICRO DEVICES, INC.  All Rights Reserved.
//
// AMD is granting you permission to use this software and documentation (if
// any) (collectively, the “Materials”) pursuant to the terms and conditions
// of the Software License Agreement included with the Materials.  If you do
// not have a copy of the Software License Agreement, contact your AMD
// representative for a copy.
// You agree that you will not reverse engineer or decompile the Materials,
// in whole or in part, except as allowed by applicable law.
//
// WARRANTY DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND.  AMD DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, THAT THE SOFTWARE
// WILL RUN UNINTERRUPTED OR ERROR-FREE OR WARRANTIES ARISING FROM CUSTOM OF
// TRADE OR COURSE OF USAGE.  THE ENTIRE RISK ASSOCIATED WITH THE USE OF THE
// SOFTWARE IS ASSUMED BY YOU.
// Some jurisdictions do not allow the exclusion of implied warranties, so
// the above exclusion may not apply to You. 
// 
// LIMITATION OF LIABILITY AND INDEMNIFICATION:  AMD AND ITS LICENSORS WILL
// NOT, UNDER ANY CIRCUMSTANCES BE LIABLE TO YOU FOR ANY PUNITIVE, DIRECT,
// INCIDENTAL, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF
// THE SOFTWARE OR THIS AGREEMENT EVEN IF AMD AND ITS LICENSORS HAVE BEEN
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.  
// In no event shall AMD's total liability to You for all damages, losses,
// and causes of action (whether in contract, tort (including negligence) or
// otherwise) exceed the amount of $100 USD.  You agree to defend, indemnify
// and hold harmless AMD and its licensors, and any of their directors,
// officers, employees, affiliates or agents from and against any and all
// loss, damage, liability and other expenses (including reasonable attorneys'
// fees), resulting from Your use of the Software or violation of the terms and
// conditions of this Agreement.  
//
// U.S. GOVERNMENT RESTRICTED RIGHTS: The Materials are provided with "RESTRICTED
// RIGHTS." Use, duplication, or disclosure by the Government is subject to the
// restrictions as set forth in FAR 52.227-14 and DFAR252.227-7013, et seq., or
// its successor.  Use of the Materials by the Government constitutes
// acknowledgement of AMD's proprietary rights in them.
// 
// EXPORT RESTRICTIONS: The Materials may be subject to export restrictions as
// stated in the Software License Agreement.
//

//--------------------------------------------------------------------------------------
// File: Deferred.hlsl
//
// HLSL file for the ComputeBasedTiledCulling sample. G-Buffer rendering.
//--------------------------------------------------------------------------------------


#include "CommonHeader.h"


//-----------------------------------------------------------------------------------------
// Textures and Buffers
//-----------------------------------------------------------------------------------------
Texture2D              g_TxDiffuse     : register( t0 );
Texture2D              g_TxNormal      : register( t1 );

//--------------------------------------------------------------------------------------
// shader input/output structure
//--------------------------------------------------------------------------------------
struct VS_INPUT_SCENE
{
    float3 Position     : POSITION;  // vertex position
    float3 Normal       : NORMAL;    // vertex normal vector
    float2 TextureUV    : TEXCOORD0; // vertex texture coords
    float3 Tangent      : TANGENT;   // vertex tangent vector
};

struct VS_OUTPUT_SCENE
{
    float4 Position     : SV_POSITION; // vertex position
    float3 Normal       : NORMAL;      // vertex normal vector
    float2 TextureUV    : TEXCOORD0;   // vertex texture coords
    float3 Tangent      : TEXCOORD1;   // vertex tangent vector
};

struct PS_OUTPUT
{
    float4 RT0 : SV_TARGET0;  // Diffuse
    float4 RT1 : SV_TARGET1;  // Normal
#if ( NUM_GBUFFER_RTS >= 3 )
    float4 RT2 : SV_TARGET2;  // Dummy
#endif
#if ( NUM_GBUFFER_RTS >= 4 )
    float4 RT3 : SV_TARGET3;  // Dummy
#endif
#if ( NUM_GBUFFER_RTS >= 5 )
    float4 RT4 : SV_TARGET4;  // Dummy
#endif
};


//--------------------------------------------------------------------------------------
// This shader transforms position, calculates world-space position, normal, 
// and tangent, and passes tex coords through to the pixel shader.
//--------------------------------------------------------------------------------------
VS_OUTPUT_SCENE RenderSceneToGBufferVS( VS_INPUT_SCENE Input )
{
    VS_OUTPUT_SCENE Output;
    
    // Transform the position from object space to homogeneous projection space
    float4 vWorldPos = mul( float4(Input.Position,1), g_mWorld );
    Output.Position = mul( vWorldPos, g_mViewProjection );

    // Normal and tangent in world space
    Output.Normal = mul( Input.Normal, (float3x3)g_mWorld );
    Output.Tangent = mul( Input.Tangent, (float3x3)g_mWorld );
    
    // Just copy the texture coordinate through
    Output.TextureUV = Input.TextureUV; 
    
    return Output;
}

//--------------------------------------------------------------------------------------
// This shader calculates diffuse and specular lighting for all lights.
//--------------------------------------------------------------------------------------
PS_OUTPUT RenderSceneToGBufferPS( VS_OUTPUT_SCENE Input )
{ 
    PS_OUTPUT Output;

    // diffuse rgb, and spec mask in the alpha channel
    float4 DiffuseTex = g_TxDiffuse.Sample( g_Sampler, Input.TextureUV );

#if ( USE_ALPHA_TEST == 1 )
    float fAlpha = DiffuseTex.a;
    if( fAlpha < g_fAlphaTest ) discard;
#endif

    // get normal from normal map
    float3 vNorm = g_TxNormal.Sample( g_Sampler, Input.TextureUV ).xyz;
    vNorm *= 2;
    vNorm -= float3(1,1,1);
    
    // transform normal into world space
    float3 vBinorm = normalize( cross( Input.Normal, Input.Tangent ) );
    float3x3 BTNMatrix = float3x3( vBinorm, Input.Tangent, Input.Normal );
    vNorm = normalize(mul( vNorm, BTNMatrix ));

#if ( USE_ALPHA_TEST == 1 )
    Output.RT0 = DiffuseTex;
    Output.RT1 = float4(0.5*vNorm + 0.5, 0);
#else
    Output.RT0 = float4(DiffuseTex.rgb, 1);
    Output.RT1 = float4(0.5*vNorm + 0.5, DiffuseTex.a);
#endif

    // write dummy data to consume more bandwidth, 
    // for performance testing
#if ( NUM_GBUFFER_RTS >= 3 )
    Output.RT2 = float4(1,1,1,1);
#endif
#if ( NUM_GBUFFER_RTS >= 4 )
    Output.RT3 = float4(1,1,1,1);
#endif
#if ( NUM_GBUFFER_RTS >= 5 )
    Output.RT4 = float4(1,1,1,1);
#endif

    return Output;
}

//-----------------------------------------------------------------------------------------
// Begin code for doing deferred lighting in a separate pixel shader 
//-----------------------------------------------------------------------------------------

#include "LightingCommonHeader.h"

//-----------------------------------------------------------------------------------------
// Textures and Buffers
//-----------------------------------------------------------------------------------------
Buffer<float4>   g_PointLightBufferCenterAndRadius  : register( t0 );
Buffer<float4>   g_SpotLightBufferCenterAndRadius   : register( t1 );
Texture2D<float> g_SceneDepthBuffer                 : register( t2 );

Buffer<float4> g_PointLightBufferColor           : register( t4 );
Buffer<float4> g_SpotLightBufferColor            : register( t5 );
Buffer<float4> g_SpotLightBufferSpotParams       : register( t6 );

Texture2D<float4> g_GBuffer0Texture : register( t7 );
Texture2D<float4> g_GBuffer1Texture : register( t8 );
#if ( NUM_GBUFFER_RTS >= 3 )
Texture2D<float4> g_GBuffer2Texture : register( t9 );
#endif
#if ( NUM_GBUFFER_RTS >= 4 )
Texture2D<float4> g_GBuffer3Texture : register( t10 );
#endif
#if ( NUM_GBUFFER_RTS >= 5 )
Texture2D<float4> g_GBuffer4Texture : register( t11 );
#endif

Buffer<uint>   g_PerTileLightIndexBuffer         : register( t12 );
Buffer<uint>   g_PerTileSpotIndexBuffer          : register( t13 );

//--------------------------------------------------------------------------------------
// shader input/output structure
//--------------------------------------------------------------------------------------
struct VS_QUAD_OUTPUT
{
    float4 Position     : SV_POSITION;
    float2 TextureUV    : TEXCOORD0;
};

//-----------------------------------------------------------------------------------------
// Deferred lighting in a separate pixel shader
//-----------------------------------------------------------------------------------------
float4 DoLightingDeferredPS( VS_QUAD_OUTPUT i ) : SV_TARGET
{
    // Convert screen coordinates to integer
    int2 globalIdx = int2(i.Position.xy);

    // get the surface normal from the G-Buffer
    float4 vNormAndSpecMask = g_GBuffer1Texture.Load( uint3(globalIdx.x,globalIdx.y,0) );
    float3 vNorm = vNormAndSpecMask.xyz;
    vNorm *= 2;
    vNorm -= float3(1,1,1);

    // convert depth and screen position to world-space position
    float fDepthBufferDepth = g_SceneDepthBuffer.Load(uint3(globalIdx.x, globalIdx.y, 0)).x;
    float4 vWorldSpacePosition = mul(float4((float)globalIdx.x+0.5, (float)globalIdx.y+0.5, fDepthBufferDepth, 1.0), g_mViewProjectionInvViewport);
    float3 vPositionWS = vWorldSpacePosition.xyz / vWorldSpacePosition.w;

    float3 vViewDir = normalize( g_vCameraPos - vPositionWS );

    float3 AccumDiffuse = float3(0,0,0);
    float3 AccumSpecular = float3(0,0,0);

    float fViewPosZ = ConvertProjDepthToView( fDepthBufferDepth );

#if ( LIGHTS_PER_TILE_MODE > 0 )
        uint nNumLightsInThisTile = 0;
        uint uMaxNumLightsPerTile = 0;
#endif

    // loop over the point lights that intersect this tile
    {
        uint nStartIndex, nLightCount;
        GetLightListInfo2(g_PerTileLightIndexBuffer, g_uMaxNumLightsPerTile, g_uMaxNumElementsPerTile, i.Position.xy, fViewPosZ, nStartIndex, nLightCount);

#if ( LIGHTS_PER_TILE_MODE > 0 )
        nNumLightsInThisTile += nLightCount;
        uMaxNumLightsPerTile += g_uMaxNumLightsPerTile;
#else
        [loop]
        for ( uint i = nStartIndex; i < nStartIndex+nLightCount; i++ )
        {
            uint nLightIndex = g_PerTileLightIndexBuffer[i];

            float3 LightColorDiffuseResult;
            float3 LightColorSpecularResult;
            DoLighting(g_PointLightBufferCenterAndRadius, g_PointLightBufferColor, nLightIndex, vPositionWS, vNorm, vViewDir, LightColorDiffuseResult, LightColorSpecularResult);

            AccumDiffuse += LightColorDiffuseResult;
            AccumSpecular += LightColorSpecularResult;
        }
#endif
    }

    // loop over the spot lights that intersect this tile
    {
        uint nStartIndex, nLightCount;
        GetLightListInfo2(g_PerTileSpotIndexBuffer, g_uMaxNumLightsPerTile, g_uMaxNumElementsPerTile, i.Position.xy, fViewPosZ, nStartIndex, nLightCount);

#if ( LIGHTS_PER_TILE_MODE > 0 )
        nNumLightsInThisTile += nLightCount;
        uMaxNumLightsPerTile += g_uMaxNumLightsPerTile;
#else
        [loop]
        for ( uint i = nStartIndex; i < nStartIndex+nLightCount; i++ )
        {
            uint nLightIndex = g_PerTileSpotIndexBuffer[i];

            float3 LightColorDiffuseResult;
            float3 LightColorSpecularResult;
            DoSpotLighting(g_SpotLightBufferCenterAndRadius, g_SpotLightBufferColor, g_SpotLightBufferSpotParams, nLightIndex, vPositionWS, vNorm, vViewDir, LightColorDiffuseResult, LightColorSpecularResult);

            AccumDiffuse += LightColorDiffuseResult;
            AccumSpecular += LightColorSpecularResult;
        }
#endif
    }

    // pump up the lights
    AccumDiffuse *= 2;
    AccumSpecular *= 8;

    // read dummy data to consume more bandwidth, 
    // for performance testing
#if ( NUM_GBUFFER_RTS >= 3 )
    float4 Dummy0 = g_GBuffer2Texture.Load( uint3(globalIdx.x,globalIdx.y,0) );
    AccumDiffuse *= Dummy0.xyz;
    AccumSpecular *= Dummy0.xyz;
#endif
#if ( NUM_GBUFFER_RTS >= 4 )
    float4 Dummy1 = g_GBuffer3Texture.Load( uint3(globalIdx.x,globalIdx.y,0) );
    AccumDiffuse *= Dummy1.xyz;
    AccumSpecular *= Dummy1.xyz;
#endif
#if ( NUM_GBUFFER_RTS >= 5 )
    float4 Dummy2 = g_GBuffer4Texture.Load( uint3(globalIdx.x,globalIdx.y,0) );
    AccumDiffuse *= Dummy2.xyz;
    AccumSpecular *= Dummy2.xyz;
#endif

    // This is a poor man's ambient cubemap (blend between an up color and a down color)
    float fAmbientBlend = 0.5f * vNorm.y + 0.5;
    float3 Ambient = g_AmbientColorUp.rgb * fAmbientBlend + g_AmbientColorDown.rgb * (1-fAmbientBlend);
    float3 DiffuseAndAmbient = AccumDiffuse + Ambient;

    // modulate mesh texture with lighting
    float3 DiffuseTex = g_GBuffer0Texture.Load( uint3(globalIdx.x,globalIdx.y,0) ).rgb;
    float fSpecMask = vNormAndSpecMask.a;

    float3 Result = DiffuseTex*(DiffuseAndAmbient + AccumSpecular*fSpecMask);

#if ( LIGHTS_PER_TILE_MODE == 1 )
    Result = ConvertNumberOfLightsToGrayscale(nNumLightsInThisTile, uMaxNumLightsPerTile).rgb;
#elif ( LIGHTS_PER_TILE_MODE == 2 )
    Result = ConvertNumberOfLightsToRadarColor(nNumLightsInThisTile, uMaxNumLightsPerTile).rgb;
#endif

    return float4(Result,1);
}
