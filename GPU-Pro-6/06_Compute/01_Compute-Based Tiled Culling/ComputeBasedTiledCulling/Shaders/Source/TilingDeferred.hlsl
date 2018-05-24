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
// File: TilingDeferred.hlsl
//
// HLSL file for the ComputeBasedTiledCulling sample. Tiled light culling.
//--------------------------------------------------------------------------------------


#include "CommonHeader.h"
#include "TilingCommonHeader.h"
#include "LightingCommonHeader.h"

//-----------------------------------------------------------------------------------------
// Textures and Buffers
//-----------------------------------------------------------------------------------------
Texture2D<float> g_SceneDepthBuffer              : register( t3 );

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

RWTexture2D<float4> g_OffScreenBufferOut : register( u0 );

//-----------------------------------------------------------------------------------------
// Light culling shader
//-----------------------------------------------------------------------------------------
[numthreads(TILE_RES, TILE_RES, 1)]
void CullLightsAndDoLightingCS( uint3 globalIdx : SV_DispatchThreadID, uint3 localIdx : SV_GroupThreadID, uint3 groupIdx : SV_GroupID )
{
    uint localIdxFlattened = localIdx.x + localIdx.y*TILE_RES;

    // after calling DoLightCulling, the per-tile list of light indices that intersect this tile 
    // will be in ldsLightIdx, and the number of lights that intersect this tile 
    // will be in ldsLightIdxCounterA and ldsLightIdxCounterB
    float fHalfZ;
    DoLightCulling( globalIdx, localIdxFlattened, groupIdx, fHalfZ );

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

    // loop over the point lights that intersect this tile
    {
        uint uStartIdx = (fViewPosZ < fHalfZ) ? 0 : MAX_NUM_LIGHTS_PER_TILE;
        uint uEndIdx = (fViewPosZ < fHalfZ) ? ldsLightIdxCounterA : ldsLightIdxCounterB;

        for(uint i=uStartIdx; i<uEndIdx; i++)
        {
            uint nLightIndex = ldsLightIdx[i];

            float3 LightColorDiffuseResult;
            float3 LightColorSpecularResult;
            DoLighting(g_PointLightBufferCenterAndRadius, g_PointLightBufferColor, nLightIndex, vPositionWS, vNorm, vViewDir, LightColorDiffuseResult, LightColorSpecularResult);

            AccumDiffuse += LightColorDiffuseResult;
            AccumSpecular += LightColorSpecularResult;
        }
    }

    // loop over the spot lights that intersect this tile
    {
        uint uStartIdx = (fViewPosZ < fHalfZ) ? 0 : MAX_NUM_LIGHTS_PER_TILE;
        uint uEndIdx = (fViewPosZ < fHalfZ) ? ldsSpotIdxCounterA : ldsSpotIdxCounterB;

        for(uint i=uStartIdx; i<uEndIdx; i++)
        {
            uint nLightIndex = ldsSpotIdx[i];

            float3 LightColorDiffuseResult;
            float3 LightColorSpecularResult;
            DoSpotLighting(g_SpotLightBufferCenterAndRadius, g_SpotLightBufferColor, g_SpotLightBufferSpotParams, nLightIndex, vPositionWS, vNorm, vViewDir, LightColorDiffuseResult, LightColorSpecularResult);

            AccumDiffuse += LightColorDiffuseResult;
            AccumSpecular += LightColorSpecularResult;
        }
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

    // override result when one of the lights-per-tile visualization modes is enabled
#if ( LIGHTS_PER_TILE_MODE > 0 )
    uint uStartIdx = (fViewPosZ < fHalfZ) ? 0 : MAX_NUM_LIGHTS_PER_TILE;
    uint uEndIdx = (fViewPosZ < fHalfZ) ? ldsLightIdxCounterA : ldsLightIdxCounterB;
    uint nNumLightsInThisTile = uEndIdx-uStartIdx;
    uEndIdx = (fViewPosZ < fHalfZ) ? ldsSpotIdxCounterA : ldsSpotIdxCounterB;
    nNumLightsInThisTile += uEndIdx-uStartIdx;
    uint uMaxNumLightsPerTile = 2*g_uMaxNumLightsPerTile;  // max for points plus max for spots
#if ( LIGHTS_PER_TILE_MODE == 1 )
    Result = ConvertNumberOfLightsToGrayscale(nNumLightsInThisTile, uMaxNumLightsPerTile).rgb;
#elif ( LIGHTS_PER_TILE_MODE == 2 )
    Result = ConvertNumberOfLightsToRadarColor(nNumLightsInThisTile, uMaxNumLightsPerTile).rgb;
#endif
#endif

    g_OffScreenBufferOut[globalIdx.xy] = float4(Result,1);
}


