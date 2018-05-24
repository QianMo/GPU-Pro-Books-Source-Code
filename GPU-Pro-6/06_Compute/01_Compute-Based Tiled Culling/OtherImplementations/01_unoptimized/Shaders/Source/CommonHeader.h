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
// File: CommonHeader.h
//
// HLSL file for the ComputeBasedTiledCulling sample. Common header file for all shaders.
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Miscellaneous constants
//--------------------------------------------------------------------------------------
static const float4 kRadarColors[14] = 
{
    {0,0.9255,0.9255,1},   // cyan
    {0,0.62745,0.9647,1},  // light blue
    {0,0,0.9647,1},        // blue
    {0,1,0,1},             // bright green
    {0,0.7843,0,1},        // green
    {0,0.5647,0,1},        // dark green
    {1,1,0,1},             // yellow
    {0.90588,0.75294,0,1}, // yellow-orange
    {1,0.5647,0,1},        // orange
    {1,0,0,1},             // bright red
    {0.8392,0,0,1},        // red
    {0.75294,0,0,1},       // dark red
    {1,0,1,1},             // magenta
    {0.6,0.3333,0.7882,1}, // purple
};

//--------------------------------------------------------------------------------------
// Light culling constants.
// These must match their counterparts in CommonUtil.h
//--------------------------------------------------------------------------------------
#define TILE_RES 8
#define MAX_NUM_LIGHTS_PER_TILE 272

//--------------------------------------------------------------------------------------
// Constant Buffers
//--------------------------------------------------------------------------------------
cbuffer cbPerObject : register( b0 )
{
    matrix  g_mWorld                   : packoffset( c0 );
}

cbuffer cbPerCamera : register( b1 )
{
    matrix  g_mViewProjection          : packoffset( c0 );
};

cbuffer cbPerFrame : register( b2 )
{
    matrix              g_mView                      : packoffset( c0 );
    matrix              g_mProjection                : packoffset( c4 );
    matrix              g_mProjectionInv             : packoffset( c8 );
    matrix              g_mViewProjectionInvViewport : packoffset( c12 );
    float4              g_AmbientColorUp             : packoffset( c16 );
    float4              g_AmbientColorDown           : packoffset( c17 );
    float3              g_vCameraPos                 : packoffset( c18 );
    float               g_fAlphaTest                 : packoffset( c18.w );
    uint                g_uNumLights                 : packoffset( c19 );
    uint                g_uNumSpotLights             : packoffset( c19.y );
    uint                g_uWindowWidth               : packoffset( c19.z );
    uint                g_uWindowHeight              : packoffset( c19.w );
    uint                g_uMaxNumLightsPerTile       : packoffset( c20 );
    uint                g_uMaxNumElementsPerTile     : packoffset( c20.y );
    uint                g_uNumTilesX                 : packoffset( c20.z );
    uint                g_uNumTilesY                 : packoffset( c20.w );
};

struct LightArrayData
{
    float4 v4PositionAndRadius;
    float4 v4Color;
};

//-----------------------------------------------------------------------------------------
// Samplers
//-----------------------------------------------------------------------------------------
SamplerState           g_Sampler       : register( s0 );

//-----------------------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------------------

// convert a point from post-projection space into view space
float3 ConvertProjToView(float4 p)
{
    p = mul(p, g_mProjectionInv);
    return (p / p.w).xyz;
}

// convert a depth value from post-projection space into view space
float ConvertProjDepthToView(float z)
{
    return (1.f / (z*g_mProjectionInv._34 + g_mProjectionInv._44));
}

uint GetTileIndex(float2 ScreenPos)
{
    float fTileRes = (float)TILE_RES;
    uint nTileIdx = floor(ScreenPos.x/fTileRes)+floor(ScreenPos.y/fTileRes)*g_uNumTilesX;
    return nTileIdx;
}

// PerTileLightIndexBuffer layout:
// | Light Count | space for max num lights per tile light indices |
void GetLightListInfo(in Buffer<uint> PerTileLightIndexBuffer, in uint uMaxNumLightsPerTile, in uint uMaxNumElementsPerTile, in float4 SVPosition, out uint uFirstLightIndex, out uint uNumLights)
{
    uint nTileIndex = GetTileIndex(SVPosition.xy);
    uint nStartIndex = uMaxNumElementsPerTile*nTileIndex;

    uFirstLightIndex = (nStartIndex + 1);
    uNumLights = PerTileLightIndexBuffer[nStartIndex];
}

// PerTileLightIndexBuffer layout:
// | Light Count | space for max num lights per tile light indices |
void GetLightListInfo2(in Buffer<uint> PerTileLightIndexBuffer, in uint uMaxNumLightsPerTile, in uint uMaxNumElementsPerTile, in float2 ScreenPos, in float fViewPosZ, out uint uFirstLightIndex, out uint uNumLights)
{
    uint nTileIndex = GetTileIndex(ScreenPos);
    uint nStartIndex = uMaxNumElementsPerTile*nTileIndex;

    uFirstLightIndex = (nStartIndex + 1);
    uNumLights = PerTileLightIndexBuffer[nStartIndex];
}

float4 ConvertNumberOfLightsToGrayscale(uint nNumLightsInThisTile, uint uMaxNumLightsPerTile)
{
    float fPercentOfMax = (float)nNumLightsInThisTile / (float)uMaxNumLightsPerTile;
    return float4(fPercentOfMax, fPercentOfMax, fPercentOfMax, 1.0f);
}

float4 ConvertNumberOfLightsToRadarColor(uint nNumLightsInThisTile, uint uMaxNumLightsPerTile)
{
    // black for no lights
    if( nNumLightsInThisTile == 0 ) return float4(0,0,0,1);
    // light purple for reaching the max
    else if( nNumLightsInThisTile == uMaxNumLightsPerTile ) return float4(0.847,0.745,0.921,1);
    // white for going over the max
    else if ( nNumLightsInThisTile > uMaxNumLightsPerTile ) return float4(1,1,1,1);
    // else use weather radar colors
    else
    {
        // use a log scale to provide more detail when the number of lights is smaller

        // want to find the base b such that the logb of uMaxNumLightsPerTile is 14
        // (because we have 14 radar colors)
        float fLogBase = exp2(0.07142857f*log2((float)uMaxNumLightsPerTile));

        // change of base
        // logb(x) = log2(x) / log2(b)
        uint nColorIndex = floor(log2((float)nNumLightsInThisTile) / log2(fLogBase));
        return kRadarColors[nColorIndex];
    }
}
