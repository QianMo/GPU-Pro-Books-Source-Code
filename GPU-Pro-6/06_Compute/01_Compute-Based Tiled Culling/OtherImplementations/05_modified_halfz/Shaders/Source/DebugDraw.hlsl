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
// File: DebugDraw.hlsl
//
// HLSL file for the ComputeBasedTiledCulling sample. Debug drawing.
//--------------------------------------------------------------------------------------


#include "CommonHeader.h"

// disable warning: pow(f, e) will not work for negative f
#pragma warning( disable : 3571 )


//-----------------------------------------------------------------------------------------
// Textures and Buffers
//-----------------------------------------------------------------------------------------

// Save two slots for CDXUTSDKMesh diffuse and normal, 
// so start with the third slot, t2
Buffer<float4> g_PointLightBufferCenterAndRadius : register( t2 );
Buffer<float4> g_PointLightBufferColor           : register( t3 );
Buffer<uint>   g_PerTileLightIndexBuffer         : register( t4 );

Buffer<float4> g_SpotLightBufferCenterAndRadius  : register( t5 );
Buffer<float4> g_SpotLightBufferColor            : register( t6 );
Buffer<float4> g_SpotLightBufferSpotParams       : register( t7 );
Buffer<uint>   g_PerTileSpotIndexBuffer          : register( t8 );

Buffer<float4> g_SpotLightBufferSpotMatrices     : register( t9 );

//-----------------------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------------------
uint GetNumLightsInThisTile(Buffer<uint> PerTileLightIndexBuffer, uint uMaxNumLightsPerTile, uint uMaxNumElementsPerTile, float4 SVPosition)
{
    uint nIndex, nNumLightsInThisTile;
    GetLightListInfo(PerTileLightIndexBuffer, uMaxNumLightsPerTile, uMaxNumElementsPerTile, SVPosition, nIndex, nNumLightsInThisTile);
    return nNumLightsInThisTile;
}

//--------------------------------------------------------------------------------------
// shader input/output structure
//--------------------------------------------------------------------------------------
struct VS_INPUT_POS_TEX
{
    float3 Position     : POSITION;    // vertex position 
    float2 TextureUV    : TEXCOORD0;   // vertex texture coords 
};

struct VS_OUTPUT_POS_TEX
{
    float4 Position     : SV_POSITION; // vertex position
    float2 TextureUV    : TEXCOORD0;   // vertex texture coords 
};

struct VS_INPUT_DRAW_POINT_LIGHTS
{
    float3 Position     : POSITION;    // vertex position 
    float2 TextureUV    : TEXCOORD0;   // vertex texture coords 
    uint   InstanceID   : SV_InstanceID;
};

struct VS_OUTPUT_DRAW_POINT_LIGHTS
{
    float4 Position     : SV_POSITION; // vertex position
    float3 Color        : COLOR0;      // vertex color
    float2 TextureUV    : TEXCOORD0;   // vertex texture coords 
};

struct VS_INPUT_DRAW_SPOT_LIGHTS
{
    float3 Position     : POSITION;    // vertex position
    float3 Normal       : NORMAL;      // vertex normal vector
    float2 TextureUV    : TEXCOORD0;   // vertex texture coords
    uint   InstanceID   : SV_InstanceID;
};

struct VS_OUTPUT_DRAW_SPOT_LIGHTS
{
    float4 Position     : SV_POSITION; // vertex position
    float3 Normal       : NORMAL;      // vertex normal vector
    float4 Color        : COLOR0;      // vertex color
    float2 TextureUV    : TEXCOORD0;   // vertex texture coords 
    float3 PositionWS   : TEXCOORD1;   // vertex position (world space)
};

struct VS_OUTPUT_POSITION_ONLY
{
    float4 Position     : SV_POSITION; // vertex position 
};

//--------------------------------------------------------------------------------------
// This shader reads from the light buffer to create a screen-facing quad
// at each light position.
//--------------------------------------------------------------------------------------
VS_OUTPUT_DRAW_POINT_LIGHTS DebugDrawPointLightsVS( VS_INPUT_DRAW_POINT_LIGHTS Input )
{
    VS_OUTPUT_DRAW_POINT_LIGHTS Output;

    // get the light position from the light buffer (this will be the quad center)
    float4 LightPositionViewSpace = mul(float4(g_PointLightBufferCenterAndRadius[Input.InstanceID].xyz, 1), g_mView);

    // move from center to corner in view space (to make a screen-facing quad)
    LightPositionViewSpace.xy = LightPositionViewSpace.xy + Input.Position.xy;

    // transform the position from view space to homogeneous projection space
    Output.Position = mul(LightPositionViewSpace, g_mProjection);

    // pass through color from the light buffer and tex coords from the vert data
    Output.Color = g_PointLightBufferColor[Input.InstanceID].rgb;
    Output.TextureUV = Input.TextureUV;
    return Output;
}

//--------------------------------------------------------------------------------------
// This shader reads from the spot light buffers to place and orient a particular 
// instance of a cone.
//--------------------------------------------------------------------------------------
VS_OUTPUT_DRAW_SPOT_LIGHTS DebugDrawSpotLightsVS( VS_INPUT_DRAW_SPOT_LIGHTS Input )
{
    VS_OUTPUT_DRAW_SPOT_LIGHTS Output;

    float4 BoundingSphereCenterAndRadius = g_SpotLightBufferCenterAndRadius[Input.InstanceID];
    float4 SpotParams = g_SpotLightBufferSpotParams[Input.InstanceID];

    // reconstruct z component of the light dir from x and y
    float3 SpotLightDir;
    SpotLightDir.xy = SpotParams.xy;
    SpotLightDir.z = sqrt(1 - SpotLightDir.x*SpotLightDir.x - SpotLightDir.y*SpotLightDir.y);

    // the sign bit for cone angle is used to store the sign for the z component of the light dir
    SpotLightDir.z = (SpotParams.z > 0) ? SpotLightDir.z : -SpotLightDir.z;

    // calculate the light position from the bounding sphere (we know the top of the cone is 
    // r_bounding_sphere units away from the bounding sphere center along the negated light direction)
    float3 LightPosition = BoundingSphereCenterAndRadius.xyz - BoundingSphereCenterAndRadius.w*SpotLightDir;

    // rotate the light to point along the light direction vector
    float4x4 LightRotation = { g_SpotLightBufferSpotMatrices[4*Input.InstanceID], 
                               g_SpotLightBufferSpotMatrices[4*Input.InstanceID+1],
                               g_SpotLightBufferSpotMatrices[4*Input.InstanceID+2],
                               g_SpotLightBufferSpotMatrices[4*Input.InstanceID+3] };
    float3 VertexPosition = mul( Input.Position, (float3x3)LightRotation ) + LightPosition;
    float3 VertexNormal = mul( Input.Normal, (float3x3)LightRotation );

    // transform the position to homogeneous projection space
    Output.Position = mul( float4(VertexPosition,1), g_mViewProjection );

    // position and normal in world space
    Output.PositionWS = VertexPosition;//, (float3x3)g_mWorld );
    Output.Normal = VertexNormal;//, (float3x3)g_mWorld );

    // pass through color from the light buffer and tex coords from the vert data
    Output.Color = g_SpotLightBufferColor[Input.InstanceID];
    Output.TextureUV = Input.TextureUV;
    return Output;
}

//--------------------------------------------------------------------------------------
// This shader creates a procedural texture to visualize the point lights.
//--------------------------------------------------------------------------------------
float4 DebugDrawPointLightsPS( VS_OUTPUT_DRAW_POINT_LIGHTS Input ) : SV_TARGET
{
    float fRad = 0.5f;
    float2 Crd = Input.TextureUV - float2(fRad, fRad);
    float fCrdLength = length(Crd);

    // early out if outside the circle
    if( fCrdLength > fRad ) discard;

    // use pow function to make a point light visualization
    float x = ( 1.f-fCrdLength/fRad );
    return float4(0.5f*pow(x,5.f)*Input.Color + 2.f*pow(x,20.f), 1);
}

//--------------------------------------------------------------------------------------
// This shader creates a procedural texture to visualize the spot lights.
//--------------------------------------------------------------------------------------
float4 DebugDrawSpotLightsPS( VS_OUTPUT_DRAW_SPOT_LIGHTS Input ) : SV_TARGET
{
    float3 vViewDir = normalize( g_vCameraPos - Input.PositionWS );
    float3 vNormal = normalize(Input.Normal);
    float fViewDirDotSurfaceNormal = dot(vViewDir,vNormal);

    float3 color = Input.Color.rgb;
    color *= fViewDirDotSurfaceNormal < 0.0 ? 0.5 : 1.0;

    return float4(color,1);
}

//--------------------------------------------------------------------------------------
// This shader visualizes the number of lights per tile, in grayscale.
//--------------------------------------------------------------------------------------
float4 DebugDrawNumLightsPerTileGrayscalePS( VS_OUTPUT_POSITION_ONLY Input ) : SV_TARGET
{
    uint nNumLightsInThisTile = GetNumLightsInThisTile(g_PerTileLightIndexBuffer, g_uMaxNumLightsPerTile, g_uMaxNumElementsPerTile, Input.Position);
    nNumLightsInThisTile += GetNumLightsInThisTile(g_PerTileSpotIndexBuffer, g_uMaxNumLightsPerTile, g_uMaxNumElementsPerTile, Input.Position);
    uint uMaxNumLightsPerTile = 2*g_uMaxNumLightsPerTile;  // max for points plus max for spots
    return ConvertNumberOfLightsToGrayscale(nNumLightsInThisTile, uMaxNumLightsPerTile);
}

//--------------------------------------------------------------------------------------
// This shader visualizes the number of lights per tile, using weather radar colors.
//--------------------------------------------------------------------------------------
float4 DebugDrawNumLightsPerTileRadarColorsPS( VS_OUTPUT_POSITION_ONLY Input ) : SV_TARGET
{
    uint nNumLightsInThisTile = GetNumLightsInThisTile(g_PerTileLightIndexBuffer, g_uMaxNumLightsPerTile, g_uMaxNumElementsPerTile, Input.Position);
    nNumLightsInThisTile += GetNumLightsInThisTile(g_PerTileSpotIndexBuffer, g_uMaxNumLightsPerTile, g_uMaxNumElementsPerTile, Input.Position);
    uint uMaxNumLightsPerTile = 2*g_uMaxNumLightsPerTile;  // max for points plus max for spots
    return ConvertNumberOfLightsToRadarColor(nNumLightsInThisTile, uMaxNumLightsPerTile);
}

//--------------------------------------------------------------------------------------
// This shader converts screen space position xy into homogeneous projection space 
// and passes through the tex coords.
//--------------------------------------------------------------------------------------
VS_OUTPUT_POS_TEX DebugDrawLegendForNumLightsPerTileVS( VS_INPUT_POS_TEX Input )
{
    VS_OUTPUT_POS_TEX Output;

    // convert from screen space to homogeneous projection space
    Output.Position.x =  2.0f * ( Input.Position.x / (float)g_uWindowWidth )  - 1.0f;
    Output.Position.y = -2.0f * ( Input.Position.y / (float)g_uWindowHeight ) + 1.0f;
    Output.Position.z = 0.0f;
    Output.Position.w = 1.0f;

    // pass through
    Output.TextureUV = Input.TextureUV;

    return Output;
}

//--------------------------------------------------------------------------------------
// This shader creates a procedural texture for a grayscale gradient, for use as 
// a legend for the grayscale lights-per-tile visualization.
//--------------------------------------------------------------------------------------
float4 DebugDrawLegendForNumLightsPerTileGrayscalePS( VS_OUTPUT_POS_TEX Input ) : SV_TARGET
{
    float fGradVal = Input.TextureUV.y;
    return float4(fGradVal, fGradVal, fGradVal, 1.0f);
}

//--------------------------------------------------------------------------------------
// This shader creates a procedural texture for the radar colors, for use as 
// a legend for the radar colors lights-per-tile visualization.
//--------------------------------------------------------------------------------------
float4 DebugDrawLegendForNumLightsPerTileRadarColorsPS( VS_OUTPUT_POS_TEX Input ) : SV_TARGET
{
    uint nBandIdx = floor(16.999f*Input.TextureUV.y);

    // black for no lights
    if( nBandIdx == 0 ) return float4(0,0,0,1);
    // light purple for reaching the max
    else if( nBandIdx == 15 ) return float4(0.847,0.745,0.921,1);
    // white for going over the max
    else if ( nBandIdx == 16 ) return float4(1,1,1,1);
    // else use weather radar colors
    else
    {
        // nBandIdx should be in the range [1,14]
        uint nColorIndex = nBandIdx-1;
        return kRadarColors[nColorIndex];
    }
}

