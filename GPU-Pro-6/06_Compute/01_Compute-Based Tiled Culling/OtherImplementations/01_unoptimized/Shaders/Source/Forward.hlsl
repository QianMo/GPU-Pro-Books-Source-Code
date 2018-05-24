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
// File: Forward.hlsl
//
// HLSL file for the ComputeBasedTiledCulling sample. Depth pre-pass and forward rendering.
//--------------------------------------------------------------------------------------


#include "CommonHeader.h"
#include "LightingCommonHeader.h"


//-----------------------------------------------------------------------------------------
// Textures and Buffers
//-----------------------------------------------------------------------------------------
Texture2D              g_TxDiffuse     : register( t0 );
Texture2D              g_TxNormal      : register( t1 );

// Save two slots for CDXUTSDKMesh diffuse and normal, 
// so start with the third slot, t2
StructuredBuffer<LightArrayData> g_PointLightBuffer : register( t2 );
Buffer<uint>   g_PerTileLightIndexBuffer         : register( t4 );

Buffer<float4> g_SpotLightBufferCenterAndRadius  : register( t5 );
Buffer<float4> g_SpotLightBufferColor            : register( t6 );
Buffer<float4> g_SpotLightBufferSpotParams       : register( t7 );
Buffer<uint>   g_PerTileSpotIndexBuffer          : register( t8 );

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
    float3 PositionWS   : TEXCOORD2;   // vertex position (world space)
};

struct VS_OUTPUT_POSITION_ONLY
{
    float4 Position     : SV_POSITION; // vertex position 
};

struct VS_OUTPUT_POSITION_AND_TEX
{
    float4 Position     : SV_POSITION; // vertex position 
    float2 TextureUV    : TEXCOORD0;   // vertex texture coords
};

//--------------------------------------------------------------------------------------
// This shader just transforms position (e.g. for depth pre-pass)
//--------------------------------------------------------------------------------------
VS_OUTPUT_POSITION_ONLY RenderScenePositionOnlyVS( VS_INPUT_SCENE Input )
{
    VS_OUTPUT_POSITION_ONLY Output;
    
    // Transform the position from object space to homogeneous projection space
    float4 vWorldPos = mul( float4(Input.Position,1), g_mWorld );
    Output.Position = mul( vWorldPos, g_mViewProjection );
    
    return Output;
}

//--------------------------------------------------------------------------------------
// This shader just transforms position and passes through tex coord 
// (e.g. for depth pre-pass with alpha test)
//--------------------------------------------------------------------------------------
VS_OUTPUT_POSITION_AND_TEX RenderScenePositionAndTexVS( VS_INPUT_SCENE Input )
{
    VS_OUTPUT_POSITION_AND_TEX Output;
    
    // Transform the position from object space to homogeneous projection space
    float4 vWorldPos = mul( float4(Input.Position,1), g_mWorld );
    Output.Position = mul( vWorldPos, g_mViewProjection );
    
    // Just copy the texture coordinate through
    Output.TextureUV = Input.TextureUV; 
    
    return Output;
}

//--------------------------------------------------------------------------------------
// This shader transforms position, calculates world-space position, normal, 
// and tangent, and passes tex coords through to the pixel shader.
//--------------------------------------------------------------------------------------
VS_OUTPUT_SCENE RenderSceneForwardVS( VS_INPUT_SCENE Input )
{
    VS_OUTPUT_SCENE Output;
    
    // Transform the position from object space to homogeneous projection space
    float4 vWorldPos = mul( float4(Input.Position,1), g_mWorld );
    Output.Position = mul( vWorldPos, g_mViewProjection );

    // Position, normal, and tangent in world space
    Output.PositionWS = vWorldPos.xyz;
    Output.Normal = mul( Input.Normal, (float3x3)g_mWorld );
    Output.Tangent = mul( Input.Tangent, (float3x3)g_mWorld );
    
    // Just copy the texture coordinate through
    Output.TextureUV = Input.TextureUV; 
    
    return Output;
}

//--------------------------------------------------------------------------------------
// This shader does alpha testing.
//--------------------------------------------------------------------------------------
float4 RenderSceneAlphaTestOnlyPS( VS_OUTPUT_POSITION_AND_TEX Input ) : SV_TARGET
{ 
    float4 DiffuseTex = g_TxDiffuse.Sample( g_Sampler, Input.TextureUV );
    float fAlpha = DiffuseTex.a;
    if( fAlpha < g_fAlphaTest ) discard;
    return DiffuseTex;
}

//--------------------------------------------------------------------------------------
// This shader calculates diffuse and specular lighting for all lights.
//--------------------------------------------------------------------------------------
float4 RenderSceneForwardPS( VS_OUTPUT_SCENE Input ) : SV_TARGET
{ 
    float3 vPositionWS = Input.PositionWS;

    float3 AccumDiffuse = float3(0,0,0);
    float3 AccumSpecular = float3(0,0,0);

    float4 DiffuseTex = g_TxDiffuse.Sample( g_Sampler, Input.TextureUV );

#if ( USE_ALPHA_TEST == 1 )
    float fSpecMask = 0.0f;
    float fAlpha = DiffuseTex.a;
    if( fAlpha < g_fAlphaTest ) discard;
#else
    float fSpecMask = DiffuseTex.a;
#endif

    // get normal from normal map
    float3 vNorm = g_TxNormal.Sample( g_Sampler, Input.TextureUV ).xyz;
    vNorm *= 2;
    vNorm -= float3(1,1,1);
    
    // transform normal into world space
    float3 vBinorm = normalize( cross( Input.Normal, Input.Tangent ) );
    float3x3 BTNMatrix = float3x3( vBinorm, Input.Tangent, Input.Normal );
    vNorm = normalize(mul( vNorm, BTNMatrix ));

    float3 vViewDir = normalize( g_vCameraPos - vPositionWS );

    // loop over the point lights
    {
        uint nStartIndex, nLightCount;
        GetLightListInfo(g_PerTileLightIndexBuffer, g_uMaxNumLightsPerTile, g_uMaxNumElementsPerTile, Input.Position, nStartIndex, nLightCount);

        [loop]
        for ( uint i = nStartIndex; i < nStartIndex+nLightCount; i++ )
        {
            uint nLightIndex = g_PerTileLightIndexBuffer[i];

            float3 LightColorDiffuseResult;
            float3 LightColorSpecularResult;
            DoLighting(g_PointLightBuffer, nLightIndex, vPositionWS, vNorm, vViewDir, LightColorDiffuseResult, LightColorSpecularResult);

            AccumDiffuse += LightColorDiffuseResult;
            AccumSpecular += LightColorSpecularResult;
        }
    }

    // loop over the spot lights
    {
        uint nStartIndex, nLightCount;
        GetLightListInfo(g_PerTileSpotIndexBuffer, g_uMaxNumLightsPerTile, g_uMaxNumElementsPerTile, Input.Position, nStartIndex, nLightCount);

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
    }

    // pump up the lights
    AccumDiffuse *= 2;
    AccumSpecular *= 8;

    // This is a poor man's ambient cubemap (blend between an up color and a down color)
    float fAmbientBlend = 0.5f * vNorm.y + 0.5;
    float3 Ambient = g_AmbientColorUp.rgb * fAmbientBlend + g_AmbientColorDown.rgb * (1-fAmbientBlend);

    // modulate mesh texture with lighting
    float3 DiffuseAndAmbient = AccumDiffuse + Ambient;
    return float4(DiffuseTex.xyz*(DiffuseAndAmbient + AccumSpecular*fSpecMask),1);
}
