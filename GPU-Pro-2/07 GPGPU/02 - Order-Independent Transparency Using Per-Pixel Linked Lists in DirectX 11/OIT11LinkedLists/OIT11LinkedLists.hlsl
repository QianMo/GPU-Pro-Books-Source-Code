//--------------------------------------------------------------------------------------
// File: OIT11LinkedLists.hlsl
//
//--------------------------------------------------------------------------------------
#include "PS_OIT_Include.hlsl"


//--------------------------------------------------------------------------------------
// Structures
//--------------------------------------------------------------------------------------
struct VS_SIMPLE_INPUT
{
    float3 vPosition : POSITION;
    float2 vTex      : TEXCOORD;
};

struct VS_INPUT
{
    float3 vPosition : POSITION;
    float3 vNormal   : NORMAL;
    float2 vTex      : TEXCOORD;
};

struct VS_SIMPLE_OUTPUT
{
    float2 vTex      : TEXCOORD;
    float4 vPosition :  SV_POSITION;
};

struct VS_OUTPUT
{
    float3 vNormal   : NORMAL;
    float2 vTex      : TEXCOORD;
    float4 vPosition :  SV_POSITION;
};

struct PS_SIMPLE_INPUT
{
    float2 vTex : TEXCOORD;
};

struct PS_INPUT
{
    float3 vNormal    : NORMAL;
    float2 vTex       : TEXCOORD;
    bool   bFrontFace : SV_ISFRONTFACE;
};

struct PS_OUTPUT_WITH_COVERAGE
{
    float4  vColor : SV_TARGET;
};


//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------
VS_OUTPUT VS( VS_INPUT input )
{
    VS_OUTPUT output = (VS_OUTPUT)0;
    
    // Transform position
    //float4 vWorldPos = mul( float4(input.vPosition, 1.0), g_mWorld );
    //output.vPosition = mul( vWorldPos, g_mViewProjection );
    output.vPosition = mul( float4(input.vPosition.xyz, 1.0), g_mWorldViewProjection );
    
    // Transform normal from model space to world space
    output.vNormal = mul(input.vNormal.xyz, (float3x3)g_mWorld);
    
    // Pass-through texture coordinates
    output.vTex = input.vTex;
    
    return output;
}

//--------------------------------------------------------------------------------------
// Vertex Shader for position only
//--------------------------------------------------------------------------------------
float4 VS_PositionOnly( VS_INPUT input ) : SV_POSITION
{
    float4 vPosition;
    
    // Transform position
    //float4 vWorldPos = mul( float4(input.vPosition, 1.0), g_mWorld );
    //vPosition = mul( vWorldPos, g_mViewProjection );
    vPosition = mul( float4(input.vPosition.xyz, 1.0), g_mWorldViewProjection );
    
    return vPosition;
}


//--------------------------------------------------------------------------------------
// Vertex Shader for pass-through
//--------------------------------------------------------------------------------------
VS_SIMPLE_OUTPUT VSPassThrough( VS_SIMPLE_INPUT input )
{
    VS_SIMPLE_OUTPUT output = (VS_SIMPLE_OUTPUT)0;
    
    output.vPosition = float4(input.vPosition.xyz, 1.0);
    output.vTex = input.vTex;
    
    return output;
}

RWByteAddressBuffer OverdrawBuffer : register(u1);
//--------------------------------------------------------------------------------------
// Pixel Shader to calculate pixel overdraw in MSAA mode
//--------------------------------------------------------------------------------------
[earlydepthstencil]
float4 PS_CalculatePixelOverdraw(float4 vPos: SV_POSITION) : SV_Target
{
    // Add 1 to overdraw buffer
    OverdrawBuffer.InterlockedAdd( 4 *( int(vPos.x)+ int(g_vScreenDimensions.x) * int(vPos.y) ), 1);
    
    // This won't write anything into the RT because color writes are off
    return float4(0,0,0,0);
}


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------
float4 PS_LightingAndTexturing( PS_INPUT input) : SV_Target
{
    // Renormalize normal
    input.vNormal = normalize(input.vNormal);
    
    // Invert normal when dealing with back faces
    if (!input.bFrontFace) input.vNormal = -input.vNormal;
    
    // Lighting
    float fLightIntensity = saturate(saturate(dot(input.vNormal.xyz, g_vLightVector.xyz)) + 0.2);
    float4 vColor = float4(g_vMeshColor.xyz * fLightIntensity, g_vMeshColor.w);
    
    // Texturing
    float4 vTextureColor = g_txDiffuse.Sample( g_samLinear, input.vTex );
    vColor.xyz *= vTextureColor.xyz;
    
    return vColor;
}

float4 PS_LightingOnly( PS_INPUT input) : SV_Target
{
    // Renormalize normal
    input.vNormal = normalize(input.vNormal);
    
    // Invert normal when dealing with back faces
    if (!input.bFrontFace) input.vNormal = -input.vNormal;
    
    // Lighting
    float fLightIntensity = saturate(saturate(dot(input.vNormal.xyz, g_vLightVector.xyz)) + 0.2);
    float4 vColor = float4(g_vMeshColor.xyz * fLightIntensity, g_vMeshColor.w);
    
    return vColor;
}

float4 PS_TextureOnly( PS_SIMPLE_INPUT input) : SV_Target
{
    return g_txDiffuse.Sample( g_samLinear, input.vTex );
}


