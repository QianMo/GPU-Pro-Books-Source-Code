#include "_Shaders/Lighting.inc"
#include "_Shaders/PCF9.inc"

#include "ASM.inc"

float4 mainVS(float4 pos : POSITION) : SV_Position
{
    return float4( pos.xy, 1, 1 );
}

cbuffer Constants : register(b0)
{
    float4x4 g_ScreenToWorld;
    float3 g_SunDir;
    float g_FrameBufferWidthQuads;
};

StructuredBuffer<float4> g_LightBuffer : register(t0);
Texture2D<float4> g_DiffuseBuffer      : register(t1);
Texture2D<float4> g_NormalBuffer       : register(t2);
Texture2D<float4> g_GeomNormalBuffer   : register(t3);
Texture2D<float> g_DepthTexture        : register(t4);

float ComputeShadowFactor( float2 screenPos )
{
    float depth = g_DepthTexture[ screenPos ];
    float4 worldPos = mul( g_ScreenToWorld, float4( screenPos, depth, 1.0 ) );
    worldPos /= worldPos.w;

    float3 G = 2.0 * g_GeomNormalBuffer[ screenPos ].xyz - 1.0;

    return SampleLongRangeShadows( worldPos.xyz ).x;
}

[earlydepthstencil]
float4 mainPS(float4 pos : SV_Position) : SV_Target
{
    float4 lighting = 0.1;

#if USE_LIGHTBUFFER
    lighting += g_LightBuffer[ GetLightBufferAddr( pos, g_FrameBufferWidthQuads ) ];
#endif

    float3 shadowFactor = ComputeShadowFactor( pos.xy );

    float3 N = 2.0 * g_GeomNormalBuffer[ pos.xy ].xyz - 1.0;
    lighting.xyz += shadowFactor * saturate( dot( g_SunDir, N ) );
    float4 diffuse = pow( g_DiffuseBuffer[pos.xy], 2.0 );

#if DEBUG_LIGHTING
    return lighting;
#elif DEBUG_DIFFUSE
    return diffuse;
#endif

    return lighting;//pow( diffuse * lighting, 0.5 );
}
