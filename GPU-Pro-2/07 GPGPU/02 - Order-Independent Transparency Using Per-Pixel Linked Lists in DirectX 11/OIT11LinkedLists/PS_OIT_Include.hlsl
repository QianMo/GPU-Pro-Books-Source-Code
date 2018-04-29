//--------------------------------------------------------------------------------------
// File: PS_OIT_Include.hlsl
// 
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Defines
//--------------------------------------------------------------------------------------
#define MAX_24BIT_UINT  ( (1<<24) - 1 )

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
Texture2D       g_txDiffuse : register( t0 );

//--------------------------------------------------------------------------------------
// Samplers
//--------------------------------------------------------------------------------------
SamplerState    g_samLinear : register( s0 );
SamplerState    g_samPoint  : register( s1 );

//--------------------------------------------------------------------------------------
// Constant Buffers
//--------------------------------------------------------------------------------------
cbuffer cbMain : register( b0 )
{
    matrix      g_mView;
    matrix      g_mProjection;
    matrix      g_mViewProjection;
    float4      g_vLightVector;
    float4      g_vViewVector;
    float4      g_vScreenDimensions;
};

cbuffer cbPerMesh : register( b1 )
{
    matrix      g_mWorld;
    matrix      g_mWorldViewProjection;
    float4      g_vMeshColor;
};

cbuffer cbTileCoordinates : register( b2 )
{
    float4      g_vRectangleCoordinates;  // The size of the rectangle area to update
    float4      g_vTileSize;              // The actual tile size that matches the buffers size
};

//--------------------------------------------------------------------------------------
// Helper functions
//--------------------------------------------------------------------------------------
uint PackFloat4IntoUint(float4 vValue)
{
    return ( ((uint)(vValue.x*255)) << 24 ) | ( ((uint)(vValue.y*255)) << 16 ) | ( ((uint)(vValue.z*255)) << 8) | (uint)(vValue.w * 255);
}

float4 UnpackUintIntoFloat4(uint uValue)
{
    return float4( ( (uValue & 0xFF000000)>>24 ) / 255.0, ( (uValue & 0x00FF0000)>>16 ) / 255.0, ( (uValue & 0x0000FF00)>>8 ) / 255.0, ( (uValue & 0x000000FF) ) / 255.0);
}

// Pack depth into 24 MSBs
uint PackDepthIntoUint(float fDepth)
{
    return ((uint)(fDepth * MAX_24BIT_UINT)) << 8;
}

// Pack depth into 24 MSBs and coverage into 8 LSBs
uint PackDepthAndCoverageIntoUint(float fDepth, uint uCoverage)
{
    return (((uint)(fDepth * MAX_24BIT_UINT)) << 8) | uCoverage;
}

uint UnpackDepthIntoUint(uint uDepthAndCoverage)
{
    return (uint)(uDepthAndCoverage >> 8);
}

uint UnpackCoverageIntoUint(uint uDepthAndCoverage)
{
    return (uDepthAndCoverage & 0xff );
}

uint PackNormalIntoUint(float3 n)
{
    uint3 i3 = (uint3) (n * 127.0f + 127.0f);
    return i3.r + (i3.g << 8) + (i3.b << 16);
}

float3 UnpackNormalIntoFloat3(uint n)
{
    float3 n3 = float3(n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff);
    return (n3 - 127.0f) / 127.0f;
}

uint PackNormalAndCoverageIntoUint(float3 n, uint uCoverage)
{
    uint3 i3 = (uint3) (n * 127.0f + 127.0f);
    return i3.r + (i3.g << 8) + (i3.b << 16) + (uCoverage << 24);
}

