// Copyright 2010 Intel Corporation
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

// GBuffer and related common utilities and structures

#ifndef H_GBUFFER
#define H_GBUFFER

struct UIConstants
{
    uint faceNormals;
    uint avsmSortingMethod;
    uint volumeShadowMethod;
    uint enableVolumeShadowLookup;
    uint pauseParticleAnimaton;
    float particleSize;
    uint particleOpacity;
    float dsmError;
    uint hairThickness;
    uint hairShadowThickness;
    uint hairOpacity;
    uint lightingOnly;
};

cbuffer PerFrameConstants
{
    float4x4    mCameraWorldViewProj;
    float4x4    mCameraWorldView;
    float4x4    mCameraViewProj;
    float4x4    mCameraProj;
    float4      mCameraPos;
    float4x4    mLightWorldViewProj;
    float4x4    mAvsmLightWorldViewProj;    
    float4x4    mCameraViewToLightProj;
    float4x4    mCameraViewToLightView;    
    float4x4    mCameraViewToAvsmLightProj;
    float4x4    mCameraViewToAvsmLightView;        
    float4      mLightDir;
    
    UIConstants mUI;
};

struct GBuffer
{
    // SV_Target0 is reserved for the back buffer when rendering
    float  viewSpaceZ : SV_Target1;
    float4 normals    : SV_Target2;
    float4 albedo     : SV_Target3;
};

Texture2D gGBufferTextures[3];


float2 EncodeSphereMap(float3 n)
{
    float oneMinusZ = 1.0f - n.z;
    float p = sqrt(n.x * n.x + n.y * n.y + oneMinusZ * oneMinusZ);
    return n.xy / p * 0.5f + 0.5f;
}

float3 DecodeSphereMap(float2 e)
{
    float2 tmp = e - e * e;
    float f = tmp.x + tmp.y;
    float m = sqrt(4.0f * f - 1.0f);
    
    float3 n;
    n.xy = m * (e * 4.0f - 2.0f);
    n.z  = 3.0f - 8.0f * f;
    return n;
}

float3 ComputePositionViewFromZ(float2 positionScreen,
                                float viewSpaceZ)
{
    float2 screenSpaceRay = float2(positionScreen.x / mCameraProj._11,
                                   positionScreen.y / mCameraProj._22);
    
    float3 positionView;
    positionView.z = viewSpaceZ;
    // Solve the two projection equations
    positionView.xy = screenSpaceRay.xy * positionView.z;
    
    return positionView;
}

// data that we can read or derived from the surface shader outputs
struct SurfaceData
{
    float3 positionView;         // View space position
    float3 positionViewDX;       // Screen space derivatives
    float3 positionViewDY;       // of view space position
    float3 normal;               // View space normal
    float4 albedo;
    float2 lightTexCoord;        // Texture coordinates in light space, [0, 1]
    float2 lightTexCoordDX;      // Screen space partial derivatives
    float2 lightTexCoordDY;      // of light space texture coordinates.
    float  lightSpaceZ;          // Z coordinate (depth) of surface in light space
};

float2 ProjectIntoLightTexCoord(float3 positionView)
{
    float4 positionLight = mul(float4(positionView, 1.0f), mCameraViewToLightProj);
    float2 texCoord = (positionLight.xy / positionLight.w) * float2(0.5f, -0.5f) + float2(0.5f, 0.5f);
    return texCoord;
}

float2 ProjectIntoAvsmLightTexCoord(float3 positionView)
{
    float4 positionLight = mul(float4(positionView, 1.0f), mCameraViewToAvsmLightProj);
    float2 texCoord = (positionLight.xy / positionLight.w) * float2(0.5f, -0.5f) + float2(0.5f, 0.5f);
    return texCoord;
}

SurfaceData ComputeSurfaceDataFromGBuffer(int2 positionViewport)
{
    // Load the raw data from the GBuffer
    int3 Coords = int3(positionViewport.xy, 0);
    GBuffer rawData;
    rawData.viewSpaceZ = gGBufferTextures[0].Load(Coords).x;
    rawData.normals    = gGBufferTextures[1].Load(Coords);
    rawData.albedo     = gGBufferTextures[2].Load(Coords);
    
    float2 gbufferDim;
    gGBufferTextures[0].GetDimensions(gbufferDim.x, gbufferDim.y);
    
    // Compute screen/clip-space position and neighbour positions
    // NOTE: Mind DX11 viewport transform and pixel center!
    float2 screenPixelOffset = float2(2.0f, -2.0f) / gbufferDim;
    float2 positionScreen = (float2(positionViewport.xy) + 0.5f) * screenPixelOffset.xy + float2(-1.0f, 1.0f);
    float2 positionScreenX = positionScreen + float2(screenPixelOffset.x, 0.0f);
    float2 positionScreenY = positionScreen + float2(0.0f, screenPixelOffset.y);
    
    // Decode into reasonable outputs
    SurfaceData data;
    
    // Solve for view-space position and derivatives
    data.positionView   = ComputePositionViewFromZ(positionScreen , rawData.viewSpaceZ);
    data.positionViewDX = ComputePositionViewFromZ(positionScreenX, rawData.viewSpaceZ + rawData.normals.z) - data.positionView;
    data.positionViewDY = ComputePositionViewFromZ(positionScreenY, rawData.viewSpaceZ + rawData.normals.w) - data.positionView;
    
    data.normal = DecodeSphereMap(rawData.normals.xy);
    
    data.albedo = rawData.albedo;
    
    // NOTE: No need for perspective divide for directional light
    data.lightSpaceZ = mul(float4(data.positionView, 1.0f), mCameraViewToLightProj).z;
    
    // Solve for light space position and screen-space derivatives
    float deltaPixels = 2.0f;    
    data.lightTexCoord   = (ProjectIntoLightTexCoord(data.positionView));
    data.lightTexCoordDX = (ProjectIntoLightTexCoord(data.positionView + deltaPixels * data.positionViewDX) - data.lightTexCoord) / deltaPixels;
    data.lightTexCoordDY = (ProjectIntoLightTexCoord(data.positionView + deltaPixels * data.positionViewDY) - data.lightTexCoord) / deltaPixels;
    
    return data;
}

#endif // H_GBUFFER
