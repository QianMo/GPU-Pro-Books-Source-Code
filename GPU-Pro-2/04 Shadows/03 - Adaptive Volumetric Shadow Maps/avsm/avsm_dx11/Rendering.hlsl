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

#include "Common.hlsl"
#include "GBuffer.hlsl"
#include "AVSM.hlsl"

#ifndef SHADOWAA_SAMPLES
#define SHADOWAA_SAMPLES 1
#endif

//--------------------------------------------------------------------------------------
// Geometry phase
//--------------------------------------------------------------------------------------
Texture2D gDiffuseTexture;
SamplerState gDiffuseSampler;

struct GeometryVSIn
{
    float3 position : position;
    float3 normal   : normal;
    float2 texCoord : texCoord;
};

struct GeometryVSOut
{
    float4 position     : SV_position;
    float3 positionView : positionView;      // View space position
    float3 normal       : normal;
    float2 texCoord     : texCoord;
};

GeometryVSOut GeometryVS(GeometryVSIn input)
{
    GeometryVSOut output;

    output.position     = mul(float4(input.position, 1.0f), mCameraWorldViewProj);
    output.positionView = mul(float4(input.position, 1.0f), mCameraWorldView).xyz;
    output.normal       = mul(float4(input.normal, 0.0f), mCameraWorldView).xyz;
    output.texCoord     = input.texCoord;
    
    return output;
}

void GeometryPS(GeometryVSOut input,
                out GBuffer outputGBuffer,
                out float4 outputLit : SV_Target0)
{
    float viewSpaceZ = input.positionView.z;

    // Map NULL diffuse textures to white 
    float4 albedo;
    uint2 textureDim; 
    gDiffuseTexture.GetDimensions(textureDim.x, textureDim.y); 
    if (textureDim.x == 0U || mUI.lightingOnly) {
    	albedo = float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    else {
    	albedo = gDiffuseTexture.Sample(gDiffuseSampler, input.texCoord);
    }

    // Optionally use face normal instead of shading normal
    float3 normal = normalize(mUI.faceNormals
        ? cross(ddx_coarse(input.positionView), ddy_coarse(input.positionView))
        : input.normal);
        
    outputGBuffer.viewSpaceZ = input.positionView.z;
    outputGBuffer.albedo = albedo;
    
    // Effectively encodes the shading and face normals
    // (The latter are used to recover derivatives in deferred passes.)
    outputGBuffer.normals = float4(EncodeSphereMap(normal),
                                   ddx_coarse(viewSpaceZ), ddy_coarse(viewSpaceZ));
    
    // Initialize the back buffer 
    outputLit = (mUI.lightingOnly ? 0.0f : 0.08f) * albedo;
}

void GeometryAlphaTestPS(GeometryVSOut input,
                         out GBuffer outputGBuffer,
                         out float4 outputLit : SV_Target0)
{
    GeometryPS(input, outputGBuffer, outputLit);
    
    // Alpha test
    clip(outputGBuffer.albedo.a - 0.3f);
}

//--------------------------------------------------------------------------------------
// List texture
//--------------------------------------------------------------------------------------


FullScreenTriangleVSOut FullScreenTriangleVS(uint vertexID : SV_VertexID)
{
    FullScreenTriangleVSOut output;

    // Parametrically work out vertex location for full screen triangle
    float2 grid = float2((vertexID << 1) & 2, vertexID & 2);
    output.positionClip = float4(grid * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    output.positionViewport = output.positionClip;
    output.texCoord = grid;
    
    return output;
}


//--------------------------------------------------------------------------------------
// Lighting phase
//--------------------------------------------------------------------------------------

float4 LightingPS(FullScreenTriangleVSOut input) : SV_Target
{
    SurfaceData surface = ComputeSurfaceDataFromGBuffer(int2(input.positionViewport.xy));
    
    float3 lit = float3(0.0f, 0.0f, 0.0f);
    
    // Sample volume occlusion term                         
    float volumeShadowContrib = 1.0f;
    if (mUI.enableVolumeShadowLookup) {
        // Compute AVSM uv coords and receiver depth
        float2 avsmLightTexCoord = ProjectIntoAvsmLightTexCoord(surface.positionView.xyz);   
        float  receiverDepth = mul(float4(surface.positionView.xyz, 1.0f), mCameraViewToAvsmLightView).z;                        
        volumeShadowContrib = VolumeSample(mUI.volumeShadowMethod, avsmLightTexCoord, receiverDepth);
    }
    float shadowContrib = volumeShadowContrib;
    
    // Also early out if we're a back face
    float nDotL = saturate(dot(-mLightDir.xyz, surface.normal));

    float3 contrib = shadowContrib * (nDotL * surface.albedo.xyz);

    // Accumulate lighting
    lit += contrib;        
    
    return float4(lit, 1.0f);
}

//--------------------------------------------------------------------------------------
// Skybox
//--------------------------------------------------------------------------------------
TextureCube gSkyboxTexture : register(t0);

struct SkyboxVSOut
{
    float4 position : SV_Position;
    float3 texCoord : texCoord;
};

SkyboxVSOut SkyboxVS(GeometryVSIn input)
{
    SkyboxVSOut output;
    
    // NOTE: Don't translate skybox, and make sure depth == 1
    output.position = mul(float4(input.position, 0.0f), mCameraViewProj).xyww;
    output.texCoord = input.position;
    
    return output;
}

float4 SkyboxPS(SkyboxVSOut input) : SV_Target0
{
    return gSkyboxTexture.Sample(gDiffuseSampler, input.texCoord);
}
