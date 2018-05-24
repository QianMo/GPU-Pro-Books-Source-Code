/////////////////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or imlied.
// See the License for the specific language governing permissions and
// limitations under the License.
/////////////////////////////////////////////////////////////////////////////////////////////

#ifndef GBUFFER_HLSL
#define GBUFFER_HLSL

#include "Rendering.hlsl"

//--------------------------------------------------------------------------------------
// GBuffer and related common utilities and structures
struct GBuffer
{
    float4 normal_specular : SV_Target0;
    float4 albedo : SV_Target1;
    float4 biased_albedo    : SV_Target2;
    float2 positionZGrad : SV_Target3;
    float  positionZ        : SV_Target4;
};

// Above values PLUS depth buffer (last element)
#if MSAA_SAMPLES > 1
Texture2DMS<float4, MSAA_SAMPLES> gGBufferTextures[4] : register(t0);
#else
Texture2D<float4>                 gGBufferTextures[5] : register(t0);
#endif // MSAA_SAMPLES > 1

#define kPI 3.1415926536f


float2 EncodeSphereMap(float3 n)
{
//     float oneMinusZ = 1.0f - n.z;
//     float p = sqrt(n.x * n.x + n.y * n.y + oneMinusZ * oneMinusZ);
//     return n.xy / p * 0.5f + 0.5f;
    return 0.5* (float2(atan2(n.y,n.x)/kPI, n.z)+1.0f);
}

float3 DecodeSphereMap(float2 enc)
{
//     float2 tmp = e - e * e;
//     float f = tmp.x + tmp.y;
//     float m = sqrt(4.0f * f - 1.0f);
//     
//     float3 n;
//     n.xy = m * (e * 4.0f - 2.0f);
//     n.z  = 3.0f - 8.0f * f;
//     return n;
    float2 ang = enc*2-1;
    float2 scth;
    sincos(ang.x * kPI, scth.x, scth.y);
    float2 scphi = float2(sqrt(1.0 - ang.y*ang.y), ang.y);
    return float3(scth.y*scphi.x, scth.x*scphi.x, scphi.y);

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

SurfaceData ComputeSurfaceDataFromGBufferSample(uint2 positionViewport, uint sampleIndex)
{
    // Load the raw data from the GBuffer
    GBuffer rawData;

#if MSAA_SAMPLES > 1
    rawData.normal_specular = gGBufferTextures[0].Load(positionViewport.xy, sampleIndex).xyzw;
    rawData.albedo = gGBufferTextures[1].Load(positionViewport.xy, sampleIndex).xyzw;
    rawData.positionZGrad = gGBufferTextures[2].Load(positionViewport.xy, sampleIndex).xy;
    float zBuffer = gGBufferTextures[3].Load(positionViewport.xy, sampleIndex).x;
#else
    rawData.normal_specular = gGBufferTextures[0][positionViewport.xy].xyzw;
    rawData.albedo = gGBufferTextures[1][positionViewport.xy].xyzw;
    rawData.biased_albedo = gGBufferTextures[2][positionViewport.xy].xyzw;
    rawData.positionZGrad = gGBufferTextures[3][positionViewport.xy].xy;
    float zBuffer = gGBufferTextures[4][positionViewport.xy].x;
#endif // MSAA_SAMPLES > 1    
    float2 gbufferDim;
    uint dummy;
#if MSAA_SAMPLES > 1
    gGBufferTextures[0].GetDimensions(gbufferDim.x, gbufferDim.y, dummy);
#else
    gGBufferTextures[0].GetDimensions(0, gbufferDim.x, gbufferDim.y, dummy);
#endif // MSAA_SAMPLES > 1
    
    // Compute screen/clip-space position and neighbour positions
    // NOTE: Mind DX11 viewport transform and pixel center!
    // NOTE: This offset can actually be precomputed on the CPU but it's actually slower to read it from
    // a constant buffer than to just recompute it.
    float2 screenPixelOffset = float2(2.0f, -2.0f) / gbufferDim;
    float2 positionScreen = (float2(positionViewport.xy) + 0.5f) * screenPixelOffset.xy + float2(-1.0f, 1.0f);
    float2 positionScreenX = positionScreen + float2(screenPixelOffset.x, 0.0f);
    float2 positionScreenY = positionScreen + float2(0.0f, screenPixelOffset.y);
        
    // Decode into reasonable outputs
    SurfaceData data;
        
    // Unproject depth buffer Z value into view space
//     float viewSpaceZ = mCameraProj._43 / (zBuffer - mCameraProj._33);
    float viewSpaceZ = zBuffer;

    data.positionView = ComputePositionViewFromZ(positionScreen, viewSpaceZ);
    data.positionViewDX = ComputePositionViewFromZ(positionScreenX, viewSpaceZ + rawData.positionZGrad.x) - data.positionView;
    data.positionViewDY = ComputePositionViewFromZ(positionScreenY, viewSpaceZ + rawData.positionZGrad.y) - data.positionView;

    data.normal = DecodeSphereMap(rawData.normal_specular.xy);
    data.albedo = rawData.albedo;
    data.biased_albedo = rawData.biased_albedo; 

    data.specularAmount = rawData.normal_specular.z;
    data.specularPower = rawData.normal_specular.w;
    
    return data;
}

void ComputeSurfaceDataFromGBufferAllSamples(uint2 positionViewport,
                                             out SurfaceData surface[MSAA_SAMPLES])
{
    // Load data for each sample
    // Most of this time only a small amount of this data is actually used so unrolling
    // this loop ensures that the compiler has an easy time with the dead code elimination.
    [unroll] for (uint i = 0; i < MSAA_SAMPLES; ++i) {
        surface[i] = ComputeSurfaceDataFromGBufferSample(positionViewport, i);
    }
}

// Check if a given pixel can be shaded at pixel frequency (i.e. they all come from
// the same surface) or if they require per-sample shading
bool RequiresPerSampleShading(SurfaceData surface[MSAA_SAMPLES])
{
    bool perSample = false;

    const float maxZDelta = abs(surface[0].positionViewDX.z) + abs(surface[0].positionViewDY.z);
    const float minNormalDot = 0.99f;        // Allow ~8 degree normal deviations

    [unroll] for (uint i = 1; i < MSAA_SAMPLES; ++i) {
        // Using the position derivatives of the triangle, check if all of the sample depths
        // could possibly have come from the same triangle/surface
        perSample = perSample || 
            abs(surface[i].positionView.z - surface[0].positionView.z) > maxZDelta;

        // Also flag places where the normal is different
        perSample = perSample || 
            dot(surface[i].normal, surface[0].normal) < minNormalDot;
    }

    return perSample;
}


// Initialize stencil mask with per-sample/per-pixel flags
void RequiresPerSampleShadingPS(FullScreenTriangleVSOut input)
{
    SurfaceData surfaceSamples[MSAA_SAMPLES];
    ComputeSurfaceDataFromGBufferAllSamples(uint2(input.positionViewport.xy), surfaceSamples);
    bool perSample = RequiresPerSampleShading(surfaceSamples);

    // Kill fragment (i.e. don't write stencil) if we don't require per sample shading
    [flatten] if (!perSample) {
        discard;
    }
}


//--------------------------------------------------------------------------------------
// G-buffer rendering
//--------------------------------------------------------------------------------------
void GBufferPS(GeometryVSOut input, out GBuffer outputGBuffer)
{
    SurfaceData surface = ComputeSurfaceDataFromGeometry(input);
    outputGBuffer.normal_specular = float4(EncodeSphereMap(surface.normal),
                                           surface.specularAmount,
                                           surface.specularPower);
    outputGBuffer.albedo = surface.albedo;
    if (mUI.faceNormals) {
        outputGBuffer.biased_albedo = surface.biased_albedo;
    }
    outputGBuffer.positionZGrad = float2(ddx_coarse(surface.positionView.z),
                                         ddy_coarse(surface.positionView.z));
    outputGBuffer.positionZ = surface.positionView.z;
}

void GBufferAlphaTestPS(GeometryVSOut input, out GBuffer outputGBuffer)
{
    GBufferPS(input, outputGBuffer);
    
    // Alpha test
    clip(outputGBuffer.albedo.a - 0.3f);

    // Always use face normal for alpha tested stuff since it's double-sided
    outputGBuffer.normal_specular.xy = EncodeSphereMap(normalize(ComputeFaceNormal(input.positionView)));
}

#endif // GBUFFER_HLSL
