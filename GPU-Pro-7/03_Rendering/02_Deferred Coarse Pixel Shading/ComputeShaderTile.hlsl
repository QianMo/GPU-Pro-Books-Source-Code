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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/////////////////////////////////////////////////////////////////////////////////////////////

#ifndef COMPUTE_SHADER_TILE_HLSL
#define COMPUTE_SHADER_TILE_HLSL

#include "GBuffer.hlsl"
#include "FramebufferFlat.hlsl"
#include "ShaderDefines.h"

RWStructuredBuffer<uint2> gFramebuffer : register(u0);

groupshared uint sMinZ;
groupshared uint sMaxZ;

// Light list for the tile
groupshared uint sTileLightIndices[MAX_LIGHTS];
groupshared uint sTileNumLights;

// List of coarse-pixels that require per-pixel shading
// We encode two 16-bit x/y coordinates in one uint to save shared memory space
groupshared uint sPerPixelCoarsePixels[COMPUTE_SHADER_TILE_GROUP_SIZE/(CPS_RATE * CPS_RATE)];
groupshared uint sNumPerPixelCoarsePixels;

//--------------------------------------------------------------------------------------
// Utility for writing to our flat MSAAed UAV
void WriteSample(uint2 coords, uint sampleIndex, float4 value)
{
    gFramebuffer[GetFramebufferSampleAddress(coords, sampleIndex)] = PackRGBA16(value);
}

// Pack and unpack two <=16-bit coordinates into a single uint
uint PackCoords(uint2 coords)
{
    return coords.y << 16 | coords.x;
}
uint2 UnpackCoords(uint coords)
{
    return uint2(coords & 0xFFFF, coords >> 16);
}



void ComputeSurfaceDataFromGBufferAllSamplesCPS(uint2 positionViewport,
    out SurfaceData surface[CPS_RATE * CPS_RATE])
{
    // Load data for each sample
    // Most of this time only a small amount of this data is actually used so unrolling
    // this loop ensures that the compiler has an easy time with the dead code elimination.
    [unroll] for (uint i = 0; i < CPS_RATE * CPS_RATE; ++i) {
        surface[i] = ComputeSurfaceDataFromGBufferSample(positionViewport + uint2( i % CPS_RATE, i / CPS_RATE), 0);
    }
}

bool RequiresPerPixelShading(SurfaceData surface[CPS_RATE * CPS_RATE])
{
    bool perPixel = false;
	// Multiply by CPS_RATE * sqrt(2.0f) to account for the fact that coarse pixels are of size CPS_RATE * CPS_RATE
    const float maxZDelta = CPS_RATE * sqrt(2.f) * (abs(surface[0].positionViewDX.z) + abs(surface[0].positionViewDY.z));
	const float kMaxAngle = 0.70710678118654752440084436210485f * (kPI/180.f);

    [unroll] for (uint i = 1; i < (CPS_RATE * CPS_RATE); ++i) {
        // Using the position derivatives of the triangle, check if all of the depths in the coarse pixel
        // could possibly have come from the same triangle/surface
        perPixel = perPixel || 
            abs(surface[i].positionView.z - surface[0].positionView.z) > maxZDelta;

        // Also flag places where the normal is different
        perPixel = perPixel || 
            any(abs(surface[i].normal - surface[0].normal) > kMaxAngle);
    }

    return perPixel;
}

// Each compute Shader works on the coarse pixel that of the size CPS_RATE * CPS_RATE 
[numthreads(COMPUTE_SHADER_TILE_GROUP_DIM/CPS_RATE, COMPUTE_SHADER_TILE_GROUP_DIM/CPS_RATE, 1)]
void ComputeShaderTileCS(uint3 groupId          : SV_GroupID,
                         uint3 dispatchThreadId : SV_DispatchThreadID,
                         uint3 groupThreadId    : SV_GroupThreadID,
                         uint groupIndex        : SV_GroupIndex
                         )
{
    // How many total lights?
    uint totalLights, dummy;
    gLight.GetDimensions(totalLights, dummy);

    uint2 globalCoords = CPS_RATE.xx * dispatchThreadId.xy;

    SurfaceData surfaceSamples[CPS_RATE*CPS_RATE];
    ComputeSurfaceDataFromGBufferAllSamplesCPS(globalCoords, surfaceSamples);
        
    // Work out Z bounds for our samples
    float minZSample = mCameraNearFar.y;
    float maxZSample = mCameraNearFar.x;
    {
        [unroll] for (uint sample = 0; sample < CPS_RATE * CPS_RATE; ++sample) {
            // Avoid shading skybox/background or otherwise invalid pixels
            float viewSpaceZ = surfaceSamples[sample].positionView.z;
            bool validPixel = 
                 viewSpaceZ >= mCameraNearFar.x &&
                 viewSpaceZ <  mCameraNearFar.y;
            [flatten] if (validPixel) {
                minZSample = min(minZSample, viewSpaceZ);
                maxZSample = max(maxZSample, viewSpaceZ);
            }
        }
    }
    
    // Initialize shared memory light list and Z bounds
    if (groupIndex == 0) {
        sTileNumLights = 0;
        sNumPerPixelCoarsePixels = 0;
        sMinZ = 0x7F7FFFFF;      // Max float
        sMaxZ = 0;
    }

    GroupMemoryBarrierWithGroupSync();
    
    // NOTE: Can do a parallel reduction here but now that we have MSAA and store sample frequency pixels
    // in shaded memory the increased shared memory pressure actually *reduces* the overall speed of the kernel.
    // Since even in the best case the speed benefit of the parallel reduction is modest on current architectures
    // with typical tile sizes, we have reverted to simple atomics for now.
    // Only scatter pixels with actual valid samples in them
    if (maxZSample >= minZSample) {
        InterlockedMin(sMinZ, asuint(minZSample));
        InterlockedMax(sMaxZ, asuint(maxZSample));
    }

    GroupMemoryBarrierWithGroupSync();

    float minTileZ = asfloat(sMinZ);
    float maxTileZ = asfloat(sMaxZ);
    
    // NOTE: This is all uniform per-tile (i.e. no need to do it per-thread) but fairly inexpensive
    // We could just precompute the frusta planes for each tile and dump them into a constant buffer...
    // They don't change unless the projection matrix changes since we're doing it in view space.
    // Then we only need to compute the near/far ones here tightened to our actual geometry.
    // The overhead of group synchronization/LDS or global memory lookup is probably as much as this
    // little bit of math anyways, but worth testing.

    // Work out scale/bias from [0, 1]
    float2 tileScale = float2(mFramebufferDimensions.xy) * rcp(float(2 * COMPUTE_SHADER_TILE_GROUP_DIM));
    float2 tileBias = tileScale - float2(groupId.xy);

    // Now work out composite projection matrix
    // Relevant matrix columns for this tile frusta
    float4 c1 = float4(mCameraProj._11 * tileScale.x, 0.0f, tileBias.x, 0.0f);
    float4 c2 = float4(0.0f, -mCameraProj._22 * tileScale.y, tileBias.y, 0.0f);
    float4 c4 = float4(0.0f, 0.0f, 1.0f, 0.0f);

    // Derive frustum planes
    float4 frustumPlanes[6];
    // Sides
    frustumPlanes[0] = c4 - c1;
    frustumPlanes[1] = c4 + c1;
    frustumPlanes[2] = c4 - c2;
    frustumPlanes[3] = c4 + c2;
    // Near/far
    frustumPlanes[4] = float4(0.0f, 0.0f,  1.0f, -minTileZ);
    frustumPlanes[5] = float4(0.0f, 0.0f, -1.0f,  maxTileZ);
    
    // Normalize frustum planes (near/far already normalized)
    [unroll] for (uint i = 0; i < 4; ++i) {
        frustumPlanes[i] *= rcp(length(frustumPlanes[i].xyz));
    }
    
    // Cull lights for this tile
    for (uint lightIndex = groupIndex; lightIndex < totalLights; lightIndex += (COMPUTE_SHADER_TILE_GROUP_SIZE/(CPS_RATE*CPS_RATE))) {
        PointLight light = gLight[lightIndex];
                
        // Cull: point light sphere vs tile frustum
        bool inFrustum = true;
        [unroll] for (uint i = 0; i < 6; ++i) {
            float d = dot(frustumPlanes[i], float4(light.positionView, 1.0f));
            inFrustum = inFrustum && (d >= -light.attenuationEnd);
        }

        [branch] if (inFrustum) {
            // Append light to list
            // Compaction might be better if we expect a lot of lights
            uint listIndex;
            InterlockedAdd(sTileNumLights, 1, listIndex);
            sTileLightIndices[listIndex] = lightIndex;
        }
    }

    GroupMemoryBarrierWithGroupSync();
    
    uint numLights = sTileNumLights;

    // Only process onscreen pixels (tiles can span screen edges)
    if (all(globalCoords < mFramebufferDimensions.xy)) {
        [branch] if (mUI.visualizeLightCount) {
            [unroll] for (uint pixel = 0; pixel < (CPS_RATE * CPS_RATE); ++pixel) {
                WriteSample(globalCoords + uint2( pixel % CPS_RATE, pixel / CPS_RATE), 0, (float(sTileNumLights) / 255.0f).xxxx);
            }
        } else if (numLights > 0) {
            bool perSampleShading = mUI.forcePerPixel ? true : RequiresPerPixelShading(surfaceSamples);
            float3 lit = float3(0.0f, 0.0f, 0.0f);
            for (uint tileLightIndex = 0; tileLightIndex < numLights; ++tileLightIndex) {
                PointLight light = gLight[sTileLightIndices[tileLightIndex]];
                AccumulateBRDF(surfaceSamples[0], light, lit, mUI.faceNormals && !perSampleShading);
            }

            // Write top left pixel's result
            WriteSample(globalCoords, 0, (mUI.visualizePerSampleShading && !perSampleShading)? float4(0.f, 1.f, 0.f, 1.0f) : float4(lit, 1.0f));
                        
            [branch] if (perSampleShading) {
                #if DEFER_PER_PIXEL
                    // Create a list of coarse pixels that need per-pixel shading
                    uint listIndex;
                    InterlockedAdd(sNumPerPixelCoarsePixels, 1, listIndex);
                    sPerPixelCoarsePixels[listIndex] = PackCoords(globalCoords);
                #else
                    // Shade the other samples for this pixel
                    for (uint pixel = 1; pixel < CPS_RATE * CPS_RATE; ++pixel) {
                        float3 litSample = float3(0.0f, 0.0f, 0.0f);
                        for (uint tileLightIndex = 0; tileLightIndex < numLights; ++tileLightIndex) {
                            PointLight light = gLight[sTileLightIndices[tileLightIndex]];
                            AccumulateBRDF(surfaceSamples[pixel], light, litSample);
                        }                        
                        WriteSample(globalCoords + uint2(pixel % 2, pixel / 2), 0, float4(litSample, 1.0f));
                    }
                #endif
            } else {
                // Otherwise per-coarse pixel shading, so splat the result to all the pixels
                [unroll] for (uint pixel = 1; pixel < CPS_RATE * CPS_RATE; ++pixel) {
                    WriteSample(globalCoords + uint2(pixel % CPS_RATE, pixel / CPS_RATE), 0, mUI.visualizePerSampleShading ? float4(0.f, 1.f, 0.f, 1.0f) : float4(lit, 1.0f));
                }
            }
        } else {
            // Otherwise no lights affect here so clear all samples
            // StephanieB5: set pixel's initial value is 0 otherwise the top left pixel is left uncleared and garbage values
            // appear in unlit areas.
            [unroll] for (uint pixel = 0; pixel < CPS_RATE * CPS_RATE; ++pixel) {
                WriteSample(globalCoords + uint2( pixel % CPS_RATE, pixel / CPS_RATE), 0, float4(0.0f, 0.0f, 0.0f, 0.0f));
            }
        }
    }

    #if DEFER_PER_PIXEL
        // NOTE: We were careful to write only top left pixel above. If we are going to do pixel
        // frequency shading below, so we don't need a device memory barrier here.
        GroupMemoryBarrierWithGroupSync();

        // Now handle any pixels that require per-pixel shading
        // NOTE: Each pixel requires CPS_RATE * CPS_RATE - 1 additional shading passes
        const uint shadingPassesPerCoarsePxiel = (CPS_RATE * CPS_RATE) - 1;
        uint globalSamples = sNumPerPixelCoarsePixels * shadingPassesPerCoarsePxiel;

        for (uint globalSample = groupIndex; globalSample < globalSamples; globalSample += (COMPUTE_SHADER_TILE_GROUP_SIZE/(CPS_RATE*CPS_RATE))) {
            uint listIndex = globalSample / shadingPassesPerCoarsePxiel;
            uint sampleIndex = globalSample % shadingPassesPerCoarsePxiel + 1;        // sample 0 has been handled earlier
            uint2 pixelOffset = uint2(sampleIndex % CPS_RATE, sampleIndex / CPS_RATE);

            uint2 sampleCoords = UnpackCoords(sPerPixelCoarsePixels[listIndex]);
            SurfaceData surface = ComputeSurfaceDataFromGBufferSample(sampleCoords + pixelOffset, 0);

            float3 lit = float3(0.0f, 0.0f, 0.0f);
            for (uint tileLightIndex = 0; tileLightIndex < numLights; ++tileLightIndex) {
                PointLight light = gLight[sTileLightIndices[tileLightIndex]];
                AccumulateBRDF(surface, light, lit, false);
            }
            WriteSample(sampleCoords + pixelOffset, 0, float4(lit, 1.0f));
        }
    #endif
}

#endif // COMPUTE_SHADER_TILE_HLSL
