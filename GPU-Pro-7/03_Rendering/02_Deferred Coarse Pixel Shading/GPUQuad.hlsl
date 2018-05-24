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

#ifndef GPU_QUAD_HLSL
#define GPU_QUAD_HLSL

#include "GBuffer.hlsl"

//--------------------------------------------------------------------------------------
// Bounds computation utilities, similar to PointLightBounds.cpp
void UpdateClipRegionRoot(float nc,          // Tangent plane x/y normal coordinate (view space)
                          float lc,          // Light x/y coordinate (view space)
                          float lz,          // Light z coordinate (view space)
                          float lightRadius,
                          float cameraScale, // Project scale for coordinate (_11 or _22 for x/y respectively)
                          inout float clipMin,
                          inout float clipMax)
{
    float nz = (lightRadius - nc * lc) / lz;
    float pz = (lc * lc + lz * lz - lightRadius * lightRadius) /
               (lz - (nz / nc) * lc);

    [flatten] if (pz > 0.0f) {
        float c = -nz * cameraScale / nc;
        [flatten] if (nc > 0.0f) {        // Left side boundary
            clipMin = max(clipMin, c);
        } else {                          // Right side boundary
            clipMax = min(clipMax, c);
        }
    }
}

void UpdateClipRegion(float lc,          // Light x/y coordinate (view space)
                      float lz,          // Light z coordinate (view space)
                      float lightRadius,
                      float cameraScale, // Project scale for coordinate (_11 or _22 for x/y respectively)
                      inout float clipMin,
                      inout float clipMax)
{
    float rSq = lightRadius * lightRadius;
    float lcSqPluslzSq = lc * lc + lz * lz;
	float d = rSq * lc * lc - lcSqPluslzSq * (rSq - lz * lz);

    if (d > 0) {
        float a = lightRadius * lc;
        float b = sqrt(d);
        float nx0 = (a + b) / lcSqPluslzSq;
        float nx1 = (a - b) / lcSqPluslzSq;
        
        UpdateClipRegionRoot(nx0, lc, lz, lightRadius, cameraScale, clipMin, clipMax);
        UpdateClipRegionRoot(nx1, lc, lz, lightRadius, cameraScale, clipMin, clipMax);
    }
}

// Returns bounding box [min.xy, max.xy] in clip [-1, 1] space.
float4 ComputeClipRegion(float3 lightPosView, float lightRadius)
{
    // Early out with empty rectangle if the light is too far behind the view frustum
    float4 clipRegion = float4(1, 1, 0, 0);
    if (lightPosView.z + lightRadius >= mCameraNearFar.x) {
        float2 clipMin = float2(-1.0f, -1.0f);
        float2 clipMax = float2( 1.0f,  1.0f);
    
        UpdateClipRegion(lightPosView.x, lightPosView.z, lightRadius, mCameraProj._11, clipMin.x, clipMax.x);
        UpdateClipRegion(lightPosView.y, lightPosView.z, lightRadius, mCameraProj._22, clipMin.y, clipMax.y);

        clipRegion = float4(clipMin, clipMax);
    }

    return clipRegion;
}

// One per quad - gets expanded in the geometry shader
struct GPUQuadVSOut
{
    float4 coords : coords;         // [min.xy, max.xy] in clip space
    float quadZ : quadZ;
    uint lightIndex : lightIndex;
};

GPUQuadVSOut GPUQuadVS(uint lightIndex : SV_VertexID) 
{
    GPUQuadVSOut output;
    output.lightIndex = lightIndex;

    // Work out tight clip-space rectangle
    PointLight light = gLight[lightIndex];
    output.coords = ComputeClipRegion(light.positionView, light.attenuationEnd);
    
    // Work out nearest depth for quad Z
    // Clamp to near plane in case this light intersects the near plane... don't want our quad to be clipped
    float quadDepth = max(mCameraNearFar.x, light.positionView.z - light.attenuationEnd);

    // Project quad depth into clip space
    float4 quadClip = mul(float4(0.0f, 0.0f, quadDepth, 1.0f), mCameraProj);
    output.quadZ = quadClip.z / quadClip.w;

    return output;
}

struct GPUQuadGSOut
{
    float4 positionViewport : SV_Position;
    // NOTE: Using a uint4 to work around a compiler bug. Otherwise the SV_SampleIndex input to the per-sample
    // shader below gets put into a '.y' which doesn't appear to work on some implementations.
    nointerpolation uint4 lightIndex : lightIndex;
};

// Takes point output and converts into screen-space quads
[maxvertexcount(4)]
void GPUQuadGS(point GPUQuadVSOut input[1], inout TriangleStream<GPUQuadGSOut> quadStream)
{
    GPUQuadGSOut output;
    output.lightIndex = input[0].lightIndex;
    output.positionViewport.zw = float2(input[0].quadZ, 1.0f);

    // Branch around empty regions (i.e. light entirely offscreen)
    if (all(input[0].coords.xy < input[0].coords.zw)) {
        output.positionViewport.xy = input[0].coords.xw;      // [-1,  1]
        quadStream.Append(output);
        output.positionViewport.xy = input[0].coords.zw;      // [ 1,  1]
        quadStream.Append(output);
        output.positionViewport.xy = input[0].coords.xy;      // [-1, -1]
        quadStream.Append(output);
        output.positionViewport.xy = input[0].coords.zy;      // [ 1, -1]
        quadStream.Append(output);
        quadStream.RestartStrip();
    }
}

float4 GPUQuad(GPUQuadGSOut input, uint sampleIndex)
{
    float3 lit = float3(0.0f, 0.0f, 0.0f);
    
    [branch] if (mUI.visualizeLightCount) {
        lit = rcp(255.0f).xxx;
    } else {
        SurfaceData surface = ComputeSurfaceDataFromGBufferSample(uint2(input.positionViewport.xy), sampleIndex);

        // Avoid shading skybox/background pixels
        // NOTE: Compiler doesn't quite seem to move all the unrelated surface computations inside here
        // We could force it to by restructuring the code a bit, but the "all skybox" case isn't useful for
        // our benchmarking anyways.
        if (surface.positionView.z < mCameraNearFar.y) {
            PointLight light = gLight[input.lightIndex.x];
            AccumulateBRDF(surface, light, lit);
        }    
    }

    return float4(lit, 1.0f);
}

float4 GPUQuadPS(GPUQuadGSOut input) : SV_Target
{
    // Shade only sample 0
    return GPUQuad(input, 0);
}

float4 GPUQuadPerSamplePS(GPUQuadGSOut input, uint sampleIndex : SV_SampleIndex) : SV_Target
{
    float4 result;
    if (mUI.visualizePerSampleShading) {
        result = float4(1, 0, 0, 1);
    } else {
        result = GPUQuad(input, sampleIndex);
    }
    return result;
}

#endif // GPU_QUAD_HLSL
