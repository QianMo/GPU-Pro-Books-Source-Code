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

#ifndef BASIC_LOOP_HLSL
#define BASIC_LOOP_HLSL

#include "GBuffer.hlsl"

//--------------------------------------------------------------------------------------
float4 BasicLoop(FullScreenTriangleVSOut input, uint sampleIndex)
{
    // How many total lights?
    uint totalLights, dummy;
    gLight.GetDimensions(totalLights, dummy);
    
    float3 lit = float3(0.0f, 0.0f, 0.0f);

    [branch] if (mUI.visualizeLightCount) {
        lit = (float(totalLights) * rcp(255.0f)).xxx;
    } else {
        SurfaceData surface = ComputeSurfaceDataFromGBufferSample(uint2(input.positionViewport.xy), sampleIndex);

        // Avoid shading skybox/background pixels
        if (surface.positionView.z < mCameraNearFar.y) {
            for (uint lightIndex = 0; lightIndex < totalLights; ++lightIndex) {
                PointLight light = gLight[lightIndex];
                AccumulateBRDF(surface, light, lit);
            }
        }
    }

    return float4(lit, 1.0f);
}

float4 BasicLoopPS(FullScreenTriangleVSOut input) : SV_Target
{
    // Shade only sample 0
    return BasicLoop(input, 0);
}

float4 BasicLoopPerSamplePS(FullScreenTriangleVSOut input, uint sampleIndex : SV_SampleIndex) : SV_Target
{
    float4 result;
    if (mUI.visualizePerSampleShading) {
        result = float4(1, 0, 0, 1);
    } else {
        result = BasicLoop(input, sampleIndex);
    }
    return result;
}

#endif // BASIC_LOOP_HLSL
