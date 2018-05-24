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

#ifndef FORWARD_HLSL
#define FORWARD_HLSL

#include "Rendering.hlsl"

//--------------------------------------------------------------------------------------
float4 ForwardPS(GeometryVSOut input) : SV_Target
{
    // How many total lights?
    uint totalLights, dummy;
    gLight.GetDimensions(totalLights, dummy);

    float3 lit = float3(0.0f, 0.0f, 0.0f);

    [branch] if (mUI.visualizeLightCount) {
        lit = (float(totalLights) * rcp(255.0f)).xxx;
    } else {
        SurfaceData surface = ComputeSurfaceDataFromGeometry(input);
        for (uint lightIndex = 0; lightIndex < totalLights; ++lightIndex) {
            PointLight light = gLight[lightIndex];
            AccumulateBRDF(surface, light, lit);
        }
    }

    return float4(lit, 1.0f);
}

float4 ForwardAlphaTestPS(GeometryVSOut input) : SV_Target
{
    // Always use face normal for alpha tested stuff since it's double-sided
    input.normal = ComputeFaceNormal(input.positionView);

    // Alpha test: dead code and CSE will take care of the duplication here
    SurfaceData surface = ComputeSurfaceDataFromGeometry(input);
    clip(surface.albedo.a - 0.3f);

    // Otherwise run the normal shader
    return ForwardPS(input);
}

// Does ONLY alpha test, not color. Useful for pre-z pass
void ForwardAlphaTestOnlyPS(GeometryVSOut input)
{
    // Alpha test: dead code and CSE will take care of the duplication here
    SurfaceData surface = ComputeSurfaceDataFromGeometry(input);
    clip(surface.albedo.a - 0.3f);
}

#endif // FORWARD_HLSL
