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

#ifndef RENDERING_HLSL
#define RENDERING_HLSL

#include "FullScreenTriangle.hlsl"
#include "PerFrameConstants.hlsl"

//--------------------------------------------------------------------------------------
// Utility
//--------------------------------------------------------------------------------------
float linstep(float min, float max, float v)
{
    return saturate((v - min) / (max - min));
}


//--------------------------------------------------------------------------------------
// Geometry phase
//--------------------------------------------------------------------------------------
Texture2D gDiffuseTexture : register(t0);
SamplerState gDiffuseSampler        : register(s0);
SamplerState gBiasedDiffuseSampler  : register(s1);

struct GeometryVSIn
{
    float3 position : position;
    float3 normal   : normal;
    float2 texCoord : texCoord;
};

struct GeometryVSOut
{
    float4 position     : SV_Position;
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

float3 ComputeFaceNormal(float3 position)
{
    return cross(ddx_coarse(position), ddy_coarse(position));
}

// Data that we can read or derive from the surface shader outputs
struct SurfaceData
{
    float3 positionView;         // View space position
    float3 positionViewDX;       // Screen space derivatives
    float3 positionViewDY;       // of view space position
    float3 normal;               // View space normal
    float4 biased_albedo;
    float4 albedo;
    float specularAmount;        // Treated as a multiplier on albedo
    float specularPower;
};

SurfaceData ComputeSurfaceDataFromGeometry(GeometryVSOut input)
{
    SurfaceData surface;
    surface.positionView = input.positionView;

    // These arguably aren't really useful in this path since they are really only used to
    // derive shading frequencies and composite derivatives but might as well compute them
    // in case they get used for anything in the future.
    // (Like the rest of these values, they will get removed by dead code elimination if
    // they are unused.)
    surface.positionViewDX = ddx_coarse(surface.positionView);
    surface.positionViewDY = ddy_coarse(surface.positionView);

    // Optionally use face normal instead of shading normal
    float3 faceNormal = ComputeFaceNormal(input.positionView);
//     surface.normal = normalize(mUI.faceNormals ? faceNormal : input.normal);
    surface.normal = normalize(input.normal);
    
    surface.albedo = gDiffuseTexture.Sample(gDiffuseSampler, input.texCoord);
    if (mUI.faceNormals) {
        surface.biased_albedo = gDiffuseTexture.Sample(gBiasedDiffuseSampler, input.texCoord);
    }
    surface.albedo.rgb = mUI.lightingOnly ? float3(1.0f, 1.0f, 1.0f) : surface.albedo.rgb;

    // Map NULL diffuse textures to white
    uint2 textureDim;
    gDiffuseTexture.GetDimensions(textureDim.x, textureDim.y);
    surface.albedo = (textureDim.x == 0U ? float4(1.0f, 1.0f, 1.0f, 1.0f) : surface.albedo);

    // We don't really have art asset-related values for these, so set them to something
    // reasonable for now... the important thing is that they are stored in the G-buffer for
    // representative performance measurement.
    surface.specularAmount = 0.9f;
    surface.specularPower = 25.0f;

    return surface;
}


//--------------------------------------------------------------------------------------
// Lighting phase utilities
//--------------------------------------------------------------------------------------
struct PointLight
{
    float3 positionView;
    float attenuationBegin;
    float3 color;
    float attenuationEnd;
};

StructuredBuffer<PointLight> gLight : register(t5);

// As below, we separate this for diffuse/specular parts for convenience in deferred lighting
void AccumulatePhongBRDF(float3 normal,
                         float3 lightDir,
                         float3 viewDir,
                         float3 lightContrib,
                         float specularPower,
                         inout float3 litDiffuse,
                         inout float3 litSpecular)
{
    // Simple Phong
    float NdotL = dot(normal, lightDir);
    [flatten] if (NdotL > 0.0f) {
        float3 r = reflect(lightDir, normal);
        float RdotV = max(0.0f, dot(r, viewDir));
        float specular = pow(RdotV, specularPower);

        litDiffuse += lightContrib * NdotL;
        litSpecular += lightContrib * specular;
    }
}

// Accumulates separate "diffuse" and "specular" components of lighting from a given
// This is not possible for all BRDFs but it works for our simple Phong example here
// and this separation is convenient for deferred lighting.
// Uses an in-out for accumulation to avoid returning and accumulating 0
void AccumulateBRDFDiffuseSpecular(SurfaceData surface, PointLight light,
                                   inout float3 litDiffuse,
                                   inout float3 litSpecular)
{
    float3 directionToLight = light.positionView - surface.positionView;
    float distanceToLight = length(directionToLight);

    [branch] if (distanceToLight < light.attenuationEnd) {
        float attenuation = linstep(light.attenuationEnd, light.attenuationBegin, distanceToLight);
        directionToLight *= rcp(distanceToLight);       // A full normalize/RSQRT might be as fast here anyways...
        
        AccumulatePhongBRDF(surface.normal, directionToLight, normalize(surface.positionView),
            attenuation * light.color, surface.specularPower, litDiffuse, litSpecular);
    }
}

void AccumulateBRDF(SurfaceData surface, PointLight light,
                    inout float3 lit, bool useBiasedSampler = false)
{
    float3 directionToLight = light.positionView - surface.positionView;
    float distanceToLight = length(directionToLight);

    [branch] if (distanceToLight < light.attenuationEnd) {
        float attenuation = linstep(light.attenuationEnd, light.attenuationBegin, distanceToLight);
        directionToLight *= rcp(distanceToLight);       // A full normalize/RSQRT might be as fast here anyways...

        float3 litDiffuse = float3(0.0f, 0.0f, 0.0f);
        float3 litSpecular = float3(0.0f, 0.0f, 0.0f);
        AccumulatePhongBRDF(surface.normal, directionToLight, normalize(surface.positionView),
            attenuation * light.color, surface.specularPower, litDiffuse, litSpecular);
        
		//
		// Biased sampler is used in case of coarse pixels so that texture-sampling is in sync with 
		// the CPS_RATE * CPS_RATE sized pixels.
		// Remember the sampler was biased with log (CPS_RATE).
		//
		lit += (useBiasedSampler ? surface.biased_albedo.rgb : surface.albedo.rgb) * (litDiffuse + surface.specularAmount * litSpecular);

		//
		// Following loop simulates the more complex shading.
		//
		//[unroll]
		//for (uint i = 0; i < 10; ++i) {
		//	lit.b += 0.000000000000000000001f * sin(distanceToLight/9999999999999999999999999.f);
		//}
    }

}

#endif // RENDERING_HLSL
