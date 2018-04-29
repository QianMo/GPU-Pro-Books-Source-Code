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

#include "HairCommon.hlsl"
#include "HairListTexture.hlsl"
#include "AVSM.hlsl"

float4
LightHair(float3 position,
          float3 tangent,
          float4 color)
{
    // Kay-Kajiya lighting for hair.

    // Id = 2 * Kd * r * sin(T, L)
    // Is = Ks * pow(cos(E, E'), p)
    // Is = ks * (dot(T, L) * dot(T, E) + sin(T, L) * sin(T, E))^p
    float3 lightDir = normalize(mLightDir.xyz);
    float3 eyeDir = normalize(position - mCameraPos.xyz);
    tangent = normalize(tangent);
    float Kd = 1.0f;
    float radius = 1.0f;

    float angle_T_L = acos(dot(tangent, lightDir));
    float angle_T_E = acos(dot(tangent, eyeDir));

    float ambient = 0.3f;

    float diffuse = 2.0f * Kd * radius * sin(angle_T_L);

    float Ks = 1.0f;
    float p = 10;
    float specular = 
        Ks * pow(saturate(dot(tangent, lightDir)) * saturate(dot(tangent, eyeDir)) + 
                 saturate(sin(angle_T_L) * sin(angle_T_E)), p);

    float volumeShadowContrib = 1.0f;
    if (mUI.enableVolumeShadowLookup) {
        float2 lightTexCoord = ProjectIntoAvsmLightTexCoord(position.xyz);
        float receiverDepth = 
            mul(float4(position.xyz, 1.0f), mCameraViewToAvsmLightView).z;                        
        
        volumeShadowContrib = VolumeSample(mUI.volumeShadowMethod, lightTexCoord, receiverDepth);
    }

    float3 litColor = (ambient + diffuse + specular) * color.rgb;

    float4 outColor;
    outColor = float4(litColor, 1.0f);
    outColor *= color.a; // premultipled alpha
    outColor.xyz *= volumeShadowContrib;

    return outColor;
}

void
CameraHairCapturePS(HairPixelInput input)
{
    const float hairOpacity = ComputeHairOpacity(input.distanceFromCenter);

    // Allocate a new node. Drop if ran out of memory.
    uint newNodeAddress;
    if (HairLTAllocNode(newNodeAddress)) {
        // Fill node
        HairLTNode node;
        node.depth = input.hposition.z;
        node.color = 
            LightHair(input.position, input.tangent, 
                      float4(input.color, hairOpacity));

        // Get fragment viewport coordinates
        int2 screenAddress = int2(input.hposition.xy);

        // Insert node!
        HairLTInsertNode(screenAddress, newNodeAddress, node);
    }
}

float4
CameraHairCapturePSDebug(HairPixelInput input) : SV_Target
{
    return float4(0, 0, 1, 1);
}
