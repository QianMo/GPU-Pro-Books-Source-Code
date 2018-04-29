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

#ifndef HAIR_COMMON_H
#define HAIR_COMMON_H

#include "GBuffer.hlsl"

struct HairVertexInput {
    float3 position : POSITION;
    float3 tangent  : TANGENT;
    float3 color    : COLOR;
};

struct HairVertexOutput {
    float4 hposition : SV_Position;
    float3 position  : POSITION;
    float3 tangent   : TANGENT;
    float3 color     : COLOR;
};

struct HairGSOutput {
    float4 hposition : SV_Position;
    float3 position  : POSITION;
    float3 tangent   : TANGENT;
    float3 color     : COLOR;
    float  distanceFromCenter : DISTANCE_FROM_CENTER;
};

typedef HairGSOutput HairPixelInput;

cbuffer HairConstants {
    float4x4 mHairProj;
    float4x4 mHairWorldView;
    float4x4 mHairWorldViewProj;
}

float ComputeHairOpacity(float distanceFromCenter)
{
    // Input distance is really from -1 to 1 based on whether the vertex is
    // "above" or "below" the hair curve. Need to remap so that it's always 
    // from 0 to 1. Throw in saturate just in case.
    float actualDistanceFromCenter = saturate(abs(distanceFromCenter));
    float smoothAlpha = sqrt(1.0f - pow(actualDistanceFromCenter, 2));

    // Scale by alpha slider value.
    return smoothAlpha * (float(mUI.hairOpacity) / 100.0f);
}

HairVertexOutput
HairVS(HairVertexInput input)
{
    HairVertexOutput output;

    output.hposition = mul(float4(input.position, 1.0f), mHairWorldViewProj);
    output.position = mul(float4(input.position, 1.0f), mHairWorldView).xyz;
    output.tangent = mul(float4(input.tangent, 0.0f), mHairWorldView).xyz;
    output.color = input.color;

    return output;
}

[maxvertexcount(4)]
void HairGS(line HairVertexOutput input[2], inout TriangleStream<HairGSOutput> TriStream)
{
    int i;    
    HairGSOutput output;    

    // We work in view space, assume viewer pos = (0,0,0) and view vector = (0,0,1)    
    // Use center of the hair strand as view target
    const float3 viewVector = normalize((input[1].position + input[0].position) / 2.0f.xxx);
    
    // bill board axis (hair direction), will use this as z axis
    float3 zAxis  = normalize(input[1].position - input[0].position);    
    //  xAxis
    float3 xAxis = normalize(cross(zAxis, viewVector));    
    // Billboard offsets
    float3 offset[2] = {xAxis, -xAxis};    
    float distanceFromCenter[2] = { 1.0f, -1.0f };

    float hairThickness = mUI.hairThickness / 200.0f; // 100.0f;

    // Generate triangle strip (2 triangles). No need to restart the strip as each GS invocation already does it.    
    output.color = input[0].color;    
    output.tangent = input[0].tangent;    
    [unroll]for (i = 0; i < 2; i++) {
        output.position  = input[0].position + hairThickness.xxx * offset[i]; 
        output.hposition = mul(float4(output.position, 1.0f), mHairProj);   
	output.distanceFromCenter = distanceFromCenter[i];
        TriStream.Append(output);      
    }

   output.color = input[1].color;    
   output.tangent = input[1].tangent;
   [unroll] for (i = 0; i < 2; i++) {
        output.position  = input[1].position + hairThickness.xxx *  offset[i]; 
        output.hposition = mul(float4(output.position, 1.0f), mHairProj);   
	output.distanceFromCenter = distanceFromCenter[i];
        TriStream.Append(output);    
    }              
}

#endif // HAIR_COMMON_H
