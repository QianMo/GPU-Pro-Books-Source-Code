/*
 * Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)
 * Copyright (C) 2009 Diego Gutierrez (diegog@unizar.es)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, 
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must display the names 'Jorge Jimenez'
 *    and 'Diego Gutierrez' as 'Real-Time Rendering R&D' in the credits of the
 *    application, if such credits exist. The authors of this work must be
 *    notified via email (jim@unizar.es) in this case of redistribution.
 * 
 * 3. Neither the name of copyright holders nor the names of its contributors 
 *    may be used to endorse or promote products derived from this software 
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS 
 * IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS 
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma warning(disable: 3571) 

struct Light {
    float3 position;
    float3 direction;
    float falloffAngle;
    float spotExponent;
    float3 color;
    float attenuation;
    float range;
    float4x4 viewProjection;
};

cbuffer UpdatedPerFrame {
    matrix view;
    matrix projection;
    float3 cameraPosition;
    Light lights[N_LIGHTS];
}

cbuffer UpdatedPerObject {
    matrix world;
    matrix worldInverseTranspose;
    int material;
    float roughness;
    float bumpiness;
    bool fade;
}

Texture2D<float> shadowMaps[N_LIGHTS];
Texture2D diffuseTex;
Texture2D normalTex;
Texture2D beckmannTex;


SamplerState PointSampler {
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};

SamplerState LinearSampler {
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

SamplerState AnisotropicSampler {
    Filter = ANISOTROPIC;
    AddressU = Clamp;
    AddressV = Clamp;
    MaxAnisotropy = 16;
};

SamplerComparisonState ShadowSampler {
    ComparisonFunc = Less;
    Filter = COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
};


struct SceneV2P {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD0;
    float3 tangentView : TEXCOORD1;
    float3 worldPosition : TEXCOORD2;
    centroid float3 tangentLight[N_LIGHTS] : TEXCOORD3;
};

SceneV2P SceneVS(float4 position : POSITION0,
                 float3 normal : NORMAL,
                 float3 tangent : TANGENT,
                 float2 texcoord : TEXCOORD0,
                 uniform float4x4 worldViewProjection) {
    SceneV2P output;
  
    output.position = mul(position, worldViewProjection);
    output.texcoord = texcoord;
    output.worldPosition = mul(position, world).xyz;
    
    float3 N = normalize(mul(normal, (float3x3) worldInverseTranspose));
    float3 T = normalize(mul(tangent, (float3x3) worldInverseTranspose));
    float3 B = cross(N, T);
    float3x3 frame = float3x3(T, B, N);

    float3 view = cameraPosition - output.worldPosition;
    output.tangentView = mul(frame, view);

    [unroll]
    for (int i = 0; i < N_LIGHTS; i++) {
        float3 light = lights[i].position - output.worldPosition;
        output.tangentLight[i] = mul(frame, light);
    }

    return output;
}


float3 BumpMap(Texture2D normalTex, float2 texcoord) {
    float3 bump;
    bump.xy = normalTex.Sample(AnisotropicSampler, texcoord).ag * 2.0 - 1.0;
    bump.z = sqrt(1.0 - bump.x * bump.x - bump.y * bump.y);
    return normalize(bump);
}

float Fresnel(float3 half, float3 view, float f0) {
    float base = 1.0 - dot(view, half);
    float exponential = pow(base, 5.0);
    return exponential + f0 * (1.0 - exponential);
}

float SpecularKSK(Texture2D beckmannTex, float3 normal, float3 light, float3 view) {
    float3 half = view + light;
    float3 halfn = normalize(half);

    float ndotl = max(dot(normal, light), 0.0);
    float ndoth = max(dot(normal, halfn), 0.0);

    float ph = pow(2.0 * beckmannTex.SampleLevel(LinearSampler, float2(ndoth, roughness), 0).r, 10.0);
    float f = Fresnel(halfn, view, 0.028);
    float ksk = max(ph * f / dot(half, half), 0.0);

    return ndotl * ksk;   
}

float Shadow(float3 worldPosition, int i) {
    float4 shadowPosition = mul(float4(worldPosition, 1.0), lights[i].viewProjection);
    shadowPosition.xy /= shadowPosition.w;
    return shadowMaps[i].SampleCmpLevelZero(ShadowSampler, shadowPosition.xy, shadowPosition.z / lights[i].range);
}

float ShadowPCF(float3 worldPosition, int i, int samples, float width) {
    float4 shadowPosition = mul(float4(worldPosition, 1.0), lights[i].viewProjection);
    shadowPosition.xy /= shadowPosition.w;
    
    float w, h;
    shadowMaps[i].GetDimensions(w, h);

    float shadow = 0.0;
    float offset = (samples - 1.0) / 2.0;
    [unroll]
    for (float x = -offset; x <= offset; x += 1.0) {
        [unroll]
        for (float y = -offset; y <= offset; y += 1.0) {
            float2 pos = shadowPosition.xy + width * float2(x, y) / w;
            shadow += shadowMaps[i].SampleCmpLevelZero(ShadowSampler, pos, shadowPosition.z / lights[i].range);
        }
    }
    shadow /= samples * samples;
    return shadow;
}


float4 SceneColor(SceneV2P input) {
    float3 tangentNormal = lerp(float3(0.0, 0.0, 1.0), BumpMap(normalTex, input.texcoord), bumpiness);
    float3 tangentView = normalize(input.tangentView);
    float4 albedo = diffuseTex.Sample(AnisotropicSampler, input.texcoord);

    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    [unroll]
    for (int i = 0; i < N_LIGHTS; i++) {
        float3 light = lights[i].position - input.worldPosition;

        float spot = dot(normalize(lights[i].direction), -normalize(light));
        [flatten]
        if (spot > lights[i].falloffAngle) {
            float3 tangentLight = normalize(input.tangentLight[i]);

            //float shadow = Shadow(input.worldPosition, i);
            float shadow = ShadowPCF(input.worldPosition, i, 3, 1.0);

            float dist = length(light);
            float curve = min(pow(dist / lights[i].range, 6.0), 1.0);
            float attenuation = lerp(1.0 / (1.0 + lights[i].attenuation * dist * dist), 0.0, curve);

            spot = pow(spot, lights[i].spotExponent);

            float3 diffuse = albedo.rgb * max(dot(tangentLight, tangentNormal), 0.0);

            float specular = SpecularKSK(beckmannTex, tangentNormal, tangentLight, tangentView);

            color.rgb += lights[i].color * shadow * attenuation * spot * (diffuse + specular);
        }
    }
    color.a = albedo.a;
    
    [flatten] 
    if (fade) {
        float d = abs(0.5 - input.texcoord.x);
        color.rgb *= 1.0 - 10.0f * max(input.texcoord.y - 0.76f + 0.22 * d * d, 0.0);
    }

    return color;
}


struct SceneMSAAP2S {
    float4 color : SV_TARGET0;
    float depth : SV_TARGET1;
    float stencil : SV_TARGET2;
};

SceneMSAAP2S SceneMSAAPS(SceneV2P input) {
    SceneMSAAP2S output;
    output.color = SceneColor(input);
    output.depth = input.position.w;
    output.stencil = material;
    return output;
}


struct SceneNoMSAAP2S {
    float4 color : SV_TARGET0;
    float4 depth : SV_TARGET1;
};

SceneNoMSAAP2S SceneNoMSAAPS(SceneV2P input) {
    SceneNoMSAAP2S output;
    output.color = SceneColor(input);
    output.depth = input.position.w;
    return output;
}


DepthStencilState EnableDepthDisableStencil {
    DepthEnable = TRUE;
    StencilEnable = FALSE;
};

DepthStencilState EnableDepthStencil {
    DepthEnable = TRUE;
    DepthWriteMask = ALL;
    DepthFunc = LESS_EQUAL;

    StencilEnable = TRUE;
    FrontFaceStencilPass = REPLACE;
};

BlendState NoBlending {
    AlphaToCoverageEnable = FALSE;
    BlendEnable[0] = FALSE;
    BlendEnable[1] = FALSE;
};


technique10 RenderMSAA {
    pass RenderMSAA {
        SetVertexShader(CompileShader(vs_4_0, SceneVS(mul(mul(world, view), projection))));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, SceneMSAAPS()));
        
        SetDepthStencilState(EnableDepthDisableStencil, material);
        SetBlendState(NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}

technique10 RenderNoMSAA {
    pass RenderNoMSAA {
        SetVertexShader(CompileShader(vs_4_0, SceneVS(mul(mul(world, view), projection))));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, SceneNoMSAAPS()));
        
        SetDepthStencilState(EnableDepthStencil, material);
        SetBlendState(NoBlending, float4( 0.0f, 0.0f, 0.0f, 0.0f ), 0xFFFFFFFF);
    }
}
