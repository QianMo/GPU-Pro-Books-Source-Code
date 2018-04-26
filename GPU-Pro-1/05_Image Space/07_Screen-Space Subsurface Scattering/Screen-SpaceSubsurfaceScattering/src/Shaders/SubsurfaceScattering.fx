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

cbuffer UpdatedOnce {
    float2 pixelSize;
}

cbuffer UpdatedUserInput {
    float sssLevel;
    float correction;
    float maxdd;
    float3 projection;
}

cbuffer UpdatedPerFrame {
    int material;
}

cbuffer UpdatedPerBlurPass {
    float depth;
    float width;
    float4 weight;
}

Texture2D tex1;
Texture2D tex2;
Texture2D depthTex;

Texture2DMS<float, N_SAMPLES> depthTexMS;
Texture2DMS<float, N_SAMPLES> stencilTexMS;


SamplerState LinearSampler {
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Clamp;
    AddressV = Clamp;
};

SamplerState PointSampler {
    Filter = MIN_MAG_MIP_POINT;
    AddressU = Clamp;
    AddressV = Clamp;
};


struct PassV2P {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD0;
};

PassV2P PassVS(float4 position : POSITION,
               float2 texcoord : TEXCOORD0) {
    PassV2P output;
    output.position = position;
    output.texcoord = texcoord;
    return output;
}


struct DownsampleP2S {
    float depth : SV_TARGET0;
    float z : SV_DEPTH;
};

DownsampleP2S DownsamplePS(PassV2P input) {
    float w, h, samples;
    depthTexMS.GetDimensions(w, h, samples);
    int2 pos = float2(w, h) * input.texcoord;

    bool skip = true;
    [unroll]
    for (int i = 0; i < samples; i++) {
        skip &= stencilTexMS.Load(pos, i) != material;
    }
    clip(-1 * skip);

    float depth = 0.0;
    [unroll]
    for (i = 0; i < samples; i++) {
        depth += depthTexMS.Load(pos, i);
    }
    depth /= samples;

    DownsampleP2S output;
    output.depth = depth;
    output.z = (projection.x * depth + projection.y) / depth * projection.z;
    return output;
}


PassV2P PassZVS(float4 position : POSITION,
               float2 texcoord : TEXCOORD0) {
    PassV2P output;
    output.position = position;
    output.position.z = depth;
    output.texcoord = texcoord;
    return output;
}


float4 BlurPS(PassV2P input, uniform float2 step) : SV_TARGET {
    float w[7] = {
        0.006,
        0.061,
        0.242,
        0.382,
        0.242,
        0.061,
        0.006
    };
    
    float4 color = tex1.Sample(LinearSampler, input.texcoord);
    color.rgb *= w[3];

    float depth = depthTex.Sample(PointSampler, input.texcoord).r;
    float2 s_x = step / (depth + correction * min(abs(ddx(depth)), maxdd));
    float2 finalWidth = color.a * s_x; // step = sssLevel * width * pixelSize * float2(1.0, 0.0)

    float2 offset = input.texcoord - finalWidth;
    [unroll]
    for (int i = 0; i < 3; i++) {
        color.rgb += w[i] * tex1.Sample(LinearSampler, offset).rgb;
        offset += finalWidth / 3.0;
    }
    offset += finalWidth / 3.0;
    [unroll]
    for (i = 4; i < 7; i++) {
        color.rgb += w[i] * tex1.Sample(LinearSampler, offset).rgb;
        offset += finalWidth / 3.0;
    }

    return color;
}


struct BlurAccumP2S {
    float4 gaussian : SV_TARGET0;
    float4 final : SV_TARGET1;
};

BlurAccumP2S BlurAccumPS(PassV2P input, uniform float2 step) {
    float w[7] = {
        0.006,
        0.061,
        0.242,
        0.382,
        0.242,
        0.061,
        0.006
    };
    
    float4 color = tex2.Sample(LinearSampler, input.texcoord);
    color.rgb *= w[3];

    float depth = depthTex.Sample(PointSampler, input.texcoord).r;
    float2 s_y = step / (depth + correction * min(abs(ddy(depth)), maxdd));
    float2 finalWidth = color.a * s_y; // step = sssLevel * width * pixelSize * float2(0.0, 1.0)

    float2 offset = input.texcoord - finalWidth;
    [unroll]
    for (int i = 0; i < 3; i++) {
        color.rgb += w[i] * tex2.Sample(LinearSampler, offset).rgb;
        offset += finalWidth / 3.0;
    }
    offset += finalWidth / 3.0;
    [unroll]
    for (i = 4; i < 7; i++) {
        color.rgb += w[i] * tex2.Sample(LinearSampler, offset).rgb;
        offset += finalWidth / 3.0;
    }
    
    BlurAccumP2S output;
    output.gaussian = color;
    output.final = color;

    return output;
}


DepthStencilState DownsampleStencil {
    DepthEnable = TRUE;
    DepthWriteMask = ALL;
    DepthFunc = LESS_EQUAL;

    StencilEnable = TRUE;
    FrontFaceStencilPass = REPLACE;
};

DepthStencilState BlurStencil {
    DepthEnable = TRUE;
    DepthWriteMask = ZERO;
    DepthFunc = GREATER;

    StencilEnable = TRUE;
    FrontFaceStencilFunc = EQUAL;
};

BlendState NoBlending {
    AlphaToCoverageEnable = FALSE;
    BlendEnable[0] = FALSE;
    BlendEnable[1] = FALSE;
};

BlendState BlendingAccum {
    AlphaToCoverageEnable = FALSE;
    BlendEnable[0] = FALSE;
    BlendEnable[1] = TRUE;
    SrcBlend = BLEND_FACTOR;
    DestBlend = INV_BLEND_FACTOR;
};


technique10 SubsurfaceScattering {
    pass Downsample {
        SetVertexShader(CompileShader(vs_4_0, PassVS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, DownsamplePS()));
        
        SetDepthStencilState(DownsampleStencil, material);
        SetBlendState(NoBlending, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xFFFFFFFF);
    }

    pass Blur {
        SetVertexShader(CompileShader(vs_4_0, PassZVS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, BlurPS(sssLevel * width * pixelSize * float2(1.0, 0.0))));
        
        SetDepthStencilState(BlurStencil, material);
        SetBlendState(NoBlending, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xFFFFFFFF);
    }

    pass BlurAccum {
        SetVertexShader(CompileShader(vs_4_0, PassZVS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, BlurAccumPS(sssLevel * width * pixelSize * float2(0.0, 1.0))));
        
        SetDepthStencilState(BlurStencil, material);
        SetBlendState(BlendingAccum, weight, 0xFFFFFFFF);
    }
}