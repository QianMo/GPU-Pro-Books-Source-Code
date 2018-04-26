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

cbuffer UpdatedOncePerFrame {
    float2 pixelSize;
    float width;
    float intensity;
}

cbuffer UpdatedPerBlurPass {
    float2 direction;
}

Texture2D finalTex;
Texture2D blurredTex;


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


float4 DownsamplePS(PassV2P input) : SV_TARGET {
    return finalTex.Sample(LinearSampler, input.texcoord);
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
    
    float2 offset = input.texcoord - 3.0 * step;
    float4 color = float4(0.0, 0.0, 0.0, 1.0);
    for (int i = 0; i < 7; i++) {
        color.rgb += w[i] * blurredTex.Sample(LinearSampler, offset).rgb;
        offset += step;
    }

    return color;
}


float4 CombinePS(PassV2P input) : SV_TARGET {
    return float4(finalTex.Sample(PointSampler, input.texcoord).rgb + 
                  blurredTex.Sample(LinearSampler, input.texcoord).rgb * intensity, 1.0);
}


DepthStencilState DisableDepthStencil {
    DepthEnable = FALSE;
    StencilEnable = FALSE;
};

BlendState NoBlending {
    AlphaToCoverageEnable = FALSE;
    BlendEnable[0] = FALSE;
};


technique10 Downsample {
    pass Downsample {
        SetVertexShader(CompileShader(vs_4_0, PassVS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, DownsamplePS()));
        
        SetDepthStencilState(DisableDepthStencil, 0);
        SetBlendState(NoBlending, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xFFFFFFFF);
    }
}

technique10 Blur {
    pass Blur {
        SetVertexShader(CompileShader(vs_4_0, PassVS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, BlurPS(pixelSize * width * direction)));
        
        SetDepthStencilState(DisableDepthStencil, 0);
        SetBlendState(NoBlending, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xFFFFFFFF);
    }
}

technique10 Combine {
    pass Combine {
        SetVertexShader(CompileShader(vs_4_0, PassVS()));
        SetGeometryShader(NULL);
        SetPixelShader(CompileShader(ps_4_0, CombinePS()));
        
        SetDepthStencilState(DisableDepthStencil, 0);
        SetBlendState(NoBlending, float4(0.0f, 0.0f, 0.0f, 0.0f), 0xFFFFFFFF);
    }
}
