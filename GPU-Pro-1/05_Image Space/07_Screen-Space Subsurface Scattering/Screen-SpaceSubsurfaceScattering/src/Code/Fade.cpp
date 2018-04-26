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

#include <DXUT.h>
#include <string>
#include "Fade.h"
using namespace std;


ID3D10Device *Fade::device;
ID3D10BlendState *Fade::spriteBlendState;
ID3D10Effect *Fade::effect;
Quad *Fade::quad;


void Fade::init(ID3D10Device *device) {
    Fade::device = device;

    HRESULT hr;

    D3D10_BLEND_DESC blendDesc;
    ZeroMemory(&blendDesc, sizeof(D3D10_BLEND_DESC));
    blendDesc.AlphaToCoverageEnable = FALSE;
    blendDesc.BlendEnable[0] = TRUE;
    blendDesc.SrcBlend = D3D10_BLEND_ZERO;
    blendDesc.DestBlend = D3D10_BLEND_SRC_ALPHA;
    blendDesc.BlendOp = D3D10_BLEND_OP_ADD;
    blendDesc.SrcBlendAlpha = D3D10_BLEND_ZERO;
    blendDesc.DestBlendAlpha = D3D10_BLEND_ZERO;
    blendDesc.BlendOpAlpha = D3D10_BLEND_OP_ADD;
    blendDesc.RenderTargetWriteMask[0] = 0xf;
    device->CreateBlendState(&blendDesc, &spriteBlendState);

    string s = "float alpha;"
               "float4 VS(float4 position: POSITION, float2 texcoord: TEXCOORD0) : SV_POSITION { return position; }"
               "float4 PS() : SV_TARGET { return float4(0.0, 0.0, 0.0, alpha); }"
               "DepthStencilState DisableDepthStencil { DepthEnable = FALSE; StencilEnable = FALSE; };"
               "technique10 Black { pass Black {"
               "SetVertexShader(CompileShader(vs_4_0, VS())); SetGeometryShader(NULL); SetPixelShader(CompileShader(ps_4_0, PS()));"
               "SetDepthStencilState(DisableDepthStencil, 0);"
               "}}";
    V(D3DX10CreateEffectFromMemory(s.c_str(), s.length(), NULL, NULL, NULL, "fx_4_0", D3DXFX_NOT_CLONEABLE, 0, device, NULL, NULL, &effect, NULL, NULL));

    D3D10_PASS_DESC desc;
    V(effect->GetTechniqueByName("Black")->GetPassByIndex(0)->GetDesc(&desc));
    quad = new Quad(device, desc);
}


void Fade::release() {
    SAFE_RELEASE(spriteBlendState);
    SAFE_RELEASE(effect);
    SAFE_DELETE(quad);
}


void Fade::render(float t, float in, float out, float inLength, float outLength) {
    render(fade(t, in, out, inLength, outLength));
}


void Fade::render(float fade) {
    device->OMSetBlendState(spriteBlendState, 0, 0xffffffff);
    HRESULT hr;
    if (fade < 1.0f) {
        V(effect->GetVariableByName("alpha")->AsScalar()->SetFloat(pow(fade, 2.2f)));
        V(effect->GetTechniqueByName("Black")->GetPassByIndex(0)->Apply(0));
        quad->setInputLayout();
        quad->draw();
    }
}


float Fade::fade(float t, float in, float out, float inLength, float outLength) {
    if (t < in) {
        return 0.0f;
    } else if (t < in + inLength) {
        return (t - in) / inLength;
    } else if (t < out) {
        return 1.0f;
    } else {
        return max(1.0f - (t - out) / outLength, 0.0f);
    }
}
