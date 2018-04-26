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
#include "BloomPass.h"


BloomPass::BloomPass(ID3D10Device *device, int width, int height, DXGI_FORMAT format, float bloomWidth, float intensity)
        : device(device),
          bloomWidth(bloomWidth),
          intensity(intensity) {
    HRESULT hr;
    V(D3DX10CreateEffectFromResource(GetModuleHandle(NULL), L"Bloom.fxo", NULL, NULL, NULL, NULL, D3DXFX_NOT_CLONEABLE, 0, DXUTGetD3D10Device(), NULL, NULL, &effect, NULL, NULL));

    D3D10_PASS_DESC desc;
    V(effect->GetTechniqueByName("Downsample")->GetPassByIndex(0)->GetDesc(&desc));
    quad = new Quad(device, desc);
    
    renderTarget[0] = new RenderTarget(device, width, height, format);
    renderTarget[1] = new RenderTarget(device, width, height, format);
}


BloomPass::~BloomPass() {
    SAFE_RELEASE(effect);
    SAFE_DELETE(quad);
    SAFE_DELETE(renderTarget[0]);
    SAFE_DELETE(renderTarget[1]);
}


void BloomPass::render(ID3D10ShaderResourceView *src, ID3D10RenderTargetView *dst) {
    HRESULT hr;

    quad->setInputLayout();

    V(effect->GetVariableByName("pixelSize")->AsVector()->SetFloatVector((float *) D3DXVECTOR2(1.0f / renderTarget[0]->getWidth(), 1.0f /renderTarget[0]->getHeight())));
    V(effect->GetVariableByName("width")->AsScalar()->SetFloat(bloomWidth));
    V(effect->GetVariableByName("intensity")->AsScalar()->SetFloat(intensity));
    V(effect->GetVariableByName("finalTex")->AsShaderResource()->SetResource(src));

    D3D10_VIEWPORT viewport;
    UINT numViewports = 1;
    device->RSGetViewports(&numViewports, &viewport);

    renderTarget[0]->setViewport();
    downsample();
    horizontalBlur();
    verticalBlur();

    device->RSSetViewports(1, &viewport);
    combine(dst);
}


void BloomPass::downsample() {
    HRESULT hr;
    V(effect->GetTechniqueByName("Downsample")->GetPassByIndex(0)->Apply(0));
    device->OMSetRenderTargets(1, *renderTarget[0], NULL);
    quad->draw();
    device->OMSetRenderTargets(0, NULL, NULL);
}


void BloomPass::horizontalBlur() {
    HRESULT hr;
    V(effect->GetVariableByName("blurredTex")->AsShaderResource()->SetResource(*renderTarget[0]));
    V(effect->GetVariableByName("direction")->AsVector()->SetFloatVector((float *) D3DXVECTOR2(1.0f, 0.0f)));
    V(effect->GetTechniqueByName("Blur")->GetPassByIndex(0)->Apply(0));
    device->OMSetRenderTargets(1, *renderTarget[1], NULL);
    quad->draw();
    device->OMSetRenderTargets(0, NULL, NULL);
}


void BloomPass::verticalBlur() {
    HRESULT hr;
    V(effect->GetVariableByName("blurredTex")->AsShaderResource()->SetResource(*renderTarget[1]));
    V(effect->GetVariableByName("direction")->AsVector()->SetFloatVector((float *) D3DXVECTOR2(0.0f, 1.0f)));
    V(effect->GetTechniqueByName("Blur")->GetPassByIndex(0)->Apply(0));
    device->OMSetRenderTargets(1, *renderTarget[0], NULL);
    quad->draw();
    device->OMSetRenderTargets(0, NULL, NULL);
}


void BloomPass::combine(ID3D10RenderTargetView *dst) {
    HRESULT hr;
    V(effect->GetVariableByName("blurredTex")->AsShaderResource()->SetResource(*renderTarget[0]));
    V(effect->GetTechniqueByName("Combine")->GetPassByIndex(0)->Apply(0));
    device->OMSetRenderTargets(1, &dst, NULL);
    quad->draw();
}
