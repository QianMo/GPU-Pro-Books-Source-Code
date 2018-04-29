/*
 * Copyright (C) 2010 Jorge Jimenez (jim@unizar.es)
 * Copyright (C) 2010 Belen Masia (bmasia@unizar.es)
 * Copyright (C) 2010 Jose I. Echevarria (joseignacioechevarria@gmail.com)
 * Copyright (C) 2010 Fernando Navarro (fernandn@microsoft.com)
 * Copyright (C) 2010 Diego Gutierrez (diegog@unizar.es)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must display the names 'Jorge Jimenez',
 *    'Belen Masia', 'Jose I. Echevarria', 'Fernando Navarro' and 'Diego
 *    Gutierrez' as 'Real-Time Rendering R&D' in the credits of the
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

#include <sstream>
#include <DXUT.h>
#include "MLAA.h"
using namespace std;


MLAA::MLAA(ID3D10Device *device, int width, int height)
        : device(device),
          maxSearchSteps(4.0f),
          threshold(0.1f) {
    HRESULT hr;

    stringstream s;
    s << "float2(1.0 / " << width << ", 1.0 / " << height << ")";
    string value = s.str();
    
    D3D10_SHADER_MACRO defines[2] = {
        {"PIXEL_SIZE", value.c_str()},
        {NULL, NULL}
    };

    V(D3DX10CreateEffectFromResource(GetModuleHandle(NULL), L"MLAA.fx", NULL, defines, NULL, "fx_4_0", D3DXFX_NOT_CLONEABLE, 0, DXUTGetD3D10Device(), NULL, NULL, &effect, NULL, NULL));

    D3D10_PASS_DESC desc;
    V(effect->GetTechniqueByName("NeighborhoodBlending")->GetPassByIndex(0)->GetDesc(&desc));
    quad = new Quad(device, desc);
    
    edgeRenderTarget = new RenderTarget(device, width, height, DXGI_FORMAT_R8G8_UNORM);
    blendRenderTarget = new RenderTarget(device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM);

    D3DX10_IMAGE_LOAD_INFO info = D3DX10_IMAGE_LOAD_INFO();
    info.MipLevels = 1;
    info.Format = DXGI_FORMAT_R8G8_UNORM;
    V(D3DX10CreateShaderResourceViewFromResource(DXUTGetD3D10Device(), GetModuleHandle(NULL), L"AreaMap32.dds", &info, NULL, &areaMapView, NULL));
    
    V(effect->GetVariableByName("areaTex")->AsShaderResource()->SetResource(areaMapView));
}


MLAA::~MLAA() {
    SAFE_RELEASE(effect);
    SAFE_DELETE(quad);
    SAFE_DELETE(edgeRenderTarget);
    SAFE_DELETE(blendRenderTarget);
    SAFE_RELEASE(areaMapView);
}


void MLAA::go(ID3D10ShaderResourceView *src, ID3D10RenderTargetView *dst, ID3D10DepthStencilView *depthStencil, ID3D10ShaderResourceView *depthResource) {
    SaveViewportScope save(device);

    edgeRenderTarget->setViewport();
    quad->setInputLayout();

    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device->ClearRenderTargetView(*edgeRenderTarget, clearColor);
    device->ClearRenderTargetView(*blendRenderTarget, clearColor);

    /**
     * Here it is the meat of the technique =)
     */
    edgesDetectionPass(src, depthResource, depthStencil);
    blendingWeightsCalculationPass(depthStencil);
    neighborhoodBlendingPass(src, dst, depthStencil);
}


void MLAA::edgesDetectionPass(ID3D10ShaderResourceView *src, ID3D10ShaderResourceView *depthResource, ID3D10DepthStencilView *depthStencil) {
    HRESULT hr;

    V(effect->GetVariableByName("threshold")->AsScalar()->SetFloat(threshold));
    V(effect->GetVariableByName("colorTex")->AsShaderResource()->SetResource(src));
    V(effect->GetVariableByName("depthTex")->AsShaderResource()->SetResource(depthResource));

    if (depthResource != NULL) {
        V(effect->GetTechniqueByName("EdgeDetectionDepth")->GetPassByIndex(0)->Apply(0));
    } else {
        V(effect->GetTechniqueByName("EdgeDetection")->GetPassByIndex(0)->Apply(0));
    }

    device->OMSetRenderTargets(1, *edgeRenderTarget, depthStencil);
    quad->draw();
    device->OMSetRenderTargets(0, NULL, NULL);
}


void MLAA::blendingWeightsCalculationPass(ID3D10DepthStencilView *depthStencil) {
    HRESULT hr;
    V(effect->GetVariableByName("edgesTex")->AsShaderResource()->SetResource(*edgeRenderTarget));
    V(effect->GetTechniqueByName("BlendingWeightCalculation")->GetPassByIndex(0)->Apply(0));

    device->OMSetRenderTargets(1, *blendRenderTarget, depthStencil);
    quad->draw();
    device->OMSetRenderTargets(0, NULL, NULL);
}


void MLAA::neighborhoodBlendingPass(ID3D10ShaderResourceView *src, ID3D10RenderTargetView *dst, ID3D10DepthStencilView *depthStencil) {
    HRESULT hr;
    V(effect->GetVariableByName("maxSearchSteps")->AsScalar()->SetFloat(maxSearchSteps));
    V(effect->GetVariableByName("colorTex")->AsShaderResource()->SetResource(src));
    V(effect->GetVariableByName("blendTex")->AsShaderResource()->SetResource(*blendRenderTarget));
    V(effect->GetTechniqueByName("NeighborhoodBlending")->GetPassByIndex(0)->Apply(0));

    device->OMSetRenderTargets(1, &dst, depthStencil);
    quad->draw();
    device->OMSetRenderTargets(0, NULL, NULL);
}
