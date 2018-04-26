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
#include <sstream>
#include "SubsurfaceScatteringPass.h"
using namespace std;


#define DEG2RAD(a) (a * D3DX_PI / 180.f)


// This class is just a helper to be able to initialize the vectors at program
// initialization.
class SkinGaussianSum : public std::vector<Gaussian> {
    public:
        SkinGaussianSum() {
            // We use the unblurred image as an aproximation to the first 
            // gaussian because it is too narrow to be noticeable. The weight
            // of the unblurred image is the first one.
            D3DXVECTOR3 weights[] = {
                D3DXVECTOR3(0.240516183695f, 0.447403391891f, 0.615796108321f), 
                D3DXVECTOR3(0.115857499765f, 0.366176401412f, 0.343917471552f),
                D3DXVECTOR3(0.183619017698f, 0.186420206697f, 0.0f),
                D3DXVECTOR3(0.460007298842f, 0.0f, 0.0402864201267f)
            };
            float variances[] = { 0.0516500425655f, 0.271928080903f, 2.00626388153f };
            std::vector<Gaussian> &gaussianSum = *this;
            gaussianSum = Gaussian::gaussianSum(variances, weights, 3);
        }
};

class MarbleGaussianSum : public std::vector<Gaussian> {
    public:
        MarbleGaussianSum() {
            // In this case the first gaussian is wide and thus we cannot
            // approximate it with the unblurred image. For this reason the
            // first weight is set to zero.
            D3DXVECTOR3 weights[] = {
                D3DXVECTOR3(0.0f, 0.0f, 0.0f),
                D3DXVECTOR3(0.0544578254963f, 0.12454890956f, 0.217724878147f),
                D3DXVECTOR3(0.243663230592f, 0.243532369381f, 0.18904245481f),
                D3DXVECTOR3(0.310530428621f, 0.315816663292f, 0.374244725886f),
                D3DXVECTOR3(0.391348515291f, 0.316102057768f, 0.218987941157f)
            };
            float variances[] = { 0.0362208693441f, 0.114450574559f, 0.455584392509f, 3.48331959682f };
            std::vector<Gaussian> &gaussianSum = *this;
            gaussianSum = Gaussian::gaussianSum(variances, weights, 4);
        }
};

const vector<Gaussian> Gaussian::SKIN = SkinGaussianSum();
const vector<Gaussian> Gaussian::MARBLE = MarbleGaussianSum();


vector<Gaussian> Gaussian::gaussianSum(float variances[], D3DXVECTOR3 weights[], int nVariances) {
    vector<Gaussian> gaussians;
    for (int i = 0; i < nVariances; i++) {
        float variance = i == 0? variances[i] : variances[i] - variances[i - 1];
        gaussians.push_back(Gaussian(variance, weights, i));
    }
    return gaussians;
}


Gaussian::Gaussian(float variance, D3DXVECTOR3 weights[], int n)
    : width(sqrt(variance)) {
    D3DXVECTOR3 total = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < n + 2; i++) {
        total += weights[i];
    }

    weight = D3DXVECTOR4(weights[n + 1], 1.0f);
    weight[0] *= 1.0f / total[0];
    weight[1] *= 1.0f / total[1];
    weight[2] *= 1.0f / total[2];
}


SubsurfaceScatteringPass::SubsurfaceScatteringPass(ID3D10Device *device,
                                                   int width,
                                                   int height,
                                                   DXGI_FORMAT format,
                                                   int samples,
                                                   const D3DXMATRIX &projection,
                                                   float sssLevel,
                                                   float correction,
                                                   float maxdd)
        : device(device),
          width(width),
          height(height),
          projection(projection),
          sssLevel(sssLevel),
          correction(correction),
          maxdd(maxdd) {
    HRESULT hr;

    wstringstream s;
    s << L"SubsurfaceScattering" << samples << ".fxo";
    V(D3DX10CreateEffectFromResource(GetModuleHandle(NULL), s.str().c_str(), NULL, NULL, NULL, NULL, D3DXFX_NOT_CLONEABLE, 0, DXUTGetD3D10Device(), NULL, NULL, &effect, NULL, NULL));

    D3D10_PASS_DESC desc;
    V(effect->GetTechniqueByName("SubsurfaceScattering")->GetPassByName("Blur")->GetDesc(&desc));
    quad = new Quad(device, desc);

    renderTarget[0] = new RenderTarget(device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
    renderTarget[1] = new RenderTarget(device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);

    V(effect->GetVariableByName("pixelSize")->AsVector()->SetFloatVector((float *) D3DXVECTOR2(1.0f / width, 1.0f / height)));
    setInputVars();
}


void SubsurfaceScatteringPass::setInputVars() {
    HRESULT hr;
    V(effect->GetVariableByName("correction")->AsScalar()->SetFloat(correction));
    V(effect->GetVariableByName("maxdd")->AsScalar()->SetFloat(maxdd));

    // This changes the SSS level depending on the viewport size and the camera
    // FOV. For a camera FOV of 20.0 and a viewport height of 720 pixels, we
    // want to use the sssLevel as is. For other cases, we will scale it
    // accordingly.
    // In D3DXMatrixPerspectiveFovLH we have _22 = cot(fovY / 2), thus:
    float scaleViewport = height / 720.0f;
    float scaleFov = projection._22 / (1.0f / tan(DEG2RAD(20.0f) / 2.0f));
    float t = scaleViewport * scaleFov * sssLevel;
    V(effect->GetVariableByName("sssLevel")->AsScalar()->SetFloat(t));

    V(effect->GetVariableByName("projection")->AsVector()->SetFloatVector(D3DXVECTOR3(projection._33, projection._43, projection._34))); 
}


SubsurfaceScatteringPass::~SubsurfaceScatteringPass() {
    SAFE_DELETE(renderTarget[0]);
    SAFE_DELETE(renderTarget[1]);
    SAFE_RELEASE(effect);
    SAFE_DELETE(quad);
}


void SubsurfaceScatteringPass::downsample(ID3D10RenderTargetView *depthRenderTarget,
                                          ID3D10ShaderResourceView *depthResource,
                                          ID3D10ShaderResourceView *stencilResource,
                                          ID3D10DepthStencilView *depthStencil) {
    HRESULT hr;

    device->ClearDepthStencilView(depthStencil, D3D10_CLEAR_DEPTH | D3D10_CLEAR_STENCIL, 1.0, 0);

    V(effect->GetVariableByName("depthTexMS")->AsShaderResource()->SetResource(depthResource));
    V(effect->GetVariableByName("stencilTexMS")->AsShaderResource()->SetResource(stencilResource));
    V(effect->GetTechniqueByName("SubsurfaceScattering")->GetPassByName("Downsample")->Apply(0));

    device->OMSetRenderTargets(1, &depthRenderTarget, depthStencil);
    quad->draw();
    device->OMSetRenderTargets(0, NULL, NULL);
}


void SubsurfaceScatteringPass::render(ID3D10RenderTargetView *mainRenderTarget, 
                                      ID3D10ShaderResourceView *mainResource, 
                                      ID3D10ShaderResourceView *depthResource, 
                                      ID3D10DepthStencilView *depthStencil, 
                                      const vector<Gaussian> &gaussians, 
                                      int stencilId) {
    HRESULT hr;

    quad->setInputLayout();

    V(effect->GetVariableByName("material")->AsScalar()->SetInt(stencilId));
    V(effect->GetVariableByName("tex2")->AsShaderResource()->SetResource(*renderTarget[1]));
    V(effect->GetVariableByName("depthTex")->AsShaderResource()->SetResource(depthResource));
    
    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device->ClearRenderTargetView(*renderTarget[0], clearColor);
    device->ClearRenderTargetView(*renderTarget[1], clearColor);

    blurPass(mainResource, *renderTarget[0], mainRenderTarget, depthStencil, gaussians[0], true);
    for (int i = 1; i < signed(gaussians.size()); i++) {
        blurPass(*renderTarget[0], *renderTarget[0], mainRenderTarget, depthStencil, gaussians[i], false);
    }
}


void SubsurfaceScatteringPass::blurPass(ID3D10ShaderResourceView *src, 
                                        ID3D10RenderTargetView *dst,
                                        ID3D10RenderTargetView *final,
                                        ID3D10DepthStencilView *depthStencil,
                                        const Gaussian &gaussian,
                                        bool firstGaussian) {
    HRESULT hr;

    float depth = firstGaussian? 1.0f : min(max(linearToDepth(0.5f * gaussian.getWidth() * sssLevel), 0.0f), 1.0f);
    V(effect->GetVariableByName("depth")->AsScalar()->SetFloat(depth));
    V(effect->GetVariableByName("width")->AsScalar()->SetFloat(gaussian.getWidth()));
    V(effect->GetVariableByName("weight")->AsVector()->SetFloatVector((float *) gaussian.getWeight()));

    V(effect->GetVariableByName("tex1")->AsShaderResource()->SetResource(src));
    effect->GetTechniqueByName("SubsurfaceScattering")->GetPassByName("Blur")->Apply(0);
    device->OMSetRenderTargets(1, *renderTarget[1], depthStencil);
    quad->draw();
    device->OMSetRenderTargets(0, NULL, NULL);

    effect->GetTechniqueByName("SubsurfaceScattering")->GetPassByName("BlurAccum")->Apply(0);
    ID3D10RenderTargetView *rt[] = { dst, final };
    device->OMSetRenderTargets(2, rt, depthStencil);
    quad->draw();
    device->OMSetRenderTargets(0, NULL, NULL);
}


float SubsurfaceScatteringPass::linearToDepth(float z) {
    D3DXVECTOR4 v = D3DXVECTOR4(0.0, 0.0, z, 1.0);
    D3DXVECTOR4 vv;
    D3DXVec4Transform(&vv, &v, &projection);
    return vv.z / vv.w;
}
