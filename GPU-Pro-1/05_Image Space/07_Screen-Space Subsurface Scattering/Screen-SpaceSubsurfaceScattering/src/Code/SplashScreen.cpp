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
#include "SplashScreen.h"
using namespace std;


SplashScreen::SplashScreen(ID3D10Device *device, ID3DX10Sprite *sprite)
        : device(device), sprite(sprite) {
    HRESULT hr;

    D3DX10_IMAGE_LOAD_INFO loadInfo;
    ZeroMemory(&loadInfo, sizeof(D3DX10_IMAGE_LOAD_INFO));
    loadInfo.BindFlags = D3D10_BIND_SHADER_RESOURCE;
    loadInfo.MipLevels = 1;
    loadInfo.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    loadInfo.Filter = D3DX10_FILTER_POINT | D3DX10_FILTER_SRGB_IN;
    V(D3DX10CreateShaderResourceViewFromResource(device, GetModuleHandle(NULL), L"SmallTitles.png", &loadInfo, NULL, &smallTitlesView, NULL));
    V(D3DX10CreateShaderResourceViewFromResource(device, GetModuleHandle(NULL), L"BigTitles.png", &loadInfo, NULL, &bigTitlesView, NULL));

    D3D10_BLEND_DESC blendDesc;
    ZeroMemory(&blendDesc, sizeof(D3D10_BLEND_DESC));
    blendDesc.AlphaToCoverageEnable = FALSE;
    blendDesc.BlendEnable[0] = TRUE;
    blendDesc.SrcBlend = D3D10_BLEND_SRC_ALPHA;
    blendDesc.DestBlend = D3D10_BLEND_INV_SRC_ALPHA;
    blendDesc.BlendOp = D3D10_BLEND_OP_ADD;
    blendDesc.SrcBlendAlpha = D3D10_BLEND_ZERO;
    blendDesc.DestBlendAlpha = D3D10_BLEND_ZERO;
    blendDesc.BlendOpAlpha = D3D10_BLEND_OP_ADD;
    blendDesc.RenderTargetWriteMask[0] = 0xf;
    device->CreateBlendState(&blendDesc, &spriteBlendState);
}


SplashScreen::~SplashScreen() {
    SAFE_RELEASE(smallTitlesView);
    SAFE_RELEASE(bigTitlesView);
    SAFE_RELEASE(spriteBlendState);
}


void SplashScreen::render(float t) {
    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    device->ClearRenderTargetView(DXUTGetD3D10RenderTargetView(), clearColor);

    D3D10_VIEWPORT viewport;
    UINT numViewports = 1;
    device->RSGetViewports(&numViewports, &viewport);

    D3DX10_SPRITE s[7];
    s[0] = buildSprite(smallTitlesView, viewport, 0,   70, 0, 5, 546, 24, fade(t, 1.0f, 6.0f, 2.0f, 2.0f));
    s[1] = buildSprite(smallTitlesView, viewport, 0,   40, 1, 5, 546, 24, fade(t, 2.0f, 6.0f, 2.5f, 2.0f));
    s[2] = buildSprite(smallTitlesView, viewport, 0,  -40, 2, 5, 546, 24, fade(t, 1.4f, 6.0f, 2.0f, 2.0f));
    s[3] = buildSprite(smallTitlesView, viewport, 0,  -70, 3, 5, 546, 24, fade(t, 2.4f, 6.0f, 2.5f, 2.0f));
    s[4] = buildSprite(bigTitlesView,   viewport, 0,    0, 0, 2, 782, 88, fade(t, 8.5f, 12.0f, 1.0f, 1.0f));
    s[5] = buildSprite(bigTitlesView,   viewport, 0,    0, 1, 2, 782, 88, fade(t, 13.5f, 17.0f, 1.0f, 3.0f));
    s[6] = buildSprite(smallTitlesView, viewport, 0, -140, 4, 5, 546, 24, fade(t, 14.5f, 17.0f, 1.0f, 3.0f));

    device->OMSetBlendState(spriteBlendState, 0, 0xffffffff);
    sprite->Begin(D3DX10_SPRITE_SAVE_STATE);
    sprite->DrawSpritesImmediate(s, 7, 0, 0);
    sprite->End();
}


float SplashScreen::fade(float t, float in, float out, float inLength, float outLength) {
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


D3DX10_SPRITE SplashScreen::buildSprite(ID3D10ShaderResourceView *view, const D3D10_VIEWPORT &viewport, int x, int y, int i, int n, int width, int height, float alpha) {
    D3DXMATRIX scale, translation;
    D3DXMatrixScaling(&scale, 2.0f * width / viewport.Width, 2.0f * height / viewport.Height, 1.0f);
    D3DXMatrixTranslation(&translation, 2.0f * float(x) / viewport.Width, 2.0f * float(y) / viewport.Height, 0.0f);

    D3DX10_SPRITE s;
    s.ColorModulate = D3DXCOLOR(1.0f, 1.0f, 1.0f, pow(alpha, 2.2f));
    s.matWorld = scale * translation;
    s.pTexture = view;
    s.TexCoord = D3DXVECTOR2(0.0f, float(i) / n);
    s.TexSize = D3DXVECTOR2(1.0f, 1.0f / n);
    s.TextureIndex = 0;
    return s;
}
