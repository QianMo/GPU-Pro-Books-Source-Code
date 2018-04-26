#ifndef SPLASHSCREEN_H
#define SPLASHSCREEN_H

#include "RenderTarget.h"

class SplashScreen {
    public:
        SplashScreen(ID3D10Device *device, ID3DX10Sprite *sprite);
        ~SplashScreen();

        void render(float t);

        bool hasFinished(float t) const { return t > 20.5f; }

    private:
        float fade(float t, float in, float out, float inLength, float outLength);
        D3DX10_SPRITE buildSprite(ID3D10ShaderResourceView *view, const D3D10_VIEWPORT &viewport, int x, int y, int i, int n, int width, int height, float alpha);

        ID3D10Device *device;
        ID3DX10Sprite *sprite;
        ID3D10ShaderResourceView *smallTitlesView;
        ID3D10ShaderResourceView *bigTitlesView;
        ID3D10BlendState *spriteBlendState;
};

#endif
