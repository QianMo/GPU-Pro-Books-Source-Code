#ifndef FADE_H
#define FADE_H

#include "RenderTarget.h"

class Fade {
    public:
        static void init(ID3D10Device *device);
        static void release();

        static void render(float t, float in, float out, float inLength, float outLength);
        static void render(float fade);

    private:
        static float fade(float t, float in, float out, float inLength, float outLength);

        static ID3D10Device *device;
        static ID3D10BlendState *spriteBlendState;
        static ID3D10Effect *effect;
        static Quad *quad;
};

#endif
