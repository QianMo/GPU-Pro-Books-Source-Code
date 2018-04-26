// Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)

#ifndef BLOOMPASS_H
#define BLOOMPASS_H

#include "RenderTarget.h"

class BloomPass {
    public:
        BloomPass(ID3D10Device *device, int width, int height, DXGI_FORMAT format, float bloomWidth, float intensity);
        ~BloomPass();

        void render(ID3D10ShaderResourceView *src, ID3D10RenderTargetView *dst);
    
    private:        
        void downsample();
        void horizontalBlur();
        void verticalBlur();
        void combine(ID3D10RenderTargetView *dst);

        ID3D10Device *device;
        float bloomWidth;
        float intensity;
        ID3D10Effect *effect;
        Quad *quad;
        RenderTarget *renderTarget[2];
};

#endif
