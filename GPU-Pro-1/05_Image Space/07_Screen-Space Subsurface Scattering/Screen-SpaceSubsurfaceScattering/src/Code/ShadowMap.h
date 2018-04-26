// Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)

#ifndef SHADOWMAP_H
#define SHADOWMAP_H

#include "RenderTarget.h"

class ShadowMap {
    public:
        static void init(ID3D10Device *device);
        static void release();

        ShadowMap(ID3D10Device *device, int width, int height);
        ~ShadowMap();

        void begin(const D3DXMATRIX &view, const D3DXMATRIX &projection);
        void setWorldMatrix(const D3DXMATRIX &world);
        void end();

        ID3D10EffectTechnique *getTechnique() const { return effect->GetTechniqueByName("ShadowMap"); }
        operator ID3D10ShaderResourceView * const () { return *depthStencil; }

        static D3DXMATRIX getViewProjectionTextureMatrix(const D3DXMATRIX &view, const D3DXMATRIX &projection, float shadowBias);

    private:
        ID3D10Device *device;
        DepthStencil *depthStencil;
        D3D10_VIEWPORT viewport;

        static ID3D10Effect *effect;
        static ID3D10InputLayout *vertexLayout;
};

#endif
