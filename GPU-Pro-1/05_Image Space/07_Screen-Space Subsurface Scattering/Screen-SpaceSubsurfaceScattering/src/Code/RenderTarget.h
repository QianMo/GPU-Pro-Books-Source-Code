// Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)

#ifndef RENDERTARGET_H
#define RENDERTARGET_H

class RenderTarget {
    public:
        static const DXGI_SAMPLE_DESC NO_MSAA;

        RenderTarget(ID3D10Device *device, int width, int height,
            DXGI_FORMAT format,
            const DXGI_SAMPLE_DESC &sampleDesc=NO_MSAA);
        ~RenderTarget();

        operator ID3D10Texture2D * () const { return texture2D; }
        operator ID3D10RenderTargetView * () const { return renderTargetView; }
        operator ID3D10RenderTargetView *const * () const { return &renderTargetView; }
        operator ID3D10ShaderResourceView * () const { return shaderResourceView; }

        int getWidth() const { return width; }
        int getHeight() const { return height; }
        
        void setViewport(float minDepth=0.0f, float maxDepth=1.0f) const;

    private:
        ID3D10Device *device;
        int width, height;
        ID3D10Texture2D *texture2D;
        ID3D10RenderTargetView *renderTargetView;
        ID3D10ShaderResourceView *shaderResourceView;
};

class DepthStencil {
    public:
        static const DXGI_SAMPLE_DESC NO_MSAA;

        DepthStencil(ID3D10Device *device, int width, int height,
            DXGI_FORMAT texture2DFormat = DXGI_FORMAT_R32_TYPELESS, 
            DXGI_FORMAT depthStencilViewFormat = DXGI_FORMAT_D32_FLOAT, 
            DXGI_FORMAT shaderResourceViewFormat = DXGI_FORMAT_R32_FLOAT,
            const DXGI_SAMPLE_DESC &sampleDesc=NO_MSAA);
        ~DepthStencil();

        operator ID3D10Texture2D * const () { return texture2D; }
        operator ID3D10DepthStencilView * const () { return depthStencilView; }
        operator ID3D10ShaderResourceView * const () { return shaderResourceView; }

        int getWidth() const { return width; }
        int getHeight() const { return height; }

        void setViewport(float minDepth=0.0f, float maxDepth=1.0f) const;

    private:
        ID3D10Device *device;
        int width, height;
        ID3D10Texture2D *texture2D;
        ID3D10DepthStencilView *depthStencilView;
        ID3D10ShaderResourceView *shaderResourceView;
};

class Quad {
    public:
        Quad(ID3D10Device *device, const D3D10_PASS_DESC &desc);
        ~Quad();
        void setInputLayout() { device->IASetInputLayout(vertexLayout); }
        void draw();

    private:
        ID3D10Device *device;
        ID3D10Buffer *buffer;
        ID3D10InputLayout *vertexLayout;
};

#endif

