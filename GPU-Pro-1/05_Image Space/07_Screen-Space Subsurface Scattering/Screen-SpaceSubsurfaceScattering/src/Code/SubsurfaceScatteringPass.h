// Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)

#ifndef SUBSURFACESCATTERINGPASS_H
#define SUBSURFACESCATTERINGPASS_H

#include <vector>
#include "RenderTarget.h"

class Gaussian {
    public:
        // This function builds a gaussian from the variances and  weights that
        // define it.
        // Two important notes:
        // - It will substract the previous variance to current one, so there
        //   is no need to do it manually.
        // - Because of optimization reasons, the first variance is implicitely
        //   0.0. So you must supply <n> variances and <n + 1> weights. If the
        //   sum of gaussians of your profile does not include a gaussian with
        //   this variance, you can set the first weight to zero.
        //   This implicit variance is useful for sum of gaussians that have a
        //   very narrow gaussian that can be approximated with the unblurred
        //   image (that is equal to 0.0 variance). See the provided gaussian
        //   sums or the README for examples.
        static std::vector<Gaussian> gaussianSum(float variances[], 
                                                 D3DXVECTOR3 weights[],
                                                 int nVariances);

        float getWidth() const { return width; }
        D3DXVECTOR4 getWeight() const { return weight; }

        static const std::vector<Gaussian> SKIN;
        static const std::vector<Gaussian> MARBLE;

    private:
        Gaussian() {} 
        Gaussian(float variance, D3DXVECTOR3 weights[], int n);


        float width;
        D3DXVECTOR4 weight;
};


class SubsurfaceScatteringPass {
    public:
        // width, height, format and samples: they should match the backbuffer.
        //     'samples' is used for downsampling the depth-stencil buffer (see
        //     downsample below).
        // projection: projection matrix used to render the scene.
        // sssLevel: specifies the global level of subsurface scattering (see
        //     article for more info).
        // correction: specifies how subsurface scattering varies with depth
        //     gradient (see article for more info).
        // maxdd: limits the effects of the derivative (see article for more
        //        info).
        SubsurfaceScatteringPass(ID3D10Device *device, 
                                 int width, int height, DXGI_FORMAT format, int samples,
                                 const D3DXMATRIX &projection,
                                 float sssLevel,
                                 float correction,
                                 float maxdd);
        ~SubsurfaceScatteringPass();

        // Both 'depthResource' and 'stencilResource' can be output in the main
        // render pass using multiple render targets. They are used to produce
        // a downsampled depth-stencil and depth map.
        //
        // depthRenderTarget: output render target where the downsampled linear
        //     depth will be stored.
        // depthResource: input multisampled resource of the *linear* depth
        //     map.
        // stencilResource: input multisampled resource of the stencil buffer.
        // depthStencil: output downsampled depth-stencil.
        void downsample(ID3D10RenderTargetView *depthRenderTarget,
                        ID3D10ShaderResourceView *depthResource,
                        ID3D10ShaderResourceView *stencilResource,
                        ID3D10DepthStencilView *depthStencil);

        // IMPORTANT NOTICE: all render targets below must not be multisampled.
        // This implies that you have to resolve the main render target and 
        // downsample the depth-stencil buffer. For this task you can use the
        // above 'downsample' function.
        //
        // mainRenderTarget: render target of the rendered final image.
        // mainResource: shader resource of the rendered final image.
        // depthResource: shader resource of the *linear* depth map. We cannot
        //     use the original depth-stencil because we need to use it as shader
        //     resource at the same time we use it as depth-stencil
        // depthStencil: depth-stencil used to render the scene (in the
        //     conventional non-linear form).
        // gaussians: sum of gaussians of the profile we want to render.
        // stencilId: stencil value used to render the objects we must apply
        //     subsurface scattering.
        void render(ID3D10RenderTargetView *mainRenderTarget,
                    ID3D10ShaderResourceView *mainResource,
                    ID3D10ShaderResourceView *depthResource,
                    ID3D10DepthStencilView *depthStencil,
                    const std::vector<Gaussian> &gaussians,
                    int stencilId);

        void setSssLevel(float sssLevel) { this->sssLevel = sssLevel; setInputVars(); }
        float getSssLevel() const { return sssLevel; }

        void setCorrection(float correction) { this->correction = correction; setInputVars(); }
        float getCorrection() const { return correction; }

        void setMaxdd(float maxdd) { this->maxdd = maxdd; setInputVars(); }
        float getMaxdd() const { return maxdd; }

        void setProjectionMatrix(const D3DXMATRIX &projection) { this->projection = projection; setInputVars(); }
        const D3DXMATRIX &getProjectionMatrix() const { return projection; }

    private:
        void setInputVars();
        void blurPass(ID3D10ShaderResourceView *src, 
                      ID3D10RenderTargetView *dst,
                      ID3D10RenderTargetView *final,
                      ID3D10DepthStencilView *depthStencil,
                      const Gaussian &gaussian,
                      bool firstGaussian);
        float linearToDepth(float z);

        ID3D10Device *device;
        int width, height;
        D3DXMATRIX projection;
        float sssLevel, correction, maxdd;

        RenderTarget *renderTarget[2];
        ID3D10Effect *effect;
        Quad *quad;
};

#endif
