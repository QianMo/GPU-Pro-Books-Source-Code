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

#ifndef MLAA_H
#define MLAA_H

#include "RenderTarget.h"

class MLAA {
    public:
        MLAA(ID3D10Device *device, int width, int height);
        ~MLAA();

        /* *
         * MLAA processing using color-based edge detection.
         * The stencil component of 'depthStencil' is used to mask the zones to be
         * processed.
         * It is assumed to be already cleared when this function is called.
         */
        void go(ID3D10ShaderResourceView *src, ID3D10RenderTargetView *dst, ID3D10DepthStencilView *depthStencil) { go(src, dst, depthStencil, NULL); }

        /* *
         * MLAA processing using depth-based edge detection.
         * The stencil component of 'depthStencil' is used to mask the zones to be
         * processed.
         * It is assumed to be already cleared when this function is called.
         * 'depthResource' should contain the linearized depth buffer to be used
         * for edge detection.
         */
        void go(ID3D10ShaderResourceView *src, ID3D10RenderTargetView *dst, ID3D10DepthStencilView *depthStencil, ID3D10ShaderResourceView *depthResource);

        float getMaxSearchSteps() const { return maxSearchSteps; }
        void setMaxSearchSteps(float maxSearchSteps) { this->maxSearchSteps = maxSearchSteps; }
    
        float getThreshold() const { return threshold; }
        void setThreshold(float threshold) { this->threshold = threshold; }

        RenderTarget *getEdgeRenderTarget() { return edgeRenderTarget; }
        RenderTarget *getBlendRenderTarget() { return blendRenderTarget; }

    private:        
        void edgesDetectionPass(ID3D10ShaderResourceView *src, ID3D10ShaderResourceView *depth, ID3D10DepthStencilView *depthStencil);
        void blendingWeightsCalculationPass(ID3D10DepthStencilView *depthStencil);
        void neighborhoodBlendingPass(ID3D10ShaderResourceView *src, ID3D10RenderTargetView *dst, ID3D10DepthStencilView *depthStencil);

        ID3D10Device *device;
        ID3D10Effect *effect;
        Quad *quad;
        RenderTarget *edgeRenderTarget;
        RenderTarget *blendRenderTarget;
        ID3D10ShaderResourceView *areaMapView;

        float maxSearchSteps;
        float threshold;
};

#endif
