/**
 *
 *  This software module was originally developed for research purposes,
 *  by Multimedia Lab at Ghent University (Belgium).
 *  Its performance may not be optimized for specific applications.
 *
 *  Those intending to use this software module in hardware or software products
 *  are advized that its use may infringe existing patents. The developers of 
 *  this software module, their companies, Ghent Universtity, nor Multimedia Lab 
 *  have any liability for use of this software module or modifications thereof.
 *
 *  Ghent University and Multimedia Lab (Belgium) retain full right to modify and
 *  use the code for their own purpose, assign or donate the code to a third
 *  party, and to inhibit third parties from using the code for their products. 
 *
 *  This copyright notice must be included in all copies or derivative works.
 *
 *  For information on its use, applications and associated permission for use,
 *  please contact Prof. Rik Van de Walle (rik.vandewalle@ugent.be). 
 *
 *  Detailed information on the activities of
 *  Ghent University Multimedia Lab can be found at
 *  http://multimedialab.elis.ugent.be/.
 *
 *  Copyright (c) Ghent University 2004-2009.
 *
 **/

#pragma once
#include "PagedTexture.h"

class AbstractPageCache;

/**
    The resolver is the thing that determines what tiles are needed by the current frame.
*/
class AbstractPageResolver {
public:
    virtual void setCache(AbstractPageCache *cache) = 0;
    virtual void resolve(void) = 0;
    virtual void captureBuffer(void) = 0;
    virtual ~AbstractPageResolver() {}
};

/*
    CUDA based GPU resolver
*/
class GpuPageResolver : public AbstractPageResolver {
    klRenderTarget *pageIdRenderTarget;
    int *buffer;
    const PagedTextureInfo *info;
    AbstractPageCache *cache;
    PerPageInfo<int> usedPagest;

    klPixelBuffer *pixelBuffer;
    klGpuBuffer *gpuPixelBuffer;
    klGpuBuffer *usedPages;
    klGpuBuffer *packedPages;
    klGpuBuffer *numPackedPages;

    klGpuConstantWrapper<GpuPagedTextureInfo> *gpuInfo;
    int frameId;

    klConVariable *r_skipResolver;
    klConVariable *r_cudaTest;
    klConVariable *r_asyncResolver;

    cudaStream_t resolveStream;
public:
    GpuPageResolver(const PagedTextureInfo *info,klRenderTarget *pageIdRenderTarget);
    virtual ~GpuPageResolver();

    virtual void resolve(void);
    virtual void setCache(AbstractPageCache *cache) { this->cache = cache; }
    virtual void captureBuffer(void);
};
