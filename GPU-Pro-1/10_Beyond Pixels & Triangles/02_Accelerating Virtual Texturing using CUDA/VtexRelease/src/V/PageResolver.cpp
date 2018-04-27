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

#include "shared.h"
#include "PageResolver.h"
#include "PageCache.h"
#include "CpuGpuShared.h"

PageId INVALID_PAGE;

/*
    CUDA based GPU resolver
*/

GpuPageResolver::GpuPageResolver(const PagedTextureInfo *info, klRenderTarget *pageIdRenderTarget) : usedPagest(info) {
    this->info = info;
    cache = NULL;
    this->pageIdRenderTarget = pageIdRenderTarget;

    frameId = 0;

    // 
    // Initialize some cuda buffers
    //
    usedPages      = new klGpuBuffer(info->totalNumPages * sizeof(int));
    usedPages->memset(0);
    packedPages    = new klGpuBuffer( (MAX_PAGE_REQUESTS_PER_FRAME+1) * sizeof(int));
    packedPages->memset(0);
    gpuInfo = new klGpuConstantWrapper<GpuPagedTextureInfo>("info");
    pixelBuffer = new klPixelBuffer(pageIdRenderTarget->getWidth(),pageIdRenderTarget->getHeight(),4);
    gpuPixelBuffer = new klGpuBuffer(pixelBuffer);
    klCudaError(cudaGLUnregisterBufferObject(pixelBuffer->handle()));
    klCheckGlErrors();

    gpuInfo->get().numLevels = info->numMipLevels;
    gpuInfo->get().numPagesOnHighestLevel = info->numPagesX[0];
    gpuInfo->update();

    // Test the mipmap functions, this also ensures the constants and CUDA, etc. are set up correctly
    {
        klGpuBuffer testResult(sizeof(int));
        testResult.memset(0);
        testMipLevels(&testResult);
        int *r = (int *)testResult.mapHost(klGpuBuffer::READ);
        assert( *r==0 );
        testResult.unmapHost();
    }

    r_skipResolver = console.getVariable("r_skipResolver", "0");
    r_cudaTest = console.getVariable("r_cudaTest", "0");
    r_asyncResolver = console.getVariable("r_asyncResolver", "1");

    klCudaError(cudaStreamCreate(&resolveStream));
};

GpuPageResolver::~GpuPageResolver() {
    delete usedPages;
    delete packedPages;
    delete gpuInfo;

    // It automatically gets registered/unregistered on create/free so re-register it here...
    klCudaError(cudaGLRegisterBufferObject(pixelBuffer->handle()));
    delete gpuPixelBuffer;

    delete pixelBuffer;
}

extern "C" void cudaDummy(int *ptr);

void GpuPageResolver::captureBuffer(void) {
    pixelBuffer->readPixels();
}

void GpuPageResolver::resolve(void) {
    frameId++;

    //
    // We switch to CUDA here... no OpenGL calls should be done...
    // Note: cudaGLRegisterBufferObject may be a gl context call?? we really need this Nexus thing :D
    //


    // For benchmarking, skips requesting new pages
    if ( r_skipResolver->getValueInt() ) {
        return;
    }


    // Map the pixel buffer object to CUDA
    klCudaError(cudaGLRegisterBufferObject(pixelBuffer->handle()));
    int *mapData = (int *)gpuPixelBuffer->mapDevice(klGpuBuffer::READ,resolveStream);

    // Check our results from last frame... they should be arrived by now
    PageId *pages = (PageId *)packedPages->mapHost(klGpuBuffer::READ);
    int numRequested = ((int *)pages)[0];
    if ( numRequested > MAX_PAGE_REQUESTS_PER_FRAME ) {
        // This means our cache is overfull, we don't handle this but should probably do something
        // like change the mipbias or whatever...
        numRequested = MAX_PAGE_REQUESTS_PER_FRAME;
    }
    for ( int i=0; i<numRequested; i++ ) {
        cache->requestPage(pages[i+1]);   
    }
    packedPages->unmapHost();

    // Mark all used pages for this frame
    markUsedPages(mapData, pageIdRenderTarget->getWidth(), pageIdRenderTarget->getHeight(), frameId, usedPages, resolveStream);
    
    // Pack the list of used pages
    packedPages->memset(0,sizeof(int));
    gatherUsedPages(usedPages,info->totalNumPages,frameId,packedPages, resolveStream);

    // Start downloading the results to system memory (via DMA)
    packedPages->startDownload(resolveStream);

    // Unregister the PBO so opengl can write to it again
    gpuPixelBuffer->unmapDevice(resolveStream);
    klCudaError(cudaGLUnregisterBufferObject(pixelBuffer->handle()));
}
