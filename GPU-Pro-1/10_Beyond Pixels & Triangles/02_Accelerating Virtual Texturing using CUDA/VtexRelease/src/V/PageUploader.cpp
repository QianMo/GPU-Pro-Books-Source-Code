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
#include "PageUploader.h"

#define DXT1_BLOCK_SIZE 8

void StreamInfo::init(int uncompressedDataSize, int compressedDataSize, int pagesInStream) {
    klCudaError(cudaStreamCreate(&id));
    size_t rawSize = pagesInStream * uncompressedDataSize;
    rawSize += (rawSize/4); //For mipmaps
    rawBuffer = new klGpuBuffer(rawSize);
    int compressedSize = pagesInStream * compressedDataSize;
    compressedSize += (compressedSize/4); //For mipmaps
    compressedPixels = new klGlBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,NULL,compressedSize);
    compressedBuffer = new klGpuBuffer(compressedPixels);
}

StreamInfo::~StreamInfo(void) {
    klCudaError(cudaStreamDestroy(id));
    delete rawBuffer;
    delete compressedBuffer;
    delete compressedPixels;
}

PageUploader::PageUploader(const PagedTextureInfo *_info, klTexture *_cacheTexture, int _numStreams, int _pagesPerStream) 
: info(_info), cacheTexture(_cacheTexture), numStreams(_numStreams), pagesPerStream(_pagesPerStream)
{
    pageDataSize = info->pageSize * info->pageSize * 4;
    compressedPageDataSize = (info->pageSize/4)*(info->pageSize/4)*DXT1_BLOCK_SIZE;
    streams = new StreamInfo[numStreams];
    for ( int i=0;i<numStreams; i++ ) {
        streams[i].init(pageDataSize,compressedPageDataSize,pagesPerStream);
    }
}

PageUploader::~PageUploader(void) {
    delete []  streams;
}

void PageUploader::UploadBatch(PageUpload *pages, int numPages) {
    int packetSize = pagesPerStream * numStreams;
    int numPackets = iDivUp(numPages,packetSize);
    for ( int p=0;p<numPackets;p++ ) {
        int numPagesInPacket = min(packetSize,numPages-(p*packetSize));
        UploadPacket(pages+p*packetSize,numPagesInPacket);
    }
}

extern "C" void EncodePages(klGpuBuffer *source, void *mappedDest, int numPages, int pageSize, int stream);

void PageUploader::UploadPacket(PageUpload *pages, int numPages) {
    assert(numPages <= pagesPerStream * numStreams);
    int usedStreams = iDivUp(numPages,pagesPerStream);

    // Map the openglBuffer
    for ( int s=0; s<usedStreams; s++ ) {
        streams[s].codedMapped = streams[s].compressedBuffer->mapDevice(klGpuBuffer::WRITE_DISCARD,streams[s].id); 
    }

    // Upload the pages 
    for ( int p=0; p<numPages; p++ ) {
        int stream = p/pagesPerStream;
        int streamPage = p-(stream*pagesPerStream);
        streams[stream].rawBuffer->copyToAsync(pages[p].data,pageDataSize,streamPage*pageDataSize,streams[stream].id);   
    }

    // Encode the pages
    for ( int s=0; s<usedStreams; s++ ) {
        int numPagesInStream = min(pagesPerStream,numPages-(s*pagesPerStream));
        EncodePages(streams[s].rawBuffer,streams[s].codedMapped,
                    numPagesInStream,info->pageSize,streams[s].id);
    }

    // Unmap the openglBuffers in their streams
    for ( int s=0; s<usedStreams; s++ ) {
        streams[s].compressedBuffer->unmapDevice(streams[s].id); 
    }

    // Switch to OpenGL
    cacheTexture->bind(0);
    int prevStream = -1;

    for ( int p=0; p<numPages; p++ ) {
        int stream = p/pagesPerStream;
        int streamPage = p-(stream*pagesPerStream);
        int numPagesInStream = min(pagesPerStream,numPages-(stream*pagesPerStream));
        int mipOffset = numPagesInStream*compressedPageDataSize;
        int mipDataSize = compressedPageDataSize>>2;

        if ( stream != prevStream ) {          
            streams[stream].compressedPixels->bind();
            prevStream = stream;
        }

#ifdef NO_DXT
        // Main texture
        glTexSubImage2D(GL_TEXTURE_2D,0,
                        pages[p].tileX*info->pageSize,pages[p].tileY*info->pageSize,
                        info->pageSize,info->pageSize,
                        GL_BGRA,GL_UNSIGNED_BYTE, ((char *)NULL)+(streamPage*compressedPageDataSize));

        // First mip down
        int halfPage = info->pageSize>>1;
        glTexSubImage2D(GL_TEXTURE_2D,1,
                        pages[p].tileX*halfPage,pages[p].tileY*halfPage,
                        halfPage,halfPage,
                        GL_BGRA,GL_UNSIGNED_BYTE, ((char *)NULL)+(mipOffset+streamPage*mipDataSize));
#else
        // Main texture
        glCompressedTexSubImage2D(GL_TEXTURE_2D,0, 
                                  pages[p].tileX*info->pageSize,pages[p].tileY*info->pageSize,
                                  info->pageSize,info->pageSize,
                                  GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                  compressedPageDataSize,
                                  ((char *)NULL)+(streamPage*compressedPageDataSize));

        // First mip down
        int halfPage = info->pageSize>>1;
        glCompressedTexSubImage2D(GL_TEXTURE_2D,1, 
                                  pages[p].tileX*halfPage,pages[p].tileY*halfPage,
                                  halfPage,halfPage,
                                  GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
                                  mipDataSize,
                                  ((char *)NULL)+(mipOffset+streamPage*mipDataSize));
#endif
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}