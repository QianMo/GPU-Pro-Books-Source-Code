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

#include "../klLib/Maths.h"
class klGlBuffer;
#include "gpu.h"
#include "CpuGpuShared.h"

#define PIXEL_SKIP 4

__constant__ GpuPagedTextureInfo info;

/**
    Given a mipmap level gives the offset of the first byte on this level in a flat mipmap.
*/
__device__ int offsetForMipLevel(int level) {
    //Partial sum of geometric series
    //This was tested on float hardware and accurate up to 2^13
    float scale = (1.0f - pow(0.25f,(float)level)) / 0.75f;
    return scale * (info.numPagesOnHighestLevel * info.numPagesOnHighestLevel);
}

__device__ int sizeForMipLevel(int level) {
    return (info.numPagesOnHighestLevel >> level);
}

__device__ float logq(float x) {
    return log(x)/log(0.25f);
}

/**
    Given a offset into a flat mipmap returns the mipmap level of this offset.
*/
__device__ int mipLevelForOffset(int offset) {
    float scale = (float)offset / (float)(info.numPagesOnHighestLevel * info.numPagesOnHighestLevel);
    scale = scale * 0.75f;
    scale = 1.0f - scale;
    scale = logq(scale);
    return (int)floor(scale);
}

/**
    This function just tests the precision of the 
    miplevel functions on the GPU
*/
#define MAX_LEVELS 16
__global__ void testMipLevelsKernel(int *error) {
    int testOffsets[MAX_LEVELS];

    // Test the offsets function
    int texSize = info.numPagesOnHighestLevel;
    testOffsets[0] = 0;
    for ( int i=1;i<info.numLevels;i++ ) {
        int size = texSize*texSize;
        testOffsets[i] = testOffsets[i-1] + size;
        texSize = texSize >> 1;
    }

    *error = 0;

    for ( int i=0;i<info.numLevels;i++ ) {
        int offset = offsetForMipLevel(i);
        int diff = testOffsets[i]-offset;
        if ( diff != 0 ) {
            *error = 1;
        }
    }

    // Test the level function
    for ( int i=0;i<info.numLevels; i++ ) {
        int ofs = offsetForMipLevel(i);
        int size = sizeForMipLevel(i);

        for ( int o=0;o<size*size; o++ ) {
            int level = mipLevelForOffset(ofs+o);
            if (level != i) {
                *error = 2;
            }
        }
    }
}

extern "C" void testMipLevels(klGpuBuffer *result) {
    int *d = (int *)result->mapDevice(klGpuBuffer::READ_WRITE);
    testMipLevelsKernel<<<1, 1>>>(d);
    result->unmapDevice();
}

texture<int4, 2, cudaReadModeElementType> renderTexture;

__global__ void markUsedPagesKernel(int *pixelBuffer, int width, int height, int frameId, int *outputBuffer) {
    int2 pixelCoord;
    pixelCoord.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelCoord.y = blockIdx.y * blockDim.y + threadIdx.y;

    // If we don't want to analyze the full resolution buffer we can only check every n-th pixel
    // doesn't seem to make much difference in practice
    pixelCoord.x *= PIXEL_SKIP;
    pixelCoord.y *= PIXEL_SKIP;

    if ( pixelCoord.x >= width || pixelCoord.y >= height ) return;

    int4 pixel = tex2D(renderTexture,pixelCoord.x, pixelCoord.y);
    //Swrizzle around...  Effin BGRA rendering graphics cards
    int tileX = pixel.z | (pixel.x>>4);
    int tileY = pixel.y | (pixel.x&0x0F);
    int level = pixel.w;

    // Do some sanity checks on the shader output...
    if ( level > info.numLevels ) {
        return;
    }

    int levelWidth = sizeForMipLevel(level);

    if ( tileX >= levelWidth || tileY >= levelWidth ) {
        return;
    }

    // Request this page and pages up the quad tree
    int hiLevel = min(info.numLevels,level+200);
    for (;level<hiLevel;level++, tileY>>=1, tileX>>=1, levelWidth>>=1) {
        // Calculate the level buffer
        int *levelData = outputBuffer + offsetForMipLevel(level);

        // Mark this page as touched
        levelData[tileY * levelWidth + tileX] = frameId;
    }
}

extern "C" void markUsedPages(int *fr, int width, int height, int frameId, klGpuBuffer *result, int stream) {
    int *res = (int *)result->mapDevice(klGpuBuffer::READ_WRITE);

    // Bind our mapped pbo to a 2D texture. (Trivia: Strangely enough this is not possible in OpenGL...)
    size_t offset;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    renderTexture.addressMode[0] = cudaAddressModeWrap;
    renderTexture.addressMode[1] = cudaAddressModeWrap;
    renderTexture.filterMode = cudaFilterModePoint;
    renderTexture.normalized = false;
    klCudaError(cudaBindTexture2D(&offset,renderTexture,fr, channelDesc, width, height, width*4));

    int blockSize = 16;
    dim3 block(blockSize, blockSize, 1);
    dim3 grid(iDivUp(width/PIXEL_SKIP, block.x), iDivUp(height/PIXEL_SKIP, block.y), 1);
    markUsedPagesKernel<<<grid, block, 0, stream>>>(fr, width, height, frameId, res); 

    result->unmapDevice();
}

__device__ int make_pageID(int x, int y, int level) {
    //Fixme: Pack highest 4 x and y bits in remaining byte
    return (level << 24) | (y<<8) | x;
}

/// Simple packer based on atomics
__global__ void gatherUsedPagesKernel(int *usedPages, int numPages, int frameId, unsigned int *outList) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;

    // Check in range
    if (offset > numPages) {
        return;
    }

    // A large portion of threads will return here, so the atomic isn't hit too much...
    if ( usedPages[offset] != frameId ) {
        return;
    }

    int level = mipLevelForOffset(offset);
    int levelOfs = offset - offsetForMipLevel(level);
    int size = sizeForMipLevel(level);

    int x = levelOfs & (size-1);
    int y = levelOfs / size;

    // This will wrap around if more than MAX_PAGE_REQUESTS_PER_FRAME are requested
    int outIndex = atomicInc(outList,0xFFFFFF);
    outList[(outIndex&MAX_PAGE_REQUESTS_PER_FRAME_MASK)+1] = make_pageID(x,y,level);
}

extern "C" void gatherUsedPages(klGpuBuffer *usedPages, int numPages, int frameId, klGpuBuffer *outList, int stream) {
    int *up = (int *)usedPages->mapDevice(klGpuBuffer::READ);
    unsigned int *ol = (unsigned int *)outList->mapDevice(klGpuBuffer::WRITE);

    int blockSize = 256;
    dim3 block(blockSize, 1, 1);
    dim3 grid(iDivUp(numPages, block.x), 1, 1);
    gatherUsedPagesKernel<<<grid, block, 0, stream>>>(up, numPages, frameId, ol); 

    outList->unmapDevice();
    usedPages->unmapDevice();
}

__global__ void cudaDummyKernel(int *ptr) {
    if ( ptr != NULL ) {
        *ptr = 1;
    }
}

extern "C" void cudaDummy(int *ptr) {
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    cudaDummyKernel<<<grid, block>>>(ptr);   
}
