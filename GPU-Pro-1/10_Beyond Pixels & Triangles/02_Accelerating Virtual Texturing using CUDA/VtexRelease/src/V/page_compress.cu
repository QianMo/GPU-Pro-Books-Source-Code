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

texture<uchar4, 2, cudaReadModeNormalizedFloat> mipTexture;
texture<int4, 2, cudaReadModeElementType> rawTexture;

#define DXT1_BLOCK_SIZE 8

#define INSET_SHIFT 4    // inset the bounding box with ( range >> shift )
#define C565_5_MASK 0xF8 // 0xFF minus last three bits
#define C565_6_MASK 0xFC // 0xFF minus last two bits

#define BIT(a,b) ( (a>>b)&1 )

/**
    This uses the algorithm and pseudocode described in:
        "Real Time DXT Compression"
        http://cache-www.intel.com/cd/00/00/32/43/324337_324337.pdf
*/
__global__ void dxtBlockEncodeKernel(unsigned int *dest, int width/*in blocks*/) {
    int2 cornerCoord;
    cornerCoord.x = (UMUL(blockIdx.x, blockDim.x) + threadIdx.x)<<2;
    cornerCoord.y = (UMUL(blockIdx.y, blockDim.y) + threadIdx.y)<<2;

    //
    // Fist we determine the min/max colors for this block
    //
    int3 minColor, maxColor, inset;
    minColor.x = minColor.y = minColor.z = 255;
    maxColor.x = maxColor.y = maxColor.z = 0;
    
    for (int i=0; i<4; i++ ) {
        for (int j=0; j<4; j++ ) {
            int4 pixel = tex2D(rawTexture,cornerCoord.x+i,cornerCoord.y+j); 
            if ( pixel.x < minColor.x ) { minColor.x = pixel.x; }
            if ( pixel.y < minColor.y ) { minColor.y = pixel.y; }
            if ( pixel.z < minColor.z ) { minColor.z = pixel.z; }
            if ( pixel.x > maxColor.x ) { maxColor.x = pixel.x; }
            if ( pixel.y > maxColor.y ) { maxColor.y = pixel.y; }
            if ( pixel.z > maxColor.z ) { maxColor.z = pixel.z; }
        }
    }
    
    inset.x = ( maxColor.x - minColor.x ) >> INSET_SHIFT;
    inset.y = ( maxColor.y - minColor.y ) >> INSET_SHIFT;
    inset.z = ( maxColor.z - minColor.z ) >> INSET_SHIFT;

    minColor.x = min(minColor.x + inset.x, 255);
    minColor.y = min(minColor.y + inset.y, 255);
    minColor.z = min(minColor.z + inset.z, 255);
    maxColor.x = max(maxColor.x - inset.x, 0);
    maxColor.y = max(maxColor.y - inset.y, 0);
    maxColor.z = max(maxColor.z - inset.z, 0);

    //
    // Pack the min/max colors
    //
    int packedColors = 0;
    packedColors = ((minColor.x>>3) << 11) | ((minColor.y>>2) << 5) | (minColor.z>>3);
    packedColors = packedColors << 16;
    packedColors |= ((maxColor.x>>3) << 11) | ((maxColor.y>>2) << 5) | (maxColor.z>>3);

    //
    // Generate the 4 entry color table
    // This really is the main register bottleneck here
    // tried shared memory but it didn't help much, there is just to much stuff happening per thread.
    //
    int3 colors0, colors1, colors2, colors3;

    colors0.x = ( maxColor.x & C565_5_MASK ) | ( maxColor.x >> 5 );
    colors0.y = ( maxColor.y & C565_6_MASK ) | ( maxColor.y >> 6 );
    colors0.z = ( maxColor.z & C565_5_MASK ) | ( maxColor.z >> 5 );
    colors1.x = ( minColor.x & C565_5_MASK ) | ( minColor.x >> 5 );
    colors1.y = ( minColor.y & C565_6_MASK ) | ( minColor.y >> 6 );
    colors1.z = ( minColor.z & C565_5_MASK ) | ( minColor.z >> 5 );
    colors2.x = ( (colors0.x << 1) + colors1.x ) / 3;
    colors2.y = ( (colors0.y << 1) + colors1.y ) / 3;
    colors2.z = ( (colors0.z << 1) + colors1.z ) / 3;
    colors3.x = ( colors0.x + (colors1.x << 1) ) / 3;
    colors3.y = ( colors0.y + (colors1.y << 1) ) / 3;
    colors3.z = ( colors0.z + (colors1.z << 1) ) / 3;
    
    unsigned int packedIndices = 0;
    for (int j=0; j<4; j++ ) {
        for (int i=0; i<4; i++ ) {
            int4 c = tex2D(rawTexture,cornerCoord.x+i,cornerCoord.y+j); 
            
            int d0 = abs( colors0.x - c.x ) + abs( colors0.y - c.y ) + abs( colors0.z - c.z );
            int d1 = abs( colors1.x - c.x ) + abs( colors1.y - c.y ) + abs( colors1.z - c.z );
            int d2 = abs( colors2.x - c.x ) + abs( colors2.y - c.y ) + abs( colors2.z - c.z );
            int d3 = abs( colors3.x - c.x ) + abs( colors3.y - c.y ) + abs( colors3.z - c.z );

            int b0 = d0 > d3;
            int b1 = d1 > d2;
            int b2 = d0 > d2;
            int b3 = d1 > d3;
            int b4 = d2 > d3;
            
            int x0 = b1 & b2;
            int x1 = b0 & b3;
            int x2 = b0 & b4;
            
            int ind = (j<<2)+i;
            packedIndices |= ( x2 | ( ( x0 | x1 ) << 1 ) ) << ( ind << 1 );
        }
    }
    
    // This is actually a single 64 bit store
    int blockIndex = UMUL(UMUL(blockIdx.y, blockDim.y) + threadIdx.y,width) + UMUL(blockIdx.x, blockDim.x) + threadIdx.x;
    dest[(blockIndex<<1)+0] = packedColors;
    dest[(blockIndex<<1)+1] = packedIndices;
}


/**
    Convert to the compressed color space, note that we write to the same memory as we sample from through the texture
    this is valid (nvidia spec) as long as we don't require the cache to see our writes.
*/
__global__ void colorSpaceConversionKernel(unsigned int *dest, int pageSize) {
    int2 pixelCoord;
    pixelCoord.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelCoord.y = blockIdx.y * blockDim.y + threadIdx.y;
    int destIndex = pixelCoord.x + pixelCoord.y*pageSize;

    // Just simple RGBA to hardware native BGRA texture format
    int4 inp = tex2D(rawTexture,pixelCoord.x, pixelCoord.y);
    unsigned int out = inp.z | (inp.y<<8) | (inp.x<<16) | (inp.w<<24);

    dest[destIndex] = out;
}

/**
    Generate mipmaps for our texture, this is just a 4-way average using the hardware's bilinear interpolator
*/
__global__ void generateMipmapsKernel(unsigned int *dest, int pageSize) {
    int2 pixelCoord;
    pixelCoord.x = blockIdx.x * blockDim.x + threadIdx.x;
    pixelCoord.y = blockIdx.y * blockDim.y + threadIdx.y;
    int destIndex = pixelCoord.x + pixelCoord.y*pageSize;

    // See appendix E.2 of the programming guide to see why this will return the correct averaged values...
    float4 inp = tex2D(mipTexture,(pixelCoord.x<<1)+1,(pixelCoord.y<<1)+1);
    int4 scaled = make_int4(
        (int)(inp.x*255.0f),
        (int)(inp.y*255.0f),
        (int)(inp.z*255.0f),
        (int)(inp.w*255.0f));
    unsigned int out = scaled.x | (scaled.y<<8) | (scaled.z<<16) | (scaled.w<<24);  
#if 0
    out = 0xFFFFFFFF;
#endif
    dest[destIndex] = out;
}

/**
    This encodes numPages pages to the cache texture format, the pages are continuous in memory so we can
    use a single kernel call to encode numPages.
        source     : Input buffer with extra allocated space to store mipmaps
        mappedDest : Pointer to GPU memory where the data needs to end up
        numPages   : Number of pages to encode
        pageSize   : Size in pixels of a single page (assumes square pages)
        stream     : Cuda stream to execute all operations in
*/
extern "C" void EncodePages(klGpuBuffer *source, void *mappedDest, int numPages, int pageSize, int stream) {
    int workWidth = pageSize;
    int workHeight = pageSize*numPages;
    unsigned char *sourceMapped = (unsigned char *)source->mapDevice(klGpuBuffer::READ);
    int mipOffset = numPages*(pageSize*pageSize*4);
    int blockSize = 16;
    size_t offset;
    int compressedMipOffset = numPages*((pageSize/4)*(pageSize/4)*DXT1_BLOCK_SIZE);

    // Set up our filtered sample texture for the mipmap generation
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    mipTexture.addressMode[0] = cudaAddressModeClamp;
    mipTexture.addressMode[1] = cudaAddressModeClamp;
    mipTexture.filterMode = cudaFilterModeLinear;
    mipTexture.normalized = false;
    klCudaError(cudaBindTexture2D(&offset,mipTexture,sourceMapped, channelDesc, workWidth, workHeight, workWidth*4));

    // Generate mipmaps (uses the extra space in the input buffer)
    dim3 mipBlock(blockSize, blockSize, 1);
    dim3 mipGrid(iDivUp(workWidth/2, mipBlock.x), iDivUp(workHeight/2, mipBlock.y), 1);
    generateMipmapsKernel<<<mipGrid, mipBlock, 0, stream>>>( (unsigned int *)((unsigned char *)sourceMapped+mipOffset), pageSize/2);

    // Set up our raw byte sampling texture for the DXT encoding
    rawTexture.addressMode[0] = cudaAddressModeClamp;
    rawTexture.addressMode[1] = cudaAddressModeClamp;
    rawTexture.filterMode = cudaFilterModePoint;
    rawTexture.normalized = false;
    klCudaError(cudaBindTexture2D(&offset,rawTexture,sourceMapped, channelDesc, workWidth, workHeight, workWidth*4));

    // Encode using dxt
    dim3 dxtBlock(8, 16, 1);
    dim3 dxtGrid(iDivUp(workWidth/4, dxtBlock.x), iDivUp(workHeight/4, dxtBlock.y), 1);
    dxtBlockEncodeKernel<<<dxtGrid, dxtBlock, 0, stream>>>((unsigned int*)mappedDest, workWidth/4);

    // Set up our raw byte sampling texture for the DXT encoding of the mipmaps
    rawTexture.addressMode[0] = cudaAddressModeClamp;
    rawTexture.addressMode[1] = cudaAddressModeClamp;
    rawTexture.filterMode = cudaFilterModePoint;
    rawTexture.normalized = false;
    klCudaError(cudaBindTexture2D(&offset,rawTexture,sourceMapped+mipOffset, channelDesc, workWidth/2, workHeight/2, workWidth*2));

    // Encode mipmaps using dxt
    dim3 dxtBlockM(8, 16, 1);
    dim3 dxtGridM(iDivUp(workWidth/8, dxtBlock.x), iDivUp(workHeight/8, dxtBlock.y), 1);
    dxtBlockEncodeKernel<<<dxtGridM, dxtBlockM, 0, stream>>>((unsigned int*)((unsigned char *)mappedDest+compressedMipOffset), workWidth/8);

    source->unmapDevice();
}