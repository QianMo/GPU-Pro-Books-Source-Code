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

/**
    A gpu buffer, encapsulates different ways of acessing gpu memory...
*/
class klGpuBuffer {
    
    enum MemorySpace {
        MS_GL,              // An opengl buffer object use GL buffer management functions to allocate/free
        MS_DEVICE_STATIC,   // Do not allocate/free memory is statically allocated by the cuda compiler
        MS_DEVICE_DYNAMIC   // cuMalloc normal dynamic cuda allocation/free
       /* MS_HOST_PINNED      Pinned host memory for direct access by device*/
    };

    klGpuPointer ptr;
    klGlBuffer *glBuff;
    size_t size;
    MemorySpace space;
    int mode;
    void *hostMap;
    void *dmaHostMap;
    bool dmaPending;
    cudaEvent_t  dmaComplete;
public:

    /*
        Create an gpu buffer from preallocated device memory, the buffer only encapsulates the memory but does not manage it
    */
    klGpuBuffer(klGpuPointer _ptr, size_t _size) : ptr(_ptr), glBuff(0), size(_size), space(MS_DEVICE_STATIC), hostMap(NULL), dmaHostMap(NULL), dmaPending(false) {}

    /*
        Create a new gpu buffer from static device memory, the buffer only encapsulates the memory but does not manage it.
    */
    klGpuBuffer(const char *globalName);

    /*
        Create an gpu buffer from an opengl buffer object, the buffer only encapsulates the memory but does not manage it
    */
    klGpuBuffer(klGlBuffer *_glBuff);

    /*
        Create a new gpu buffer of the given size.
    */
    klGpuBuffer(size_t _size);
    
    ~klGpuBuffer(void);

    // Bitmask, can be or-ed together
    static const int READ    = 1;
    static const int WRITE   = 2;
    static const int DISCARD = 4;
    static const int READ_WRITE    = READ | WRITE;
    static const int WRITE_DISCARD = WRITE | DISCARD;

    // Map/unmap the buffer to host memory,
    // NOTE: this may be a slow emulation and require large heap
    // allocations. Use for intitialization/debug only.
    void *mapHost(int accessMode);
    void unmapHost(void);
    
    // Map/unmap the buffer to device memory for use by kernels
    klGpuPointer mapDevice(int accessMode, int stream=0);
    void unmapDevice(int stream=0);

    // Start downloading the buffer for CPU access in the given stream, returns immediately
    // mapHost will block untill completion of this transfer
    void startDownload(int stream = 0);

    // Start uploading the buffer to the GPU from pagelocked CPU mem, GPU side syncrhro
    // is handled using streams, CPU synchro is user handled.
    void copyToAsync(const void *sourcePageLocked, size_t bytes, size_t offset, int stream=0);

    // These provide slightly more efficient ways to put data into the buffers than mapHost/unmapHost
    void copyTo(const void *source, size_t bytes);
    void copyFrom(void *dest, size_t bytes);
    void memset(int value, size_t length=0); // if length==0 the full buffer is cleared

    // These allow easy readback of single scalar results
    int   asInt(void);
    float asFloat(void);
};
