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

#include "Shared.h"
#include "Gpu.h"
#include <GL\glew.h>

void klCudaError(cudaError_t error) {
    if ( error != cudaSuccess ) {
        const char *msg = cudaGetErrorString (error);
        klFatalError("Cuda Error: %s",msg);
    }
}

void klGpuInit(void) {
    klLog("Initializing cuda");
    klCudaError(cudaSetDevice(0));
    //klCudaError(cudaSetDeviceFlags(cudaDeviceBlockingSync));
    //klCudaError(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    //klCudaError(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    klCudaError(cudaGLSetGLDevice(0));

    int driverVersion;
    int runtimeVersion;

    klCudaError(cudaDriverGetVersion(&driverVersion));
    klCudaError(cudaRuntimeGetVersion(&runtimeVersion));
    
    klLog("Cuda initialized driver %i, runtime %i",driverVersion,runtimeVersion);
}

klGpuBuffer::klGpuBuffer(size_t _size) : size(_size), space(MS_DEVICE_DYNAMIC), hostMap(NULL), dmaHostMap(NULL), dmaPending(false) {
    klCudaError(cudaMalloc(&ptr, size)); 
}

klGpuBuffer::klGpuBuffer(klGlBuffer *_glBuff) : ptr(NULL), glBuff(_glBuff), space(MS_GL), hostMap(NULL), dmaHostMap(NULL), dmaPending(false) {
    size = glBuff->sizeInBytes();
    klCudaError(cudaGLRegisterBufferObject(glBuff->handle()));
}

klGpuBuffer::~klGpuBuffer(void) {
    if ( dmaPending ) {
        //wait if we are still dma-ing to this buffer...
        klCudaError(cudaEventSynchronize(dmaComplete));
        dmaPending = false;
    }
    if ( dmaHostMap ) {
        klCudaError(cudaFreeHost(dmaHostMap)); 
        klCudaError(cudaEventDestroy(dmaComplete));
    }

    switch (space) {
        case MS_DEVICE_STATIC:
            assert( hostMap == NULL ); // Still mapped and tried to free!!
            break;
        case MS_DEVICE_DYNAMIC:
            assert( hostMap == NULL ); // Still mapped and tried to free!!
            klCudaError(cudaFree(ptr));
            break;
        case MS_GL:
            assert( ptr == NULL );     // Still mapped and tried to free!!
            klCudaError(cudaGLUnregisterBufferObject(glBuff->handle()));
            break;
        default:
            assert(0);//todo
    }
}

void klGpuBuffer::startDownload(int stream) {
    assert(space == MS_DEVICE_STATIC || space == MS_DEVICE_DYNAMIC);
    assert(dmaPending == false);

    if ( dmaHostMap == NULL ) {
        klCudaError(cudaHostAlloc(&dmaHostMap,  size, cudaHostAllocDefault));
        klCudaError(cudaEventCreate(&dmaComplete));
    }

    klCudaError(cudaMemcpyAsync(dmaHostMap,ptr,size,cudaMemcpyDeviceToHost,stream));
    klCudaError(cudaEventRecord(dmaComplete,stream));
    dmaPending = true;
}

void *klGpuBuffer::mapHost(int accessMode) {
    assert( hostMap == NULL );
    mode = accessMode;
    switch (space) {
        case MS_DEVICE_STATIC:
        case MS_DEVICE_DYNAMIC:
            if ( dmaPending ) {
                assert(mode == READ);
                cudaError_t err =  cudaEventQuery(dmaComplete);
                if ( err == cudaErrorNotReady ) {
                    klCudaError(cudaEventSynchronize(dmaComplete));
                } else {
                    klCudaError(err);
                }
                return dmaHostMap;
            } else {
                hostMap = malloc(size);
                if ( mode & READ ) { 
                    klCudaError(cudaMemcpy(hostMap, ptr, size, cudaMemcpyDeviceToHost)); 
                }
            }
            break;
        case MS_GL:
            glBindBuffer(GL_ARRAY_BUFFER, glBuff->handle());
            hostMap = glMapBuffer(GL_ARRAY_BUFFER,GL_READ_WRITE);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            break;
        default:
            assert(0);//todo
    }
    return hostMap;
}

void klGpuBuffer::unmapHost() {
    assert(hostMap || dmaPending);
    switch (space) {
        case MS_DEVICE_STATIC:
        case MS_DEVICE_DYNAMIC:
            if ( dmaPending ) {
                dmaPending = false;
            } else {
                if ( mode & WRITE ) { 
                    klCudaError(cudaMemcpy(ptr, hostMap, size, cudaMemcpyHostToDevice));
                }
                free(hostMap);
            }
            break;
        case MS_GL:
            glBindBuffer(GL_ARRAY_BUFFER, glBuff->handle());
            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            break;
        default:
            assert(0);//todo
    }   
    hostMap = NULL;
}

void  klGpuBuffer::copyTo(const void *source, size_t bytes) {
    assert(bytes <= size);
    assert(hostMap == NULL);

    switch (space) {
        case MS_DEVICE_STATIC:
        case MS_DEVICE_DYNAMIC:
            klCudaError(cudaMemcpy(ptr, source, bytes, cudaMemcpyHostToDevice));
            break;
        case MS_GL:
            assert(0);//todo
        default:
            assert(0);//todo
    }   
}

void  klGpuBuffer::copyToAsync(const void *sourcePageLocked, size_t bytes, size_t offset, int stream) {
    assert(bytes <= size);
    assert(hostMap == NULL);

    switch (space) {
        case MS_DEVICE_STATIC:
        case MS_DEVICE_DYNAMIC:
            klCudaError(cudaMemcpyAsync(((unsigned char *)ptr)+offset, sourcePageLocked, bytes, cudaMemcpyHostToDevice,stream));
            break;
        case MS_GL:
            assert(0);//should use PBO or something ....
        default:
            assert(0);//todo
    }   
}


void klGpuBuffer::copyFrom(void *dest, size_t bytes) {
    assert(bytes <= size);
    assert(hostMap == NULL);

    switch (space) {
        case MS_DEVICE_STATIC:
        case MS_DEVICE_DYNAMIC:
            klCudaError(cudaMemcpy(dest, ptr, bytes, cudaMemcpyDeviceToHost));
            break;
        case MS_GL:
            assert(0);//todo
        default:
            assert(0);//todo
    }   
}

int klGpuBuffer::asInt(void) {
    int result;
    copyFrom(&result,sizeof(int));
    return result;
}

float klGpuBuffer::asFloat(void) {
    float result;
    copyFrom(&result,sizeof(float));
    return result;
}

void  klGpuBuffer::memset(int value, size_t length) {
    assert(hostMap == NULL);
    if ( length == 0 ) {
        length = size;
    }

    switch (space) {
        case MS_DEVICE_STATIC:
        case MS_DEVICE_DYNAMIC:
            klCudaError(cudaMemset(ptr,value,length));
            break;
        case MS_GL:
            assert(0);//todo
        default:
            assert(0);//todo
    }   
}

klGpuPointer klGpuBuffer::mapDevice(int accessMode, int stream) {
    
    // We are still dma'ing from this avoid writing it on the GPU
    assert(dmaPending == false);

    switch (space) {
        case MS_DEVICE_STATIC:
        case MS_DEVICE_DYNAMIC:
            return ptr;
            break;
        case MS_GL: {
            assert(ptr == NULL); //Still mapped and try to map again!

            if ( accessMode == READ ) {
                klCudaError(cudaGLSetBufferObjectMapFlags(glBuff->handle(),cudaGLMapFlagsReadOnly));
            }

            if ( cudaGLMapBufferObjectAsync((void**)(&ptr), glBuff->handle(),stream) != cudaSuccess) {
                klLog("Error mapping opengl buffer\n");
            }

            break;
        }
        default:
            assert(0);//todo
    }  
    return ptr;
}

void klGpuBuffer::unmapDevice(int stream) {
    switch (space) {
        case MS_DEVICE_STATIC:
        case MS_DEVICE_DYNAMIC:
            // nothing to do
            break;
        case MS_GL:
            klCudaError(cudaGLUnmapBufferObjectAsync(glBuff->handle(),stream));
            ptr = NULL;
            break;
        default:
            assert(0);//todo
    }  
}
