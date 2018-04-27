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

#ifndef CPU_GPU_SHARED_H
#define CPU_GPU_SHARED_H

/**
    This header contains datastructures and constants shared by the cpu and gpu (i.e. cuda) compilers
*/

struct GpuPagedTextureInfo {
    int numLevels;
    int numPagesOnHighestLevel;
};

#define MAX_PAGE_REQUESTS_PER_FRAME 1024
#define MAX_PAGE_REQUESTS_PER_FRAME_MASK 0x3FF

extern "C" void testMipLevels(klGpuBuffer *result);
extern "C" void markUsedPages(/*klGpuBuffer *renderedFrame*/int *fr, int width, int height, int frameId, klGpuBuffer *result, int stream);
extern "C" void gatherUsedPages(klGpuBuffer *usedPages, int numPages, int frameId, klGpuBuffer *outList, int stream);

#if USE_MUL32
#define SMUL(a,b) ((a)*(b))
#define UMUL(a,b) ((a)*(b))
#else
#define UMUL(a,b) __umul24((a),(b))
#define SMUL(a,b) __mul24((a),(b))
#endif

#endif
