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

struct PageCacheItem;

class PageTableGenerator {
public:
    virtual void generate(klTexture *pageTableTexture, const unsigned int *numPagesOnLevel, PageCacheItem *pages) = 0;
    virtual ~PageTableGenerator(void) {}
};

/**
    This provides an (unoptimized) CPU based reference table genrator
*/
class SoftwarePageTableGenerator : public PageTableGenerator {
    
    const PagedTextureInfo *info;
    int numPagesInCache;
    unsigned char *pageTableData;
    unsigned char *pageTables[MAX_MIP_LEVELS];

public:

    SoftwarePageTableGenerator(const PagedTextureInfo *_info, int _numPagesInCache);
    virtual void generate(klTexture *pageTableTexture, const unsigned int *numPagesOnLevel, PageCacheItem *pages);
    virtual ~SoftwarePageTableGenerator(void);
};

/**
    This table generator generates the pageTable on the GPU by transferring the page information to the GPU (several KiB) and using
    the geometry shader to epand this data into the pageTableTexture.
*/
class HardwarePageTableGenerator : public PageTableGenerator {

    const PagedTextureInfo *info;
    int numPagesInCache;
    klGlBuffer *pageTableVertexBuffer;
    klEffect   *pageTableEffect;
    klEffect::ParamHandle pageTableEffectParam;
    klRenderTarget *pageTableRenderTarget;

public:

    HardwarePageTableGenerator(const PagedTextureInfo *_info, int _numPagesInCache);
    virtual void generate(klTexture *pageTableTexture, const unsigned int *numPagesOnLevel, PageCacheItem *pages);

    virtual ~HardwarePageTableGenerator(void);
};
