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
#include "SingleProducerConsumerBuffer.h"
#include "FpsCounter.h"
#include "SimpleDCTDec.h"

class AbstractPageCache;

/*
    This class generates pixel data for a requested page. 
    Does its work asynchronously and then submits to the cache
*/
class AbstractPageProvider {
public:
    // Get info about the texture provided
    virtual const PagedTextureInfo *getTextureInfo(void) = 0;

    // Set the cache to submit to
    virtual void setCache(AbstractPageCache *cache) = 0;
    
    // Update the cache for this frame
    virtual void frameSync(void) = 0;
    
    // Request a page (will be submitted to the cache when ready)
    // returns false if the request can't be handled now (re-request later)
    virtual bool requestPage(PageId page) = 0;

    // Loads a page, this immediately loads the page to the specified buffer
    // should be used in a different thread or for toos/debug only
    virtual void loadPage(unsigned char *buffer, PageId page) = 0;

    virtual ~AbstractPageProvider() {}
};

struct DiskPagePending {
    DiskPagePending(void) {
        page = PageId(0,0,0);
        memory = NULL;
    }

    PageId page;
    unsigned char *memory;
    LARGE_INTEGER startTime;
};

struct PageBuffer {
    PageBuffer *next;  
};

/*
    A page provider that loads pages stored on disc.
*/
class DiskPageProvider : public AbstractPageProvider {
    FILE              *pageFile;
    PagedTextureInfo  textureInfo;
    AbstractPageCache *cache;
    PageBuffer        *freeBufferList;
    unsigned char     *bufferMemory;
    unsigned char     *bufferMemoryLimit;
    unsigned char     *readBuffer;
    int bufferSize;
    int compressedBufferSize;
    SimpleDCTDec decoder;
    bool run;
    HANDLE            processingThread;
    klStatisticsCounter stats;

    struct PageInfo {
        int fileDataSize;
        int format;
        long long fileDataOffset;

        // Dynamic stuff
        bool dirty;
    };

    PerPageInfo<PageInfo> *pageInfo;
    klConVariable *r_slowDisk;
    klConVariable *r_showPageInfo;
    klConVariable *r_forceUploads;

    SingleProducerConsumerBuffer<DiskPagePending> requests;
    SingleProducerConsumerBuffer<DiskPagePending> responses;

    unsigned char *allocPageBuffer(void);
    void freePageBuffer(void *data);

public:
    DiskPageProvider(const char *pageFileName);
    virtual ~DiskPageProvider(void);

    virtual void loadPage(unsigned char *buffer, PageId page);
    virtual bool requestPage(PageId page);
    virtual const PagedTextureInfo *getTextureInfo(void) { return &textureInfo; }
    virtual void setCache(AbstractPageCache *cache) { this->cache = cache; }
    virtual void frameSync(void);

    void threadMain(void);
    void benchmarkUpload(void);
};