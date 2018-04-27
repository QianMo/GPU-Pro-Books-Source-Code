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
#include "PageTableGenerator.h"
#include "PageUploader.h"

class AbstractPageProvider;

class AbstractPageCache {
public:

    // Set the page provider for this cache
    virtual void setProvider(AbstractPageProvider *provider) = 0;

    // Reload/Regenerate all the data contained in the cache
    virtual void flush(void) = 0;

    // Update the cache for this frame
    virtual void frameSync(void) = 0;

    // Register the tiles needed by this frame
    virtual void requestPage(PageId page) = 0;

    // A new tile is loaded and needs to be accomodated in the cache
    virtual klVec2i submitPage(PageId page) = 0;

    // Upload all the submitted pages at once
    virtual void uploadPages(PageUpload *pages, int numPages) = 0;

    // Write debug data 
    virtual void writeDebugData(void) {};

    virtual ~AbstractPageCache() {}
};

/**
    Linked list cache item
*/
struct PageCacheItem {
    klVec2i reference;
    PageId  id;

    PageCacheItem *prev;
    PageCacheItem *next;
};

/**
    Object identifying a single page in the cache
*/
class CacheId : public PageId {
public:
    CacheId(int x, int y, int level) : PageId(x,y,level) {}
};

/**
    The actual cache implementation.
*/
class SimplePageCache : public AbstractPageCache {
    const PagedTextureInfo *info;
    AbstractPageProvider *provider;

    // CPU side page table: translates pageID -> cache items
    PerPageInfo<PageCacheItem *>available;

    // GPU side page texture: translates texCoords -> physical cache locations
    klTexture *pageTableTexture;
    PageTableGenerator *pageTableGenerator;
    bool invalidPageTable;

    // CPU side physical cache information
    int numPages;
    PageCacheItem *pages;
    PageCacheItem *mru;
    PageCacheItem *lru;
    PageCacheItem *REQUESTED_PAGE;
    unsigned int numPagesOnLevel[MAX_MIP_LEVELS];

    // GPU phisical cache texture
    klTexture *cacheTexture;
    PageUploader *pageUploader;

    // Performance counter variables
    int frameRequests;
    int newFrameRequests;
    int hitFrameRequests;
    int reFrameRequests;
    int submittedRequests;
    int droppedRequests;

    PageCacheItem *allocateTile(void);

    klConVariable *r_forcePageTableUpdate;
public:
    enum PageTableUpdateOp {
        OP_PURGE,
        OP_SUBMIT
    };

    struct PageTableUpdate {
        PageId id;
        PageTableUpdateOp op;    
        PageCacheItem *tile;
    };

public:
    SimplePageCache(const PagedTextureInfo *info);
    virtual ~SimplePageCache();

    virtual void setProvider(AbstractPageProvider *provider) { this->provider = provider; }
    virtual void flush(void);
    virtual void frameSync(void);
    virtual void requestPage(PageId page);
    virtual klVec2i submitPage(PageId page);
    virtual void uploadPages(PageUpload *pages, int numPages);
    virtual void writeDebugData(void);

};
