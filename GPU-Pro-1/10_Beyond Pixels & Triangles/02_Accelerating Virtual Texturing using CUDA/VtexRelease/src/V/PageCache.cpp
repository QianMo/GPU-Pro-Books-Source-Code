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
#include "PageCache.h"
#include "PageProvider.h"

// The physical page cache will contain CACHE_SIZE*CACHE_SIZE tiles
#define  CACHE_SIZE 32

SimplePageCache::SimplePageCache(const PagedTextureInfo *info) : available(info), provider(NULL) {
    this->info = info;
    numPages = CACHE_SIZE*CACHE_SIZE;
    int cacheTextureSize = CACHE_SIZE * info->pageSize;

    pages = (PageCacheItem *)malloc(sizeof(PageCacheItem)*numPages);

    for ( int i=0;i<CACHE_SIZE; i++ ) {
        for ( int j=0;j<CACHE_SIZE; j++ ) {
            pages[j*CACHE_SIZE+i].reference.set(i,j);
            pages[j*CACHE_SIZE+i].id = INVALID_PAGE;
            pages[j*CACHE_SIZE+i].next = NULL;
            pages[j*CACHE_SIZE+i].prev = NULL;
        }
    }

    REQUESTED_PAGE = (PageCacheItem *)1;

    flush();    

    // Allocate the physical cache texture
    cacheTexture = textureManager.getForName("_cache");
    cacheTexture->setData(KLTX_RGB8,KLTX_2D,cacheTextureSize,cacheTextureSize);
         
    cacheTexture->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 1);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, 0.0f);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 4);
#ifdef NO_DXT
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,cacheTextureSize,cacheTextureSize,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
    glTexImage2D(GL_TEXTURE_2D,1,GL_RGBA8,cacheTextureSize/2,cacheTextureSize/2,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
#else
    int dataSize = (cacheTextureSize/4)*(cacheTextureSize/4)*8;
    void *tempBuffer = malloc(dataSize); //glCompressedTexImage2D doesn't like NULL pointers...
    for ( int i=0;i<13;i++) {
        int mipSize=cacheTextureSize>>i;
        int mipDataSize=max(dataSize>>(2*i),8); 
        glCompressedTexImage2D(GL_TEXTURE_2D,i,GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,mipSize,mipSize,0,
            mipDataSize,tempBuffer);
    }
    free(tempBuffer);
#endif

    pageUploader = new PageUploader(info,cacheTexture,2,4);

    // Allocate the page table texture
    pageTableTexture = textureManager.getForName("_page_table");
    pageTableTexture->setData(KLTX_RGB8,KLTX_2D,info->numPagesX[0],info->numPagesY[0]);
    pageTableTexture->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.5f);

    tempBuffer = malloc(info->numPagesX[0]*info->numPagesY[0]*4);
    for ( int level=0; level<info->numMipLevels; level++ ) {
        for ( int i=0;i<info->numPagesX[level]*info->numPagesY[level];i++) {
            unsigned char *pix = ((unsigned char*)tempBuffer)+i*4;
            pix[0] = 0;
            pix[1] = 0;
            pix[2] = level;
            pix[3] = info->numMipLevels;
        }
        glTexImage2D(GL_TEXTURE_2D,level,GL_RGBA8,info->numPagesX[level],info->numPagesY[level],0,GL_RGBA,GL_UNSIGNED_BYTE,tempBuffer);
    }
    free(tempBuffer);

    // Allocate resources for the page translation table generation
#ifdef SOFTWARE_PAGE_TABLE
    pageTableGenerator = new SoftwarePageTableGenerator(info,numPages);
#else
    pageTableGenerator = new HardwarePageTableGenerator(info,numPages);    
#endif

    r_forcePageTableUpdate = console.getVariable("r_forcePageTableUpdate", "0");
}

SimplePageCache::~SimplePageCache() {
    delete pageTableGenerator;
    delete pageUploader;
    free(pages);
}


PageCacheItem *SimplePageCache::allocateTile(void) {
    // Mark the tile as unavailable
    // and remove it from the level count
    if ( lru->id != INVALID_PAGE ) {
        available.get(lru->id) = NULL;
        numPagesOnLevel[lru->id.getMipLevel()]--;
    }

    // Get the lru and reuse it
    PageCacheItem *tile = lru;
    tile->prev->next = NULL;
    lru = tile->prev;

    // Move it to the head of the list
    tile->prev = NULL;
    tile->next = mru;
    mru->prev = tile;
    mru = tile;

    invalidPageTable = true;

    return tile;
}

void SimplePageCache::flush(void) {
    
    for (int i=0;i<MAX_MIP_LEVELS; i++) {
        numPagesOnLevel[i] = 0;
    }

    // Mark all tiles as unavailable
    available.clear();

    // Put all the tiles in the lru marked as invalid
    for ( int i=0;i<numPages; i++ ) {
        PageCacheItem *page = &pages[i];
        page->prev = (i>0) ? &pages[i-1] : NULL;
        page->next = (i<(numPages-1)) ? &pages[i+1] : NULL;
        page->id = INVALID_PAGE;
    }    

    mru = &pages[0];
    lru = &pages[numPages-1];

    // Request the lowest miplevel page so we always have something to display
    if (provider) {
        requestPage(PageId(0,0,info->numMipLevels-1));
    }
    invalidPageTable = true;
}

void SimplePageCache::frameSync(void) {

    // Hack: request the lowest miplevel tile every frame so it never gets swapped out
    // FIX: Should really get a proper system to lock certain pages in the cache...
    requestPage(PageId(0,0,info->numMipLevels-1));

    if ( invalidPageTable || r_forcePageTableUpdate->getValueInt() ) {
        pageTableGenerator->generate(pageTableTexture,numPagesOnLevel,pages);
    }

    invalidPageTable = false;

    // Draw debug information
    char buffer[1024];
    sprintf(buffer,"Pages: %i Hit: %i Requested: %i Waiting: %i Subtmitted: %i Dropped: %i",
            frameRequests,hitFrameRequests,newFrameRequests,reFrameRequests,submittedRequests,droppedRequests);
    renderBackend.drawString(2,20,buffer,0xFFFFFFFF);

    frameRequests = 0;
    newFrameRequests = 0;
    hitFrameRequests = 0;
    reFrameRequests = 0;
    submittedRequests = 0;
    droppedRequests = 0;
}

void SimplePageCache::requestPage(PageId page) {
    if ( page.getMipLevel() > info->numMipLevels ) return;
    if ( page.getIdX() > info->numPagesX[page.getMipLevel()] ) return;
    if ( page.getIdY() > info->numPagesY[page.getMipLevel()] ) return;

    frameRequests++;
    PageCacheItem *tile = available.get(page);
    if ( tile == NULL ) {
        // Not in cache -> request it
        if( provider->requestPage(page) ) {
            available.get(page) = REQUESTED_PAGE;
            newFrameRequests++;
        } else {
            droppedRequests++;
        }
    } else if ( tile == REQUESTED_PAGE ) {
        //Still waiting for it to be loaded ... 
        //will just use lower mipmap for now
        reFrameRequests++;
    } else {
        // The requested frame is in the cache...
        hitFrameRequests++;

        //If it is already loaded put it at the front of the mru
        if ( tile == mru ) return;
        
        if ( tile->next ) {
            // unlink from old pos
            tile->prev->next = tile->next;
            tile->next->prev = tile->prev;
        } else {
            // this was the lru
            assert( tile == lru );
            tile->prev->next = NULL;     
            lru = tile->prev;
        }

        // link at front
        tile->prev = NULL;
        tile->next = mru;
        mru->prev = tile;

        // save for later
        mru = tile;
    }
}

klVec2i SimplePageCache::submitPage(PageId page) {
    PageCacheItem *tile = available.get(page);

    if ( tile != REQUESTED_PAGE ) {
        //This happens when we flush the cache and write black to all the tiles
        //If we ever have an editor this could also happen because the page changed in the editor...
    } else {
        tile = allocateTile();
        assert(tile);
        available.get(page) = tile;
        tile->id = page;

        // Add to the count for this level
        numPagesOnLevel[tile->id.getMipLevel()]++;
    }

    submittedRequests++;
    return tile->reference;
}

void SimplePageCache::uploadPages(PageUpload *pages, int numPages) {
    pageUploader->UploadBatch(pages,numPages);
}

int ReverseBits(int a) {
    int r = 0;
    for ( int i=0;i<8;i++) {
        int mask = 1<<i;
        bool bit = (a&mask) != 0;
        if (bit) {
            mask = 1<<(7-i);
            r |= mask;
        }
    }
    return r;
}

void SimplePageCache::writeDebugData(void) {

    int textureSize = cacheTexture->getWidth()*cacheTexture->getHeight()*4;
    unsigned char *buffer = (unsigned char *)malloc(textureSize);
    cacheTexture->bind(0);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,buffer);
    FILE *out = fopen("cache.raw","wb");
    fwrite(buffer,1,textureSize,out);
    fclose(out);
    free(buffer);

    textureSize = pageTableTexture->getWidth()*pageTableTexture->getHeight()*4;
    buffer = (unsigned char *)malloc(textureSize);
    pageTableTexture->bind(0);
    glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,buffer);
    
    for (int i=0;i<textureSize; i++) {
        buffer[i] = ReverseBits(buffer[i]);
    }
    
    out = fopen("pages.raw","wb");
    fwrite(buffer,1,textureSize,out);
    fclose(out);
}