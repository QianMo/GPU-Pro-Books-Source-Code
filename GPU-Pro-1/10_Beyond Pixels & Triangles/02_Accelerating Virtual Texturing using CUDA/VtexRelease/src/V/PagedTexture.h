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

#include "CpuGpuShared.h"
#include "DiscPageStore.h"

struct PagedTextureInfo {
    int pageSize;
    int pageContentSize;
    int numMipLevels;
    int widths[MAX_MIP_LEVELS];
    int heights[MAX_MIP_LEVELS];
    int numPagesX[MAX_MIP_LEVELS];
    int numPagesY[MAX_MIP_LEVELS];
    int totalNumPages; // Number of pages in all levels
};

struct PageId {
    unsigned char bytes[4];
    
    PageId(void) {
        // invalid tile ID
        bytes[0] = bytes[1] = bytes[2] = bytes[3] = 0xFF;
    }

    PageId(int x, int y, int level) {
        bytes[0] = x&0xFF;
        bytes[1] = y&0xFF;
        bytes[2] = ((x >> 8) << 4) | (y >> 8);
        bytes[3] = level;
    }

    int getMipLevel(void) const {
        return bytes[3];
    }

    int getIdX(void) const {
        return bytes[0] | (bytes[2] & 0xF) << 8;
    }

    int getIdY(void) const {
        return bytes[1] | (bytes[2] >> 4) << 8;
    }

    bool operator== (const PageId &other) {
        return memcmp(bytes,other.bytes,4) == 0;
    }

    bool operator!= (const PageId &other) {
        return !(*this == other);
    }
};

extern PageId INVALID_PAGE;

/*
    Allows you to store per page information
*/
template <class Type> class PerPageInfo {
    Type **levels;
    const PagedTextureInfo *info;
public:

    PerPageInfo(const PagedTextureInfo *info) {
        this->info = info;
        levels = (Type **)malloc(sizeof(Type *)*info->numMipLevels);
        for ( int i=0; i<info->numMipLevels; i++ ) {
            levels[i] = (Type *)malloc(sizeof(Type)*info->numPagesX[i]*info->numPagesY[i]);
        }
        clear();
    }

    ~PerPageInfo(void) {
        for ( int i=0; i<info->numMipLevels; i++ ) {
            free(levels[i]);
        }        
        free(levels);
    }

    Type &get(PageId id) {
        int level = id.getMipLevel();
        int idX = id.getIdX();
        int idY = id.getIdY(); 
        return get(level, idX, idY);
    }

    Type &get(int level, int idX, int idY) {
        assert(level < info->numMipLevels);
        assert(idX < info->numPagesX[level]);
        assert(idY < info->numPagesY[level]);

        Type *levelData = levels[level];
        int levelSize = info->numPagesX[level];
        return levelData[idY*levelSize+idX];
    }

    void clear(void) {
        for ( int i=0; i<info->numMipLevels; i++ ) {
            int numTiles = info->numPagesX[i]*info->numPagesY[i];
            memset(levels[i],0,sizeof(Type)*numTiles);
        }
    }
};