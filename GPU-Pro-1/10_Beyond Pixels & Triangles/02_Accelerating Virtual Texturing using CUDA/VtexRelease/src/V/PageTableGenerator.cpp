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
#include "PageTableGenerator.h"
#include "PageCache.h"

/**
    This provides an (unoptimized) CPU based reference table genrator
*/

SoftwarePageTableGenerator::SoftwarePageTableGenerator(const PagedTextureInfo *_info, int _numPagesInCache) 
    : info(_info), numPagesInCache(_numPagesInCache)
{
    int pageTableSize = 0;
    for ( int i=0;i<info->numMipLevels; i++ ) {
        pageTableSize += info->numPagesX[i]*info->numPagesY[i]*4;
    }

    pageTableData = (unsigned char*)malloc(pageTableSize);
    memset(pageTableData,0,sizeof(pageTableSize));

    pageTableSize = 0;
    for ( int i=0;i<info->numMipLevels; i++ ) {
        pageTables[i] = pageTableData + pageTableSize;
        pageTableSize += info->numPagesX[i]*info->numPagesY[i]*4;
    }
}

void SoftwarePageTableGenerator::generate(klTexture *pageTableTexture, const unsigned int *numPagesOnLevel, PageCacheItem *pages) {
    for ( int level=info->numMipLevels-1; level>=0; level-- ) {
        for ( int i=0;i<numPagesInCache; i++ ) {
            PageCacheItem *page = &pages[i];

            if ( page->id == INVALID_PAGE ) continue;
            if ( page->id.getMipLevel() != level ) continue;

            // Draw yourself to this level
            unsigned char *pixels = pageTables[level];
            int x = page->id.getIdX();
            int y = page->id.getIdY();
            *(CacheId *)(pixels+(y*info->numPagesX[level]+x)*4) = CacheId(page->reference[0],
                                                                          page->reference[1],
                                                                          info->numPagesX[0]>>page->id.getMipLevel());
        }    
                
        // Upsample for next...
        if ( level != 0 ) {
            unsigned int *npixels = (unsigned int *)pageTables[level-1];
            unsigned int *pixels = (unsigned int *)pageTables[level];
            int nw = info->numPagesX[level-1];
            int w = info->numPagesX[level];

            for ( int y=0;y<info->numPagesY[level-1]; y++ ) {
                for ( int x=0;x<info->numPagesX[level-1]; x++ ) {
                    int px = x>>1;
                    int py = y>>1;
                    npixels[x+y*nw] = pixels[px+py*w];
                }
            }
        }
    }

    // Upload to gl
    pageTableTexture->bind(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    for ( int level=0; level<info->numMipLevels; level++ ) {
        glTexImage2D(GL_TEXTURE_2D,level,GL_RGBA8,info->numPagesX[level],info->numPagesY[level],0,GL_RGBA,GL_UNSIGNED_BYTE,pageTables[level]);
    }
}

SoftwarePageTableGenerator::~SoftwarePageTableGenerator(void) {
    free(pageTableData);
}


///////////////////////////////////////////////////////////////////////////////////////////


/**
    This table generator generates the pageTable on the GPU by transferring the page information to the GPU (several KiB) and using
    the geometry shader to epand this data into the pageTableTexture.
*/

struct PageTableVertex {
    float   x;
    float   y;
    PageId  id;
};

HardwarePageTableGenerator::HardwarePageTableGenerator(const PagedTextureInfo *_info, int _numPagesInCache)
    : info(_info), numPagesInCache(_numPagesInCache)
{
    pageTableVertexBuffer = new klGlBuffer(GL_ARRAY_BUFFER_ARB,NULL,numPagesInCache*sizeof(PageTableVertex));
    pageTableEffect = effectManager.getForName("virtual_pagetable");
    pageTableEffectParam = pageTableEffect->getParameter("LevelInfo");
    pageTableRenderTarget = new klRenderTarget();        
}

void HardwarePageTableGenerator::generate(klTexture *pageTableTexture, const unsigned int *numPagesOnLevel, PageCacheItem *pages) {
    // Initialize our per level list counts and offsets
    int *levelOffsets = (int *)_alloca((MAX_MIP_LEVELS+1)*sizeof(int));
    int *numPerLevel = (int *)_alloca(MAX_MIP_LEVELS*sizeof(int));
    memset(numPerLevel,0,sizeof(int)*info->numMipLevels);

    // Accumulate the offsets over the levels
    levelOffsets[0] = 0;
    for ( int i=1; i<=info->numMipLevels; i++ ) {
        levelOffsets[i] =  levelOffsets[i-1] + numPagesOnLevel[i-1];  
    }
    
    // Reverse the offsets from the end so we will render the higher mipmaps first
    for ( int i=0; i<info->numMipLevels; i++ ) {
        levelOffsets[i] =  (levelOffsets[info->numMipLevels]) - (levelOffsets[i]+numPagesOnLevel[i]);  
    }

    pageTableVertexBuffer->bind();

    // Flag to gl that we are rewriting the whole buffer
    glBufferData(GL_ARRAY_BUFFER, pageTableVertexBuffer->sizeInBytes(), 0, GL_DYNAMIC_DRAW);
    PageTableVertex *pageTableVertices = (PageTableVertex *)glMapBuffer(GL_ARRAY_BUFFER,GL_WRITE_ONLY);

    // Copy into the struct that will be read by the geometry shader
    // note that this full structure is only 12k
    for ( int i=0;i<numPagesInCache;i++) {
        PageCacheItem *page = &pages[i];
        if ( page->id == INVALID_PAGE ) continue; 

        int level = page->id.getMipLevel();          

        // Calculate index in the vertex array to write to.
        int dstIndex = levelOffsets[level]+numPerLevel[level];
        numPerLevel[level]++;

        // Write to the vertex array
        PageTableVertex *vert = pageTableVertices+dstIndex;
        vert->x = (float)page->reference[0];
        vert->y = (float)page->reference[1];
        vert->id = page->id;
    }

    glUnmapBuffer(GL_ARRAY_BUFFER);

    // Set up for rendering
    glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, true, sizeof(PageTableVertex), (void *)offsetof(PageTableVertex,id));
    glVertexPointer(2, GL_FLOAT, sizeof(PageTableVertex), (void *)offsetof(PageTableVertex,x));
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);
    glDisableVertexAttribArray(5);

    pageTableEffect->setup();
    klInstanceParameters inst;
    inst.instanceId = 1;

    for ( int level=0; level<info->numMipLevels; level++ ) {

        // Set-up rendering to the mipmap level
        pageTableRenderTarget->setColor(pageTableTexture, 0, level);
        pageTableRenderTarget->startRendering();
        glViewport(0,0,info->numPagesX[level],info->numPagesY[level]);

        glClearColor(0.0f,0.0f,0.0f,0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Set the parameters, this is done with setupInstance instead of setParameter
        // to avoid having to call pageTableEffect->setup() (binding shaders, textures, etc...) for
        // every level
        inst.userParams[0] = (float)level;
        inst.userParams[1] = 1.0f/(float)info->numPagesX[level];
        inst.instanceId++;
        pageTableEffect->setupInstance(inst);

        // Process the structs, higher mipmap levels will draw only the first few elements in the array
        glDrawArrays(GL_POINTS,0,levelOffsets[level]+numPerLevel[level]);
    }
    pageTableEffect->reset();
    systemRenderTarget.startRendering();
}

HardwarePageTableGenerator::~HardwarePageTableGenerator(void) {
    delete pageTableVertexBuffer;
    delete pageTableRenderTarget;
}

