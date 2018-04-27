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
#include "PageProvider.h"
#include "PageCache.h"
#include "..\klLib\PixieFont.h"

// Max number of "in flight" page requests
#define MAX_OUTSTANDING_REQUESTS 128
#define MAX_OUTSTANDING_RESPONSES 128

//Colors for debugging, Needs to have MAX_MIP_LEVELS entries...
static unsigned char MipLevelColors[] = {
    255,255,255,255,
    255,255,128,255,
    128,255,255,255,
    128,255,128,255,
    255,128,255,255,
    255,128,128,255,
    128,128,255,255,
    255,255,000,255,
    000,255,255,255,
    000,255,000,255,
    255,000,255,255,
    255,000,000,255,
    000,000,255,255,
    128,064,000,255,
};

static DWORD WINAPI RealThreadMain( LPVOID lpParam );

DiskPageProvider::DiskPageProvider(const char *pageFileName)
    : requests(MAX_OUTSTANDING_REQUESTS), responses(MAX_OUTSTANDING_RESPONSES), stats(100)
{
    cache = NULL;

    // Open the pagefile
    pageFile = fopen(pageFileName,"rb");
    if (pageFile == NULL) {
        klFatalError("Page file not found: '%s'",pageFileName);
    }

    DiscPageStoreHeader header;
    fread(&header,sizeof(header),1,pageFile);
    if (header.magic != DISC_TEXTURE_MAGIC ||
        header.version != DISC_TEXTURE_VERSION ||
        header.numMipLevels >= MAX_MIP_LEVELS)
    {
        klFatalError("Invalid page file or version: '%s'",pageFileName);
    }

    textureInfo.pageSize        = header.pageSize;
    textureInfo.pageContentSize = header.pageContentSize;
    textureInfo.numMipLevels    = header.numMipLevels;

    // First parse the file to fill in the textureInfo struct
    long levelStartPos = ftell(pageFile);
    for ( int i=0;i<textureInfo.numMipLevels; i++ ) {
        DiscPageStoreLevel level;
        fread(&level,sizeof(level),1,pageFile);
        fseek(pageFile,sizeof(DiscPageStorePage) * level.numPagesX*level.numPagesY,SEEK_CUR);
    
        textureInfo.widths[i]  = level.width;
        textureInfo.heights[i] = level.height;
        textureInfo.numPagesX[i] = level.numPagesX;
        textureInfo.numPagesY[i] = level.numPagesY;

        if ( (textureInfo.widths[i] % textureInfo.pageContentSize) != 0 ||
             (textureInfo.heights[i] % textureInfo.pageContentSize) != 0 ||
             !isPow2(textureInfo.numPagesX[i]) ||
             !isPow2(textureInfo.numPagesY[i]) )
        {
            klFatalError("Invalid page layout: '%s'",pageFileName);         
        }
    }

    // Calculate total number of pages
    textureInfo.totalNumPages = 0;
    for ( int i=0;i<textureInfo.numMipLevels; i++ ) {
        textureInfo.totalNumPages += (textureInfo.numPagesX[i] * textureInfo.numPagesY[i]);
    }

    pageInfo = new PerPageInfo<PageInfo>(&textureInfo);
    compressedBufferSize = 0;
    bufferSize = textureInfo.pageSize*textureInfo.pageSize*4;

    // Now read it again to set up page info
    fseek(pageFile,levelStartPos,SEEK_SET);
    for ( int i=0;i<textureInfo.numMipLevels; i++ ) {
        DiscPageStoreLevel level;
        fread(&level,sizeof(level),1,pageFile);

        for ( int y=0; y<level.numPagesY; y++ ) {
            for ( int x=0; x<level.numPagesX; x++ ) {
                DiscPageStorePage page;
                fread(&page,sizeof(page),1,pageFile);
                PageInfo &info = pageInfo->get(i,x,y);
                
                info.fileDataOffset = page.dataOffset;
                info.fileDataSize   = page.size;
                info.format         = page.format;
                info.dirty          = false;
                compressedBufferSize = max(compressedBufferSize,page.size); 
            }
        }
    }

    r_slowDisk = console.getVariable("r_slowDisk", "0");
    r_showPageInfo = console.getVariable("r_showPageInfo", "0");
    r_forceUploads = console.getVariable("r_forceUploads", "0");

    // Allocate memory for the read buffer
    readBuffer = (unsigned char *)malloc(compressedBufferSize);

    // Allocate the page buffer list
    int numPageBuffers = MAX_OUTSTANDING_REQUESTS + MAX_OUTSTANDING_RESPONSES;
    klCudaError(cudaMallocHost((void **)&bufferMemory,bufferSize*numPageBuffers));
    bufferMemoryLimit = bufferMemory + bufferSize*numPageBuffers;
    freeBufferList = (PageBuffer *)bufferMemory;
    for ( int i=0; i<numPageBuffers; i++ ){
        PageBuffer *buff = (PageBuffer *)(bufferMemory+(i*bufferSize));
        if ( i != (numPageBuffers-1) ) {
            buff->next = (PageBuffer *)(bufferMemory+((i+1)*bufferSize));
        } else {
            buff->next = NULL;
        }
    }

    // Start the processing thread
    run = true;
#ifndef NO_THREAD
    processingThread = CreateThread(NULL,0,RealThreadMain,this,0,NULL);
#endif
}

DiskPageProvider::~DiskPageProvider(void) {
    // Shut down the thread (sends a dummy request so it wakes up when it was sleeping)
    run = false;
    DiskPagePending dummy;
    requests.push(dummy);

    // Now wait for it to clean itself up...
    WaitForSingleObject(processingThread,INFINITE);

    // Free the rest
    fclose(pageFile);
    delete pageInfo;
    klCudaError(cudaFreeHost(bufferMemory));
    free(readBuffer);
}

int ReverseBits(int a);

#define CHECK_PTR(a) assert((unsigned char *)(a) < bufferMemoryLimit && (unsigned char *)(a) >= bufferMemory);

unsigned char *DiskPageProvider::allocPageBuffer(void) {

    if (freeBufferList == NULL) {
        return NULL;
    }
    CHECK_PTR(freeBufferList);
    if (freeBufferList->next) CHECK_PTR(freeBufferList->next);
    PageBuffer *result = freeBufferList;
    freeBufferList = freeBufferList->next;  
    if (freeBufferList) CHECK_PTR(freeBufferList);
    CHECK_PTR(result);
    return (unsigned char *)result;
}

void DiskPageProvider::freePageBuffer(void *data) {
    CHECK_PTR(data);
    if ( freeBufferList ) CHECK_PTR(freeBufferList);
    PageBuffer *toFree = (PageBuffer *)data;
    toFree->next = freeBufferList;
    freeBufferList = toFree;

    if ( freeBufferList->next ) CHECK_PTR(freeBufferList->next);
    CHECK_PTR(freeBufferList);
}

bool DiskPageProvider::requestPage(PageId page) {
    DiskPagePending pend;

    pend.page = page;
    pend.memory = allocPageBuffer();
    QueryPerformanceCounter(&pend.startTime);

    if( pend.memory == NULL ) {
        // No free buffers, drop the request (it will be re-requested next frame anyway)
        return false;
    }

    if (!requests.pushNonBl(pend)) {
        // If the request list was full, free the buffer again...
        freePageBuffer(pend.memory);
        return false;
    }

    return true;
}

int ReqSort(DiskPagePending *a, DiskPagePending *b) {
    return a->page.getMipLevel() - b->page.getMipLevel();
}

void DiskPageProvider::frameSync(void) {

    // Lower level mipmaps get higher priority
    // Fixme, this is hacky...
    requests.sort(ReqSort);

    // Bundle all pages ready to upload this frame
    PageUpload uploads[64];
    int numToUpload = 0;

    DiskPagePending response;
    while ( responses.popNonBl(response) && numToUpload<64 ) {
        klVec2i cacheID = cache->submitPage(response.page);
        uploads[numToUpload].data = response.memory;
        uploads[numToUpload].tileX = cacheID[0];
        uploads[numToUpload].tileY = cacheID[1];
        numToUpload++;
    }

    cache->uploadPages(uploads,numToUpload);

    // Just to be sure... doesn't ever seem to block in practice
    if(numToUpload) {
        klCudaError(cudaThreadSynchronize());
    }

    // Release the upload buffers
    for ( int i=0; i<numToUpload; i++ ) {
        freePageBuffer(uploads[i].data);
    }

    // Warning: this will corrupt the cache contents temporarily but can be used to
    // benchmark upload performance at run time
    if ( r_forceUploads->getValueInt() ) {
        numToUpload = r_forceUploads->getValueInt();
        for ( int i=0;i<numToUpload;i++ ) {
            uploads[i].data = bufferMemory + i*bufferSize;
            uploads[i].tileX = rand() % 32;
            uploads[i].tileY = rand() % 32;       
        }
        cache->uploadPages(uploads,numToUpload);
        klCudaError(cudaThreadSynchronize());
    }
}

static DWORD WINAPI RealThreadMain( LPVOID lpParam ) {
    DiskPageProvider *provider = (DiskPageProvider *)lpParam;
    provider->threadMain();
    return 0;
}

void DiskPageProvider::loadPage(unsigned char *buffer, PageId page) {
    // Get info on the requested page
    PageInfo &info = pageInfo->get(page);

    // Read the data from disk
    if ( info.format == TFM_DCTHUFF_RGBA ) {
        fseek(pageFile,info.fileDataOffset,SEEK_SET);
        fread(readBuffer,info.fileDataSize,1,pageFile);

        // Extract quality
        int quality = *((int *)readBuffer);
        decoder.setQuality(quality);
        decoder.setColorSpace(SimpleDCTDec::CS_RGBA);

        // Quality is followed by raw DCTHUFF data
        unsigned char *dctData = readBuffer+sizeof(int);
        int dctSize = info.fileDataSize-sizeof(int);

        // Decode to rgba
        decoder.decompress(buffer,dctData,textureInfo.pageSize,textureInfo.pageSize,dctSize);
    } else if ( info.format == TFM_RGBA ) {
        fseek(pageFile,info.fileDataOffset,SEEK_SET);
        fread(buffer,info.fileDataSize,1,pageFile);
    } else {
        assert(0);
    }

    // Debugging, encode page information into the pixels
    if ( r_showPageInfo->getValueInt() ) {
        char buff[256];
        sprintf(buff,"L%i(%i,%i)",page.getMipLevel(),page.getIdX(),page.getIdY());
        int lengthPx = 0;

        for( char *c=buff; *c; c++) {
            lengthPx+=lentbl_S[(*c)-firstchr_S];
            lengthPx++;
        }

        int color   = *(int *)(MipLevelColors+page.getMipLevel()*4);
        int startX  = textureInfo.pageSize/2-lengthPx/2;
        int startY  = textureInfo.pageSize/2;
        int *pixels = (int *)buffer;
        int pitch   = textureInfo.pageSize;

        for( char *c=buff; *c; c++) {
            // Draw the character
            int col = DrawChar((unsigned char *)(pixels+startX+startY*pitch), pitch*4, *c, color);
            startX += (col+1);
        }

        // Draw the outline of the border

        int border = (textureInfo.pageSize - textureInfo.pageContentSize)/2;
        int firstPix = border/*-1*/;
        int lastPix = textureInfo.pageSize-1-(border/*-1*/);

        for ( int i=firstPix;i<=lastPix;i++) {
            pixels[firstPix*textureInfo.pageSize+i] = color;
            pixels[lastPix*textureInfo.pageSize+i] = color;
            pixels[i*textureInfo.pageSize+firstPix] = color;
            pixels[i*textureInfo.pageSize+lastPix] = color;
        }
    }
}

void DiskPageProvider::threadMain(void) {
    while (run) {
        DiskPagePending request = requests.pop();
        if (!run) break;

        // Load a page ...
        loadPage(request.memory, request.page);

        // ... and send it to the cache
        responses.push(request);
    }
}

void DiskPageProvider::benchmarkUpload(void) {
    //Warning: This will place the cache and everything in an undefined state...
    //the application will exit after running this
    
    static const int numToUpload = 256;
    PageUpload uploads[numToUpload];

    for ( int i=0; i<numToUpload; i++ ) {
        uploads[i].tileX = i>>4;
        uploads[i].tileY = i&0xFF;
        uploads[i].data = bufferMemory + i*bufferSize;

        // upload random data in case black pixels affect the DXT compressor
        for ( int j=0;j<bufferSize;j++) {
            ((unsigned char *)uploads[i].data)[j] = rand()&0xFF;
        }
    }

    LARGE_INTEGER startTime;
    LARGE_INTEGER endTime;
    LARGE_INTEGER endSyncTime;
    LARGE_INTEGER counterFrequency;
    cudaEvent_t start;
    cudaEvent_t end;
    klCudaError(cudaEventCreate(&start));
    klCudaError(cudaEventCreate(&end));
    QueryPerformanceFrequency(&counterFrequency);

    QueryPerformanceCounter(&startTime);
    cudaEventRecord(start,0);
    for ( int i=0;i<10;i++) {
        cache->uploadPages(uploads,numToUpload);
    }
    cudaEventRecord(end,0);
    QueryPerformanceCounter(&endTime);
    
    float elapsedGPU;
    klCudaError(cudaThreadSynchronize());
    klCudaError(cudaEventElapsedTime(&elapsedGPU,start,end));

    double elapsedCPU = ((double)(endTime.QuadPart - startTime.QuadPart)/(double)(counterFrequency.QuadPart))*1000.0;


    QueryPerformanceCounter(&endSyncTime);
    double elapsedCPUSync = ((double)(endSyncTime.QuadPart - startTime.QuadPart)/(double)(counterFrequency.QuadPart))*1000.0;

    klFatalError("Elapsed GPU:%fms CPU:%fms CPU+s:%fms",elapsedGPU,(float)elapsedCPU,(float)elapsedCPUSync);

    for ( int i=0; i<numToUpload; i++ ) {
        freePageBuffer(uploads[i].data);
    }
}