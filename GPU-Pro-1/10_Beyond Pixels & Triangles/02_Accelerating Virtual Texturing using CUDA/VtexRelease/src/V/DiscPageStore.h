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

/*
    This header describes the on-disc format of the page store.
    Warning: This header is shared with MakeV it should not include any klubnika stuff
*/

#define int64 long long
#define DISC_TEXTURE_MAGIC ('P'|('A'<<8)|('G'<<16)|('E'<<24))
#define DISC_TEXTURE_VERSION 1

/* This means 16k tiles on one axis with a tile size of 128 pixels 
   that is an 2m x 2m image, enough for now ;-) */
#define MAX_MIP_LEVELS       14

struct DiscPageStoreHeader {
    int magic;
    int version;
    int width;             // Width of the original image
    int height;            // Height of the original image
    int numMipLevels;      // Number of mipmap layers stored in the image (can be smaller than the theroretical number of mipmaps)
    int pageSize;          // Size of a single page (this includes any border data)
    int pageContentSize;   // The size of the actual content in the page (can be smaller than pageNumPixels due to borders)
    //followed by DiscPageStoreLevel[numMipLevels] mipmap level information
};

struct DiscPageStoreLevel {
    int width;      // Width of this level (may be wider than teoretically possible if non square/pow2 textures)
    int height;     // Height of this level (may be wider than teoretically possible if non square/pow2 textures)
    int numPagesX;  // Number of pages on X-axis        
    int numPagesY;  // Number of pages on Y-axis
    //followed by DiscPageStorePage[numPagesX*numPagesY] information about the individual pages
};

enum DiscPageStoreFormat {
    TFM_RGBA,
    TFM_DCTHUFF_RGBA
};

struct DiscPageStorePage {
    int format;         // DiscPageStoreFormat of this tile
    int size;           // Size (in bytes) of this tile
    int64 dataOffset;   // offset from the start of the file where this tile's data starts
};
