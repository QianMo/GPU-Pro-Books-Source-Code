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

#ifndef __KLTEXTUREFILE_H
#define __KLTEXTUREFILE_H

enum klTextureType {
    KLTX_2D,
    KLTX_3D,
    KLTX_CUBE,
};

enum klTextureFilter {
    KLTX_NEAREST,
    KLTX_LINEAR, // Default used by klTextureFileWriter
    KLTX_NEAREST_MIPMAP_NEAREST,
    KLTX_LINEAR_MIPMAP_NEAREST,
    KLTX_NEAREST_MIPMAP_LINEAR,
    KLTX_LINEAR_MIPMAP_LINEAR,
};

enum klTextureWrap {
    KLTX_CLAMP,
    KLTX_REPEAT, // Default used by klTextureFileWriter
};

// Add new formats to the end so existing files don't get invalidated
enum klTextureFileFormat {
    KLTX_RGB8,
    KLTX_RGBX8,
    KLTX_RGBA8,
    KLTX_RGB565,
    KLTX_RGBA5551,

    KLTX_RGB_DXT1,
    KLTX_RGBA_DXT1,
    KLTX_RGBA_DXT3,
    KLTX_RGBA_DXT5,

    KLTX_DEPTH24,
    KLTX_L32F,
    KLTX_LA32F,
    KLTX_RGBA32F,

    KLTX_R_RGTC,
    KLTX_SR_RGTC
};

// These are on-disc structures

#define KLTX_MAGIC 'XTLK'

#define KLTX_CURRENT_VERSION 2

struct klTextureFileHeader {
    unsigned int magic;
    unsigned int version;
    unsigned int width;
    unsigned int height;
    unsigned int depth;     // 0 for 2D textures
    unsigned int numLayers; // >1 for texture arrays, will be 6 for cubemaps
    unsigned int numMips;   // 1 for no mips
    klTextureType texType;   // 2d, 3d, cube, ...
    klTextureFileFormat format;
    klTextureFilter     filter;
    klTextureWrap       wrap[3];
    // followed by klTextureFileLayer[numLayers]
};

struct klTextureFileLayer {
    unsigned int numMips;
    // followed by klTextureFileMipmap[numMips]
};

struct klTextureFileMipmap {
    unsigned int dataSize;
    // followed by char[dataSize]
};

class klTextureFileWriter {

    klTextureWrap wrapModes[3];
    klTextureFilter filter;

    FILE *outFile;
    bool startedFile;
    int expectMips;
    int excpectLayers;
    int lastSize;
    int numLayers;

    klTextureFileHeader initHeader(klTextureFileFormat format, int numMipMaps);

public:
    klTextureFileWriter(const char *fileName);
    ~klTextureFileWriter();

    void write2DHeader(klTextureFileFormat format, int width, int height, int numMipMaps = 1,  int numLayers = 1);
    void write3DHeader(klTextureFileFormat format, int width, int height, int depth, int numMipMaps = 1);
    void writeCubeHeader(klTextureFileFormat format, int width, int height, int numMipMaps = 1);

    void writeMipMap(unsigned int dataSize);
    void writeLayerData(const void *data);

    void setParameters(klTextureWrap s, klTextureWrap t, klTextureWrap r, klTextureFilter filt);
};

class klTextureFileReader {
    std::istream &inFile;
    bool startedFile;
    int expectMips;
    int excpectLayers;
    int lastSize;
    int numLayers;
public:
    klTextureFileReader(std::istream &stream);
    ~klTextureFileReader();

    void readHeader(klTextureType &texType, klTextureFileFormat &format, int &width, int &height, int &depth, int &numMipMaps, int &numLayers);
    void readMipMap(unsigned int &dataSize);
    void readLayerData(void *data);
};

#endif //__KLTEXTUREFILE_H