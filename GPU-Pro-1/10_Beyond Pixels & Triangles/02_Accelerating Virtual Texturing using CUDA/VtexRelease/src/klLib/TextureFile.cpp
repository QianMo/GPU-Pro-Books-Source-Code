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
#include "texturefile.h"

////////////////////// Writer ///////////////////////////////////////

klTextureFileWriter::klTextureFileWriter(const char *fileName) {
    startedFile = false;
    expectMips = 0;
    excpectLayers = 0;
    outFile = fopen(fileName,"wb");

    wrapModes[0] = wrapModes[1] = wrapModes[2] = KLTX_REPEAT;
    filter = KLTX_LINEAR;
}

void klTextureFileWriter::setParameters(klTextureWrap s, klTextureWrap t, klTextureWrap r, klTextureFilter filt) {
    wrapModes[0] = s;
    wrapModes[1] = t;
    wrapModes[2] = r;
    filter = filt;
}

klTextureFileWriter::~klTextureFileWriter() {
    fclose(outFile);
}

klTextureFileHeader klTextureFileWriter::initHeader(klTextureFileFormat format, int numMipMaps) {
    klTextureFileHeader h;
    h.magic = KLTX_MAGIC;
    h.version = KLTX_CURRENT_VERSION;
    h.format = format;
    h.numMips = numMipMaps;
    h.filter = filter;
    for (int i=0; i<3; i++ ) {
        h.wrap[i] = wrapModes[i];
    }

    // Set state for consistency checking
    startedFile = true;
    expectMips = numMipMaps;
    excpectLayers = 0;
    lastSize = 0;

    return h;
}

void klTextureFileWriter::write2DHeader(klTextureFileFormat format, int width, int height, int numMipMaps,  int numLayers) {
    assert(!startedFile);
    klTextureFileHeader h = initHeader(format,numMipMaps);
    h.texType = KLTX_2D;
    h.width = width;
    h.height = height;
    h.depth = 0;
    h.numLayers = numLayers;  
    this->numLayers = numLayers;

    fwrite(&h,sizeof(klTextureFileHeader),1,outFile);
}

void klTextureFileWriter::write3DHeader(klTextureFileFormat format, int width, int height, int depth, int numMipMaps) {
    assert(!startedFile);
    klTextureFileHeader h = initHeader(format,numMipMaps);
    h.texType = KLTX_3D;
    h.width = width;
    h.height = height;
    h.depth = depth;
    h.numLayers = 1;  
    this->numLayers = 1;

    fwrite(&h,sizeof(klTextureFileHeader),1,outFile);
}

void klTextureFileWriter::writeCubeHeader(klTextureFileFormat format, int width, int height, int numMipMaps) {
    assert(!startedFile);
    klTextureFileHeader h = initHeader(format,numMipMaps);
    h.texType = KLTX_CUBE;
    h.width = width;
    h.height = height;
    h.depth = 0;
    h.numLayers = 6;  
    this->numLayers = 6;

    fwrite(&h,sizeof(klTextureFileHeader),1,outFile);
}

void klTextureFileWriter::writeMipMap(unsigned int dataSize) {
    assert(startedFile);
    assert(expectMips);
    assert(!excpectLayers);

    klTextureFileMipmap mipmap;
    mipmap.dataSize = dataSize;
    fwrite(&mipmap,sizeof(klTextureFileMipmap),1,outFile);
    lastSize = dataSize; 
    excpectLayers = numLayers;

    expectMips--;    
}

void klTextureFileWriter::writeLayerData(const void *data) {
    assert(startedFile);
    assert(excpectLayers);

    fwrite(data,1,lastSize,outFile);

    excpectLayers--;
}

////////////////////// Reader ///////////////////////////////////////

klTextureFileReader::klTextureFileReader(std::istream &stream) : inFile(stream) {
    startedFile = false;
    expectMips = 0;
    excpectLayers = 0;
}

klTextureFileReader::~klTextureFileReader() {
}

void klTextureFileReader::readHeader(klTextureType &texType, klTextureFileFormat &format, int &width, int &height, int &depth, int &numMipMaps, int &numLayers) {
    assert(!startedFile);

    klTextureFileHeader h;
    inFile.read((char *)&h,sizeof(klTextureFileHeader));
    assert(h.magic == KLTX_MAGIC);
    assert(h.version == KLTX_CURRENT_VERSION);

    width = h.width;
    height = h.height;
    depth = h.depth;
    format = h.format;
    numLayers = h.numLayers;    
    numMipMaps = h.numMips;
    texType = h.texType;

    this->numLayers = numLayers;
    excpectLayers = 0;
    startedFile = true;
    expectMips = numMipMaps;
}

void klTextureFileReader::readMipMap(unsigned int &dataSize) {
    assert(startedFile);
    assert(expectMips);
    assert(!excpectLayers);

    klTextureFileMipmap mipmap;
    inFile.read((char *)&mipmap,sizeof(klTextureFileMipmap));
    dataSize = mipmap.dataSize;
    lastSize = dataSize; 
    excpectLayers = numLayers;

    expectMips--;    
}


void klTextureFileReader::readLayerData(void *data) {
    assert(startedFile);
    assert(excpectLayers);

    inFile.read((char *)data,lastSize);

    excpectLayers--;
}
