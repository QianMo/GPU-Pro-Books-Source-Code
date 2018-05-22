#include "microimage.h"

#include <iostream>
#include <string>
#include <vector>

#include <stdio.h>

MicroImage::MicroImage(void) : width_(0), height_(0), data_(0)
{
}

MicroImage::~MicroImage(void)
{
    if (data_!=0) delete [] data_;
}

MicroImage::MicroImage(unsigned int w, unsigned int h) : width_(w), height_(h)
{
    data_ = new unsigned char[w*h*3];
}


unsigned char *MicroImage::pixel(unsigned int x, unsigned int y)
{
    return data_+3*(x+y*width_);
}

unsigned int getUInt32(const unsigned char *p, unsigned int o)
{
    return *((unsigned int *) (p+o));
}

unsigned short getUInt16(const unsigned char *p, unsigned int o)
{
    return *((unsigned short *) (p+o));
}

bool MicroImage::loadBMPFromMemory(const unsigned char *data, unsigned int size, bool flip)
{
    if (size<54) return false;
    if (data==0) return false;

	// check for magic characters
    if((data[0] != 'B') || (data[1] != 'M')) { 
		// resource seems not to be in BMP format
		return false;
	}

    // read header information
    width_  = getUInt32(data, 18);
    height_ = getUInt32(data, 22);
    
    const unsigned short planes      = getUInt16(data, 26);
    const unsigned short bitCount    = getUInt16(data, 28);
    const unsigned int   compression = getUInt32(data, 30);

    const unsigned int clrUsed = getUInt32(data, 46);
    data += 54; // 54 bytes header size

	if(compression != 0) {
        std::cout << "cannot handle compression!" << std::endl;
        return false;
	}

    const unsigned int bitsPerPixel = planes * bitCount;
    const unsigned int paddedWidth = ((width_ * (bitsPerPixel / 8)) + 3) & (~3);

    if (bitsPerPixel!=24) {
        std::cout << "cannot handle bitdepths other than 24!" << std::endl;
        return false;
    }

    data_ = new unsigned char[width_*height_*3];
    if (flip) {
        unsigned char *d = data_+(height_-1)*width_*3;
        for(unsigned int y = 0; y < height_; ++y, data += paddedWidth, d -= width_*3) {
            ::memcpy(d, data, width_ * 3);
        } 
    } else {
        unsigned char *d = data_;
        for(unsigned int y = 0; y < height_; ++y, data += paddedWidth, d += width_*3) {
            ::memcpy(d, data, width_ * 3);
        } 
    }

    // turn BGR into RGB
    unsigned char *d = data_;
    unsigned char *dEnd = data_+height_*width_*3;
    for (;d!=dEnd; d+=3) {
        unsigned char t = d[0];
        d[0] = d[2];
        d[2] = t;
    }

    return true;
}

bool MicroImage::loadBMPFromFile(const char * name, bool flip)
{
    //must read files as binary to prevent problems from newline translation
    FILE *imgFile = fopen( name, "rb");
    if (imgFile==0) return false;

    fseek( imgFile, 0, SEEK_END);
    unsigned int size = ftell(imgFile);

    fseek( imgFile, 0, SEEK_SET);
    unsigned char *data = new unsigned char[size];
    fread( data, size, 1, imgFile);
    fclose( imgFile);

    bool res = loadBMPFromMemory(data, size, flip);

    delete[] data;

    return res;
}

