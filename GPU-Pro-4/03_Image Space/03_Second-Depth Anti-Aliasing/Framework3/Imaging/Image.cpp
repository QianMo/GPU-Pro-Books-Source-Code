
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "Image.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "../Math/Vector.h"

#ifndef NO_JPEG
extern "C" {
#include "jpeglib.h"
}
#ifdef _WIN32
#pragma comment (lib, "../Framework3/Libs/libjpeg.lib")
#endif // _WIN32
#endif // NO_JPEG

#ifndef NO_PNG
#include "png.h"
#ifdef _WIN32
#pragma comment (lib, "../Framework3/Libs/libpng.lib")
#pragma comment (lib, "../Framework3/Libs/zlib.lib")
#endif // _WIN32


void user_write_data(png_structp png_ptr, png_bytep data, png_size_t length){
	fwrite(data, length, 1, (FILE *) png_get_io_ptr(png_ptr));
}

void user_flush_data(png_structp png_ptr){
	fflush((FILE *) png_get_io_ptr(png_ptr));
}

void user_read_data(png_structp png_ptr, png_bytep data, png_size_t length){
	fread(data, length, 1, (FILE *) png_get_io_ptr(png_ptr));
}

png_voidp malloc_fn(png_structp png_ptr, png_size_t size){
	return malloc(size);
}
void free_fn(png_structp png_ptr, png_voidp ptr){
	free(ptr);
}

#endif // NO_PNG

#pragma pack (push, 1)

struct HTexHeader {
	uint32 identifier;
	uint32 version;
	uint64 format;
	uint32 width;
	uint32 height;
	uint32 depth;
	uint32 nMipMaps;
};

struct HTexHeader2 {
	uint32 flags;
	uint32 nExtraData;
};

#define DDPF_ALPHAPIXELS 0x00000001 
#define DDPF_FOURCC      0x00000004 
#define DDPF_RGB         0x00000040

#define DDSD_CAPS        0x00000001
#define DDSD_HEIGHT      0x00000002
#define DDSD_WIDTH       0x00000004
#define DDSD_PITCH       0x00000008
#define DDSD_PIXELFORMAT 0x00001000
#define DDSD_MIPMAPCOUNT 0x00020000
#define DDSD_LINEARSIZE  0x00080000
#define DDSD_DEPTH       0x00800000

#define DDSCAPS_COMPLEX  0x00000008 
#define DDSCAPS_TEXTURE  0x00001000 
#define DDSCAPS_MIPMAP   0x00400000 

#define DDSCAPS2_CUBEMAP 0x00000200 
#define DDSCAPS2_VOLUME  0x00200000 

#define DDSCAPS2_CUBEMAP_POSITIVEX 0x00000400
#define DDSCAPS2_CUBEMAP_NEGATIVEX 0x00000800
#define DDSCAPS2_CUBEMAP_POSITIVEY 0x00001000
#define DDSCAPS2_CUBEMAP_NEGATIVEY 0x00002000
#define DDSCAPS2_CUBEMAP_POSITIVEZ 0x00004000
#define DDSCAPS2_CUBEMAP_NEGATIVEZ 0x00008000
#define DDSCAPS2_CUBEMAP_ALL_FACES (DDSCAPS2_CUBEMAP_POSITIVEX | DDSCAPS2_CUBEMAP_NEGATIVEX | DDSCAPS2_CUBEMAP_POSITIVEY | DDSCAPS2_CUBEMAP_NEGATIVEY | DDSCAPS2_CUBEMAP_POSITIVEZ | DDSCAPS2_CUBEMAP_NEGATIVEZ)

#define D3D10_RESOURCE_MISC_TEXTURECUBE 0x4
#define D3D10_RESOURCE_DIMENSION_BUFFER    1
#define D3D10_RESOURCE_DIMENSION_TEXTURE1D 2
#define D3D10_RESOURCE_DIMENSION_TEXTURE2D 3
#define D3D10_RESOURCE_DIMENSION_TEXTURE3D 4

struct DDSHeader {
	uint32 dwMagic;
	uint32 dwSize;
	uint32 dwFlags;
	uint32 dwHeight;
	uint32 dwWidth;
	uint32 dwPitchOrLinearSize;
	uint32 dwDepth; 
	uint32 dwMipMapCount;
	uint32 dwReserved[11];

	struct {
        uint32 dwSize;
		uint32 dwFlags;
		uint32 dwFourCC;
		uint32 dwRGBBitCount;
		uint32 dwRBitMask;
		uint32 dwGBitMask;
		uint32 dwBBitMask;
		uint32 dwRGBAlphaBitMask; 
	} ddpfPixelFormat;

	struct {
		uint32 dwCaps1;
		uint32 dwCaps2;
		uint32 Reserved[2];
	} ddsCaps;

	uint32 dwReserved2;
};

struct DDSHeaderDX10 {
    uint32 dxgiFormat;
    uint32 resourceDimension;
    uint32 miscFlag;
    uint32 arraySize;
    uint32 reserved;
};

struct TGAHeader {
	uint8  descriptionlen;
	uint8  cmaptype;
	uint8  imagetype;
	uint16 cmapstart;
	uint16 cmapentries;
	uint8  cmapbits;
	uint16 xoffset;
	uint16 yoffset;
	uint16 width;
	uint16 height;
	uint8  bpp;
	uint8  attrib;
};

struct BMPHeader {
	uint16 bmpIdentifier;
	uint8  junk[16];
	uint32 width;
	uint32 height;
	uint16 junk2;
	uint16 bpp;
	uint16 compression;
	uint8  junk3[22];
};

struct PCXHeader {
	uint8  junk[3];
	uint8  bitsPerChannel;
	uint8  junk2[4];
	uint16 width;
	uint16 height;
	uint8  junk3[53];
	uint8  nChannels;
	uint16 scanlineLen;
	uint8  junk4[60];
};

#pragma pack (pop)


/* ---------------------------------------------- */

struct FormatString {
	FORMAT format;
	const char *string;
};

static const FormatString formatStrings[] = {
	{ FORMAT_NONE,   "NONE"  },

	{ FORMAT_R8,     "R8"    },
	{ FORMAT_RG8,    "RG8"   },
	{ FORMAT_RGB8,   "RGB8"  },
	{ FORMAT_RGBA8,  "RGBA8" },
	
	{ FORMAT_R16,    "R16"   },
	{ FORMAT_RG16,   "RG16"  },
	{ FORMAT_RGB16,  "RGB16" },
	{ FORMAT_RGBA16, "RGBA16"},

	{ FORMAT_R16F,   "R16F"    },
	{ FORMAT_RG16F,  "RG16F"   },
	{ FORMAT_RGB16F, "RGB16F"  },
	{ FORMAT_RGBA16F,"RGBA16F" },

	{ FORMAT_R32F,   "R32F"    },
	{ FORMAT_RG32F,  "RG32F"   },
	{ FORMAT_RGB32F, "RGB32F"  },
	{ FORMAT_RGBA32F,"RGBA32F" },

	{ FORMAT_RGBE8,  "RGBE8"   },
	{ FORMAT_RGB565, "RGB565"  },
	{ FORMAT_RGBA4,  "RGBA4"   },
	{ FORMAT_RGB10A2,"RGB10A2" },

	{ FORMAT_DXT1,   "DXT1"  },
	{ FORMAT_DXT3,   "DXT3"  },
	{ FORMAT_DXT5,   "DXT5"  },
	{ FORMAT_ATI1N,  "ATI1N" },
	{ FORMAT_ATI2N,  "ATI2N" },
};

const char *getFormatString(const FORMAT format){
	for (unsigned int i = 0; i < elementsOf(formatStrings); i++){
		if (format == formatStrings[i].format) return formatStrings[i].string;
	}
	return NULL;
}

FORMAT getFormatFromString(char *string){
	for (unsigned int i = 0; i < elementsOf(formatStrings); i++){
		if (stricmp(string, formatStrings[i].string) == 0) return formatStrings[i].format;
	}
	return FORMAT_NONE;
}


template <typename DATA_TYPE>
inline void swapChannels(DATA_TYPE *pixels, int nPixels, const int channels, const int ch0, const int ch1){
	do {
		DATA_TYPE tmp = pixels[ch1];
		pixels[ch1] = pixels[ch0];
		pixels[ch0] = tmp;
		pixels += channels;
	} while (--nPixels);
}


/* ---------------------------------------------- */

Image::Image(){
	pixels = NULL;
	width  = 0;
	height = 0;
	depth  = 0;
	nMipMaps = 0;
	arraySize = 0;
	format = FORMAT_NONE;

	nExtraData = 0;
	extraData = NULL;
}

Image::Image(const Image &img){
	width  = img.width;
	height = img.height;
	depth  = img.depth;
	nMipMaps = img.nMipMaps;
	arraySize = img.arraySize;
	format = img.format;

	int size = getMipMappedSize(0, nMipMaps) * arraySize;
	pixels = new unsigned char[size];
	memcpy(pixels, img.pixels, size);

	nExtraData = img.nExtraData;
	extraData = new unsigned char[nExtraData];
	memcpy(extraData, img.extraData, nExtraData);
}

Image::~Image(){
	delete [] pixels;
	delete [] extraData;
}

unsigned char *Image::create(const FORMAT fmt, const int w, const int h, const int d, const int mipMapCount, const int arraysize){
    format = fmt;
    width  = w;
	height = h;
	depth  = d;
	nMipMaps = mipMapCount;
	arraySize = arraysize;

	return (pixels = new unsigned char[getMipMappedSize(0, nMipMaps) * arraySize]);
}

void Image::free(){
	delete [] pixels;
	pixels = NULL;

	delete [] extraData;
	extraData = NULL;
}

void Image::clear(){
	free();

	width  = 0;
	height = 0;
	depth  = 0;
	nMipMaps = 0;
	arraySize = 0;
	format = FORMAT_NONE;

	nExtraData = 0;
}

unsigned char *Image::getPixels(const int mipMapLevel) const {
	return (mipMapLevel < nMipMaps)? pixels + getMipMappedSize(0, mipMapLevel) : NULL;
}

unsigned char *Image::getPixels(const int mipMapLevel, const int arraySlice) const {
	if (mipMapLevel >= nMipMaps || arraySlice >= arraySize) return NULL;

	return pixels + getMipMappedSize(0, nMipMaps) * arraySlice + getMipMappedSize(0, mipMapLevel);
}


int Image::getMipMapCountFromDimesions() const {
	int m = max(width, height);
	m = max(m, depth);

	int i = 0;
	while (m > 0){
		m >>= 1;
		i++;
	}

	return i;
}

int Image::getMipMappedSize(const int firstMipMapLevel, int nMipMapLevels, FORMAT srcFormat) const {
	int w = getWidth (firstMipMapLevel);
	int h = getHeight(firstMipMapLevel);
	int d = getDepth (firstMipMapLevel);

	if (srcFormat == FORMAT_NONE) srcFormat = format;
	
	int size = 0;
	while (nMipMapLevels){
		if (isCompressedFormat(srcFormat)){
			size += ((w + 3) >> 2) * ((h + 3) >> 2) * d;
		} else {
			size += w * h * d;
		}
		w >>= 1;
		h >>= 1;
		d >>= 1;
		if (w + h + d == 0) break;
		if (w == 0) w = 1;
		if (h == 0) h = 1;
		if (d == 0) d = 1;

		nMipMapLevels--;
	}

	if (isCompressedFormat(srcFormat)){
		size *= getBytesPerBlock(srcFormat);
	} else {
		size *= getBytesPerPixel(srcFormat);
	}

	return (depth == 0)? 6 * size : size;
}

int Image::getSliceSize(const int mipMapLevel, FORMAT srcFormat) const {
	int w = getWidth (mipMapLevel);
	int h = getHeight(mipMapLevel);

	if (srcFormat == FORMAT_NONE) srcFormat = format;
	
	int size;
	if (isCompressedFormat(srcFormat)){
		size = ((w + 3) >> 2) * ((h + 3) >> 2) * getBytesPerBlock(srcFormat);
	} else {
		size = w * h * getBytesPerPixel(srcFormat);
	}

	return size;
}

int Image::getPixelCount(const int firstMipMapLevel, int nMipMapLevels) const {
	int w = getWidth (firstMipMapLevel);
	int h = getHeight(firstMipMapLevel);
	int d = getDepth (firstMipMapLevel);
	int size = 0;
	while (nMipMapLevels){
		size += w * h * d;
		w >>= 1;
		h >>= 1;
		d >>= 1;
		if (w + h + d == 0) break;
		if (w == 0) w = 1;
		if (h == 0) h = 1;
		if (d == 0) d = 1;

		nMipMapLevels--;
	}

	return (depth == 0)? 6 * size : size;
}

int Image::getWidth(const int mipMapLevel) const {
	int a = width >> mipMapLevel;
	return (a == 0)? 1 : a;
}

int Image::getHeight(const int mipMapLevel) const {
	int a = height >> mipMapLevel;
	return (a == 0)? 1 : a;
}

int Image::getDepth(const int mipMapLevel) const {
	int a = depth >> mipMapLevel;
	return (a == 0)? 1 : a;
}

bool Image::loadHTEX(const char *fileName){
	FILE *file = fopen(fileName, "rb");
	if (file == NULL) return false;

	HTexHeader header;
	fread(&header, sizeof(header), 1, file);

	if (header.identifier != MCHAR4('H','T','E','X') || header.version > 2){
		fclose(file);
		return false;
	}

	HTexHeader2 header2;
	if (header.version >= 2){
		fread(&header2, sizeof(header2), 1, file);
	} else {
		memset(&header2, 0, sizeof(header2));
	}

	format = getFormatFromString((char *) &header.format);
	width  = header.width;
	height = header.height;
	depth  = header.depth;
	nMipMaps = header.nMipMaps;
	arraySize = 1;

	nExtraData = header2.nExtraData;
	
	pixels = new unsigned char[getMipMappedSize(0, nMipMaps)];

	if (isPlainFormat(format)){
		/*int nChannels = getChannelCount(format);
		int bpc = getBytesPerChannel(format);

		for (int ch = 0; ch < nChannels; ch++){
			for (int level = 0; level < nMipMaps; level++){
				int nPixels = getPixelCount(level, 1);

				ubyte *dest = getPixels(level) + ch * bpc;
				for (int i = 0; i < nPixels; i++){
					fread(dest + i * nChannels * bpc, bpc, 1, file);
				}
			}
		}*/
		fread(pixels, getMipMappedSize(0, nMipMaps), 1, file);

	} else if (isCompressedFormat(format)){
		if (format == FORMAT_DXT1 || format == FORMAT_DXT5){
			char shift = (format == FORMAT_DXT1)? 3 : 4;

			for (int level = 0; level < nMipMaps; level++){
				ubyte *dest = getPixels(level);
				if (format != FORMAT_DXT1) dest += 8;
				int w = getWidth(level);
				int h = getHeight(level);
				int d = getDepth(level);

				int nBlocks = d * ((w + 3) >> 2) * ((h + 3) >> 2);

				for (int n = 0; n < nBlocks; n++){
					fread(dest + (n << shift), 1, 2, file);
				}

				for (int n = 0; n < nBlocks; n++){
					fread(dest + (n << shift) + 2, 1, 2, file);
				}

				for (int n = 0; n < nBlocks; n++){
					fread(dest + (n << shift) + 4, 1, 4, file);
				}
			}
		}
		if (format == FORMAT_DXT5 || format == FORMAT_ATI1N || format == FORMAT_ATI2N){
			char shift = (format != FORMAT_DXT5)? 3 : 4;

			for (int level = 0; level < nMipMaps; level++){
				ubyte *src = getPixels(level);
				int w = getWidth(level);
				int h = getHeight(level);
				int d = getDepth(level);

				int nBlocks = d * ((w + 3) >> 2) * ((h + 3) >> 2);

				for (int n = 0; n < nBlocks; n++){
					fread(src + (n << shift), 1, 1, file);
				}
				if (format == FORMAT_ATI2N){
					for (int n = 0; n < nBlocks; n++){
						fread(src + 8 * (nBlocks + n), 1, 1, file);
					}
				}

				for (int n = 0; n < nBlocks; n++){
					fread(src + (n << shift) + 1, 1, 1, file);
				}
				if (format == FORMAT_ATI2N){
					for (int n = 0; n < nBlocks; n++){
						fread(src + 8 * (nBlocks + n) + 1, 1, 1, file);
					}
				}

				for (int n = 0; n < nBlocks; n++){
					fread(src + (n << shift) + 2, 1, 6, file);
				}
				if (format == FORMAT_ATI2N){
					for (int n = 0; n < nBlocks; n++){
						fread(src + 8 * (nBlocks + n) + 2, 1, 6, file);
					}
				}
			}
		}
	}

	if (nExtraData){
		extraData = new unsigned char[nExtraData];
		fread(extraData, 1, nExtraData, file);
	}

	fclose(file);

	return true;
}

bool Image::loadDDS(const char *fileName, uint flags){
	DDSHeader header;

	FILE *file;
	if ((file = fopen(fileName, "rb")) == NULL) return false;

	fread(&header, sizeof(header), 1, file);
	if (header.dwMagic != MCHAR4('D','D','S',' ')){
		fclose(file);
		return false;
	}

	width  = header.dwWidth;
	height = header.dwHeight;
	depth  = (header.ddsCaps.dwCaps2 & DDSCAPS2_CUBEMAP)? 0 : (header.dwDepth == 0)? 1 : header.dwDepth;
	nMipMaps = ((flags & DONT_LOAD_MIPMAPS) || (header.dwMipMapCount == 0))? 1 : header.dwMipMapCount;
	arraySize = 1;

	if (header.ddpfPixelFormat.dwFourCC == MCHAR4('D','X','1','0')){
		DDSHeaderDX10 dx10Header;
		fread(&dx10Header, sizeof(dx10Header), 1, file);

		switch (dx10Header.dxgiFormat){
			case 61: format = FORMAT_R8; break;
			case 49: format = FORMAT_RG8; break;
			case 28: format = FORMAT_RGBA8; break;

			case 56: format = FORMAT_R16; break;
			case 35: format = FORMAT_RG16; break;
			case 11: format = FORMAT_RGBA16; break;

			case 54: format = FORMAT_R16F; break;
			case 34: format = FORMAT_RG16F; break;
			case 10: format = FORMAT_RGBA16F; break;

			case 41: format = FORMAT_R32F; break;
			case 16: format = FORMAT_RG32F; break;
			case 6:  format = FORMAT_RGB32F; break;
			case 2:  format = FORMAT_RGBA32F; break;

			case 67: format = FORMAT_RGB9E5; break;
			case 26: format = FORMAT_RG11B10F; break;
			case 24: format = FORMAT_RGB10A2; break;

			case 71: format = FORMAT_DXT1; break;
			case 74: format = FORMAT_DXT3; break;
			case 77: format = FORMAT_DXT5; break;
			case 80: format = FORMAT_ATI1N; break;
			case 83: format = FORMAT_ATI2N; break;
			default:
				fclose(file);
				return false;
		}

	} else {

		switch (header.ddpfPixelFormat.dwFourCC){
			case 34:  format = FORMAT_RG16; break;
			case 36:  format = FORMAT_RGBA16; break;
			case 111: format = FORMAT_R16F; break;
			case 112: format = FORMAT_RG16F; break;
			case 113: format = FORMAT_RGBA16F; break;
			case 114: format = FORMAT_R32F; break;
			case 115: format = FORMAT_RG32F; break;
			case 116: format = FORMAT_RGBA32F; break;
			case MCHAR4('D','X','T','1'): format = FORMAT_DXT1; break;
			case MCHAR4('D','X','T','3'): format = FORMAT_DXT3; break;
			case MCHAR4('D','X','T','5'): format = FORMAT_DXT5; break;
			case MCHAR4('A','T','I','1'): format = FORMAT_ATI1N; break;
			case MCHAR4('A','T','I','2'): format = FORMAT_ATI2N; break;
			default:
				switch (header.ddpfPixelFormat.dwRGBBitCount){
					case 8: format = FORMAT_I8; break;
					case 16:
						format = (header.ddpfPixelFormat.dwRGBAlphaBitMask == 0xF000)? FORMAT_RGBA4 : 
								 (header.ddpfPixelFormat.dwRGBAlphaBitMask == 0xFF00)? FORMAT_IA8 : 
								 (header.ddpfPixelFormat.dwBBitMask == 0x1F)? FORMAT_RGB565 : FORMAT_I16;
						break;
					case 24: format = FORMAT_RGB8; break;
					case 32:
						format = (header.ddpfPixelFormat.dwRBitMask == 0x3FF00000)? FORMAT_RGB10A2 : FORMAT_RGBA8;
						break;
					default:
						fclose(file);
						return false;
				}
		}
	}

	int size = getMipMappedSize(0, nMipMaps);
	pixels = new unsigned char[size];
	if (isCube()){
		for (int face = 0; face < 6; face++){
			for (int mipMapLevel = 0; mipMapLevel < nMipMaps; mipMapLevel++){
				int faceSize = getMipMappedSize(mipMapLevel, 1) / 6;
                unsigned char *src = getPixels(mipMapLevel) + face * faceSize;

				fread(src, 1, faceSize, file);
			}
			if ((flags & DONT_LOAD_MIPMAPS) && header.dwMipMapCount > 1){
				fseek(file, getMipMappedSize(1, header.dwMipMapCount - 1) / 6, SEEK_CUR);
			}
		}
	} else {
		fread(pixels, 1, size, file);
	}

	if ((format == FORMAT_RGB8 || format == FORMAT_RGBA8) && header.ddpfPixelFormat.dwBBitMask == 0xFF){
		int nChannels = getChannelCount(format);
		swapChannels(pixels, size / nChannels, nChannels, 0, 2);
	}

	fclose(file);
	return true;
}

#ifndef NO_HDR

bool Image::loadHDR(const char *fileName){
	FILE *file = fopen(fileName, "rb");
	if (file == NULL) return false;

	char *str, header[256];
	fread(header, 1, sizeof(header), file);
	header[255] = '\0';

	if ((str = strstr(header, "-Y ")) != NULL){
		height = atoi(str + 3);
		if ((str = strstr(header, "+X ")) != NULL){
			str += 3;
            width = atoi(str);
			// Find the end of the width field, which should be where the header ends
			while (*str >= '0' && *str <= '9') str++;
			uint headerSize = (uint) (str + 1 - header);

			depth = 1;
			nMipMaps = 1;
			arraySize = 1;
			format = FORMAT_RGBE8;

			// Load the whole file
			fseek(file, 0, SEEK_END);
			long size = ftell(file) - headerSize;
			fseek(file, headerSize, SEEK_SET);
			ubyte *buffer = new ubyte[size];
			fread(buffer, 1, size, file);
			fclose(file);

			// No RLE encoding
			if (width < 8 || width > 0x7FFF || buffer[0] != 2 || buffer[1] != 2){
				pixels = buffer;
			} else {
				int w = width * 4;

				pixels = new ubyte[w * height];
				ubyte *src = buffer;

				for (int j = 0; j < height; j++){
					src += 4;

					ubyte *sEnd = pixels + (j + 1) * w;
					for (int i = 0; i < 4; i++){
						ubyte *sDest = pixels + j * w + i;
						do {
							uint count = *src++;
							if (count > 128){
								count -= 128;
								do {
									*sDest = *src;
									sDest += 4;
								} while (--count);
								src++;
							} else {
								do {
									*sDest = *src++;
									sDest += 4;
								} while (--count);
							}
						} while (sDest < sEnd);
					}
				}

		        delete [] buffer;
			}

			return true;
		}
	}

	fclose(file);
	return false;
}

#endif

#ifndef NO_JPEG
bool Image::loadJPEG(const char *fileName){
	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);

	FILE *file;
	if ((file = fopen(fileName, "rb")) == NULL) return false;
	
	jpeg_stdio_src(&cinfo, file);
	jpeg_read_header(&cinfo, TRUE);
	jpeg_start_decompress(&cinfo);

	switch (cinfo.num_components){
		case 1:
			format = FORMAT_I8;
			break;
		case 3:
			format = FORMAT_RGB8;
			break;
		case 4:
			format = FORMAT_RGBA8;
			break;
	}
	width  = cinfo.output_width;
	height = cinfo.output_height;
	depth  = 1;
	nMipMaps = 1;
	arraySize = 1;

	pixels = new unsigned char[width * height * cinfo.num_components];
	unsigned char *curr_scanline = pixels;

	while (cinfo.output_scanline < cinfo.output_height){
		jpeg_read_scanlines(&cinfo, &curr_scanline, 1);
		curr_scanline += width * cinfo.num_components;
	}
	
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);

	fclose(file);

	return true;
}
#endif // NO_JPEG

#ifndef NO_PNG
bool Image::loadPNG(const char *fileName){
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
    FILE *file;

    // open the PNG input file
    if ((file = fopen(fileName, "rb")) == NULL) return false;

    // first check the eight byte PNG signature
    png_byte pbSig[8];
    fread(pbSig, 1, 8, file);
    if (!png_check_sig(pbSig, 8)){
		fclose(file);
		return false;
	}

    // create the two png(-info) structures
    if ((png_ptr = png_create_read_struct_2(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL, NULL, malloc_fn, free_fn)) == NULL){
		fclose(file);
        return false;
    }

    if ((info_ptr = png_create_info_struct(png_ptr)) == NULL){
        png_destroy_read_struct(&png_ptr, NULL, NULL);
		fclose(file);
        return false;
    }

	// initialize the png structure
	png_set_read_fn(png_ptr, file, user_read_data);
	png_set_sig_bytes(png_ptr, 8);
	
	// read all PNG info up to image data
	png_read_info(png_ptr, info_ptr);

	// get width, height, bit-depth and color-type
	png_uint_32 w, h;
    int bitDepth, colorType;
	png_get_IHDR(png_ptr, info_ptr, &w, &h, &bitDepth, &colorType, NULL, NULL, NULL);

	width = w;
	height = h;
	depth = 1;
	nMipMaps = 1;
	arraySize = 1;

	int nChannels = png_get_channels(png_ptr, info_ptr);
	switch (nChannels){
		case 1:
			format = (bitDepth > 8)? FORMAT_I16 : FORMAT_I8;
			break;
		case 2:
			format = (bitDepth > 8)? FORMAT_IA16 : FORMAT_IA8;
			break;
		case 3:
			format = (bitDepth > 8)? FORMAT_RGB16 : FORMAT_RGB8;
			break;
		case 4:
			format = (bitDepth > 8)? FORMAT_RGBA16 : FORMAT_RGBA8;
			break;
	}

	int rowSize = width * nChannels * bitDepth / 8;

	// now we can allocate memory to store the image
	pixels = new ubyte[rowSize * height];
	
	// set the individual row-pointers to point at the correct offsets
    png_byte **ppbRowPointers = new png_bytep[height];
	for (int i = 0; i < height; i++)
		ppbRowPointers[i] = pixels + i * rowSize;

	// now we can go ahead and just read the whole image
	png_read_image(png_ptr, ppbRowPointers);

	// read the additional chunks in the PNG file (not really needed)
	png_read_end(png_ptr, NULL);
	
	delete [] ppbRowPointers;

	if (colorType == PNG_COLOR_TYPE_PALETTE){
		png_colorp palette;
		int num_palette;
		png_get_PLTE(png_ptr, info_ptr, &palette, &num_palette);

		ubyte *newPixels = new ubyte[width * height * 3];
		if (bitDepth == 4){
			for (int i = 0; i < rowSize * height; i++){
				uint i0 = pixels[i] >> 4;
				uint i1 = pixels[i] & 0xF;
				newPixels[6 * i    ] = palette[i0].red;
				newPixels[6 * i + 1] = palette[i0].green;
				newPixels[6 * i + 2] = palette[i0].blue;
				newPixels[6 * i + 3] = palette[i1].red;
				newPixels[6 * i + 4] = palette[i1].green;
				newPixels[6 * i + 5] = palette[i1].blue;
			}
		} else {
			for (int i = 0; i < rowSize * height; i++){
				newPixels[3 * i    ] = palette[pixels[i]].red;
				newPixels[3 * i + 1] = palette[pixels[i]].green;
				newPixels[3 * i + 2] = palette[pixels[i]].blue;
			}
		}
		format = FORMAT_RGB8;

		delete [] pixels;
		pixels = newPixels;
	}

	if (bitDepth == 16){
		// Fix endian
		int size = width * height * nChannels * sizeof(ushort);
		for (int i = 0; i < size; i += 2){
			ubyte tmp = pixels[i];
			pixels[i] = pixels[i + 1];
			pixels[i + 1] = tmp;
		}
	}

	// and we're done
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    fclose(file);

    return true;
}
#endif // NO_PNG

#ifndef NO_TGA
bool Image::loadTGA(const char *fileName){
	TGAHeader header;

	int size, x, y, pixelSize, palLength;
	unsigned char *tempBuffer, *fBuffer, *dest, *src;
	unsigned int tempPixel;
	unsigned char palette[768];
	FILE *file;


	if ((file = fopen(fileName, "rb")) == NULL) return false;
	
	// Find file size
	fseek(file, 0, SEEK_END);
	size = ftell(file);
	fseek(file, 0, SEEK_SET);

	// Read the header
	fread(&header, sizeof(header), 1, file);
	
	width  = header.width;
	height = header.height;
	depth  = 1;
	nMipMaps = 1;
	arraySize = 1;

	pixelSize = header.bpp / 8;

	if ((palLength = header.descriptionlen + header.cmapentries * header.cmapbits / 8) > 0) fread(palette, sizeof(palette), 1, file);
	
	// Read the file data
	fBuffer = new unsigned char[size - sizeof(header) - palLength];
	fread(fBuffer, size - sizeof(header) - palLength, 1, file);
	fclose(file);

	size = width * height * pixelSize;

	tempBuffer = new unsigned char[size];

	// Decode if rle compressed. Bit 3 of .imagetype tells if the file is compressed
	if (header.imagetype & 0x08){
		unsigned int c,count;

		dest = tempBuffer;
		src = fBuffer;

		while (size > 0) {
			// Get packet header
			c = *src++;

			count = (c & 0x7f) + 1;
			size -= count * pixelSize;
			
			if (c & 0x80) {
				// Rle packet
				do {
					memcpy(dest,src,pixelSize);
					dest += pixelSize;
				} while (--count);
				src += pixelSize;
			} else {
				// Raw packet
				count *= pixelSize;
				memcpy(dest,src,count);
				src += count;
				dest += count;
			}
		}
		
		src = tempBuffer;
	} else {
		src = fBuffer;
	}

	src += (header.bpp / 8) * width * (height - 1);	

	switch(header.bpp) {
	case 8:
		if (palLength > 0){
			format = FORMAT_RGB8;
			dest = pixels = new unsigned char[width * height * 3];
			for (y = 0; y < height; y++){
				for (x = 0; x < width; x++){
					tempPixel = 3 * (*src++);
					*dest++ = palette[tempPixel + 2];
					*dest++ = palette[tempPixel + 1];
					*dest++ = palette[tempPixel];
				}
				src -= 2 * width;
			}
		} else {
			format = FORMAT_I8;
			dest = pixels = new unsigned char[width * height];
			for (y = 0; y < height; y++){
				memcpy(dest, src, width);
				dest += width;
				src -= width;
			}
		}
		break;
	case 16:
		format = FORMAT_RGBA8;
		dest = pixels = new unsigned char[width * height * 4];
		for (y = 0; y < height; y++){
			for (x = 0; x < width; x++){
				tempPixel = *((unsigned short *) src);

				dest[0] = ((tempPixel >> 10) & 0x1F) << 3;
				dest[1] = ((tempPixel >>  5) & 0x1F) << 3;
				dest[2] = ((tempPixel      ) & 0x1F) << 3;
				dest[3] = ((tempPixel >> 15) ? 0xFF : 0);
				dest += 4;
				src += 2;
			}
			src -= 4 * width;
		}
		break;
	case 24:
		format = FORMAT_RGB8;
		dest = pixels = new unsigned char[width * height * 3];
		for (y = 0; y < height; y++){
			for (x = 0; x < width; x++){
				*dest++ = src[2];
				*dest++ = src[1];
				*dest++ = src[0];
				src += 3;
			}
			src -= 6 * width;
		}
		break;
	case 32:
		format = FORMAT_RGBA8;
		dest = pixels = new unsigned char[width * height * 4];
		for (y = 0; y < height; y++){
			for (x = 0; x < width; x++){
				*dest++ = src[2];
				*dest++ = src[1];
				*dest++ = src[0];
				*dest++ = src[3];
				src += 4;
			}
			src -= 8 * width;
		}
		break;
	}

	delete [] tempBuffer;
	delete [] fBuffer;
	return true;
}
#endif

#ifndef NO_BMP
bool Image::loadBMP(const char *fileName){
	BMPHeader header;

	FILE *file;
	unsigned char *dest, *src, *tmp;
	int i, len;
	unsigned char palette[1024];

	if ((file = fopen(fileName, "rb")) == NULL) return false;
	
	// Read the header
	fread(&header, sizeof(header), 1, file);
	if (header.bmpIdentifier != MCHAR2('B', 'M')){
		fclose(file);
		return false;
	}

	width  = header.width;
	height = header.height;
	depth  = 1;
	nMipMaps = 1;
	arraySize = 1;

	switch (header.bpp){
	case 8:
		if (header.compression){
			// No support for RLE compressed bitmaps
			fclose(file);
			return false;
		}
		format = FORMAT_RGB8;
		pixels = new unsigned char[width * height * 3];

		fread(palette, sizeof(palette), 1, file);
		swapChannels(palette, 256, 4, 0, 2);

		// Normal unencoded 8 bit paletted bitmap
		tmp = new unsigned char[width];
		for (i = height - 1; i >= 0; i--){
			dest = pixels + i * width * 3;
			fread(tmp, width, 1, file);
			len = width;
			do {
				src = palette + ((*tmp++) << 2);
				*dest++ = *src++;
				*dest++ = *src++;
				*dest++ = *src++;
			} while (--len);
			tmp -= width;
		}
		delete [] tmp;
		break;
	case 24:
	case 32:
		int nChannels;
		nChannels = (header.bpp == 24)? 3 : 4;
		format    = (header.bpp == 24)? FORMAT_RGB8 : FORMAT_RGBA8;
		pixels = new unsigned char[width * height * nChannels];
		for (i = height - 1; i >= 0; i--){
			dest = pixels + i * width * nChannels;
			fread(dest, width * nChannels, 1, file);
			swapChannels(dest, width, nChannels, 0, 2);
		}
		break;
	default:
		fclose(file);
		return false;
	}

	fclose(file);

	return true;
}
#endif // NO_BMP

#ifndef NO_PCX
bool Image::loadPCX(const char *fileName){
	PCXHeader header;

	FILE *file;
	int size, bpp, i, x,y,n, length, len, col;
	unsigned char *fBuffer, *src, *palette;

	if ((file = fopen(fileName, "rb")) == NULL) return false;
	
	// Find file size
	fseek(file, 0, SEEK_END);
	size = ftell(file) - sizeof(header);
	fseek(file, 0, SEEK_SET);

	// Read the header
	fread(&header, sizeof(header), 1, file);
	
	width    = header.width  + 1;
	height   = header.height + 1;
	depth    = 1;
	format   = FORMAT_RGB8;
	nMipMaps = 1;
	arraySize = 1;

	bpp = header.nChannels * header.bitsPerChannel;

	pixels = new unsigned char[width * height * 3];
	fBuffer = new unsigned char[size];
	fread(fBuffer, size, 1, file);
	fclose(file);

	src = fBuffer;

	switch(bpp){
	case 8:
		palette = fBuffer + size - 768;
		len = width * height * 3;
		i = 0;
		do {
			if (*src < 192){
				col = 3 * (*src++);
				pixels[i++] = palette[col];
				pixels[i++] = palette[col + 1];
				pixels[i++] = palette[col + 2];
			} else {
				length = (*src++) - 192;
				col = 3 * (*src++);
				do {
					pixels[i++] = palette[col];
					pixels[i++] = palette[col + 1];
					pixels[i++] = palette[col + 2];
				} while (--length);
			}
		} while (i < len);
		break;
	case 24:
		len = width * height * 3;
		i = 0;

		x = width;
		y = n = 0;

		do {
			if (x == 0){
				x = width;
				if (n == 2){
					if (++y == height) break;
					n = 0;
				} else n++;
				i = y * width * 3 + n;
			}

			if (*src < 192){
				pixels[i += 3] = *src++;
				x--;
			} else {
				length = (*src++) - 192;
				x -= length;
				do {
					pixels[i += 3] = *src;
				} while (--length);
				src++;
			}

		} while (true);
		break;
	default:
		delete [] fBuffer;
		return false;
	}

	delete [] fBuffer;

	return true;
}
#endif // NO_PCX

bool Image::loadImage(const char *fileName, uint flags){
	const char *extension = strrchr(fileName, '.');

	clear();

	if (extension == NULL) return false;

	if (stricmp(extension, ".htex") == 0){
		if (!loadHTEX(fileName)) return false;
	} else if (stricmp(extension, ".dds") == 0){
		if (!loadDDS(fileName, flags)) return false;
	} else
#ifndef NO_HDR
	if (stricmp(extension, ".hdr") == 0){
		if (!loadHDR(fileName)) return false;
	}
#endif // NO_HDR
#ifndef NO_JPEG
	else if (stricmp(extension, ".jpg") == 0 || 
			 stricmp(extension, ".jpeg") == 0){
		if (!loadJPEG(fileName)) return false;
	}
#endif // NO_JPEG
#ifndef NO_PNG
	else if (stricmp(extension, ".png") == 0){
		if (!loadPNG(fileName)) return false;
	}
#endif // NO_PNG
#ifndef NO_TGA
	else if (stricmp(extension, ".tga") == 0){
		if (!loadTGA(fileName)) return false;
	}
#endif // NO_TGA
#ifndef NO_BMP
	else if (stricmp(extension, ".bmp") == 0){
		if (!loadBMP(fileName)) return false;
	}
#endif // NO_BMP
#ifndef NO_PCX
	else if (stricmp(extension, ".pcx") == 0){
		if (!loadPCX(fileName)) return false;
	}
#endif // NO_PCX
	else {
		return false;
	}
	return true;
}

bool Image::loadSlicedImage(const char **fileNames, const int nImages, const int nArraySlices, uint flags){
	int maxImage = nImages? nImages : 6;

	Image *images = new Image[maxImage * nArraySlices];

	for (int i = 0; i < maxImage * nArraySlices; i++){
		if (!images[i].loadImage(fileNames[i], flags)){
			delete [] images;
			return false;
		}
	}

	uint nMipMaps = images[0].nMipMaps;
	ubyte *dest = create(images[0].format, images[0].width, images[0].height, nImages, nMipMaps, nArraySlices);

	for (int arraySlice = 0; arraySlice < nArraySlices; arraySlice++){
		int base = arraySlice * maxImage;

		for (uint level = 0; level < nMipMaps; level++){
			int size = images[0].getMipMappedSize(level, 1);
			for (int i = 0; i < maxImage; i++){
				memcpy(dest, images[base + i].getPixels(level), size);
				dest += size;
			}
		}
	}

	delete [] images;

	return true;
}

bool Image::saveHTEX(const char *fileName){
	if (!(isPlainFormat(format) || format == FORMAT_DXT1 || format == FORMAT_DXT5 || format == FORMAT_ATI1N || format == FORMAT_ATI2N)) return false;

	FILE *file = fopen(fileName, "wb");
	if (file == NULL) return false;


	HTexHeader header;
	header.identifier = MCHAR4('H','T','E','X');
	header.version = 2;

	header.format = 0;
	strcpy((char *) &header.format, getFormatString(format));

	header.width  = width;
	header.height = height;
	header.depth  = depth;
	header.nMipMaps = nMipMaps;

	HTexHeader2 header2;
	header2.flags = 0;
	header2.nExtraData = nExtraData;

	fwrite(&header,  sizeof(header),  1, file);
	fwrite(&header2, sizeof(header2), 1, file);


	if (isPlainFormat(format)){
		/*int nChannels = getChannelCount(format);
		int bpc = getBytesPerChannel(format);

		for (int ch = 0; ch < nChannels; ch++){
			for (int level = 0; level < nMipMaps; level++){
				int nPixels = getPixelCount(level, 1);

				ubyte *src = getPixels(level) + ch * bpc;
				for (int i = 0; i < nPixels; i++){
					fwrite(src + i * nChannels * bpc, bpc, 1, file);
				}
			}
		}*/
		fwrite(pixels, getMipMappedSize(0, nMipMaps), 1, file);

	} else if (isCompressedFormat(format)){
		if (format == FORMAT_DXT1 || format == FORMAT_DXT5){
			char shift = (format == FORMAT_DXT1)? 3 : 4;

			for (int level = 0; level < nMipMaps; level++){
				ubyte *src = getPixels(level);
				if (format != FORMAT_DXT1) src += 8;
				int w = getWidth(level);
				int h = getHeight(level);
				int d = getDepth(level);

				int nBlocks = d * ((w + 3) >> 2) * ((h + 3) >> 2);

				for (int n = 0; n < nBlocks; n++){
					fwrite(src + (n << shift), 1, 2, file);
				}

				for (int n = 0; n < nBlocks; n++){
					fwrite(src + (n << shift) + 2, 1, 2, file);
				}

				for (int n = 0; n < nBlocks; n++){
					fwrite(src + (n << shift) + 4, 1, 4, file);
				}
			}
		}

		if (format == FORMAT_DXT5 || format == FORMAT_ATI1N || format == FORMAT_ATI2N){
			char shift = (format != FORMAT_DXT5)? 3 : 4;

			for (int level = 0; level < nMipMaps; level++){
				ubyte *src = getPixels(level);
				int w = getWidth(level);
				int h = getHeight(level);
				int d = getDepth(level);

				int nBlocks = d * ((w + 3) >> 2) * ((h + 3) >> 2);

				for (int n = 0; n < nBlocks; n++){
					fwrite(src + (n << shift), 1, 1, file);
				}
				if (format == FORMAT_ATI2N){
					for (int n = 0; n < nBlocks; n++){
						fwrite(src + 8 * (nBlocks + n), 1, 1, file);
					}
				}

				for (int n = 0; n < nBlocks; n++){
					fwrite(src + (n << shift) + 1, 1, 1, file);
				}
				if (format == FORMAT_ATI2N){
					for (int n = 0; n < nBlocks; n++){
						fwrite(src + 8 * (nBlocks + n) + 1, 1, 1, file);
					}
				}

				for (int n = 0; n < nBlocks; n++){
					fwrite(src + (n << shift) + 2, 1, 6, file);
				}
				if (format == FORMAT_ATI2N){
					for (int n = 0; n < nBlocks; n++){
						fwrite(src + 8 * (nBlocks + n) + 2, 1, 6, file);
					}
				}
			}
		}
	}

	if (nExtraData > 0){
		fwrite(extraData, 1, nExtraData, file);
	}

	fclose(file);

	return true;
}

bool Image::saveDDS(const char *fileName){
	// Set up the header
	DDSHeader header;
	memset(&header, 0, sizeof(header));
	DDSHeaderDX10 headerDX10;
	memset(&headerDX10, 0, sizeof(headerDX10));

	header.dwMagic = MCHAR4('D', 'D', 'S', ' ');
	header.dwSize = 124;
	header.dwFlags = DDSD_CAPS | DDSD_PIXELFORMAT | DDSD_WIDTH | DDSD_HEIGHT | (nMipMaps > 1? DDSD_MIPMAPCOUNT : 0) | (depth > 1? DDSD_DEPTH : 0);
	header.dwHeight = height;
	header.dwWidth  = width;
	header.dwPitchOrLinearSize = 0;
	header.dwDepth = (depth > 1)? depth : 0;
	header.dwMipMapCount = (nMipMaps > 1)? nMipMaps : 0;

	int nChannels = getChannelCount(format);

	header.ddpfPixelFormat.dwSize = 32;
	if (format <= FORMAT_I16 || format == FORMAT_RGB10A2){
		header.ddpfPixelFormat.dwFlags = ((nChannels < 3)? 0x00020000 : DDPF_RGB) | ((nChannels & 1)? 0 : DDPF_ALPHAPIXELS);
		if (format <= FORMAT_RGBA8){
			header.ddpfPixelFormat.dwRGBBitCount = 8 * nChannels;
			header.ddpfPixelFormat.dwRBitMask = (nChannels > 2)? 0x00FF0000 : 0xFF;
			header.ddpfPixelFormat.dwGBitMask = (nChannels > 1)? 0x0000FF00 : 0;
			header.ddpfPixelFormat.dwBBitMask = (nChannels > 1)? 0x000000FF : 0;
			header.ddpfPixelFormat.dwRGBAlphaBitMask = (nChannels == 4)? 0xFF000000 : (nChannels == 2)? 0xFF00 : 0;
		} else if (format == FORMAT_I16){
			header.ddpfPixelFormat.dwRGBBitCount = 16;
			header.ddpfPixelFormat.dwRBitMask = 0xFFFF;
		} else {
			header.ddpfPixelFormat.dwRGBBitCount = 32;
			header.ddpfPixelFormat.dwRBitMask = 0x3FF00000;
			header.ddpfPixelFormat.dwGBitMask = 0x000FFC00;
			header.ddpfPixelFormat.dwBBitMask = 0x000003FF;
			header.ddpfPixelFormat.dwRGBAlphaBitMask = 0xC0000000;
		}
	} else {
		header.ddpfPixelFormat.dwFlags = DDPF_FOURCC;

		switch (format){
			case FORMAT_RG16:    header.ddpfPixelFormat.dwFourCC = 34; break;
			case FORMAT_RGBA16:  header.ddpfPixelFormat.dwFourCC = 36; break;
			case FORMAT_R16F:    header.ddpfPixelFormat.dwFourCC = 111; break;
			case FORMAT_RG16F:   header.ddpfPixelFormat.dwFourCC = 112; break;
			case FORMAT_RGBA16F: header.ddpfPixelFormat.dwFourCC = 113; break;
			case FORMAT_R32F:    header.ddpfPixelFormat.dwFourCC = 114; break;
			case FORMAT_RG32F:   header.ddpfPixelFormat.dwFourCC = 115; break;
			case FORMAT_RGBA32F: header.ddpfPixelFormat.dwFourCC = 116; break;
			case FORMAT_DXT1:    header.ddpfPixelFormat.dwFourCC = MCHAR4('D','X','T','1'); break;
			case FORMAT_DXT3:    header.ddpfPixelFormat.dwFourCC = MCHAR4('D','X','T','3'); break;
			case FORMAT_DXT5:    header.ddpfPixelFormat.dwFourCC = MCHAR4('D','X','T','5'); break;
			case FORMAT_ATI1N:   header.ddpfPixelFormat.dwFourCC = MCHAR4('A','T','I','1'); break;
			case FORMAT_ATI2N:   header.ddpfPixelFormat.dwFourCC = MCHAR4('A','T','I','2'); break;
			default:
				header.ddpfPixelFormat.dwFourCC = MCHAR4('D','X','1','0');
				headerDX10.arraySize = 1;
				headerDX10.miscFlag = (depth == 0)? D3D10_RESOURCE_MISC_TEXTURECUBE : 0;
				headerDX10.resourceDimension = is1D()? D3D10_RESOURCE_DIMENSION_TEXTURE1D : is3D()? D3D10_RESOURCE_DIMENSION_TEXTURE3D : D3D10_RESOURCE_DIMENSION_TEXTURE2D;
				switch (format){
					//case FORMAT_RGBA8:    headerDX10.dxgiFormat = 28; break;
					case FORMAT_RGB32F:   headerDX10.dxgiFormat = 6; break;
					case FORMAT_RGB9E5:   headerDX10.dxgiFormat = 67; break;
					case FORMAT_RG11B10F: headerDX10.dxgiFormat = 26; break;
					default:
						return false;
				}
		}
	}

	header.ddsCaps.dwCaps1 = DDSCAPS_TEXTURE | (nMipMaps > 1? DDSCAPS_MIPMAP | DDSCAPS_COMPLEX : 0) | (depth != 1? DDSCAPS_COMPLEX : 0);
	header.ddsCaps.dwCaps2 = (depth > 1)? DDSCAPS2_VOLUME : (depth == 0)? DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_ALL_FACES : 0;
	header.ddsCaps.Reserved[0] = 0;
	header.ddsCaps.Reserved[1] = 0;
	header.dwReserved2 = 0;

	FILE *file;
	if ((file = fopen(fileName, "wb")) == NULL) return false;

	fwrite(&header, sizeof(header), 1, file);
	if (headerDX10.dxgiFormat) fwrite(&headerDX10, sizeof(headerDX10), 1, file);


	int size = getMipMappedSize(0, nMipMaps);

	// RGB to BGR
	if (format == FORMAT_RGB8 || format == FORMAT_RGBA8) swapChannels(pixels, size / nChannels, nChannels, 0, 2);

	if (isCube()){
		for (int face = 0; face < 6; face++){
			for (int mipMapLevel = 0; mipMapLevel < nMipMaps; mipMapLevel++){
				int faceSize = getMipMappedSize(mipMapLevel, 1) / 6;
                ubyte *src = getPixels(mipMapLevel) + face * faceSize;
				fwrite(src, 1, faceSize, file);
			}
		}
	} else {
		fwrite(pixels, size, 1, file);
	}
	fclose(file);

	// Restore to RGB
	if (format == FORMAT_RGB8 || format == FORMAT_RGBA8) swapChannels(pixels, size / nChannels, nChannels, 0, 2);

	return true;
}

#ifndef NO_HDR

bool Image::saveHDR(const char *fileName){
	if (format != FORMAT_RGBE8 && format != FORMAT_RGB32F) return false;

	FILE *file = fopen(fileName, "wb");
	if (file == NULL) return false;

	fprintf(file, "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y %d +X %d\n", height, width);

	int nPixels = width * height;

	if (format == FORMAT_RGBE8){
		fwrite(pixels, sizeof(uint32), nPixels, file);
	} else {

		vec3 *src = (vec3 *) pixels;
		uint32 *dst = new uint32[nPixels];
		for (int i = 0; i < nPixels; i++){
			dst[i] = rgbToRGBE8(src[i]);
		}

		fwrite(dst, sizeof(uint32), nPixels, file);

		delete [] dst;
	}

	fclose(file);
	return true;
}

#endif

#ifndef NO_JPEG
bool Image::saveJPEG(const char *fileName, const int quality){
	if (format != FORMAT_I8 && format != FORMAT_RGB8) return false;

	jpeg_compress_struct cinfo;
	jpeg_error_mgr jerr;

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);

	FILE *file;
	if ((file = fopen(fileName, "wb")) == NULL) return false;

	int nChannels = getChannelCount(format);

	cinfo.in_color_space = (nChannels == 1)? JCS_GRAYSCALE : JCS_RGB;
	jpeg_set_defaults(&cinfo);

	cinfo.input_components = nChannels;
	cinfo.num_components   = nChannels;
	cinfo.image_width  = width;
	cinfo.image_height = height;
	cinfo.data_precision = 8;
	cinfo.input_gamma = 1.0;

	jpeg_set_quality(&cinfo, quality, FALSE);

	jpeg_stdio_dest(&cinfo, file);
	jpeg_start_compress(&cinfo, TRUE);

	unsigned char *curr_scanline = pixels;

	for (int y = 0; y < height; y++){
		jpeg_write_scanlines(&cinfo, &curr_scanline, 1);
		curr_scanline += nChannels * width;
	}

	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);

	fclose(file);

	return true;
}
#endif // NO_JPEG


#ifndef NO_PNG
bool Image::savePNG(const char *fileName){
	int type;

	switch (format){
		case FORMAT_I8:
		case FORMAT_I16:
			type = PNG_COLOR_TYPE_GRAY;
			break;
		case FORMAT_IA8:
		case FORMAT_IA16:
			type = PNG_COLOR_TYPE_GRAY_ALPHA;
			break;
		case FORMAT_RGB8:
		case FORMAT_RGB16:
			type = PNG_COLOR_TYPE_RGB;
			break;
		case FORMAT_RGBA8:
		case FORMAT_RGBA16:
			type = PNG_COLOR_TYPE_RGBA;
			break;
		default:
			return false;
	}

	png_structp png_ptr;
	png_infop info_ptr;
	FILE *file;

    if ((file = fopen(fileName, "wb")) == NULL) return false;

    if ((png_ptr = png_create_write_struct_2(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL, NULL, malloc_fn, free_fn)) == NULL){
		fclose(file);
        return false;
    }

    if ((info_ptr = png_create_info_struct(png_ptr)) == NULL){
        png_destroy_write_struct(&png_ptr, NULL);
		fclose(file);
        return false;
    }

	png_set_write_fn(png_ptr, file, user_write_data, user_flush_data);

	int bpp = (format >= FORMAT_I16)? 16 : 8;

	png_set_IHDR(png_ptr, info_ptr, width, height, bpp, type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	//png_set_compression_level(png_ptr, 9);
	png_write_info(png_ptr, info_ptr);

	int nElements = width * getChannelCount(format);
	if (format >= FORMAT_I16){
		ushort *line = new ushort[nElements];
		for (int y = 0; y < height; y++){
			ushort *src = ((ushort *) pixels) + y * nElements;
			// Swap endian
			for (int x = 0; x < nElements; x++){
				line[x] = (src[x] >> 8) | (src[x] << 8);
			}
			png_write_row(png_ptr, (ubyte *) line);
		}
		delete [] line;
	} else {
		for (int y = 0; y < height; y++){
			png_write_row(png_ptr, pixels + y * nElements);
		}
	}

	png_write_end(png_ptr, info_ptr);
	
    png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(file);

	return true;
}
#endif // NO_PNG


#ifndef NO_TGA
bool Image::saveTGA(const char *fileName){
	if (format != FORMAT_I8 && format != FORMAT_RGB8 && format != FORMAT_RGBA8) return false;

	FILE *file;
	if ((file = fopen(fileName, "wb")) == NULL) return false;

	int nChannels = getChannelCount(format);

	TGAHeader header = {
		0x00,
		(format == FORMAT_I8)? 1 : 0,
		(format == FORMAT_I8)? 1 : 2,
		0x0000,
		(format == FORMAT_I8)? 256 : 0,
		(format == FORMAT_I8)? 24  : 0,
		0x0000,
		0x0000,
		width,
		height,
		nChannels * 8,
		0x00
	};

	fwrite(&header, sizeof(header), 1, file);

	ubyte *dest, *src, *buffer;

	if (format == FORMAT_I8){
		ubyte pal[768];
		int p = 0;
		for (int i = 0; i < 256; i++){
			pal[p++] = i;
			pal[p++] = i;
			pal[p++] = i;
		}
		fwrite(pal, sizeof(pal), 1, file);

		src = pixels + width * height;
		for (int y = 0; y < height; y++){
			src -= width;
			fwrite(src, width, 1, file);
		}

	} else {
		bool useAlpha = (nChannels == 4);
		int lineLength = width * (useAlpha? 4 : 3);

		buffer = new ubyte[height * lineLength];
		int len;

		for (int y = 0; y < height; y++){
			dest = buffer + (height - y - 1) * lineLength;
			src  = pixels + y * width * nChannels;
			len = width;
			do {
				*dest++ = src[2];
				*dest++ = src[1];
				*dest++ = src[0];
				if (useAlpha) *dest++ = src[3];
				src += nChannels;
			} while (--len);
		}

		fwrite(buffer, height * lineLength, 1, file);
		delete [] buffer;
	}

	fclose(file);

	return true;
}
#endif // NO_TGA

#ifndef NO_BMP
bool Image::saveBMP(const char *fileName){
	if (format != FORMAT_I8 && format != FORMAT_RGB8 && format != FORMAT_RGBA8) return false;

	FILE *file;
	if ((file = fopen(fileName, "wb")) == NULL) return false;

	int nChannels = getChannelCount(format);
	BMPHeader header = {
		MCHAR2('B','M'), 
		{0x36, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00},
		width,
		height,
		0x0001,
		nChannels * 8,
		0,
		{0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x13, 0x0b, 0x00, 0x00, 0x13, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	};

	fwrite(&header, sizeof(header), 1, file);

	if (format == FORMAT_I8){
		unsigned char pal[1024];
		for (int i = 0; i < 1024; i++){
			pal[i] = i >> 2;
		}
		fwrite(pal, sizeof(pal), 1, file);

		unsigned char *src = pixels + width * height;
		for (int y = 0; y < height; y++){
			src -= width;
			fwrite(src, width, 1, file);
		}
	} else {
		bool useAlpha = (nChannels == 4);
		int paddBytes = (width * nChannels) & 3;
		if (paddBytes) paddBytes = 4 - paddBytes;

		int len, size = (width * nChannels + paddBytes) * height;
		unsigned char *dest, *src, *buffer = new unsigned char[size];

		for (int y = 0; y < height; y++){
			dest = buffer + (height - y - 1) * (width * nChannels + paddBytes);
			src  = pixels + y * width * nChannels;
			len = width;
			do {
				*dest++ = src[2];
				*dest++ = src[1];
				*dest++ = src[0];
				if (useAlpha) *dest++ = src[3];
				src += nChannels;
			} while (--len);
		}
		fwrite(buffer, size, 1, file);
		delete [] buffer;
	}

	fclose(file);

	return true;
}
#endif // NO_BMP

#ifndef NO_PCX
bool Image::savePCX(const char *fileName){
	if (format != FORMAT_I8 && format != FORMAT_RGB8 && format != FORMAT_RGBA8) return false;

	int nChannels = getChannelCount(format);
	int destChannels = (format == FORMAT_RGBA8)? 3 : nChannels;


	PCXHeader header = {
		{0x0a, 0x05, 0x01},
		8,
		{0x00, 0x00, 0x00, 0x00}, 
		width  - 1,
		height - 1,
		{0x48, 0x00, 0x48, 0x00, 0x0f, 0x0f, 0x0f, 0x0e, 0x0e, 0x0e,
		 0x0d, 0x0d, 0x0d, 0x0c, 0x0c, 0x0c, 0x0b, 0x0b, 0x0b, 0x0a,
		 0x0a, 0x0a, 0x09, 0x09, 0x09, 0x08, 0x08, 0x08, 0x07, 0x07,
		 0x07, 0x06, 0x06, 0x06, 0x05, 0x05, 0x05, 0x04, 0x04, 0x04,
		 0x03, 0x03, 0x03, 0x02, 0x02, 0x02, 0x01, 0x01, 0x01, 0x00,
		 0x00, 0x00, 0x00},
		destChannels, // 3 channels
		width,
		{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	};

	FILE *file;
	if ((file = fopen(fileName, "wb")) == NULL) return false;

	fwrite(&header, sizeof(header), 1, file);


	unsigned char *src, *buffer = new unsigned char[width * 2];
	int pos;

	for (int y = 0; y < height; y++){
		for (int i = 0; i < destChannels; i++){
			src = pixels + y * width * nChannels + i;
			pos = 0;

			unsigned int count;
			int prevx, x = 0;
			unsigned char curr = *src;

			do {
				prevx = x;
				curr = src[x * nChannels];
				do {
					x++;
				} while (x < width && src[x * nChannels] == curr);

				count = x - prevx;
				if (count == 1){
					if (curr < 192){
						buffer[pos++] = curr;
					} else {
						buffer[pos++] = 193;
						buffer[pos++] = curr;
					}
				} else {
					do {
						unsigned char len = (count > 63)? 63 : count;
						
						buffer[pos++] = 192 + len;
						buffer[pos++] = curr;
						
						count -= len;
					} while (count);
				}

			} while (x < width);

			fwrite(buffer, pos, 1, file);			
		}
	}

	delete [] buffer;

	if (format == FORMAT_I8){
		unsigned char pal[768];
		int p = 0;
		for (int i = 0; i < 256; i++){
			pal[p++] = i;
			pal[p++] = i;
			pal[p++] = i;
		}
		fwrite(pal, sizeof(pal), 1, file);
	}

	fclose(file);

	return true;
}
#endif // NO_PCX

bool Image::saveImage(const char *fileName){
	const char *extension = strrchr(fileName, '.');

	if (extension != NULL){
		if (stricmp(extension, ".htex") == 0){
			return saveHTEX(fileName);
		} else if (stricmp(extension, ".dds") == 0){
			return saveDDS(fileName);
		}
#ifndef NO_JPEG
		else if (stricmp(extension, ".jpg") == 0 || 
                   stricmp(extension, ".jpeg") == 0){
			return saveJPEG(fileName, 75);
		}
#endif // NO_JPEG
#ifndef NO_PNG
		else if (stricmp(extension, ".png") == 0){
			return savePNG(fileName);
		}
#endif // NO_PNG
#ifndef NO_TGA
		else if (stricmp(extension, ".tga") == 0){
			return saveTGA(fileName);
		}
#endif // NO_TGA
#ifndef NO_BMP
		else if (stricmp(extension, ".bmp") == 0){
			return saveBMP(fileName);
		}
#endif // NO_BMP
#ifndef NO_PCX
		else if (stricmp(extension, ".pcx") == 0){
			return savePCX(fileName);
		}
#endif // NO_PCX
	}
	return false;
}

void Image::loadFromMemory(void *mem, const FORMAT frmt, const int w, const int h, const int d, const int mipMapCount, bool ownsMemory){
	free();

	width  = w;
	height = h;
	depth  = d;
    format = frmt;
	nMipMaps = mipMapCount;
	arraySize = 1;

	if (ownsMemory){
		pixels = (unsigned char *) mem;
	} else {
		int size = getMipMappedSize(0, nMipMaps);
		pixels = new unsigned char[size];
		memcpy(pixels, mem, size);
	}
}


template <typename DATA_TYPE>
void buildMipMap(DATA_TYPE *dst, const DATA_TYPE *src, const uint w, const uint h, const uint d, const uint c){
	uint xOff = (w < 2)? 0 : c;
	uint yOff = (h < 2)? 0 : c * w;
	uint zOff = (d < 2)? 0 : c * w * h;

	for (uint z = 0; z < d; z += 2){
		for (uint y = 0; y < h; y += 2){
			for (uint x = 0; x < w; x += 2){
				for (uint i = 0; i < c; i++){
					*dst++ = (src[0] + src[xOff] + src[yOff] + src[yOff + xOff] + src[zOff] + src[zOff + xOff] + src[zOff + yOff] + src[zOff + yOff + xOff]) / 8;
					src++;
				}
				src += xOff;
			}
			src += yOff;
		}
		src += zOff;
	}
}

bool Image::createMipMaps(const int mipMaps){
	if (isCompressedFormat(format)) return false;
	if (!isPowerOf2(width) || !isPowerOf2(height) || !isPowerOf2(depth)) return false;

	int actualMipMaps = min(mipMaps, getMipMapCountFromDimesions());

	if (nMipMaps != actualMipMaps){
		int size = getMipMappedSize(0, actualMipMaps);
		if (arraySize > 1){
			ubyte *newPixels = new ubyte[size * arraySize];

			// Copy top mipmap of all array slices to new location
			int firstMipSize = getMipMappedSize(0, 1);
			int oldSize = getMipMappedSize(0, nMipMaps);

			for (int i = 0; i < arraySize; i++){
				memcpy(newPixels + i * size, pixels + i * oldSize, firstMipSize);
			}

			delete [] pixels;
			pixels = newPixels;
		} else {
			pixels = (ubyte *) realloc(pixels, size);
		}
		nMipMaps = actualMipMaps;
	}

	int nChannels = getChannelCount(format);


	int n = isCube()? 6 : 1;

	for (int arraySlice = 0; arraySlice < arraySize; arraySlice++){
		ubyte *src = getPixels(0, arraySlice);
		ubyte *dst = getPixels(1, arraySlice);

		for (int level = 1; level < nMipMaps; level++){
			int w = getWidth (level - 1);
			int h = getHeight(level - 1);
			int d = getDepth (level - 1);

			int srcSize = getMipMappedSize(level - 1, 1) / n;
			int dstSize = getMipMappedSize(level,     1) / n;

			for (int i = 0; i < n; i++){
				if (isPlainFormat(format)){
					if (isFloatFormat(format)){
						buildMipMap((float *) dst, (float *) src, w, h, d, nChannels);
					} else if (format >= FORMAT_I16){
						buildMipMap((ushort *) dst, (ushort *) src, w, h, d, nChannels);
					} else {
						buildMipMap(dst, src, w, h, d, nChannels);
					}
				}
				src += srcSize;
				dst += dstSize;
			}
		}
	}

	return true;
}

bool Image::removeMipMaps(const int firstMipMap, int mipMapsToSave){
	if (arraySize > 1) return false; // TODO: Implement ...
	if (firstMipMap > nMipMaps) return false;

	int mipMapCount = min(firstMipMap + mipMapsToSave, nMipMaps) - firstMipMap;

	int size = getMipMappedSize(firstMipMap, mipMapCount);
	unsigned char *newPixels = new unsigned char[size];

	memcpy(newPixels, getPixels(firstMipMap), size);

	delete [] pixels;
	pixels = newPixels;
	width = getWidth(firstMipMap);
	height = getHeight(firstMipMap);
	depth = depth? getDepth(firstMipMap) : 0;
	nMipMaps = mipMapCount;

	return true;
}

void decodeColorBlock(unsigned char *dest, int w, int h, int xOff, int yOff, FORMAT format, int red, int blue, unsigned char *src){
	unsigned char colors[4][3];

	uint16 c0 = *(uint16 *) src;
	uint16 c1 = *(uint16 *) (src + 2);
	
	colors[0][0] = ((c0 >> 11) & 0x1F) << 3;
	colors[0][1] = ((c0 >>  5) & 0x3F) << 2;
	colors[0][2] =  (c0        & 0x1F) << 3;
	
	colors[1][0] = ((c1 >> 11) & 0x1F) << 3;
	colors[1][1] = ((c1 >>  5) & 0x3F) << 2;
	colors[1][2] =  (c1        & 0x1F) << 3;

	if (c0 > c1 || format == FORMAT_DXT5){
		for (int i = 0; i < 3; i++){
			colors[2][i] = (2 * colors[0][i] +     colors[1][i] + 1) / 3;
			colors[3][i] = (    colors[0][i] + 2 * colors[1][i] + 1) / 3;
		}
	} else {
		for (int i = 0; i < 3; i++){
			colors[2][i] = (colors[0][i] + colors[1][i] + 1) >> 1;
			colors[3][i] = 0;
		}
	}

	src += 4;
	for (int y = 0; y < h; y++){
		unsigned char *dst = dest + yOff * y;
		unsigned int indexes = src[y];
		for (int x = 0; x < w; x++){
			unsigned int index = indexes & 0x3;
			dst[red]  = colors[index][0];
			dst[1]    = colors[index][1];
			dst[blue] = colors[index][2];
			indexes >>= 2;

			dst += xOff;
		}
	}
}

void decodeDXT3AlphaBlock(unsigned char *dest, int w, int h, int xOff, int yOff, unsigned char *src){
	for (int y = 0; y < h; y++){
		unsigned char *dst = dest + yOff * y;
		unsigned int alpha = ((unsigned short *) src)[y];
		for (int x = 0; x < w; x++){
			*dst = (alpha & 0xF) * 17;
			alpha >>= 4;
			dst += xOff;
		}
	}
}

void decodeDXT5AlphaBlock(unsigned char *dest, int w, int h, int xOff, int yOff, unsigned char *src){
	unsigned char a0 = src[0];
	unsigned char a1 = src[1];
	uint64 alpha = (*(uint64 *) src) >> 16;

	for (int y = 0; y < h; y++){
		unsigned char *dst = dest + yOff * y;
		for (int x = 0; x < w; x++){
			int k = ((unsigned int) alpha) & 0x7;
			if (k == 0){
				*dst = a0;
			} else if (k == 1){
				*dst = a1;
			} else if (a0 > a1){
				*dst = ((8 - k) * a0 + (k - 1) * a1) / 7;
			} else if (k >= 6){
				*dst = (k == 6)? 0 : 255;
			} else {
				*dst = ((6 - k) * a0 + (k - 1) * a1) / 5;
			}
			alpha >>= 3;

			dst += xOff;
		}
		if (w < 4) alpha >>= (3 * (4 - w));
	}
}

void decodeCompressedImage(unsigned char *dest, unsigned char *src, const int width, const int height, const FORMAT format){
	int sx = (width  < 4)? width  : 4;
	int sy = (height < 4)? height : 4;

	int nChannels = getChannelCount(format);

	for (int y = 0; y < height; y += 4){
		for (int x = 0; x < width; x += 4){
			unsigned char *dst = dest + (y * width + x) * nChannels;
			if (format == FORMAT_DXT3){
				decodeDXT3AlphaBlock(dst + 3, sx, sy, nChannels, width * nChannels, src);
				src += 8;
			} else if (format == FORMAT_DXT5){
				decodeDXT5AlphaBlock(dst + 3, sx, sy, nChannels, width * nChannels, src);
				src += 8;
			}
			if (format <= FORMAT_DXT5){
				decodeColorBlock(dst, sx, sy, nChannels, width * nChannels, format, 0, 2, src);
				src += 8;
			} else {
				if (format == FORMAT_ATI1N){
					decodeDXT5AlphaBlock(dst, sx, sy, 1, width, src);
					src += 8;
				} else {
					decodeDXT5AlphaBlock(dst,     sx, sy, 2, width * 2, src + 8);
					decodeDXT5AlphaBlock(dst + 1, sx, sy, 2, width * 2, src);
					src += 16;
				}
			}
		}
	}
}

bool Image::uncompressImage(){
	if (isCompressedFormat(format)){
		FORMAT destFormat;
		if (format >= FORMAT_ATI1N){
			destFormat = (format == FORMAT_ATI1N)? FORMAT_I8 : FORMAT_IA8;
		} else {
			destFormat = (format == FORMAT_DXT1)? FORMAT_RGB8 : FORMAT_RGBA8;
		}

		ubyte *newPixels = new ubyte[getMipMappedSize(0, nMipMaps, destFormat)];

		int level = 0;
		ubyte *src, *dst = newPixels;
		while ((src = getPixels(level)) != NULL){
			int w = getWidth(level);
			int h = getHeight(level);
			int d = (depth == 0)? 6 : getDepth(level);

			int dstSliceSize = getSliceSize(level, destFormat);
			int srcSliceSize = getSliceSize(level, format);

			for (int slice = 0; slice < d; slice++){
				decodeCompressedImage(dst, src, w, h, format);

				dst += dstSliceSize;
				src += srcSliceSize;
			}
			level++;
		}

		format = destFormat;
		
		free();
		pixels = newPixels;
	}

	return true;
}

bool Image::unpackImage(){
	int pixelCount = getPixelCount(0, nMipMaps);

	ubyte *newPixels;
	if (format == FORMAT_RGBE8){
		format = FORMAT_RGB32F;
		newPixels = new unsigned char[getMipMappedSize(0, nMipMaps)];

		for (int i = 0; i < pixelCount; i++){
			((vec3 *) newPixels)[i] = rgbeToRGB(pixels + 4 * i);
		}

	} else if (format == FORMAT_RGB565){
		format = FORMAT_RGB8;
		newPixels = new unsigned char[getMipMappedSize(0, nMipMaps)];

		for (int i = 0; i < pixelCount; i++){
			unsigned int rgb565 = ((unsigned short *) pixels)[i];
			newPixels[3 * i    ] = ((rgb565 >> 11) * 2106) >> 8;
			newPixels[3 * i + 1] = ((rgb565 >> 5) & 0x3F) * 1037 >> 8;
			newPixels[3 * i + 2] = ((rgb565 & 0x1F) * 2106) >> 8;
		}
	} else if (format == FORMAT_RGBA4){
		format = FORMAT_RGBA8;
		newPixels = new unsigned char[getMipMappedSize(0, nMipMaps)];

		for (int i = 0; i < pixelCount; i++){
			newPixels[4 * i    ] = (pixels[2 * i + 1] & 0xF) * 17;
			newPixels[4 * i + 1] = (pixels[2 * i]     >>  4) * 17;
			newPixels[4 * i + 2] = (pixels[2 * i]     & 0xF) * 17;
			newPixels[4 * i + 3] = (pixels[2 * i + 1] >>  4) * 17;
		}
	} else if (format == FORMAT_RGB10A2){
		format = FORMAT_RGBA16;
		newPixels = new unsigned char[getMipMappedSize(0, nMipMaps)];

		for (int i = 0; i < pixelCount; i++){
			uint32 src = ((uint32 *) pixels)[i];
			((ushort *) newPixels)[4 * i    ] = (((src      ) & 0x3FF) * 4198340) >> 16;
			((ushort *) newPixels)[4 * i + 1] = (((src >> 10) & 0x3FF) * 4198340) >> 16;
			((ushort *) newPixels)[4 * i + 2] = (((src >> 20) & 0x3FF) * 4198340) >> 16;
			((ushort *) newPixels)[4 * i + 3] = (((src >> 30) & 0x003) * 21845);
		}
	} else {
		return false;
	}

	delete [] pixels;
	pixels = newPixels;

	return true;
}

bool Image::convert(const FORMAT newFormat){
	ubyte *newPixels;
	uint nPixels = getPixelCount(0, nMipMaps) * arraySize;

	if (format == FORMAT_RGBE8 && (newFormat == FORMAT_RGB32F || newFormat == FORMAT_RGBA32F)){
		newPixels = new ubyte[getMipMappedSize(0, nMipMaps, newFormat) * arraySize];
		float *dest = (float *) newPixels;

		bool writeAlpha = (newFormat == FORMAT_RGBA32F);
		ubyte *src = pixels;
		do {
			*((vec3 *) dest) = rgbeToRGB(src);
			if (writeAlpha){
				dest[3] = 1.0f;
				dest += 4;
			} else {
				dest += 3;
			}
			src += 4;
		} while (--nPixels);

	} else {
		if (!isPlainFormat(format) || !(isPlainFormat(newFormat) || newFormat == FORMAT_RGB10A2 || newFormat == FORMAT_RGBE8 || newFormat == FORMAT_RGB9E5)) return false;
		if (format == newFormat) return true;

		ubyte *src = pixels;
		ubyte *dest = newPixels = new ubyte[getMipMappedSize(0, nMipMaps, newFormat) * arraySize];

		if (format == FORMAT_RGB8 && newFormat == FORMAT_RGBA8){
			// Fast path for RGB->RGBA8
			do {
				dest[0] = src[0];
				dest[1] = src[1];
				dest[2] = src[2];
				dest[3] = 255;
				dest += 4;
				src += 3;
			} while (--nPixels);

		} else {
			int srcSize = getBytesPerPixel(format);
			int nSrcChannels = getChannelCount(format);

			int destSize = getBytesPerPixel(newFormat);
			int nDestChannels = getChannelCount(newFormat);

			do {
				float rgba[4];

				if (isFloatFormat(format)){
					if (format <= FORMAT_RGBA16F){
						for (int i = 0; i < nSrcChannels; i++) rgba[i] = ((half *) src)[i];
					} else {
						for (int i = 0; i < nSrcChannels; i++) rgba[i] = ((float *) src)[i];
					}
				} else if (format >= FORMAT_I16 && format <= FORMAT_RGBA16){
					for (int i = 0; i < nSrcChannels; i++) rgba[i] = ((ushort *) src)[i] * (1.0f / 65535.0f);
				} else {
					for (int i = 0; i < nSrcChannels; i++) rgba[i] = src[i] * (1.0f / 255.0f);
				}
				if (nSrcChannels  < 4) rgba[3] = 1.0f;
				if (nSrcChannels == 1) rgba[2] = rgba[1] = rgba[0];
				
				if (nDestChannels == 1)	rgba[0] = 0.30f * rgba[0] + 0.59f * rgba[1] + 0.11f * rgba[2];

				if (isFloatFormat(newFormat)){
					if (newFormat <= FORMAT_RGBA32F){
						if (newFormat <= FORMAT_RGBA16F){
							for (int i = 0; i < nDestChannels; i++)	((half *) dest)[i] = rgba[i];
						} else {
							for (int i = 0; i < nDestChannels; i++)	((float *) dest)[i] = rgba[i];
						}
					} else {
						if (newFormat == FORMAT_RGBE8){
							*(uint32 *) dest = rgbToRGBE8(vec3(rgba[0], rgba[1], rgba[2]));
						} else {
							*(uint32 *) dest = rgbToRGB9E5(vec3(rgba[0], rgba[1], rgba[2]));
						}
					}
				} else if (newFormat >= FORMAT_I16 && newFormat <= FORMAT_RGBA16){
					for (int i = 0; i < nDestChannels; i++)	((ushort *) dest)[i] = (ushort) (65535 * saturate(rgba[i]) + 0.5f);
				} else if (/*isPackedFormat(newFormat)*/newFormat == FORMAT_RGB10A2){
					*(uint *) dest =
						(uint(1023.0f * saturate(rgba[0]) + 0.5f) << 22) |
						(uint(1023.0f * saturate(rgba[1]) + 0.5f) << 12) |
						(uint(1023.0f * saturate(rgba[2]) + 0.5f) <<  2) |
						(uint(   3.0f * saturate(rgba[3]) + 0.5f));
				} else {
					for (int i = 0; i < nDestChannels; i++)	dest[i] = (unsigned char) (255 * saturate(rgba[i]) + 0.5f);
				}

				src  += srcSize;
				dest += destSize;
			} while (--nPixels);
		}
	}
	delete [] pixels;
	pixels = newPixels;
	format = newFormat;

	return true;
}

bool Image::swap(const int ch0, const int ch1){
	if (!isPlainFormat(format)) return false;

	unsigned int nPixels = getPixelCount(0, nMipMaps) * arraySize;
	unsigned int nChannels = getChannelCount(format);

	if (format <= FORMAT_RGBA8){
		swapChannels((unsigned char *) pixels, nPixels, nChannels, ch0, ch1);
	} else if (format <= FORMAT_RGBA16F){
		swapChannels((unsigned short *) pixels, nPixels, nChannels, ch0, ch1);
	} else {
		swapChannels((float *) pixels, nPixels, nChannels, ch0, ch1);
	}

	return true;
}

bool Image::flipX(){
	// TODO: Implement ...
	return false;
}

bool Image::flipY(){
	if (arraySize > 1) return false; // TODO: Implement ...
	if (isCompressedFormat(format)) return false;

	ubyte *newPixels = new ubyte[getMipMappedSize(0, nMipMaps)];
	ubyte *dst = newPixels;

	int bytes = getBytesPerPixel(format);

	for (int i = 0; i < nMipMaps; i++){
		int w = getWidth(i);
		int h = getHeight(i);

		int lineWidth = w * bytes;

		ubyte *src = getPixels(i) + (h - 1) * lineWidth;

		for (int y = 0; y < h; y++){
			memcpy(dst, src, lineWidth);

			dst += lineWidth;
			src -= lineWidth;
		}
	}


	delete [] pixels;
	pixels = newPixels;

	return true;
}

bool Image::dilate(){
	if (format != FORMAT_R8)
		return false;

	ubyte *newPixels = new ubyte[getMipMappedSize(0, nMipMaps)];
	ubyte *dst = newPixels;

	//int ch = getChannelCount(format);

	for (int i = 0; i < nMipMaps; i++){
		int w = getWidth(i);
		int h = getHeight(i);

		ubyte *src = getPixels(i);

		for (int y = 0; y < h; y++){
			for (int x = 0; x < w; x++){
				int startX = max(x - 1, 0);
				int endX = min(x + 2, w);
				int startY = max(y - 1, 0);
				int endY = min(y + 2, h);

				ubyte max = 0;
				for (int iy = startY; iy < endY; iy++)
				{
					for (int ix = startX; ix < endX; ix++)
					{
						ubyte c = src[iy * w + ix];
						if (c > max)
							max = c;
					}
				}

				*dst++ = max;
			}
		}
	}


	delete [] pixels;
	pixels = newPixels;

	return true;
}

bool Image::erode(){
	if (format != FORMAT_R8)
		return false;

	ubyte *newPixels = new ubyte[getMipMappedSize(0, nMipMaps)];
	ubyte *dst = newPixels;

	//int ch = getChannelCount(format);

	for (int i = 0; i < nMipMaps; i++){
		int w = getWidth(i);
		int h = getHeight(i);

		ubyte *src = getPixels(i);

		for (int y = 0; y < h; y++){
			for (int x = 0; x < w; x++){
				int startX = max(x - 1, 0);
				int endX = min(x + 2, w);
				int startY = max(y - 1, 0);
				int endY = min(y + 2, h);

				ubyte min = 0xFF;
				for (int iy = startY; iy < endY; iy++)
				{
					for (int ix = startX; ix < endX; ix++)
					{
						ubyte c = src[iy * w + ix];
						if (c < min)
							min = c;
					}
				}

				*dst++ = min;
			}
		}
	}


	delete [] pixels;
	pixels = newPixels;

	return true;
}

bool Image::toRGBD16(){
	if (arraySize > 1) return false; // TODO: Implement ...
	if (format != FORMAT_RGB32F) return false;

	uint nPixels = getPixelCount(0, nMipMaps);
	float *src = (float *) pixels;

	ushort *newPixels = new ushort[nPixels * 4];
	for (uint i = 0; i < nPixels; i++){
		float maxChannel = max(max(max(src[3 * i], src[3 * i + 1]), src[3 * i + 2]), 1.0f);

		newPixels[4 * i + 0] = (ushort) (65535 * (src[3 * i + 0] / maxChannel));
		newPixels[4 * i + 1] = (ushort) (65535 * (src[3 * i + 1] / maxChannel));
		newPixels[4 * i + 2] = (ushort) (65535 * (src[3 * i + 2] / maxChannel));
		newPixels[4 * i + 3] = (ushort) (65535 * (1.0f / maxChannel));
	}

	delete [] pixels;
	pixels = (ubyte *) newPixels;
	format = FORMAT_RGBA16;

	return true;
}


// TODO: Take care of black pixels ...
//       Best would be to set RGB to zero and use an exponent that's the average of the surrounding pixels
bool Image::toRGBE16(float &scale, float &bias){
	if (arraySize > 1) return false; // TODO: Implement ...
	if (format != FORMAT_RGB32F) return false;

	uint nPixels = getPixelCount(0, nMipMaps);

	float maxExp = -1e10f;
	float minExp =  1e10f;

	float *src = (float *) pixels;
	for (uint i = 0; i < nPixels; i++){
        float maxChannel = max(max(src[3 * i], src[3 * i + 1]), src[3 * i + 2]);

		if (maxChannel > maxExp) maxExp = maxChannel;
		if (maxChannel < minExp) minExp = maxChannel;
	}

	const float invLog2 = 1.0f / logf(2.0f);

	maxExp = logf(maxExp) * invLog2;
	minExp = logf(minExp) * invLog2;

	scale = maxExp - minExp;
	bias  = minExp;

	ushort *newPixels = new ushort[nPixels * 4];
	for (uint i = 0; i < nPixels; i++){
		float maxChannel = max(max(src[3 * i], src[3 * i + 1]), src[3 * i + 2]);
		float chExp = logf(maxChannel) * invLog2;

		newPixels[4 * i + 0] = (ushort) (65535 * src[3 * i + 0] / maxChannel);
		newPixels[4 * i + 1] = (ushort) (65535 * src[3 * i + 1] / maxChannel);
		newPixels[4 * i + 2] = (ushort) (65535 * src[3 * i + 2] / maxChannel);
		newPixels[4 * i + 3] = (ushort) (65535 * saturate((chExp - bias) / scale));
	}


	delete [] pixels;
	pixels = (ubyte *) newPixels;
	format = FORMAT_RGBA16;

	return true;
}

bool Image::toE16(float *scale, float *bias, const bool useAllSameRange, const float minValue, const float maxValue){
	if (arraySize > 1) return false; // TODO: Implement ...
	if (format < FORMAT_I32F || format > FORMAT_RGBA32F) return false;

	uint nPixels = getPixelCount(0, nMipMaps);
	uint nChannels = getChannelCount(format);
	uint nElements = nPixels * nChannels;

	uint nRanges = useAllSameRange? 1 : nChannels;


	float maxExp[4] = { minValue, minValue, minValue, minValue };
	float minExp[4] = { maxValue, maxValue, maxValue, maxValue };

	float *src = (float *) pixels;
	for (uint i = 0; i < nElements; i += nRanges){
		for (uint k = 0; k < nRanges; k++){
			float val = min(max(src[i + k], minValue), maxValue);
			if (val > maxExp[k]) maxExp[k] = val;
			if (val < minExp[k]) minExp[k] = val;
		}
	}

	const float invLog2 = 1.0f / logf(2.0f);

	for (uint k = 0; k < nRanges; k++){
		maxExp[k] = logf(maxExp[k]) * invLog2;
		minExp[k] = logf(minExp[k]) * invLog2;

		scale[k] = maxExp[k] - minExp[k];
		bias[k]  = minExp[k];
	}

	ushort *newPixels = new ushort[nElements];
	for (uint i = 0; i < nElements; i += nRanges){
		for (uint k = 0; k < nRanges; k++){
			if (src[i + k] > minValue){
				if (src[i + k] < maxValue){
					float chExp = logf(src[i + k]) * invLog2;
					newPixels[i + k] = (ushort) (65535 * (chExp - bias[k]) / scale[k]);
				} else {
					newPixels[i + k] = 65535;
				}
			} else {
				newPixels[i + k] = 0;
			}
		}
	}

	delete [] pixels;
	pixels = (ubyte *) newPixels;
	format = (FORMAT) ((FORMAT_I16 - 1) + nChannels);

	return true;
}


bool Image::toFixedPointHDR(float *maxValue, const int finalRgbBits, const int finalRangeBits){
	if (arraySize > 1) return false; // TODO: Implement ...
	if (format != FORMAT_RGB32F) return false;

	float rgbScale = float(1 << finalRgbBits);
	float rangeScale = float(1 << finalRangeBits);

	float clampVal = *maxValue;

	float *src = (float *) pixels;
	uint nPixels = getPixelCount(0, nMipMaps);

	float maxVal = 0;
	for (uint i = 0; i < 3 * nPixels; i++){
        if (src[i] > maxVal) maxVal = src[i];
	}
	if (maxVal > clampVal) maxVal = clampVal;

	ushort *newPixels = new ushort[4 * nPixels];
	for (uint i = 0; i < nPixels; i++){
		float r = min(src[3 * i + 0], clampVal);
		float g = min(src[3 * i + 1], clampVal);
		float b = min(src[3 * i + 2], clampVal);

		float maxChannel = max(max(r, g), b);
		//float minChannel = min(min(r, g), b);

		r /= maxChannel;
		g /= maxChannel;
		b /= maxChannel;
		float range = maxChannel / maxVal;

/*
		r = sqrtf(r);
		g = sqrtf(g);
		b = sqrtf(b);
		range = sqrtf(range);
*/

		/*
			If range gets too small, scale it up and scale down rgb so that the lowest
			channel's value is equal to range. This gives us a much better distribution
			of bits for low values.
		*/
		//float rgbMin = minChannel / maxChannel;
		//if (range * rangeScale < rgbMin * rgbScale){
			//float f = sqrtf((rgbMin * rgbScale) / (range * rangeScale));
			float f = sqrtf(range * rangeScale / rgbScale);

			//float f = powf(range * rangeScale / rgbScale, 1.0f / 3.0f);

//			f = 1.0f;

			range /= f;
//			f *= f;
			r *= f;
			g *= f;
			b *= f;
		//}

		newPixels[4 * i + 0] = (ushort) (65535 * r);
		newPixels[4 * i + 1] = (ushort) (65535 * g);
		newPixels[4 * i + 2] = (ushort) (65535 * b);
		newPixels[4 * i + 3] = (ushort) (65535 * range);
	}


	format = FORMAT_RGBA16;

	delete [] pixels;
	pixels = (ubyte *) newPixels;

	*maxValue = maxVal;
	return true;
}

bool Image::toNormalMap(FORMAT destFormat, float sZ, float mipMapScaleZ){
	if (arraySize > 1) return false; // TODO: Implement ...
	if (format == FORMAT_RGB8 || format == FORMAT_RGBA8) toGrayScale();
	if (format != FORMAT_I8) return false;
	
	uint xMask = 0, yMask = 0, zMask = 0, hMask = 0;
	uint xShift = 0, yShift = 0, zShift = 0, hShift = 0, hFactor = 1;

	switch (destFormat){
		case FORMAT_RG8:
		case FORMAT_RG8S:
			xMask = yMask = 0xFF;
			xShift = 8;
			break;
		case FORMAT_RGB565:
			xMask = zMask = 0x1F;
			yMask = 0x3F;
			xShift = 11;
			yShift = 5;
			break;
		case FORMAT_RGBA4:
			xMask = yMask = zMask = hMask = 0xF;
			yShift = 4;
			zShift = 8;
			hShift = 12;
			break;
		case FORMAT_RGBA8:
		case FORMAT_RGBA8S:
			xMask = yMask = zMask = hMask = 0xFF;
			yShift = 8;
			zShift = 16;
			hShift = 24;
			break;
		case FORMAT_RGB10A2:
			xMask = yMask = zMask = 0x3FF;
			hMask = 0x03;
			yShift = 10;
			zShift = 20;
			hShift = 30;
			break;
		case FORMAT_RGBA16:
		case FORMAT_RGBA16S:
			xMask = yMask = zMask = hMask = 0xFFFF;
			yShift = 16;
			zShift = 32;
			hShift = 48;
			hFactor = 257;
			break;
		default:
			return false;
	}

	const float sobelX[5][5] = {
		{ 1,  2,  0,  -2, -1 },
		{ 4,  8,  0,  -8, -4 },
		{ 6, 12,  0, -12, -6 },
		{ 4,  8,  0,  -8, -4 },
		{ 1,  2,  0,  -2, -1 },
	};
	const float sobelY[5][5] = {
		{  1,  4,   6,  4,  1 },
		{  2,  8,  12,  8,  2 },
		{  0,  0,   0,  0,  0 },
		{ -2, -8, -12, -8, -2 },
		{ -1, -4,  -6, -4, -1 },
	};

	int bpp = getBytesPerPixel(destFormat);
	bool use16 = (bpp == 2);
	bool use32 = (bpp == 4);

	float xFactor = 0.5f * xMask;
	float yFactor = 0.5f * yMask;
	float zFactor = 0.5f * zMask;
	float bias = isSignedFormat(destFormat)? 0.0f : 1.0f;


	// Size of the z component
	sZ *= 128.0f / max(width, height);

	ubyte *newPixels = new ubyte[getMipMappedSize(0, nMipMaps, destFormat)];

	union {
		ushort *dest16;
		uint32 *dest32;
		uint64 *dest64;
	};
	dest32 = (uint32 *) newPixels;

	for (int mipMap = 0; mipMap < nMipMaps; mipMap++){
		ubyte *src = getPixels(mipMap);

		int w = getWidth(mipMap);
		int h = getHeight(mipMap);

		for (int y = 0; y < h; y++){
			for (int x = 0; x < w; x++){
				// Apply a 5x5 Sobel filter
				float sX = 0;
				float sY = 0;
				for (int dy = 0; dy < 5; dy++){
					int fy = (y + dy - 2 + h) % h;
					for (int dx = 0; dx < 5; dx++){
						int fx = (x + dx - 2 + w) % w;
						sX += sobelX[dy][dx] * src[fy * w + fx];
						sY += sobelY[dy][dx] * src[fy * w + fx];
					}
				}
				// Construct the components
				sX *= 1.0f / (48 * 255);
				sY *= 1.0f / (48 * 255);

				// Normalize and store
				float invLen = 1.0f / sqrtf(sX * sX + sY * sY + sZ * sZ);
				float rX = xFactor * (sX * invLen + bias);
				float rY = yFactor * (sY * invLen + bias);
				float rZ = zFactor * (sZ * invLen + bias);

				uint64 result = 0;
				result |= uint64(int(rX) & xMask) << xShift;
				result |= uint64(int(rY) & yMask) << yShift;
				result |= uint64(int(rZ) & zMask) << zShift;
				result |= uint64((src[y * w + x] * hFactor) & hMask) << hShift;

				if (use32){
                    *dest32++ = (uint32) result;
				} else if (use16){
                    *dest16++ = (uint16) result;
				} else {
					*dest64++ = result;
				}
			}
		}
		sZ *= mipMapScaleZ;
	}

	format = destFormat;
	delete [] pixels;
	pixels = newPixels;

	return true;
}

bool Image::toGrayScale(){
	int nChannels = getChannelCount(format);

	if (!isPlainFormat(format) || isFloatFormat(format) || nChannels < 3) return false;

	uint len = getPixelCount(0, nMipMaps) * arraySize;
	uint size = len;

	if (format <= FORMAT_RGBA8){
		ubyte *src = pixels, *dest = pixels;
		do {
			*dest++ = (77 * src[0] + 151 * src[1] + 28 * src[2] + 128) >> 8;
			src += nChannels;
		} while (--len);

		format = FORMAT_I8;
	} else {
		ushort *src = (ushort *) pixels, *dest = (ushort *) pixels;
		do {
			*dest++ = (77 * src[0] + 151 * src[1] + 28 * src[2] + 128) >> 8;
			src += nChannels;
		} while (--len);

		format = FORMAT_I16;
		size *= 2;
	}

	pixels = (ubyte *) realloc(pixels, size);

	return true;
}

bool Image::getRange(float &min, float &max){
	// Only float supported at this point
	if (format < FORMAT_R32F || format > FORMAT_RGBA32F) return false;

	int nElements = getPixelCount(0, nMipMaps) * getChannelCount(format) * arraySize;
	if (nElements <= 0) return false;

	float minVal =  FLT_MAX;
	float maxVal = -FLT_MAX;
	for (int i = 0; i < nElements; i++){
		float d = ((float *) pixels)[i];
		if (d < minVal) minVal = d;
		if (d > maxVal) maxVal = d;
	}
	min = minVal;
	max = maxVal;

	return true;
}

bool Image::scaleBias(const float scale, const float bias){
	// Only float supported at this point
	if (format < FORMAT_R32F || format > FORMAT_RGBA32F) return false;

	int nElements = getPixelCount(0, nMipMaps) * getChannelCount(format) * arraySize;

	for (int i = 0; i < nElements; i++){
		float d = ((float *) pixels)[i];
		((float *) pixels)[i] = d * scale + bias;
	}

	return true;
}

bool Image::normalize(){
	// Only float supported at this point
	if (format < FORMAT_R32F || format > FORMAT_RGBA32F) return false;

	float min, max;
	getRange(min, max);

	int nElements = getPixelCount(0, nMipMaps) * getChannelCount(format) * arraySize;

	float s = 1.0f / (max - min);
	float b = -min * s;
	for (int i = 0; i < nElements; i++){
		float d = ((float *) pixels)[i];
		((float *) pixels)[i] = d * s + b;
	}

	return true;
}

bool Image::removeChannels(bool keepCh0, bool keepCh1, bool keepCh2, bool keepCh3){
	if (!isPlainFormat(format)) return false;

	uint srcChannels = getChannelCount(format);
	if (srcChannels < 4) keepCh3 = false;
	if (srcChannels < 3) keepCh2 = false;
	if (srcChannels < 2) keepCh1 = false;

	if (!(keepCh0 || keepCh1 || keepCh2 || keepCh3)) return false;
	uint destChannels = int(keepCh0) + int(keepCh1) + int(keepCh2) + int(keepCh3);
	if (srcChannels == destChannels) return true;


	uint nPixels = getPixelCount(0, nMipMaps) * arraySize;
	uint bpc = getBytesPerChannel(format);

	format = (FORMAT) (format + (destChannels - srcChannels));
	ubyte *newPixels = new ubyte[getMipMappedSize(0, nMipMaps) * arraySize];

	if (bpc == 1){
		ubyte *src = pixels;
		ubyte *dst = newPixels;
		do {
			if (keepCh0) *dst++ = src[0];
			if (keepCh1) *dst++ = src[1];
			if (keepCh2) *dst++ = src[2];
			if (keepCh3) *dst++ = src[3];
			src += srcChannels;
		} while (--nPixels);
	} else if (bpc == 2){
		ushort *src = (ushort *) pixels;
		ushort *dst = (ushort *) newPixels;
		do {
			if (keepCh0) *dst++ = src[0];
			if (keepCh1) *dst++ = src[1];
			if (keepCh2) *dst++ = src[2];
			if (keepCh3) *dst++ = src[3];
			src += srcChannels;
		} while (--nPixels);
	} else {
		uint *src = (uint *) pixels;
		uint *dst = (uint *) newPixels;
		do {
			if (keepCh0) *dst++ = src[0];
			if (keepCh1) *dst++ = src[1];
			if (keepCh2) *dst++ = src[2];
			if (keepCh3) *dst++ = src[3];
			src += srcChannels;
		} while (--nPixels);
	}

	delete [] pixels;
	pixels = newPixels;

	return true;
}
