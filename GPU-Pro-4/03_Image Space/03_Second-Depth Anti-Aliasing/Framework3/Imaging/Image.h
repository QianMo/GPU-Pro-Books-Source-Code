
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

#ifndef _IMAGE_H_
#define _IMAGE_H_

#include "../Platform.h"

#define ALL_MIPMAPS 127

// Image loading flags
#define DONT_LOAD_MIPMAPS 0x1

enum FORMAT {
	FORMAT_NONE     = 0,

	// Unsigned formats
	FORMAT_R8       = 1,
	FORMAT_RG8      = 2,
	FORMAT_RGB8     = 3,
	FORMAT_RGBA8    = 4,

	FORMAT_R16      = 5,
	FORMAT_RG16     = 6,
	FORMAT_RGB16    = 7,
	FORMAT_RGBA16   = 8,

	// Signed formats
	FORMAT_R8S      = 9,
	FORMAT_RG8S     = 10,
	FORMAT_RGB8S    = 11,
	FORMAT_RGBA8S   = 12,

	FORMAT_R16S     = 13,
	FORMAT_RG16S    = 14,
	FORMAT_RGB16S   = 15,
	FORMAT_RGBA16S  = 16,

	// Float formats
	FORMAT_R16F     = 17,
	FORMAT_RG16F    = 18,
	FORMAT_RGB16F   = 19,
	FORMAT_RGBA16F  = 20,

	FORMAT_R32F     = 21,
	FORMAT_RG32F    = 22,
	FORMAT_RGB32F   = 23,
	FORMAT_RGBA32F  = 24,

	// Signed integer formats
	FORMAT_R16I     = 25,
	FORMAT_RG16I    = 26,
	FORMAT_RGB16I   = 27,
	FORMAT_RGBA16I  = 28,

	FORMAT_R32I     = 29,
	FORMAT_RG32I    = 30,
	FORMAT_RGB32I   = 31,
	FORMAT_RGBA32I  = 32,

	// Unsigned integer formats
	FORMAT_R16UI    = 33,
	FORMAT_RG16UI   = 34,
	FORMAT_RGB16UI  = 35,
	FORMAT_RGBA16UI = 36,

	FORMAT_R32UI    = 37,
	FORMAT_RG32UI   = 38,
	FORMAT_RGB32UI  = 39,
	FORMAT_RGBA32UI = 40,

	// Packed formats
	FORMAT_RGBE8    = 41,
	FORMAT_RGB9E5   = 42,
	FORMAT_RG11B10F = 43,
	FORMAT_RGB565   = 44,
	FORMAT_RGBA4    = 45,
	FORMAT_RGB10A2  = 46,

	// Depth formats
	FORMAT_D16      = 47,
	FORMAT_D24      = 48,
	FORMAT_D24S8    = 49,
	FORMAT_D32F     = 50,

	// Compressed formats
	FORMAT_DXT1     = 51,
	FORMAT_DXT3     = 52,
	FORMAT_DXT5     = 53,
	FORMAT_ATI1N    = 54,
	FORMAT_ATI2N    = 55,
};

#define FORMAT_I8    FORMAT_R8
#define FORMAT_IA8   FORMAT_RG8
#define FORMAT_I16   FORMAT_R16
#define FORMAT_IA16  FORMAT_RG16
#define FORMAT_I16F  FORMAT_R16F
#define FORMAT_IA16F FORMAT_RG16F
#define FORMAT_I32F  FORMAT_R32F
#define FORMAT_IA32F FORMAT_RG32F

// Compatibility with old demos
#define FORMAT_DEPTH16 FORMAT_D16
#define FORMAT_DEPTH24 FORMAT_D24

inline bool isPlainFormat(const FORMAT format){
	return (format <= FORMAT_RGBA32UI);
}

inline bool isPackedFormat(const FORMAT format){
	return (format >= FORMAT_RGBE8 && format <= FORMAT_RGB10A2);
}

inline bool isDepthFormat(const FORMAT format){
	return (format >= FORMAT_D16 && format <= FORMAT_D32F);
}

inline bool isStencilFormat(const FORMAT format){
	return (format == FORMAT_D24S8);
}

inline bool isSignedFormat(const FORMAT format){
	return ((format >= FORMAT_R8S) && (format <= FORMAT_RGBA16S)) || ((format >= FORMAT_R16I) && (format <= FORMAT_RGBA32I));
}

inline bool isCompressedFormat(const FORMAT format){
	return (format >= FORMAT_DXT1) && (format <= FORMAT_ATI2N);
}

inline bool isFloatFormat(const FORMAT format){
//	return (format >= FORMAT_R16F && format <= FORMAT_RGBA32F);
	return (format >= FORMAT_R16F && format <= FORMAT_RG11B10F) || (format == FORMAT_D32F);
}

inline bool isIntegerFormat(const FORMAT format){
	return (format >= FORMAT_R16I && format <= FORMAT_RGBA32UI);
}

inline int getChannelCount(const FORMAT format){
	static const int chCount[] = {
		0,
		1, 2, 3, 4,       //  8-bit unsigned
		1, 2, 3, 4,       // 16-bit unsigned
		1, 2, 3, 4,       //  8-bit signed
		1, 2, 3, 4,       // 16-bit signed
		1, 2, 3, 4,       // 16-bit float
		1, 2, 3, 4,       // 32-bit float
		1, 2, 3, 4,       // 16-bit unsigned integer
		1, 2, 3, 4,       // 32-bit unsigned integer
		1, 2, 3, 4,       // 16-bit signed integer
		1, 2, 3, 4,       // 32-bit signed integer
		3, 3, 3, 3, 4, 4, // Packed
		1, 1, 2, 1,       // Depth
		3, 4, 4, 1, 2,    // Compressed
	};
	return chCount[format];
}

// Accepts only plain formats
inline int getBytesPerChannel(const FORMAT format){
	static const int bytesPC[] = {
		1, //  8-bit unsigned
		2, // 16-bit unsigned
		1, //  8-bit signed
		2, // 16-bit signed
		2, // 16-bit float
		4, // 32-bit float
		2, // 16-bit unsigned integer
		4, // 32-bit unsigned integer
		2, // 16-bit signed integer
		4, // 32-bit signed integer
	};

	return bytesPC[(format - 1) >> 2];
}

// Does not accept compressed formats
inline int getBytesPerPixel(const FORMAT format){
	static const int bytesPP[] = {
		0,
		1, 2, 3, 4,       //  8-bit unsigned
		2, 4, 6, 8,       // 16-bit unsigned
		1, 2, 3, 4,       //  8-bit signed
		2, 4, 6, 8,       // 16-bit signed
		2, 4, 6, 8,       // 16-bit float
		4, 8, 12, 16,     // 32-bit float
		2, 4, 6, 8,       // 16-bit unsigned integer
		4, 8, 12, 16,     // 32-bit unsigned integer
		2, 4, 6, 8,       // 16-bit signed integer
		4, 8, 12, 16,     // 32-bit signed integer
		4, 4, 4, 2, 2, 4, // Packed
		2, 4, 4, 4,       // Depth
	};
	return bytesPP[format];
}

// Accepts only compressed formats
inline int getBytesPerBlock(const FORMAT format){
	return (format == FORMAT_DXT1 || format == FORMAT_ATI1N)? 8 : 16;
}

const char *getFormatString(const FORMAT format);
FORMAT getFormatFromString(char *string);

class Image {
public:
	Image();
	Image(const Image &img);
	~Image();

	unsigned char *create(const FORMAT fmt, const int w, const int h, const int d, const int mipMapCount, const int arraysize = 1);
	void free();
	void clear();

	unsigned char *getPixels() const { return pixels; }
	unsigned char *getPixels(const int mipMapLevel) const;
	unsigned char *getPixels(const int mipMapLevel, const int arraySlice) const;
	int getMipMapCount() const { return nMipMaps; }
	int getMipMapCountFromDimesions() const;
	int getMipMappedSize(const int firstMipMapLevel = 0, int nMipMapLevels = ALL_MIPMAPS, FORMAT srcFormat = FORMAT_NONE) const;
	int getSliceSize(const int mipMapLevel = 0, FORMAT srcFormat = FORMAT_NONE) const;
	int getPixelCount(const int firstMipMapLevel = 0, int nMipMapLevels = ALL_MIPMAPS) const;

	int getWidth () const { return width;  }
	int getHeight() const { return height; }
	int getDepth () const { return depth;  }
	int getWidth (const int mipMapLevel) const;
	int getHeight(const int mipMapLevel) const;
	int getDepth (const int mipMapLevel) const;
	int getArraySize() const { return arraySize; }

	bool is1D()    const { return (depth == 1 && height == 1); }
	bool is2D()    const { return (depth == 1 && height >  1); }
	bool is3D()    const { return (depth >  1); }
	bool isCube()  const { return (depth == 0); }
	bool isArray() const { return (arraySize > 1); }

	FORMAT getFormat() const { return format; }
	void setFormat(const FORMAT form){ format = form; }

	int getExtraDataBytes() const { return nExtraData; }
	ubyte *getExtraData() const { return extraData; }
	void setExtraData(void *data, const int nBytes){
		nExtraData = nBytes;
		extraData = (unsigned char *) data;
	}

	bool loadHTEX(const char *fileName);
	bool loadDDS(const char *fileName, uint flags = 0);
#ifndef NO_HDR
	bool loadHDR(const char *fileName);
#endif // NO_HDR
#ifndef NO_JPEG
	bool loadJPEG(const char *fileName);
#endif // NO_JPEG
#ifndef NO_PNG
	bool loadPNG(const char *fileName);
#endif // NO_PNG
#ifndef NO_TGA
	bool loadTGA(const char *fileName);
#endif // NO_TGA
#ifndef NO_BMP
	bool loadBMP(const char *fileName);
#endif // NO_BMP
#ifndef NO_PCX
	bool loadPCX(const char *fileName);
#endif // NO_PCX

	bool saveHTEX(const char *fileName);
	bool saveDDS(const char *fileName);
#ifndef NO_HDR
	bool saveHDR(const char *fileName);
#endif // NO_HDR
#ifndef NO_JPEG
	bool saveJPEG(const char *fileName, const int quality);
#endif // NO_JPEG
#ifndef NO_PNG
	bool savePNG(const char *fileName);
#endif // NO_PNG
#ifndef NO_TGA
	bool saveTGA(const char *fileName);
#endif // NO_TGA
#ifndef NO_BMP
	bool saveBMP(const char *fileName);
#endif // NO_BMP
#ifndef NO_PCX
	bool savePCX(const char *fileName);
#endif // NO_PCX

	bool loadImage(const char *fileName, uint flags = 0);
	bool loadSlicedImage(const char **fileNames, const int nImages, const int nArraySlices = 1, uint flags = 0);
	bool saveImage(const char *fileName);

	void loadFromMemory(void *mem, const FORMAT frmt, const int w, const int h, const int d, const int mipMapCount, bool ownsMemory);

	bool createMipMaps(const int mipMaps = ALL_MIPMAPS);
	bool removeMipMaps(const int firstMipMap, const int mipMapsToSave = ALL_MIPMAPS);

	bool uncompressImage();
	bool unpackImage();

	bool convert(const FORMAT newFormat);
	bool swap(const int ch0, const int ch1);
	bool flipX();
	bool flipY();
	bool dilate();
	bool erode();

	bool toRGBD16();
	bool toRGBE16(float &scale, float &bias);
	bool toE16(float *scale, float *bias, const bool useAllSameRange = false, const float minValue = FLT_MIN, const float maxValue = FLT_MAX);
	bool toFixedPointHDR(float *maxValue, const int finalRgbBits, const int finalRangeBits);
	bool toNormalMap(FORMAT destFormat, float sZ = 1.0f, float mipMapScaleZ = 2.0f);
	bool toGrayScale();
	bool getRange(float &min, float &max);
	bool scaleBias(const float scale, const float bias);
	bool normalize();

	bool removeChannels(bool keepCh0, bool keepCh1 = true, bool keepCh2 = true, bool keepCh3 = true);

protected:
	unsigned char *pixels;
	int width, height, depth;
	int nMipMaps;
	int arraySize;
	FORMAT format;

	int nExtraData;
	unsigned char *extraData;
};

#endif // _IMAGE_H_
