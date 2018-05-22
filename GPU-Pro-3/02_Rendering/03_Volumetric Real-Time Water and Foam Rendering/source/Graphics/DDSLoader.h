#ifndef __DDSLOADER__H__
#define __DDSLOADER__H__

#include "../Util/Singleton.h"

class DDSLoader : public Singleton<DDSLoader>
{
	friend class Singleton<DDSLoader>;

public:
	DDSLoader(void);

	// loads a dds texture from spec. file
	int LoadTexture(const char* filename, unsigned char*** pixelData, unsigned int* width, unsigned int* height, unsigned int* bitsPerPixel, bool mipMaps = true);
};

#endif