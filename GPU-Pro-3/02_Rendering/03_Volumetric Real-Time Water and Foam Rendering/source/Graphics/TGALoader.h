#ifndef __TGALOADER__H__
#define __TGALOADER__H__

#include "../Util/Singleton.h"

class TGALoader : public Singleton<TGALoader>
{
	friend class Singleton<TGALoader>;

public:
	TGALoader(void);

	// loads a 32 bit tga texture from spec. file
	int LoadTexture(const char* filename, unsigned char*** pixelData, unsigned int* width, unsigned int* height, unsigned int* bitsPerPixel, bool mipMaps = true);
private:
	// swaps a data buffer from bgr to rgb
	void BGR_TO_RGB(unsigned char* data, unsigned int numPixels, unsigned int bitsPerPixel);
};

#endif