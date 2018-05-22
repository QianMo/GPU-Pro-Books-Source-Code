#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

#include <IL/il.h>
#include <IL/ilu.h>

#include "../Graphics/DDSLoader.h"

#include <GL/glut.h>


// -----------------------------------------------------------------------------
// ----------------------- DDSLoader::DDSLoader --------------------------------
// -----------------------------------------------------------------------------
DDSLoader::DDSLoader(void)
{
}

// -----------------------------------------------------------------------------
// ----------------------- DDSLoader::LoadTexture ------------------------------
// -----------------------------------------------------------------------------
int DDSLoader::LoadTexture(const char* filename, unsigned char*** pixelData, unsigned int* width, unsigned int* height, unsigned int* bitsPerPixel, bool mipMaps)
{
	unsigned int numMipMaps = 0;

	ilInit();

	ILuint ImageName;
	ilGenImages(1, &ImageName);
	ilBindImage(ImageName);

	if (!ilLoadImage(filename))
	{
		assert(false);
	}

	*width = ilGetInteger(IL_IMAGE_WIDTH);
	*height = ilGetInteger(IL_IMAGE_HEIGHT);
	*bitsPerPixel = ilGetInteger(IL_IMAGE_BPP);
	numMipMaps = ilGetInteger(IL_NUM_MIPMAPS);
	
	numMipMaps += 1;

	if (!mipMaps)
	{
		*pixelData    = (unsigned char**) malloc(sizeof(unsigned char**));
		*pixelData[0] = (unsigned char*)malloc((*bitsPerPixel) * (*width) * (*height));
		numMipMaps = 1;
	}
	else if (*bitsPerPixel != 1)
	{
		unsigned int _width = *width;
		unsigned int _height = *height;

		*pixelData    = (unsigned char**)malloc(numMipMaps * sizeof(unsigned char**));
		unsigned int i;
		for (i=0; i<numMipMaps; i++)
		{
			(*pixelData)[i] = (unsigned char*)malloc((*bitsPerPixel) * _width * _height);
			_width /= 2;
			_height /= 2;
		}
	}
	else
	{
		assert(false);
	}

	unsigned int i;
	unsigned char* pixels = NULL;
	unsigned int _width = *width;
	unsigned int _height = *height;
	for (i=0; i<numMipMaps; i++)
	{
		ilActiveMipmap(i);

		pixels = ilGetData();
		memcpy((*pixelData)[i], pixels, (*bitsPerPixel) * _width * _height);

		_width /= 2;
		_height /= 2;

		ilBindImage(ImageName);
	}
	
	ilDeleteImages(1, &ImageName);

	return numMipMaps;
}