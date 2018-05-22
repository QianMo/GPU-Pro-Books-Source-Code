#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

#include "..\Graphics\TGALoader.h"

#include <GL/glut.h>

#pragma pack(push,1)
typedef struct
{
	unsigned char  imageIdent;
	unsigned char  colourMapType;
	unsigned char  imageType;
	unsigned short colourMapOrigin;
	unsigned short colourMapSize;
	unsigned char  colourMapESize;
	unsigned short xOrigin;
	unsigned short yOrigin;
	unsigned short width;
	unsigned short height;
	unsigned char  pixelIndexSize;
	unsigned char  imageDescByte;
}
TGA_HEADER;
#pragma pack(pop)


// -----------------------------------------------------------------------------
// ----------------------- TGALoader::TGALoader --------------------------------
// -----------------------------------------------------------------------------
TGALoader::TGALoader(void)
{
}

// -----------------------------------------------------------------------------
// ----------------------- TGALoader::LoadTexture ------------------------------
// -----------------------------------------------------------------------------
int TGALoader::LoadTexture(const char* filename, unsigned char*** pixelData, unsigned int* width, unsigned int* height, unsigned int* bitsPerPixel, bool mipMaps)
{
	TGA_HEADER header;
	unsigned int numMipMaps = 0;

	FILE *file;
	fopen_s(&file, filename, "rb");

	if (file)
	{
		unsigned int i;

		fread(&header,1,sizeof(TGA_HEADER),file);

		*width = header.width;
		*height = header.height;

		assert(*width == *height);

		for (i=0; i!=header.imageIdent; ++i)
		{
			fgetc(file);
		}

		*bitsPerPixel = header.pixelIndexSize/8;
		if (!mipMaps)
		{
			*pixelData    = (unsigned char**) malloc(1);
			*pixelData[0] = (unsigned char*)malloc((*bitsPerPixel) * header.width * header.height);
			numMipMaps = 1;
		}
		else if (*bitsPerPixel != 1)
		{
			unsigned int _width = *width;
			unsigned int _height;
			while (_width > 0)
			{
				numMipMaps++;
				_width /= 2;
			}

			_width = *width;
			_height = *height;

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

		assert(*pixelData != NULL);

		unsigned int rowsize = 0;
		unsigned char* pd = NULL;

		switch (header.imageType)
		{
		case 2:
			rowsize = (*bitsPerPixel) * header.width;
			pd = *pixelData[0] + (*bitsPerPixel) * (header.height-1) * header.width;

			for ( i = 0; i < header.height; ++i, pd -= rowsize )
			{
				fread(pd, 1, rowsize, file);
			}

			if (*bitsPerPixel != 1)
			{
				BGR_TO_RGB((*pixelData[0]), header.height * header.width, *bitsPerPixel);
			}
			break;
		default:
			assert(false);
			break;
		}

		if ((mipMaps) && (*bitsPerPixel != 1))
		{
			unsigned int i;
			unsigned int _width = *width / 2;
			unsigned int _height = *height / 2;
			for (i=1; i<numMipMaps; i++)
			{
				gluScaleImage(GL_RGBA, *width, *height, GL_UNSIGNED_BYTE, (*pixelData)[0], _width, _height, GL_UNSIGNED_BYTE, (*pixelData)[i]);
				_width /= 2;
				_height /= 2;
			}
		}	

	}
	else
	{
		assert(false);
	}
	
	fclose(file);

	return numMipMaps;
}

// -----------------------------------------------------------------------------
// ----------------------- TGALoader::BGR_TO_RGB ---------------------------------
// -----------------------------------------------------------------------------
void TGALoader::BGR_TO_RGB(unsigned char* data, unsigned int numPixels, unsigned int bitsPerPixel)
{
	unsigned char *end = data + (bitsPerPixel * numPixels);

	for (; data != end ; data += bitsPerPixel)
	{
		unsigned char temp = *data;
		*data = data[2];
		data[2] = temp;
	}
}