#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>

#include "../Graphics/TextureManager.h"
#include "../Graphics/TGALoader.h"
#include "../Graphics/DDSLoader.h"

#include "../Util/ConfigLoader.h"

#include "GL/glew.h"
#include <GL/glut.h>

#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>


#define GL_CLAMP_TO_EDGE 0x812F

#ifndef GL_COMPRESSED_RGB_S3TC_DXT1_EXT
#define GL_COMPRESSED_RGB_S3TC_DXT1_EXT 0x83F0
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define GL_COMPRESSED_RGBA_S3TC_DXT3_EXT 0x83F2
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3
#endif

#define GL_REFLECTION_MAP					 0x8512
#define GL_TEXTURE_CUBE_MAP					 0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X_EXT  0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X_EXT  0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y_EXT  0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_EXT  0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z_EXT  0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_EXT  0x851A

#ifndef GL_TEXTURE_COMPRESSED_IMAGE_SIZE
#define GL_TEXTURE_COMPRESSED_IMAGE_SIZE 0x86A0
#endif

GLenum cubefaces[6] = {
	GL_TEXTURE_CUBE_MAP_POSITIVE_X_EXT,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_X_EXT,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Y_EXT,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_EXT,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Z_EXT,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_EXT,
};


// -----------------------------------------------------------------------------
// ----------------------- TextureManager::TextureManager ----------------------
// -----------------------------------------------------------------------------
TextureManager::TextureManager(void) :
	filterMethod(FILTER_LINEAR),
	mipMapMethod(FILTER_LINEAR)
{
}

// -----------------------------------------------------------------------------
// ----------------------- TextureManager::~TextureManager ---------------------
// -----------------------------------------------------------------------------
TextureManager::~TextureManager(void)
{
}

// -----------------------------------------------------------------------------
// ----------------------- TextureManager::LoadTexture -------------------------
// -----------------------------------------------------------------------------
unsigned int TextureManager::LoadTexture(const char* filename, bool mipMaps, bool compressed, bool isSkyBoxTexture)
{
	iter = textures.find(filename);
	if (iter != textures.end())
	{
		iter->second.ref++;
		return iter->second.idx;
	}

	unsigned int width = 0, height = 0, bitsPerPixel = 0;
	unsigned char** pixels = NULL;
	unsigned int numMipMaps = 0;

	int len = static_cast<int>(strlen(filename));

	if ((filename[len-3] == 't' || filename[len-3] == 'T') &&
		(filename[len-2] == 'g' || filename[len-2] == 'G') &&
		(filename[len-1] == 'a' || filename[len-1] == 'A'))
	{
		numMipMaps = TGALoader::Instance()->LoadTexture(filename, &pixels, &width, &height, &bitsPerPixel, mipMaps);
	}
	else if ((filename[len-3] == 'd' || filename[len-3] == 'D') &&
			 (filename[len-2] == 'd' || filename[len-2] == 'D') &&
			 (filename[len-1] == 's' || filename[len-1] == 'S'))
	{
		numMipMaps = DDSLoader::Instance()->LoadTexture(filename, &pixels, &width, &height, &bitsPerPixel, mipMaps);
		compressed = false;
	}
	else
	{
		assert(false);
	}

	unsigned int tex_object=0;
	switch (bitsPerPixel)
	{
	case 1:
		tex_object = CreateTexture(GL_ALPHA, pixels, width, height, numMipMaps, compressed, isSkyBoxTexture);
		break;
	case 2:
		break;
	case 3:
		tex_object = CreateTexture(GL_RGB, pixels, width, height, numMipMaps, compressed, isSkyBoxTexture);
		break;
	case 4:
		tex_object = CreateTexture(GL_RGBA, pixels, width, height, numMipMaps, compressed, isSkyBoxTexture);
		break;
	default:
		break;
	}

	int data_size=0;
	if (compressed) {
		glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &data_size);
	}
	else {
		data_size = bitsPerPixel * width * height;
	}

	unsigned int i;
	for (i=0; i<numMipMaps; i++)
	{
		free(pixels[i]);
	}
	
	if (numMipMaps >= 1)
	{
		free(pixels);
	}

	textures.insert(std::make_pair(std::string(filename), TextureReference(tex_object, data_size, numMipMaps)));

	return tex_object;
}

// -----------------------------------------------------------------------------
// ------------------------ TextureManager::LoadCubeMap ------------------------
// -----------------------------------------------------------------------------
unsigned int TextureManager::LoadCubeMap(const char* filename[6], bool mipMaps)
{
	unsigned int width = 0, height = 0, bitsPerPixel = 0;
	unsigned char** pixels[6];
	unsigned int numMipMaps = 0;

	int i;
	for (i=0; i<6; i++)
	{
		unsigned int currentWidth = 0, currentHeight = 0, currentBitsPerPixel = 0;
		unsigned int currentNumMipMaps = 0;

		int len = static_cast<int>(strlen(filename[i]));

		if ((filename[i][len-3] == 't' || filename[i][len-3] == 'T') &&
			(filename[i][len-2] == 'g' || filename[i][len-2] == 'G') &&
			(filename[i][len-1] == 'a' || filename[i][len-1] == 'A'))
		{
			currentNumMipMaps = TGALoader::Instance()->LoadTexture(filename[i], &pixels[i], &currentWidth, &currentHeight, &currentBitsPerPixel, mipMaps);
		}
		else if ((filename[i][len-3] == 'd' || filename[i][len-3] == 'D') &&
			(filename[i][len-2] == 'd' || filename[i][len-2] == 'D') &&
			(filename[i][len-1] == 's' || filename[i][len-1] == 'S'))
		{
			currentNumMipMaps = DDSLoader::Instance()->LoadTexture(filename[i], &pixels[i], &currentWidth, &currentHeight, &currentBitsPerPixel, mipMaps);
		}
		else
		{
			assert(false);
		}

		if (width == 0)
			width = currentWidth;
		if (height == 0)
			height = currentHeight;
		if (bitsPerPixel == 0)
			bitsPerPixel = currentBitsPerPixel;
		if (numMipMaps == 0)
			numMipMaps = currentNumMipMaps;

		assert(width == currentWidth);
		assert(height == currentHeight);
		assert(bitsPerPixel == currentBitsPerPixel);
		assert(numMipMaps == currentNumMipMaps);
	}

	// assert(bitsPerPixel == ..)

	unsigned int tex_object;
	tex_object = CreateCubeMap(pixels, width, height, numMipMaps);

	for (i=0; i<6; i++)
	{
		unsigned int j;
		for (j=0; j<numMipMaps; j++)
		{
			free(pixels[i][j]);
		}

		if (numMipMaps >= 1)
		{
			free(pixels[i]);
		}
	}

	return tex_object;
}

// -----------------------------------------------------------------------------
// ----------------------- TextureManager::ReleaseTexture ----------------------
// -----------------------------------------------------------------------------
void TextureManager::ReleaseTexture(const unsigned int& idx)
{
	iter = textures.begin();
	for (; iter != textures.end(); ++iter )
	{
		if (idx == iter->second.idx)
		{
			if (--iter->second.ref == 0)
			{
				glDeleteTextures(1,&(iter->second.idx));
				textures.erase(iter);
			}
			return;
		}
	}
}

// -----------------------------------------------------------------------------
// ----------------------- TextureManager::DeleteAllTextures -------------------
// -----------------------------------------------------------------------------
void TextureManager::DeleteAllTextures(void)
{
	iter = textures.begin();
	for (; iter != textures.end(); ++iter)
	{
		glDeleteTextures(1,&iter->second.idx);
	}
	textures.clear();
}


// -----------------------------------------------------------------------------
// ----------------------- TextureManager::GetTextureSize ----------------------
// -----------------------------------------------------------------------------
unsigned int TextureManager::GetTextureSize(const unsigned int& idx)
{
	iter = textures.begin();
	for (; iter != textures.end(); ++iter)
	{
		if (idx == iter->second.idx)
		{
			return iter->second.size;
		}
	}
	return 0;
}

// -----------------------------------------------------------------------------
// ----------------------- TextureManager::GetTotalTextureSize -----------------
// -----------------------------------------------------------------------------
unsigned int TextureManager::GetTotalTextureSize(void)
{
	unsigned int sz=0;
	iter = textures.begin();
	for (; iter != textures.end(); ++iter)
	{
		sz += iter->second.size;		
	}
	return sz;
}

// -----------------------------------------------------------------------------
// ----------------------- TextureManager::ChangeFilterMethod ------------------
// -----------------------------------------------------------------------------
void TextureManager::ChangeFilterMethod(const FilterMethod& method)
{
	iter = textures.begin();
	for (; iter != textures.end(); ++iter)
	{
		glBindTexture (GL_TEXTURE_2D, iter->second.idx);

		switch(method)
		{
		case FILTER_NONE:
		case FILTER_NEAREST:
			filterMethod=1;
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			if ((mipMapMethod > 0) && (iter->second.numMipMaps > 1))
			{
				if (mipMapMethod == 1)
				{
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
				}
				else if (mipMapMethod == 2)
				{
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
				}
				else
				{
					assert(false);
				}
				
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			}
			break;
		case FILTER_LINEAR:
			filterMethod=2;
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			if ((mipMapMethod > 0) && (iter->second.numMipMaps > 1))
			{
				if (mipMapMethod == 1)
				{
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
				}
				else if (mipMapMethod == 2)
				{
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				}
				else
				{
					assert(false);
				}
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			}
			break;
		default:
			assert(false);
			break;
		}
	}
}

// -----------------------------------------------------------------------------
// ----------------------- TextureManager::ChangeMipMapMethod ------------------
// -----------------------------------------------------------------------------
void TextureManager::ChangeMipMapMethod(const FilterMethod& method)
{
	iter = textures.begin();
	for (; iter != textures.end(); ++iter)
	{
		if (iter->second.numMipMaps <= 1)
		{
			continue;
		}

		glBindTexture (GL_TEXTURE_2D, iter->second.idx);

		switch(method)
		{
		case FILTER_NONE:
			mipMapMethod=0;
			if (filterMethod == 1)
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			}
			break;
		case FILTER_NEAREST:
			mipMapMethod=1;
			if (filterMethod == 1)
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
			}
			break;
		case FILTER_LINEAR:
			mipMapMethod=2;
			if (iter->second.numMipMaps == 1)
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			}
			break;
		default:
			assert(false);
			break;
		}
	}
}

// -----------------------------------------------------------------------------
// ----------------------- TextureManager::createTexture -----------------------
// -----------------------------------------------------------------------------
unsigned int TextureManager::CreateTexture(const unsigned int& Format, unsigned char **pixels, const unsigned int& width, const unsigned int& height, const unsigned int& numMipMaps, const bool& compressed, const bool& isSkyBoxTexture)
{
	unsigned int texObject;

	unsigned int _width = width;
	unsigned int _height = height;

	glGenTextures(1, &texObject);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindTexture(GL_TEXTURE_2D, texObject);

	if (isSkyBoxTexture)
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}
	else
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	}

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	if (numMipMaps == 1)
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	}
	else
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	}

	float maxAnisotrophy;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotrophy);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAnisotrophy);

	if (compressed)
	{
		switch (Format)
		{
		case GL_ALPHA:
			for (unsigned int i=0; i<(numMipMaps); i++)
			{
				glTexImage2D(GL_TEXTURE_2D, i, GL_ALPHA, _width, _height, 0, GL_ALPHA, GL_UNSIGNED_BYTE, pixels[i]);
				_width /= 2;
				_height /= 2;
			}
			break;
		case GL_RGB:
			for (unsigned int i=0; i<(numMipMaps); i++)
			{
				glTexImage2D(GL_TEXTURE_2D, i, GL_COMPRESSED_RGB_S3TC_DXT1_EXT, _width, _height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels[i]);
				_width /= 2;
				_height /= 2;
			}
			break;
		case GL_RGBA:
			for (unsigned int i=0; i<(numMipMaps); i++)
			{
				glTexImage2D(GL_TEXTURE_2D, i, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels[i]);
				_width /= 2;
				_height /= 2;
			}
			break;
		default:
			assert(false);
			break;
		}
	}
	else {
		for (unsigned int i=0; i<(numMipMaps); i++)
		{
			glTexImage2D(GL_TEXTURE_2D, i, Format, _width, _height, 0, Format, GL_UNSIGNED_BYTE, pixels[i]);
			_width /= 2;
			_height /= 2;
		}
	}

	return texObject;
}

unsigned int TextureManager::CreateCubeMap(unsigned char** pixels[6], const unsigned int& width, const unsigned int& height, const unsigned int& numMipMaps)
{
	unsigned int texObject;

	unsigned int _width;
	unsigned int _height;

	glGenTextures(1, &texObject);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glBindTexture(GL_TEXTURE_CUBE_MAP, texObject);

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	//glEnable(GL_TEXTURE_GEN_S);
	//glEnable(GL_TEXTURE_GEN_T);
	//glEnable(GL_TEXTURE_GEN_R);

	//glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
	//glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
	//glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP); 

	int i;
	for (i=0; i<6; i++)
	{
		if (numMipMaps == 1)
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		else
			glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		_width = width;
		_height = height;

		unsigned int j;
		for (j=0; j<numMipMaps; j++)
		{
			glTexImage2D(cubefaces[i], j, GL_RGBA, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels[i][j]);
			_width /= 2;
			_height /= 2;
		}
	}

	return texObject;
}