///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#include "Bitmask.h"

#include "Utils/EmptyTexture.h"
#include "Utils/FullScreenQuad.h"
#include "Utils/GLError.h"
#include "Utils/ShaderPool.h"
#include "Utils/TexturePool.h"

#include <cmath>

////////////// BITMASK CREATION ////////////////////////////////

void Bitmask::createBitmasks()
{
   create1DBitmaskOR();
   create1DBitmaskXOR();
   create2DBitmaskXORRays();
}

///
/// Creates the bitmask-texture for z-lookup in voxelization fragment shader
/// 
void Bitmask::create1DBitmaskOR()
{

	int bits = 32; // bits per texture channel
	GLuint bitposition =(GLuint)(pow(2.0,(double)bits-1));//0x80000000U;
	GLuint R,G,B,A;
	int counter = 0;

	vector<GLuint> bitmaskORData;
	for(int i = 0; i< 4*bits ; i++)
	{
		if (counter == bits) // reset
		{
			counter = 0;
			bitposition = (GLuint)(pow(2.0,(double)bits-1));
		}

		if (i < bits) // first 31 texels: 1-bit in R
		{
			R = bitposition;
			G = 0;
			B = 0;
			A = 0;
		}
		if (bits <= i && i < 2*bits) // G
		{
			R = 0;
			G = bitposition;
			B = 0;
			A = 0;
		}
		if (2*bits <= i && i < 3*bits) // B
		{
			R = 0;
			G = 0;
			B = bitposition;
			A = 0;
		}
		if (3*bits <= i && i < 4*bits) // A
		{
			R = 0;
			G = 0;
			B = 0;
			A = bitposition;
		}
		bitmaskORData.push_back(R); // R
		bitmaskORData.push_back(G); // G
		bitmaskORData.push_back(B); // B
		bitmaskORData.push_back(A); // A

		counter++;
		bitposition = bitposition >> 1;

	}

	// assign lookup-data to new 1D texture
   GLuint bitmaskOR;
	glGenTextures(1, &bitmaskOR);
	glBindTexture(GL_TEXTURE_1D, bitmaskOR);
	glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32UI_EXT, 4*bits, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, &bitmaskORData[0]);

   TexturePool::addTexture("bitmaskOR", bitmaskOR);
}

///
/// Creates the bitmask-texture used for creation 
/// of the voxel ray bitmasks.
///
void Bitmask::create1DBitmaskXOR()
{
	int bits = 32; // bits per texture channel
	GLuint all_1 =(GLuint)(pow(2.0,(double)bits)-1);//)0x7FFFFFFFU; // (2^31 -1)
	GLuint shifted_ones = all_1;
	GLuint R,G,B,A;
	int counter = 0;
	vector<GLuint> bitmaskData;
	for(int i = 0; i < 4*bits; i++)
	{
		if (counter == bits) //reset
		{
			counter = 0;
			shifted_ones = all_1;
		}
		
		if (i < bits) // first 31 texels: 1-bit in R
		{
			R = shifted_ones;
			G = all_1;
			B = all_1;
			A = all_1;
		}
		if (bits <= i && i < 2*bits) // G
		{
			R = 0;
			G = shifted_ones;
			B = all_1;
			A = all_1;
		}
		if (2*bits <= i && i < 3*bits) // B
		{
			R = 0;
			G = 0;
			B = shifted_ones;
			A = all_1;
		}
		if (3*bits <= i && i < 4*bits) // A
		{
			R = 0;
			G = 0;
			B = 0;
			A = shifted_ones;
		}
		bitmaskData.push_back(R); // R
		bitmaskData.push_back(G); // G
		bitmaskData.push_back(B); // B
		bitmaskData.push_back(A); // A

		counter++;
		shifted_ones = shifted_ones >> 1;

	}

	// assign lookup-data to new empty tex
   GLuint bitmaskXOR;
	glGenTextures(1, &bitmaskXOR);
	glBindTexture(GL_TEXTURE_1D, bitmaskXOR);
	glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32UI_EXT, 4*bits, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, &bitmaskData[0]);

   TexturePool::addTexture("bitmaskXOR", bitmaskXOR);
}

/// Ray Bitmasks for hierarchical voxel mipmap intersection test
void Bitmask::create2DBitmaskXORRays()
{
   if(TexturePool::getTexture("bitmaskXORRays") != 0)
      return;

   ShaderProgram* pCreateVoxelRays = ShaderPool::getCreateVoxelRaysShaderProgram();

   GLuint bitmaskXORRays = EmptyTexture::create2D(128, 128, GL_RGBA32UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT);

   GLuint fboTemp;

	V(glGenFramebuffersEXT(1, &fboTemp));
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboTemp));
	V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, bitmaskXORRays, 0));                                                                           
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

   glPushAttrib(GL_VIEWPORT_BIT);
   glViewport(0, 0, 128, 128);

   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_1D, TexturePool::getTexture("bitmaskXOR"));

   pCreateVoxelRays->useProgram();
   glUniform1i(pCreateVoxelRays->getUniformLocation("bitmaskXOR"), 0);
   FullScreenQuad::drawComplete();
   glPopAttrib();

   glUseProgram(0);
	V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, 0, 0));                                                                           
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

   V(glDeleteFramebuffersEXT(1, &fboTemp));

   TexturePool::addTexture("bitmaskXORRays", bitmaskXORRays);

}
