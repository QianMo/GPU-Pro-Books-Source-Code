///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#ifndef MIPMAPRENDERER_H
#define MIPMAPRENDERER_H

#include "OpenGL.h"

#include <cmath>

class ShaderProgram;

/// Creates custom mipmaps for binary voxel textures 
/// generated from voxelization.

class MipmapRenderer
{
public:
   MipmapRenderer();

   void renderMipmapsFor(GLuint voxelTexture, unsigned int resolution);

   void attachMipmapsTo(GLuint voxelTexture, unsigned int resolution);

   static bool isPowerofTwo(unsigned int n);
   static unsigned int computeMaxLevel(unsigned int resolution);

private:
   static ShaderProgram* getMipmapShaderProgram();

   void createFBO();
   void createShader();

   GLuint fboMipmap;

   ShaderProgram* pMipmap;

};


#endif
