///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#ifndef ATLASVOXELIZATION_H
#define ATLASVOXELIZATION_H

#include "OpenGL.h"

class AtlasRenderer;
class Camera;
class ShaderProgram;

class AtlasVoxelization
{
public:
   AtlasVoxelization(unsigned int volumeResX, unsigned int volumeResY, unsigned int volumeResZ);

   void voxelizeBinary(AtlasRenderer* atlasRenderer, const Camera* const voxelCamera);

   GLuint getBinaryVoxelTexture() { return mBinaryVoxelTexture; }
   
   /// Change the x-y-Dimensions of the binary voxel texture.
   void changeVoxelTextureResolution(unsigned int resX, unsigned int resY);

   unsigned int getVoxelTextureResolutionX() { return mVoxelTextureResX; }
   unsigned int getVoxelTextureResolutionY() { return mVoxelTextureResY; }
   unsigned int getVoxelTextureResolutionZ() { return 128; }

   static ShaderProgram* getAtlasToBinaryShader();

private:
   AtlasVoxelization();

   void createFBO();
   void createAndAttachVoxelTexture();
   void createAndAttachAtlases();
   void createShader();

   GLuint fboBinaryVoxels; 
   GLuint mBinaryVoxelTexture;

   unsigned int mVoxelTextureResX;
   unsigned int mVoxelTextureResY;

   ShaderProgram* pAtlasToBinaryVoxels;

};

#endif
