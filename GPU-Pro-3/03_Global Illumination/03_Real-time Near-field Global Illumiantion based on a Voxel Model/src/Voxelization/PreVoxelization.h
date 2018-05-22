///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#ifndef PREVOXELIZATION_H
#define PREVOXELIZATION_H

#include "OpenGL.h"

class AtlasRenderer;
class AtlasVoxelization;
class PreVoxelizationData;
class ShaderProgram;

class PreVoxelization
{
public:
   PreVoxelization(AtlasRenderer* atlasRenderer, PreVoxelizationData* preVoxData);

   /// Preprocess: Voxelize all static scene elements
   void voxelizeStaticSceneElements();

   /// Updates static and final texture resolution,
   /// voxelizes static again.
   void update();

   /// Runtime: Voxelize all dynamic objects by inserting them into a copy of the static voxel texture
   void insertDynamicObjects();

   GLuint getStaticVoxelTexture() const { return mStaticVoxelTexture; }
   GLuint getFinalVoxelTexture() const  { return mFinalVoxelTexture; }

private:
   PreVoxelization() {}

   void createFBO();
   void createShader();

   void createAndAttachVoxelTextures();

   /// Copy the original voxel texture containing only static voxels
   void copyVoxelTexture();

   /// Voxel insertion (all dynamic objects are inserted by atlas voxelization)
   void atlasVoxelizeDynamicObjects();

   /// FBO and voxel textures
   GLuint fboVoxel; 
   GLuint mStaticVoxelTexture; ///< a single binary voxel texture with 128 depth bits containing static voxelized objects
   GLuint mFinalVoxelTexture; ///< contains static and dynamic voxelized objects

   /// This shader just copies a single unsigned integer texture
   ShaderProgram* pCopyVoxelTexture;

   /// The texture atlas renderer whose results (atlases) are used for pre-voxelization
   AtlasRenderer* mAtlasRenderer;
   // Defines voxelization camera and resolution
   PreVoxelizationData* mPreVoxData;
 
};

#endif
