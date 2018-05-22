///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#ifndef PREVOXELIZATIONDATA_H
#define PREVOXELIZATIONDATA_H

#include <vector>
using std::vector;

#include "OpenGL.h"
#include "glm/glm.hpp"

class Camera;

struct Cube3D
{
   float minX;
   float minZ;
   float minY;
   float sideLength;
};

/// Holds information about voxel camera setups
/// and voxel texture resolutions
/// (depending on scene extensions).

class PreVoxelizationData
{
public:
   PreVoxelizationData();
   
   /// @param dx dy dz scene's bounding box dimensions 
   /// @param centerX centerY centerZ scene's bounding box center (world coordinates)
   /// @param requestedXYResolution 
   void computeVoxelTextureData(float dx, float dy, float dz,
      float centerX, float centerY, float centerZ, unsigned int requestedXYResolution);

   // COUTs
   void coutVoxelTextureData();

   // GETTER
   const Camera* getVoxelCamera() const      { return mVoxelCamera; }
   const int getVoxelTextureResolution() const  { return mVoxelTextureResolution; }

private:
   static int getNextPowerOfTwo(int  value);

   //--- Single Voxel texture for mipmapping  --- //

   Camera* mVoxelCamera; /// frustum defines the area to be voxelized; may be non-cubic
   int mVoxelCameraLookAlong; // 0 = X Axis, 1 = Y Axis, 2 = Z Axis (world space)

   int mVoxelTextureResolution;

   glm::ivec3 mVoxelGridRes;
};


#endif
