///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#include "PreVoxelizationData.h"

#include "Scene/Camera.h"
#include "Utils/CoutMethods.h"
#include "Voxelization/MipmapRenderer.h" // for computeMaxLevel

PreVoxelizationData::PreVoxelizationData()
{
   mVoxelCamera = 0;
}


void PreVoxelizationData::computeVoxelTextureData(float dx, float dy, float dz,
                                                  float centerX, float centerY, float centerZ,
                                                  unsigned int requestedXYResolution)
{

   // for mipmapping, voxel texture must have resolution (2^n x 2^n)
   if(!MipmapRenderer::isPowerofTwo(requestedXYResolution))
   {
      std::cerr << "[PreVoxelizationData] Voxel texture resolution must be 2^n x 2^n!" << std::endl;
      std::cerr << "                      Change " << requestedXYResolution << " to ";
      requestedXYResolution = getNextPowerOfTwo(requestedXYResolution);
      std::cerr << requestedXYResolution  << std::endl;
   }

   if(mVoxelCamera != 0) delete mVoxelCamera;
   mVoxelCamera = new Camera();
   // default cam is in (0, 0, 0) and looks along (0, 0, -1)

   float lookAlongDim = dz;
   mVoxelCameraLookAlong = 2; // Z

   bool lookAlongShortestDimension = requestedXYResolution >= 128;

   if( lookAlongShortestDimension && (dx < lookAlongDim)
   || !lookAlongShortestDimension && (dx > lookAlongDim))
   {
      lookAlongDim = dx;
      mVoxelCameraLookAlong = 0; // X
   }
   if( lookAlongShortestDimension && (dy < lookAlongDim)
   || !lookAlongShortestDimension && (dy > lookAlongDim))
   {
      lookAlongDim = dy;
      mVoxelCameraLookAlong = 1; // Y
   }

   // voxel length is lookAlongDim / 128.0
   const float voxelLength = lookAlongDim / 128.0f;

   mVoxelGridRes = glm::ivec3(0);
   mVoxelTextureResolution = requestedXYResolution;

   // center of static bounding box (centerX, centerY, centerZ)

   float left, right, top, bottom, zNear, zFar;

   // make voxelization frustum a little bit bigger than the scene bb
   float scaledDx = dx + 2.0f * dx / ( mVoxelCameraLookAlong == 0 ? 128 : mVoxelTextureResolution );
   float scaledDy = dy + 2.0f * dy / ( mVoxelCameraLookAlong == 1 ? 128 : mVoxelTextureResolution );
   float scaledDz = dz + 2.0f * dz / ( mVoxelCameraLookAlong == 2 ? 128 : mVoxelTextureResolution );

   switch(mVoxelCameraLookAlong)
   {
   case 0:     // X

      mVoxelGridRes.x = 128;

      left  = -scaledDz / 2.0f;
      right = scaledDz / 2.0f;
      bottom = -scaledDy / 2.0f;
      top    = scaledDy / 2.0f;
      zNear  = 1.0f;
      zFar   = 1.0f + scaledDx;

      mVoxelCamera->setAngleY(-90);
      mVoxelCamera->move(-centerZ, centerY, -(centerX + scaledDx / 2.0f + 1.0f));

      break;

   case 1:     // Y

      mVoxelGridRes.y = 128;

      left  = -scaledDx / 2.0f;
      right = scaledDx / 2.0f;
      bottom = -scaledDz / 2.0f;
      top    = scaledDz / 2.0f;
      zNear  = 1.0f;
      zFar   = 1.0f + scaledDy;

      mVoxelCamera->setAngleX(90);
      mVoxelCamera->move(centerX, -centerZ, -(centerY + scaledDy / 2.0f + 1.0f));

      break;

   case 2:     // Z 

      mVoxelGridRes.z = 128;

      left  = -scaledDx / 2.0f;
      right = scaledDx / 2.0f;
      bottom = -scaledDy / 2.0f;
      top    = scaledDy / 2.0f;
      zNear  = 1.0f;
      zFar   = 1.0f + scaledDz;

      // no rotation necessary
      mVoxelCamera->move(centerX, centerY, -(centerZ + scaledDz / 2.0f + 1.0f));

      break;
   }

   if(mVoxelGridRes.x == 0)
      mVoxelGridRes.x = mVoxelTextureResolution;
   if(mVoxelGridRes.y == 0)
      mVoxelGridRes.y = mVoxelTextureResolution;
   if(mVoxelGridRes.z == 0)
      mVoxelGridRes.z = mVoxelTextureResolution;


   mVoxelCamera->setOrthographicFrustum(left, right, bottom, top, zNear, zFar);
}


int PreVoxelizationData::getNextPowerOfTwo(int value)
{
   int result = static_cast<int>(pow(2.0, int(ceil(log(double(value)) / log(2.0)))));
   int diffSmaller = int(value) - result / 2;
   int diffBigger  = result - int(value);
   //cout << "diffSmaller: " << diffSmaller << endl;
   //cout << "diffBigger:  " << diffBigger << endl;
   if(diffSmaller <= diffBigger)
      result /= 2;
   cout << "Get next power of 2 for: " << value << " ==> " << result << endl;
   return result;
}

void PreVoxelizationData::coutVoxelTextureData()
{

   cout << endl;
   cout << "Voxel Texture with mipmapping constraints  " << endl;
   cout << "Voxel Dimensions: " << 
            mVoxelCamera->getFrustum().width  / mVoxelTextureResolution 
            << " x " <<
            mVoxelCamera->getFrustum().height / mVoxelTextureResolution
            << " x " <<
            mVoxelCamera->getFrustum().zRange / 128 << endl;
   cout << "Voxelization camera looking along axis: " << ((mVoxelCameraLookAlong == 0) ? "X" : ((mVoxelCameraLookAlong == 1) ? "Y" : "Z")) << endl;
   cout << "Resolution: " << mVoxelTextureResolution << " x " << mVoxelTextureResolution << endl;
   cout << "Grid: " << mVoxelGridRes.x << " x " << mVoxelGridRes.y << " x " << mVoxelGridRes.z << endl;
   cout << endl;

}
