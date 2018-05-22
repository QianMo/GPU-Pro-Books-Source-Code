///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#ifndef INDIRECTLIGHT_H
#define INDIRECTLIGHT_H

#include <vector>

#include "Qt/Settings.h"
#include "OpenGL.h"

class Camera;
class EnvMap;
class ShaderProgram;
class Scene;

using namespace std;

class IndirectLight
{
public:
   IndirectLight();

   const glm::vec2* getHammersleySequence();

   // Computes indirect light with directional occlusion from environment map
   // via hierarchical voxel texture intersection test
   void computeVGIWithMipmapTest(GLuint binaryVoxelTexture,
      int voxelTextureResolution, int maxMipmapLevel,
      const Camera* const voxelCamera, EnvMap* envMap);

   // Computes global ambient color from RSM
   void computeAmbientTerm();
   
   // Sets the size of the indirect light buffer to FULL, HALF or QUARTER
   void setBufferSize(BufferSize size);
   BufferSize getCurrentBufferSize() const { return mCurrentBufferSize; }

   int getCurrentBufferWidth() const { return mBufferWidth.at(mCurrentBufferSize); }
   int getCurrentBufferHeight() const { return mBufferHeight.at(mCurrentBufferSize); }

   GLuint getResult() const { return mIndirectLightResultBuffer.at(mCurrentBufferSize); }

   void createRandom2DTexture();

   static void createPIPixelTexture();

private:
   void createFBO();
   void createShader();

   void createSamplingSequence();

    // set uniform variables for RSM lookup shader
   void setupSpotMapLookupShader(   
      float sampleContrib, float voxelDiagonal,
      int texSlotPositionMap, int texSlotColorMap, int texSlotNormalMap, 
      int texSlotHitBuffer, int texSlotHitRayOriginBuffer, int texSlotMaterialBuffer
      );

   // calls RSM lookup shader for each spot light
   void spotLookupDrawOnly(int texSlotPositionMap, int texSlotColorMap, int texSlotNormalMap); 

   // set uniform variables for voxel intersection test shader
   void setupMipmapTestShaderUniforms(
      ShaderProgram* prog,
      int texSlotVoxelTexture,
      int texSlotBitmaskXORRays,
      int texSlotPositionBuffer,
      int texSlotNormalBuffer,
      int texSlotMaterialBuffer,
      int maxMipMapLevel,
      float voxelDiagonal,
      const Camera* voxelCamera);


   // BUFFER -----------------------------------
   
   BufferSize mCurrentBufferSize; ///< Full, half or quarter resolution
   vector<int> mBufferWidth;  ///< Current result buffer resolution in pixel
   vector<int> mBufferHeight; ///< Current result buffer resolution in pixel

   GLuint fboIndirectLight;
   vector<GLuint> mIndirectLightResultBuffer;
   vector<GLuint> mIntermediateBuffer;
   vector<GLuint> mHitBuffer; 

   GLuint sumTexture;
   
   // SHADER -----------------------------------

   ShaderProgram* pSpotLookup; ///< processes hit buffer, computes luminance emitted from hit point to ray origin
   
   ShaderProgram* pIntersectMipmap; ///< hierarchical intersection test with mipmapped voxel texture

   // For ambient term computation and display
   ShaderProgram* pSum; ///< computes the sum of a given texture
   ShaderProgram* pWriteColorRGB; ///< writes a full screen quad with a given color to the framebuffer

   GLuint mTexRand2D; ///< small 2D texture holding random values in [0, 1]

   // Sampling
   glm::vec2* mHammersleySequence2D; // low-discrepancy sequence for first bounce ray generation
   float mSamplingSequence[MAX_RAYS][2*MAX_RAYS]; // contains the hammersley values for the current number of rays, passed to the shader

};


#endif
