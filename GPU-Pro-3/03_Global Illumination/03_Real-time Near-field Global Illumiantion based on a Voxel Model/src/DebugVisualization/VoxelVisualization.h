///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#ifndef VOXELVISUALIZATION_H
#define VOXELVISUALIZATION_H

#include "OpenGL.h"

#include <iostream>
#include <algorithm> 
#include <cmath>
#include <vector>

#include "glm/glm.hpp"

class Camera;
class IndirectLight;
class PreVoxelizationData;
class ShaderProgram;

using namespace std;

/// This class provides methods for displaying voxel grids,
/// voxelized scenes, 2D and 3D digital lines.
class VoxelVisualization
{
public:
   VoxelVisualization(PreVoxelizationData* preVoxData);
   ~VoxelVisualization();

   /// Draws an axis-aligned box
   static void drawBox(float minX, float minY, float minZ,
      float maxX, float maxY, float maxZ, bool withContour = false);

   /// convenience method that calls the drawBox function above
   /// \param[in] box: minX, minY, minZ, maxX, maxY, maxZ
   static void drawBox(const GLfloat box[6]);

   /// Draws voxels as grey cubes from a binary voxel texture
   /// with given voxelization camera and resolution.
   void drawVoxelsAsCubesInstanced(GLuint voxelTexture,
      unsigned int resX, unsigned int resY, 
      const Camera* const voxelCam);

   void drawVoxelsAsCubesInstancedMipmapped
      (GLuint voxelTexture,
      unsigned int level0_resX, unsigned int level0_resY, 
      const Camera* const voxelCam);

   void drawVoxelsAsCubesInstancedMipmapTestVis
      (GLuint voxelTexture,
      unsigned int level0_resX, unsigned int level0_resY, 
      int maxMipMapLevel,
      const Camera* const voxelCam,
      glm::ivec2 userClickPos,
      const IndirectLight* indirectLight);


   void rayCastBinaryVoxelTexture(GLuint voxelTexture,
      const Camera* voxelCamera);

   void rayCastMipmappedBinaryVoxelTexture(GLuint voxelTexture,
      unsigned int resolution, int level,
      const Camera* voxelCamera);

   static void drawCylinderFromTo(GLUquadric* quad, float radius, glm::vec3 from, glm::vec3 to);

   static float voxelAlpha;

private:
   VoxelVisualization() {}
   double log2(double x);

   static GLuint* binarySlicemapData;
   static GLfloat* volumeData;
   static const int range; ///< number of display lists

   void createFBO();
   void createShader();
   void renderPositionsTextures(const Camera* voxelCamera, bool withNearPlane);
   
   bool intersectBits(glm::uvec4 bitRay, glm::ivec2 texel, int level);

   bool IntersectHierarchy(int level, float& tFar);
   glm::uvec4 getBitRay(float z1, float z2);

   void drawSphereAt(float radius, float posX, float posY, float posZ);
   void drawSphereAt(float radius, glm::vec3 pos);
   void drawCylinderFromTo(float radius, float fromX, float fromY, float fromZ, float toX, float toY, float toZ);
   void drawCylinderFromTo(float radius, glm::vec3 from, glm::vec3 to);
   void drawCylinderBox(float cylinderRadius, glm::vec3 box_min, glm::vec3 box_max);
   void drawCone(float baseRadius, glm::vec3 from, glm::vec3 to);
   void drawTorusAt(float innerRadius, float outerRadius, glm::vec3 pos, glm::vec3 rotAxis);
   
   PreVoxelizationData* mPreVoxData;
   
   GLUquadric* mQuadric; // for sphere and cylinder

   /// For volume rendering of voxel textures
   //@{
   GLuint fboCubePositions;
   GLuint mDepthRenderBuffer;

   GLuint mTexPositionRayStart;   
   GLuint mTexPositionRayEnd;
   //@}

   GLuint fboRaycastingResult;
   GLuint mTexRaycastingResult;

   GLuint fboBitRay;
   GLuint mTexBitRay;

   ShaderProgram* pCreatePositions;
   
   ShaderProgram* pRayCastingSlicemap;
   ShaderProgram* pRayCastingSlicemapMipmap; 
   ShaderProgram* pRayCastingSlicemapDirectLight;

   ShaderProgram* pInstancedCubes;

   ShaderProgram* pBitRay;


   // for instanced cube rendering
   GLuint tboTranslate;
   GLuint vboTranslate;

   GLuint* mSliceData;
   int mCurrentSliceSize;
   bool mFirstSliceRun;
   int mNumVoxels;


};

#endif
