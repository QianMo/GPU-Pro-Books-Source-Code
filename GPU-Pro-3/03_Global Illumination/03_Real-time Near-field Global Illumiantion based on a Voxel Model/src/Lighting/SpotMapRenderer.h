#ifndef SPOTMAPRENDERER_H
#define SPOTMAPRENDERER_H

#include "OpenGL.h"

#include <iostream>
#include <vector>

using namespace std;

class Scene;
class ShaderProgram;

/// Standard shadow maps for hard shadows.
/// "Spot Maps" (RSM) for indirect light
class SpotMapRenderer
{
public:
   SpotMapRenderer(int shadowMapResolution, int spotMapResolution);

   void addSpotLight();
   void deleteSpotLight(int spotLightIndex);

   void createShadowMap(int index);
   void createSpotMap(int index, bool renderPosNormMat);
   void changeMapResolution(int delta);

   void modifyOffsetFactor(float x) {mOffsetFactor += x; cout << "offset factor " << mOffsetFactor << endl;}
   void modifyOffsetUnits(float x)  {mOffsetUnits += x;}


   GLuint getSpotMap(SpotBuffer buffer, int index) const { return mMap.at(index).at(buffer); }
   GLuint getDepthSpotMap(int index)   const {return mMapDepth.at(index);}
   GLuint getShadowMap(int index)   const {return mShadowMaps.at(index);}

   int getSpotMapResolution() const   { return mSpotMapResolution; }
   int getShadowMapResolution() const { return mShadowMapResolution; }

   float getOffsetFactor() const {return mOffsetFactor;}

private:

   SpotMapRenderer() {}
   void createShadowTextures();
   void createMapTextures();
   void createFBO();
   void attachMapTexturesToFBO();
   void createShader();
   void updateLookupMatrices();

   int mNumSpotMaps;

   // texture resolutions
   int mShadowMapResolution;
   int mSpotMapResolution;

   GLfloat* mLookupMatrices; // lookup matrices for spot maps

   // shadow mapping parameters
   float mOffsetFactor;
   float mOffsetUnits;

   //fbo and shadow map texture
   GLuint fboShadow;
   vector<GLuint> mShadowMaps; ///< seperate shadow maps for higher resolution than the mapDepths

   GLuint fboMap;
   vector<GLuint> mMapDepth;

   vector<vector<GLuint> > mMap;

   ShaderProgram* pSpotMap;
};

#endif
