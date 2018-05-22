#ifndef FILTER_H
#define FILTER_H

#include "OpenGL.h"

class ShaderProgram;
class EmptyTexture;

class Filter
{
public:
   Filter();

   void upsampleSpatial(GLuint inputTex, int inputWidth, int inputHeight);
   void addSurfaceDetail(GLuint inputTex);

   GLuint getResult() const { return mTexFilteredXY; }

private:

   void createShader();
   void createFBO();

   GLuint fboFilter;
   GLuint mTexFilteredX;  ///< filtered in X direction
   GLuint mTexFilteredXY; ///< filtered in X and Y direction
 
   ShaderProgram* pSpatialUpsampling;
   ShaderProgram* pSpatialUpsamplingOnePass;

   ShaderProgram* pSurfaceDetail;
};

#endif
