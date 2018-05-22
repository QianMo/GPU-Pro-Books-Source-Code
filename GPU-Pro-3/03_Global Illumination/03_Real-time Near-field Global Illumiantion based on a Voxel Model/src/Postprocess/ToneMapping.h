#ifndef TONEMAPPING_H
#define TONEMAPPING_H

#include "OpenGL.h"
#include "Utils/EmptyTexture.h"
#include "Utils/ShaderProgram.h"
#include "Utils/ShaderPool.h"
#include "Utils/FullScreenQuad.h"

#include <vector>

using namespace std;

class ToneMapping
{
public:
   ToneMapping(int imageWidth, int imageHeight);
   ~ToneMapping();

   void renderToFBO();
   void tonemapLinear(); // linear tone mapping with gamma correction
   void tonemapLog();    // log tone mapping with gamma correction
   void onlyGammaCorrection();    // only gamma correction
   GLuint getResult();
   

private:
   void createTextures();
   void createFBO();
   void createShader();

   int mImageWidth, mImageHeight;

   GLuint fboTonemap;
   GLuint mInputTex;

   ShaderProgram* pToneMapSimple;

};

#endif
