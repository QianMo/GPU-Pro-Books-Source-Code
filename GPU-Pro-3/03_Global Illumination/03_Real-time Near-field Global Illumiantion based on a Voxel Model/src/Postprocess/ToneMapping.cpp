#include "ToneMapping.h"
#include "Utils/GLError.h"

#include "Qt/Settings.h"

ToneMapping::ToneMapping(int imageWidth, int imageHeight)
: mImageWidth(imageWidth), mImageHeight(imageHeight)
{
   createTextures();
   createFBO();
   createShader();
}

ToneMapping::~ToneMapping(){}

void ToneMapping::createTextures()
{
   int w = mImageWidth;
   int h = mImageHeight;

   mInputTex = EmptyTexture::create2D(w, h, GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT);
}


void ToneMapping::createFBO()
{
   glGenFramebuffersEXT(1, &fboTonemap);
   
}
void ToneMapping::createShader()
{  
   pToneMapSimple = new ShaderProgram("src/shader/Quad.vert", "src/shader/ToneMapSimple.frag");
   pToneMapSimple->useProgram();
   glUniform1i(pToneMapSimple->getUniformLocation("Lrgb"), 0);
}

void ToneMapping::renderToFBO()
{
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboTonemap);
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mInputTex, 0);                                                                           
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, 0, 0);                                                                           
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
}

void ToneMapping::tonemapLinear()
{
   // tone map:

   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

   pToneMapSimple->useProgram();
   glUniform1f(pToneMapSimple->getUniformLocation("maxRadiance"), SETTINGS->getSimpleMaxRadiance());
   glUniform1f(pToneMapSimple->getUniformLocation("gammaExponent"), 1.0f/SETTINGS->getGammaExponent());
   glUniform1i(pToneMapSimple->getUniformLocation("logToneMap"), 0);

   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mInputTex);

   FullScreenQuad::drawComplete();
}

double log2(double d) {return log(d)/log(2.0) ;}

void ToneMapping::tonemapLog()
{
   // tone map:

   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

   pToneMapSimple->useProgram();
   // c = 1.0 / log2(1.0 + maxRadiance)
   glUniform1f(pToneMapSimple->getUniformLocation("c"), 1.0 / log2(1.0 + SETTINGS->getSimpleMaxRadiance()));
   glUniform1f(pToneMapSimple->getUniformLocation("gammaExponent"), 1.0f/SETTINGS->getGammaExponent());
   glUniform1i(pToneMapSimple->getUniformLocation("logToneMap"), 1);

   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mInputTex);

   FullScreenQuad::drawComplete();
}


void ToneMapping::onlyGammaCorrection()
{
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

   ShaderPool::getQuadGamma()->useProgram();
   glUniform1f(ShaderPool::getQuadGamma()->getUniformLocation("gammaExponent"), 1.0f/SETTINGS->getGammaExponent());

   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mInputTex);

   FullScreenQuad::drawComplete();
}