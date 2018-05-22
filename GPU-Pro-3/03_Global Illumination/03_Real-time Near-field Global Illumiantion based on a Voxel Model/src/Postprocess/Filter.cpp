#include "Filter.h"

#include "Utils/EmptyTexture.h"
#include "Utils/FBOUtil.h"
#include "Utils/FullScreenQuad.h"
#include "Utils/GLError.h"
#include "Utils/ShaderPool.h"
#include "Utils/ShaderProgram.h"

#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include "Qt/Settings.h"

Filter::Filter()
{
   createShader();
   createFBO();
}


void Filter::createShader()
{
   pSpatialUpsampling = new ShaderProgram("src/shader/Quad.vert", "src/shader/SpatialUpsampling.frag");
   pSurfaceDetail   = new ShaderProgram("src/shader/Quad.vert", "src/shader/AddSurfaceDetail.frag");
}

void Filter::createFBO()
{
   mTexFilteredX = EmptyTexture::create2D(SCENE->getWindowWidth(), SCENE->getWindowHeight(), GL_RGB16F_ARB, GL_RGB, GL_FLOAT, GL_LINEAR, GL_LINEAR);
   mTexFilteredXY = EmptyTexture::create2D(SCENE->getWindowWidth(), SCENE->getWindowHeight(), GL_RGB16F_ARB, GL_RGB, GL_FLOAT, GL_LINEAR, GL_LINEAR);

   glGenFramebuffersEXT(1, &fboFilter);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboFilter);

   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mTexFilteredX, 0);                                                                           
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, mTexFilteredXY, 0);                                                                           

   glDrawBuffers(2, FBOUtil::buffers01);
   glClearColor(0,0,0,0);
   glClear(GL_COLOR_BUFFER_BIT);

   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

   std::cout << "  <>   [FBO Status] Filter: " << checkFramebufferStatus()<< endl;

   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void Filter::addSurfaceDetail(GLuint inputTex)
{
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboFilter);
// input is a filtered texture, so the inputTex may equal mTexFilteredXY
   if(inputTex == mTexFilteredXY)
   {
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT); // draw to mTexFilteredX
   }
   else
   {
      glDrawBuffer(GL_COLOR_ATTACHMENT1_EXT); // draw to mTexFilteredXY
   }

   FullScreenQuad::setupRendering();

   pSurfaceDetail->useProgram();
	glUniform1i(pSurfaceDetail->getUniformLocation("positionBuffer"), 0); 
	glUniform1i(pSurfaceDetail->getUniformLocation("normalBuffer"), 1);
	glUniform1i(pSurfaceDetail->getUniformLocation("inputTex"), 2);
	glUniform1f(pSurfaceDetail->getUniformLocation("alpha"), SETTINGS->getSurfaceDetailAlpha());
   glUniform3fv(pSurfaceDetail->getUniformLocation("cameraPosWorldSpace"), 1, &SCENE->getCamera()->getEye()[0]);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, inputTex);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(NORMAL));

   glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(POSITION));

   FullScreenQuad::drawOnly();


   // now copy result to mTexFilteredXY again
   if(inputTex == mTexFilteredXY)
   {
      glDrawBuffer(GL_COLOR_ATTACHMENT1_EXT);
      ShaderPool::getQuad();
	   glBindTexture(GL_TEXTURE_2D, mTexFilteredX); // unit 0
      FullScreenQuad::drawOnly();

   }

   FullScreenQuad::resetRendering();

}


void Filter::upsampleSpatial(GLuint inputTex, int inputWidth, int inputHeight)
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboFilter);

   // X Pass
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(POSITION));
	V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_NEAREST));
   V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_NEAREST));

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(NORMAL));
	V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_NEAREST));
   V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_NEAREST));

	glActiveTexture(GL_TEXTURE3);
   glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(MATERIAL, SETTINGS->getCurrentILBufferSize()));
	V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR));
   V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR));

   glActiveTexture(GL_TEXTURE4);
   glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(MATERIAL));
	V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_NEAREST));
   V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_NEAREST));

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, inputTex);
	V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR));
   V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR));

   pSpatialUpsampling->useProgram();

	glUniform1i(pSpatialUpsampling->getUniformLocation("positionBuffer"), 0); 
	glUniform1i(pSpatialUpsampling->getUniformLocation("normalBuffer"), 1);
   glUniform1i(pSpatialUpsampling->getUniformLocation("materialBufferHighRes"), 4);
   glUniform1i(pSpatialUpsampling->getUniformLocation("inputTex"), 2);
   glUniform1i(pSpatialUpsampling->getUniformLocation("materialBufferLowRes"), 3);
	glUniform1i(pSpatialUpsampling->getUniformLocation("filterRadius"), SETTINGS->getFilterRadius());
	glUniform1f(pSpatialUpsampling->getUniformLocation("distanceLimit_sqr"), SETTINGS->getFilterDistanceLimit() * SETTINGS->getFilterDistanceLimit() );
	glUniform1f(pSpatialUpsampling->getUniformLocation("normalLimit_sqr"),   SETTINGS->getFilterNormalLimit()   * SETTINGS->getFilterNormalLimit());
   glUniform1f(pSpatialUpsampling->getUniformLocation("materialLimit"),   SETTINGS->getFilterMaterialLimit());

	glUniform1f(pSpatialUpsampling->getUniformLocation("lowResDiag"),  sqrt(float(inputWidth * inputWidth + inputHeight * inputHeight)));
   // normalLimit = cos(90.0 - value)

   FullScreenQuad::setupRendering();

	// x-dir
   glUniform2f(pSpatialUpsampling->getUniformLocation("filterDirection"), 1.0f/inputWidth, 0);
   FullScreenQuad::drawOnly();

	//y-dir
	//write blurred result to mTexFilteredXY
   glDrawBuffer(GL_COLOR_ATTACHMENT1_EXT);
   glBindTexture(GL_TEXTURE_2D, mTexFilteredX); // result of x-pass
   glUniform2f(pSpatialUpsampling->getUniformLocation("filterDirection"), 0, 1.0f/inputHeight); 
	FullScreenQuad::drawOnly();


	glUniform1f(pSpatialUpsampling->getUniformLocation("lowResDiag"),  sqrt(float(SCENE->getWindowWidth() * SCENE->getWindowWidth() + SCENE->getWindowHeight() * SCENE->getWindowHeight())));
	glUniform1i(pSpatialUpsampling->getUniformLocation("filterRadius"), SETTINGS->getFilterIterationRadius());

   for(int i = 0; i < SETTINGS->getFilterIterations() - 1; i++)
   {
      // x-dir
      glUniform2f(pSpatialUpsampling->getUniformLocation("filterDirection"), 1.0f/SCENE->getWindowWidth(), 0);
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
      glBindTexture(GL_TEXTURE_2D, mTexFilteredXY); // result of xy-pass
      FullScreenQuad::drawOnly();

      //y-dir
      //write blurred result to mTexFilteredXY
      glDrawBuffer(GL_COLOR_ATTACHMENT1_EXT);
      glBindTexture(GL_TEXTURE_2D, mTexFilteredX); // result of x-pass
      glUniform2f(pSpatialUpsampling->getUniformLocation("filterDirection"), 0, 1.0f/SCENE->getWindowHeight());
      FullScreenQuad::drawOnly();
   }

   FullScreenQuad::resetRendering();
}


