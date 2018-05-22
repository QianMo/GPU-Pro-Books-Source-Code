///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#include "PreVoxelization.h"

#include "Common.h"

#include "Scene/Camera.h"
#include "Scene/ObjectSequence.h"
#include "Scene/Scene.h"

#include "Voxelization/AtlasRenderer.h"
#include "Voxelization/AtlasVoxelization.h"
#include "Voxelization/PreVoxelizationData.h"

#include "Utils/EmptyTexture.h"
#include "Utils/FullScreenQuad.h"
#include "Utils/GLError.h"
#include "Utils/ShaderProgram.h"
#include "Utils/TexturePool.h"

PreVoxelization::PreVoxelization(AtlasRenderer* atlasRenderer, PreVoxelizationData* preVoxData)
: mAtlasRenderer(atlasRenderer),
  mPreVoxData(preVoxData),
  mStaticVoxelTexture(0),
  mFinalVoxelTexture(0)
{
   createFBO();
   createShader();
}

void PreVoxelization::createFBO()
{
   glGenFramebuffersEXT(1, &fboVoxel);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboVoxel);
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

   createAndAttachVoxelTextures();

   cout << "  <>   [FBO Status] Pre-Voxelization: " << checkFramebufferStatus() << endl;
}

void PreVoxelization::createAndAttachVoxelTextures()
{
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, 0, 0);

   if(glIsTexture(mStaticVoxelTexture)) glDeleteTextures(1, &mStaticVoxelTexture);
   if(glIsTexture(mFinalVoxelTexture)) glDeleteTextures(1, &mFinalVoxelTexture);

   int voxelTextureWidth  = mPreVoxData->getVoxelTextureResolution();
   int voxelTextureHeight = mPreVoxData->getVoxelTextureResolution();

   mStaticVoxelTexture = EmptyTexture::create2D(voxelTextureWidth, voxelTextureHeight, GL_RGBA32UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT);
   mFinalVoxelTexture  = EmptyTexture::create2D(voxelTextureWidth, voxelTextureHeight, GL_RGBA32UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT);

   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mStaticVoxelTexture, 0);

}

void PreVoxelization::createShader()
{
   pCopyVoxelTexture = new ShaderProgram("src/shader/CopyVoxelTexture.vert", "src/shader/CopyVoxelTexture.frag");
   pCopyVoxelTexture->useProgram();
   glUniform1i(pCopyVoxelTexture->getUniformLocation("voxelTexture"), 0);
}

void PreVoxelization::update()
{
   createAndAttachVoxelTextures();
   voxelizeStaticSceneElements();
}


void PreVoxelization::voxelizeStaticSceneElements()
{
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboVoxel);
   // write to:
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mStaticVoxelTexture, 0);
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT); 

   // We do _not_ have a cubic frustum 

   // set viewport resolution
   glPushAttrib(GL_VIEWPORT_BIT);
   glViewport(0, 0, mPreVoxData->getVoxelTextureResolution(), mPreVoxData->getVoxelTextureResolution());
	// clear 
   V(glClearColorIuiEXT(0, 0, 0, 0));
	V(glClear(GL_COLOR_BUFFER_BIT)); 

   V(glEnable(GL_COLOR_LOGIC_OP));
   glLogicOp(GL_OR); // fragments are ORed
	V(glDisable(GL_DEPTH_TEST));
	V(glDisable(GL_CULL_FACE));

   ShaderProgram* p = AtlasVoxelization::getAtlasToBinaryShader();
   p->useProgram();
   V(glUniform1i(p->getUniformLocation("textureAtlas"), 0)); // slot 0
	V(glUniform1i(p->getUniformLocation("bitmask"), 1)); // slot 1 
   glUniformMatrix4fv(p->getUniformLocation("viewProjMatrixVoxelCam"), 1, GL_FALSE, &mPreVoxData->getVoxelCamera()->getViewProjectionMatrix()[0][0]);

	V(glActiveTexture(GL_TEXTURE1));
   V(glBindTexture(GL_TEXTURE_1D, TexturePool::getTexture("bitmaskOR")));

   // read from position atlas
   glActiveTexture(GL_TEXTURE0);

   p->useProgram();
   
   // loop over static objects
   for(unsigned int e = 0; e < SCENE->getSceneElements().size(); e++)
   {
      if(SCENE->getSceneElements().at(e)->isStatic())
      {
         // read from position atlas
         V(glBindTexture(GL_TEXTURE_2D, mAtlasRenderer->getTextureAtlas(e, 0))); // position atlas
         V(glCallList(mAtlasRenderer->getPixelDisplayList(e)));
      }
   }

   // reset viewport
   glPopAttrib();


   // restore states 
   V(glDisable(GL_COLOR_LOGIC_OP));
}


void PreVoxelization::insertDynamicObjects()
{
   // set viewport size to size of voxel texture
   glPushAttrib(GL_VIEWPORT_BIT);
   glViewport(0, 0, mPreVoxData->getVoxelTextureResolution(), mPreVoxData->getVoxelTextureResolution()); 

   // 1. Copy mStaticVoxelTexture to mFinalVoxelTexture

   copyVoxelTexture();

   // 2. Voxelize dynamic objects (OR the results into the single final texture)

   atlasVoxelizeDynamicObjects();

   glPopAttrib(); // reset viewport

}

void PreVoxelization::copyVoxelTexture()
{
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboVoxel);
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
   // write to:
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mFinalVoxelTexture, 0);

   // read from:
   pCopyVoxelTexture->useProgram();
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mStaticVoxelTexture);

   FullScreenQuad::drawComplete();

}


void PreVoxelization::atlasVoxelizeDynamicObjects()
{
   ShaderProgram* p = AtlasVoxelization::getAtlasToBinaryShader();
   p->useProgram();

   V(glEnable(GL_COLOR_LOGIC_OP));
   glLogicOp(GL_OR); // fragments are ORed
   V(glDisable(GL_DEPTH_TEST));
   V(glDisable(GL_CULL_FACE));

   p->useProgram();
   V(glUniform1i(p->getUniformLocation("textureAtlas"), 0)); // slot 0
   V(glUniform1i(p->getUniformLocation("bitmask"), 1)); // slot 1 
   glUniformMatrix4fv(p->getUniformLocation("viewProjMatrixVoxelCam"), 1, GL_FALSE, &mPreVoxData->getVoxelCamera()->getViewProjectionMatrix()[0][0]);

   // read from position atlas
   V(glActiveTexture(GL_TEXTURE1));
   V(glBindTexture(GL_TEXTURE_1D, TexturePool::getTexture("bitmaskOR")));

   glActiveTexture(GL_TEXTURE0);

   // Loop over dynamic objects
   for(unsigned int e = 0; e < SCENE->getSceneElements().size(); e++)
   {
      if(!SCENE->getSceneElements().at(e)->isStatic())
      {
         for(unsigned int inst = 0; inst < SCENE->getSceneElements().at(e)->getNumInstances(); inst++)
         {
            // read from position atlas
            V(glBindTexture(GL_TEXTURE_2D, mAtlasRenderer->getTextureAtlas(e, inst))); 
            // draw vertices
            V(glCallList(mAtlasRenderer->getPixelDisplayList(e)));
         }
      }
   }

   // restore states after drawing:
   V(glDisable(GL_COLOR_LOGIC_OP));
   //V(glEnable(GL_CULL_FACE));
}
