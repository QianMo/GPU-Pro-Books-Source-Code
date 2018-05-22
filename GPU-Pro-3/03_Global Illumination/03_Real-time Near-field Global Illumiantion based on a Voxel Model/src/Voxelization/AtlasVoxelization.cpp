///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#include "AtlasVoxelization.h"

#include "Scene/Camera.h"
#include "Scene/ObjectSequence.h"
#include "Scene/Scene.h"

#include "Utils/EmptyTexture.h"
#include "Utils/ShaderProgram.h" 
#include "Utils/TexturePool.h"

#include "Voxelization/AtlasRenderer.h"

#include "Utils/GLError.h"
//#include "glm/gtx/string_cast.hpp" // cast glm types with glm::string() to std::string

AtlasVoxelization::AtlasVoxelization(unsigned int volumeResX, unsigned int volumeResY, unsigned int volumeResZ)
: mVoxelTextureResX(volumeResX),
  mVoxelTextureResY(volumeResY)
{
   createFBO();
   createShader();
}


void AtlasVoxelization::createFBO()
{
   // Binary voxel texture
   V(glGenFramebuffersEXT(1, &fboBinaryVoxels));

   createAndAttachVoxelTexture();

   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

}


void AtlasVoxelization::createAndAttachVoxelTexture()
{
   // Binary Voxel Texture
   mBinaryVoxelTexture = EmptyTexture::create2D(mVoxelTextureResX, mVoxelTextureResY, GL_RGBA32UI_EXT, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT);
	
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboBinaryVoxels));
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mBinaryVoxelTexture, 0));                                                                           

	// Draw Buffers
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
   
   if(glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT) != GL_FRAMEBUFFER_COMPLETE_EXT)
      cout << "  <>   [FBO Status] Atlas Binary Voxel Texture: " << checkFramebufferStatus() << endl;

	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

}


void AtlasVoxelization::changeVoxelTextureResolution(unsigned int newResX, unsigned int newResY)
{
	if(newResX != mVoxelTextureResX || newResY != mVoxelTextureResY )
   {
		mVoxelTextureResX = newResX;
      mVoxelTextureResY = newResY;
   }
	else
   {
      // nothing to do
		return;
   }

	// detach texture from framebuffer
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboBinaryVoxels));
	V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, 0, 0));

   // delete texture(s)
	glDeleteTextures(1, &mBinaryVoxelTexture);

	createAndAttachVoxelTexture();
}

ShaderProgram* AtlasVoxelization::getAtlasToBinaryShader()
{
   static ShaderProgram* aS = new ShaderProgram("src/shader/AtlasToBinaryVoxels.vert", "src/shader/AtlasToBinaryVoxels.frag");
   return aS;
}

void AtlasVoxelization::createShader()
{
   pAtlasToBinaryVoxels = getAtlasToBinaryShader();
   pAtlasToBinaryVoxels->useProgram();
   V(glUniform1i(pAtlasToBinaryVoxels->getUniformLocation("textureAtlas"), 0)); // slot 0
	V(glUniform1i(pAtlasToBinaryVoxels->getUniformLocation("bitmask"), 1)); // slot 1 

   glUseProgram(0);

}


void AtlasVoxelization::voxelizeBinary(AtlasRenderer* atlasRenderer,
                                       const Camera* const voxelCamera)
{
   glPushAttrib(GL_VIEWPORT_BIT);

   // render to voxel texture
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboBinaryVoxels));

   // set viewport according to voxel texture resolution
   V(glViewport(0, 0, mVoxelTextureResX, mVoxelTextureResY));

   // clear unsigned int texture bound to fboBinaryVoxels
   V(glClearColorIuiEXT(0, 0, 0, 0));
	V(glClear(GL_COLOR_BUFFER_BIT)); 

	glLogicOp(GL_OR); // fragments are ORed
	V(glEnable(GL_COLOR_LOGIC_OP));
	//V(glEnable(GL_DEPTH_CLAMP_NV));

   pAtlasToBinaryVoxels->useProgram();
   V(glUniform1i(pAtlasToBinaryVoxels->getUniformLocation("textureAtlas"), 0)); // slot 0
	V(glUniform1i(pAtlasToBinaryVoxels->getUniformLocation("bitmask"), 1)); // slot 1 

   // bind one-dim. lookup texture that is used in the shader for determining 
   // the position of the filled voxels within the voxel grid
	V(glActiveTexture(GL_TEXTURE1));
   V(glBindTexture(GL_TEXTURE_1D, TexturePool::getTexture("bitmaskOR")));

   glUniformMatrix4fv(pAtlasToBinaryVoxels->getUniformLocation("viewProjMatrixVoxelCam"), 1, GL_FALSE, &voxelCamera->getViewProjectionMatrix()[0][0]);

   glActiveTexture(GL_TEXTURE0);

   for(unsigned int e = 0; e < SCENE->getSceneElements().size(); e++)
   {
      for(unsigned int inst = 0; inst < SCENE->getSceneElements().at(e)->getNumInstances(); inst++)
      {
         // read from position atlas
         V(glBindTexture(GL_TEXTURE_2D, atlasRenderer->getTextureAtlas(e, inst))); // position atlas

         // render vertices (one for each valid atlas texel)
         V(glCallList(atlasRenderer->getPixelDisplayList(e)));
      }
   }

   // restore states after drawing:
	V(glDisable(GL_COLOR_LOGIC_OP));
	//V(glDisable(GL_DEPTH_CLAMP_NV));

   // reset viewport
   glPopAttrib();

}

