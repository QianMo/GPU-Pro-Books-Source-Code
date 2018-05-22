///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#include "MipmapRenderer.h"

#include "Scene/Scene.h"
#include "Utils/FullScreenQuad.h"
#include "Utils/GLError.h"
#include "Utils/ShaderProgram.h"

MipmapRenderer::MipmapRenderer()
{
   createFBO();
   createShader();
}


void MipmapRenderer::attachMipmapsTo(GLuint voxelTexture, unsigned int resolution)
{
   unsigned int maxLevel = computeMaxLevel(resolution);
   //cout << "[MipmapRenderer] Attach mipmap to voxel texture with max. level = " << maxLevel << endl;

   // attach mipmap levels to this voxel texture
	glBindTexture(GL_TEXTURE_2D, voxelTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	int div = 2;
	for(unsigned int level = 1; level <= maxLevel; level++)
	{
		glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA32UI_EXT, resolution/div, resolution/div, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, 0);
		div = div << 1;
	}

}

void MipmapRenderer::createFBO()
{
   glGenFramebuffersEXT(1, &fboMipmap);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboMipmap);
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
}

ShaderProgram* MipmapRenderer::getMipmapShaderProgram()
{
   static ShaderProgram* s = new ShaderProgram("src/shader/MipmapVoxelTexture.vert", "src/shader/MipmapVoxelTexture.frag");
   return s;
}

void MipmapRenderer::createShader()
{
   pMipmap = getMipmapShaderProgram();
}


bool MipmapRenderer::isPowerofTwo(unsigned int n)
{
	return ((n & (n - 1)) == 0) && n != 0;
}




unsigned int MipmapRenderer::computeMaxLevel(unsigned int resolution)
{
	unsigned int count = 0;
	while(resolution%2==0)
	{
		resolution /= 2;
		count++;
	}
	return count;
}


void MipmapRenderer::renderMipmapsFor(GLuint voxelTexture, unsigned int resolution)
{
   glPushAttrib(GL_VIEWPORT_BIT);

   unsigned int maxLevel = computeMaxLevel(resolution);

	pMipmap->useProgram();
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboMipmap);

	//start writing to level 1

   //read from:
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, voxelTexture);
   glUniform1i(pMipmap->getUniformLocation("voxelTexture"), 0);
   FullScreenQuad::setupRendering();

   for(unsigned int i = 0; i < maxLevel; i++)
   {
      // read from      
      glUniform1i(pMipmap->getUniformLocation("level"), i);

      //write to:
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, voxelTexture, i+1);                                                                           

      //if(glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT) != GL_FRAMEBUFFER_COMPLETE_EXT)
      //{
      //   cout << "  <>   [FBO Status] MIPMAP " << checkFramebufferStatus() << endl;
      //}

      float invResFloat = static_cast<float>( 1.0/(resolution/pow(2.0,double(i))) );
      glUniform1f(pMipmap->getUniformLocation("inverseTexSize"), invResFloat   );
      glViewport(0, 0, static_cast<GLsizei>(resolution/pow(2.0,double(i+1))), static_cast<GLsizei>(resolution/pow(2.0,double(i+1))));


      FullScreenQuad::drawOnly();
   }
   FullScreenQuad::resetRendering();

	//glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	//glActiveTexture(GL_TEXTURE0);

	// reset viewport
   glPopAttrib();


}