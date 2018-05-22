#include "SpotMapRenderer.h"

#include "Scene/ObjModel.h"
#include "Scene/Scene.h" // for drawing 
#include "Scene/SpotLight.h"

#include "Utils/EmptyTexture.h"
#include "Utils/FBOUtil.h"
#include "Utils/FullScreenQuad.h"
#include "Utils/GLError.h"
#include "Utils/ShaderProgram.h"

SpotMapRenderer::SpotMapRenderer(int shadowMapResolution, int spotMapResolution) 
                   : mShadowMapResolution(shadowMapResolution), mSpotMapResolution(spotMapResolution)
{
   mNumSpotMaps = SCENE->getSpotLights().size();
   mLookupMatrices = new GLfloat[mNumSpotMaps * 16];

	// shadow mapping parameters (plattform-dependent!)
	mOffsetFactor = 6.1f;              
	mOffsetUnits = 1.0f; 

   createShadowTextures();
   createMapTextures();
   createFBO();
   createShader();

}

void SpotMapRenderer::createShader()
{
   pSpotMap = new ShaderProgram("src/shader/SpotMap.vert", "src/shader/SpotMap.frag");
   pSpotMap->useProgram();
   glUniform1i(pSpotMap->getUniformLocation("diffuseTexture"), 0); // slot 0
}

void SpotMapRenderer::createShadowTextures()
{
   // delete if necessary
   for(unsigned int i = 0; i < mShadowMaps.size(); i++)
   {
      if(glIsTexture(mShadowMaps.at(i)))
         glDeleteTextures(1, &mShadowMaps.at(i));
   }
   mShadowMaps.clear();

   // generate
   for(int i = 0; i < mNumSpotMaps; i++)
   {
      mShadowMaps.push_back(EmptyTexture::create2D(mShadowMapResolution, mShadowMapResolution,
         GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT,
         GL_LINEAR, GL_LINEAR));
      EmptyTexture::setComparisonModesShadow(GL_TEXTURE_2D);
   }

}
void SpotMapRenderer::createMapTextures()
{
   // delete if necessary
   for(unsigned int i = 0; i < mMapDepth.size(); i++)
   {
      if(glIsTexture(mMapDepth.at(i)))
         glDeleteTextures(1, &mMapDepth.at(i));
   }
   mMapDepth.clear();

   for(unsigned int m = 0; m < mMap.size(); m++)
   {
      for(unsigned int b = 0; b < mMap.at(m).size(); b++)
      {
         if(glIsTexture(mMap.at(m).at(b)))
            glDeleteTextures(1, &mMap.at(m).at(b));
      }
      mMap.at(m).clear();
   }
   mMap.clear();

   for(int i = 0; i < mNumSpotMaps; i++)
   {
      mMapDepth.push_back(EmptyTexture::create2D(mSpotMapResolution, mSpotMapResolution,
         GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT,
         GL_LINEAR, GL_LINEAR));
      EmptyTexture::setComparisonModesShadow(GL_TEXTURE_2D);

      vector<GLuint> buff;
      //MAP_POSITION, MAP_NORMAL, MAP_MATERIAL, MAP_DIRECTLIGHT

      buff.push_back(EmptyTexture::create2D(mSpotMapResolution, mSpotMapResolution,
         GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT,
         GL_NEAREST, GL_NEAREST));
      EmptyTexture::setClampToBorder(GL_TEXTURE_2D, 0, 0, 0, 1);

      buff.push_back(EmptyTexture::create2D(mSpotMapResolution, mSpotMapResolution,
         GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT,
         GL_NEAREST, GL_NEAREST));
      EmptyTexture::setClampToBorder(GL_TEXTURE_2D, 0, 0, 0, 1);

      buff.push_back(EmptyTexture::create2D(mSpotMapResolution, mSpotMapResolution,
         GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT,
         GL_NEAREST, GL_NEAREST));
      EmptyTexture::setClampToBorder(GL_TEXTURE_2D, 0, 0, 0, 1);

      buff.push_back(EmptyTexture::create2D(mSpotMapResolution, mSpotMapResolution,
         GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT,
         GL_LINEAR, GL_LINEAR)); // or GL_LINEAR, GL_LINEAR for Directlight
      EmptyTexture::setClampToBorder(GL_TEXTURE_2D, 0, 0, 0, 1);

      mMap.push_back(buff);
   }

}

void SpotMapRenderer::createFBO()
{
	// init shadow map (use fbo, not copytex) //////////////////////////

	V(glGenFramebuffersEXT(1, &fboShadow));
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboShadow));

	V(glDrawBuffer(GL_NONE));
	V(glReadBuffer(GL_NONE));
	
	// attach the texture to FBO depth attachment point
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, mShadowMaps.front(), 0));

   cout << "  <>   [FBO Status] Shadow Maps: " << checkFramebufferStatus() << endl;

	V(glGenFramebuffersEXT(1, &fboMap));
	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboMap));
	V(glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT));

   attachMapTexturesToFBO();

   cout << "  <>   [FBO Status] Spot Maps: " << checkFramebufferStatus() << endl;


	V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0));

}

void SpotMapRenderer::changeMapResolution(int delta)
{
   mSpotMapResolution += delta;

   mSpotMapResolution = max(16, mSpotMapResolution);

   // detach everything from fbo
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboMap));
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, 0, 0));
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, 0, 0));
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_2D, 0, 0));
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT3_EXT, GL_TEXTURE_2D, 0, 0));
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, 0, 0));

   createMapTextures();

   SpotLight::changeResolution(mSpotMapResolution);

   for(unsigned int i = 0; i < SCENE->getSpotLights().size(); i++)
   {
      SCENE->getSpotLights().at(i)->updatePixelSide();
   }


   cout << "[Spot Map] Updated resolution : " << mSpotMapResolution << " x " << mSpotMapResolution << endl;
}

void SpotMapRenderer::attachMapTexturesToFBO()
{
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mMap.front().at(MAP_DIRECTLIGHT), 0));
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, mMap.front().at(MAP_POSITION), 0));
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, mMapDepth.front(), 0));

}

void SpotMapRenderer::addSpotLight()
{
   mNumSpotMaps++;

   delete[] mLookupMatrices;
   mLookupMatrices = new GLfloat[mNumSpotMaps * 16];

   mShadowMaps.push_back(EmptyTexture::create2D(mShadowMapResolution, mShadowMapResolution,
      GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT,
      GL_LINEAR, GL_LINEAR));
   EmptyTexture::setComparisonModesShadow(GL_TEXTURE_2D);

   mMapDepth.push_back(EmptyTexture::create2D(mSpotMapResolution, mSpotMapResolution,
      GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT,
      GL_LINEAR, GL_LINEAR));
   EmptyTexture::setComparisonModesShadow(GL_TEXTURE_2D);

   vector<GLuint> buff;
   //MAP_POSITION, MAP_NORMAL, MAP_MATERIAL, MAP_DIRECTLIGHT

   buff.push_back(EmptyTexture::create2D(mSpotMapResolution, mSpotMapResolution,
      GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT,
      GL_NEAREST, GL_NEAREST));
   EmptyTexture::setClampToBorder(GL_TEXTURE_2D, 0, 0, 0, 1);

   buff.push_back(EmptyTexture::create2D(mSpotMapResolution, mSpotMapResolution,
      GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT,
      GL_NEAREST, GL_NEAREST));
   EmptyTexture::setClampToBorder(GL_TEXTURE_2D, 0, 0, 0, 1);

   buff.push_back(EmptyTexture::create2D(mSpotMapResolution, mSpotMapResolution,
      GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT,
      GL_NEAREST, GL_NEAREST));
   EmptyTexture::setClampToBorder(GL_TEXTURE_2D, 0, 0, 0, 1);

   buff.push_back(EmptyTexture::create2D(mSpotMapResolution, mSpotMapResolution,
      GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT,
      GL_LINEAR, GL_LINEAR)); // or GL_LINEAR, GL_LINEAR for Directlight
   EmptyTexture::setClampToBorder(GL_TEXTURE_2D, 0, 0, 0, 1);

   mMap.push_back(buff);


}

void SpotMapRenderer::deleteSpotLight(int spotLightIndex)
{
   mNumSpotMaps--;
   glDeleteTextures(1, &mShadowMaps.at(spotLightIndex));
   mShadowMaps.erase(mShadowMaps.begin() + spotLightIndex);

   glDeleteTextures(1, &mMapDepth.at(spotLightIndex));
   mMapDepth.erase(mMapDepth.begin() + spotLightIndex);

   for(int b = 0; b < 4; b++)
      glDeleteTextures(1, &mMap.at(spotLightIndex).at(b));

   mMap.erase(mMap.begin() + spotLightIndex);


   delete[] mLookupMatrices;
   mLookupMatrices = new GLfloat[mNumSpotMaps * 16];

}



void SpotMapRenderer::createShadowMap(int index)
{
   const SpotLight* spotLight = SCENE->getSpotLights().at(index);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboShadow);
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, mShadowMaps.at(index), 0));

   glPushAttrib(GL_VIEWPORT_BIT);

	// set viewport (shadow resolution)
	glViewport(0, 0, mShadowMapResolution, mShadowMapResolution);  
	glClear(GL_DEPTH_BUFFER_BIT);

	// set light projection
	glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadMatrixf(&spotLight->getProjectionMatrix()[0][0]);
	//end 

	// set light modelview
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
   glLoadMatrixf(&spotLight->getLightViewMatrix()[0][0]);	
   
   //end

	// enable offsets
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(mOffsetFactor, mOffsetUnits);   

	//Disable color rendering, we only want to write to the Z-Buffer
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE); 
   glDepthMask(GL_TRUE); // Enable depth writing

	// Culling switching, rendering only backface, this is done to avoid self-shadowing
   // this does not work for single-face objects
   //glEnable(GL_CULL_FACE);
   //glCullFace(GL_FRONT);

	glDisable(GL_LIGHTING); // for speed

	// now: draw content
   glUseProgram(0);
   glEnable(GL_DEPTH_TEST);
   SCENE->drawAllElementsWithMode(GLM_NONE);
   glDisable(GL_DEPTH_TEST);

   // restore states

	//glCullFace(GL_BACK);
 //  glDisable(GL_CULL_FACE);

	//Enabling color write (previously disabled for light POV z-buffer rendering)
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE); 
   //glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0); // system framebuffer

	// deactivate offsetting
	glDisable(GL_POLYGON_OFFSET_FILL);                          
	
	// reset viewport
   glPopAttrib();

	// reset projection/modelview matrices
	glMatrixMode( GL_PROJECTION );                            
	glPopMatrix();								
	
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();   
}


void SpotMapRenderer::updateLookupMatrices()
{
   // check all spot lights for updates
   for(int i = 0; i < mNumSpotMaps; i++)
   {
      SpotLight* const spot = SCENE->getSpotLights().at(i);
      // if spot map moved or rotated
      if(spot->lookupMatrixChanged())
      {
         // copy new lookup matrix to array
         const GLfloat* matrix = &spot->getMapLookupMatrix()[0][0];
         memcpy(&(mLookupMatrices[i * 16]), matrix, 16*sizeof(GLfloat));
/*         for(int m = 0; m < 16; m++)
         {
            mLookupMatrices[i * 16 + m] = matrix[m];
         }
  */       spot->setChanged(false);
      }
   }
   
}

void SpotMapRenderer::createSpotMap(int index, bool renderPosNormMat)
{
   const SpotLight* spotLight = SCENE->getSpotLights().at(index);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboMap);

   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mMap.at(index).at(MAP_DIRECTLIGHT), 0));
   V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, mMapDepth.at(index), 0));
   if(renderPosNormMat)
   {
      V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, mMap.at(index).at(MAP_POSITION), 0));
      V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_2D, mMap.at(index).at(MAP_NORMAL), 0));
      V(glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT3_EXT, GL_TEXTURE_2D, mMap.at(index).at(MAP_MATERIAL), 0));
      glDrawBuffers(4, FBOUtil::buffers0123);
   }
   else
   {
      V(glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT));

   }
   V(glPushAttrib(GL_VIEWPORT_BIT));

   // set viewport (shadow resolution)
   glViewport(0, 0, mSpotMapResolution, mSpotMapResolution);  
   glClearColor(-1, -1, -1, 0);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set light projection
	glMatrixMode(GL_PROJECTION);
   glPushMatrix();
   glLoadMatrixf(&spotLight->getProjectionMatrix()[0][0]);
	//end 

	// set light modelview
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
   glLoadIdentity();
   //glLoadMatrixf(&spotLight->getLightViewMatrix()[0][0]);	

   
	// enable offsets
	//glEnable(GL_POLYGON_OFFSET_FILL);
	//glPolygonOffset(offsetFactor, offsetUnits);   

	// Culling switching, rendering only backface, this is done to avoid self-shadowing
   //glEnable(GL_CULL_FACE);
   //glCullFace(GL_FRONT);

	// now: draw content
   pSpotMap->useProgram();

   //V(glUniformMatrix4fv(pSpotMap->getUniformLocation("inverseViewMatrix"),
      //1, GL_FALSE, &spotLight->getInverseLightViewMatrix()[0][0]));
   V(glUniformMatrix4fv(pSpotMap->getUniformLocation("viewMatrix"),
      1, GL_FALSE, &spotLight->getLightViewMatrix()[0][0]));

   glUniform3fv(pSpotMap->getUniformLocation("I"), 1, &spotLight->getI()[0]);
   glUniform1f(pSpotMap->getUniformLocation("constantAttenuation"), spotLight->getConstantAttenuation());
   glUniform1f(pSpotMap->getUniformLocation("quadraticAttenuation"), spotLight->getQuadraticAttenuation());

   glUniform1f(pSpotMap->getUniformLocation("spotCosCutoff"), spotLight->getCosCutoffAngle());
   glUniform1f(pSpotMap->getUniformLocation("spotInnerCosCutoff"), spotLight->getInnerCosCutoffAngle());
   glUniform1f(pSpotMap->getUniformLocation("spotExponent"), spotLight->getExponent());


   V(glEnable(GL_DEPTH_TEST));
   SCENE->drawAllElementsDefault();
   V(glDisable(GL_DEPTH_TEST));

   // restore states

	//glCullFace(GL_BACK);
 //  glDisable(GL_CULL_FACE);

	// deactivate offsetting
	//V(glDisable(GL_POLYGON_OFFSET_FILL));                          
	
	// reset viewport
   glPopAttrib();

	// reset projection/modelview matrices
	glMatrixMode( GL_PROJECTION );                            
	glPopMatrix();								
	
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();   
}

