///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#include "IndirectLight.h"

#include "DebugVisualization/Debugging.h"
#include "Lighting/EnvMap.h"
#include "Lighting/Sampling.h"
#include "Lighting/SpotMapRenderer.h"

#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include "Scene/SpotLight.h"

#include "Utils/EmptyTexture.h"
#include "Utils/FBOUtil.h"
#include "Utils/FullScreenQuad.h"
#include "Utils/GLError.h"
#include "Utils/ShaderPool.h"
#include "Utils/ShaderProgram.h"
#include "Utils/TexturePool.h"

#include "Qt/Settings.h"

IndirectLight::IndirectLight()
{
   mHammersleySequence2D = 0;

   createShader();
   createFBO();
   createRandom2DTexture();
   createSamplingSequence();
}

void IndirectLight::createShader()
{
   pIntersectMipmap = new ShaderProgram("src/shader/Quad.vert", "src/shader/IntersectionTestMipmap.frag");
   pSpotLookup = new ShaderProgram("src/shader/Quad.vert", "src/shader/HitPointSpotLookup.frag");

   // utility
   pSum = new ShaderProgram("src/shader/Quad.vert", "src/shader/Sum.frag");
   pWriteColorRGB = ShaderPool::getWriteColorRGB();

}

void IndirectLight::createFBO()
{
   const int w = SCENE->getWindowWidth();
   const int h = SCENE->getWindowHeight();

   mCurrentBufferSize = FULL;

   // FULL Res
   mIndirectLightResultBuffer.push_back(EmptyTexture::create2D(w, h, GL_RGB16F_ARB, GL_RGB, GL_FLOAT, GL_LINEAR, GL_LINEAR));
   // HALF
   mIndirectLightResultBuffer.push_back(EmptyTexture::create2D(w/2, h/2, GL_RGB16F_ARB, GL_RGB, GL_FLOAT, GL_LINEAR, GL_LINEAR));
   // QUARTER
   mIndirectLightResultBuffer.push_back(EmptyTexture::create2D(w/4, h/4, GL_RGB16F_ARB, GL_RGB, GL_FLOAT, GL_LINEAR, GL_LINEAR));
   
   sumTexture = EmptyTexture::create2D(w, h, GL_RGB16F_ARB, GL_RGB, GL_FLOAT);
 
   // buffer for bounce ray hit positions
   mHitBuffer.push_back(EmptyTexture::create2D(w, h,     GL_RGB16F_ARB, GL_RGB, GL_FLOAT, GL_NEAREST, GL_NEAREST));
   mHitBuffer.push_back(EmptyTexture::create2D(w/2, h/2, GL_RGB16F_ARB, GL_RGB, GL_FLOAT, GL_NEAREST, GL_NEAREST));
   mHitBuffer.push_back(EmptyTexture::create2D(w/4, h/4, GL_RGB16F_ARB, GL_RGB, GL_FLOAT, GL_NEAREST, GL_NEAREST));

   // Intermediate Buffer
   mIntermediateBuffer.push_back(EmptyTexture::create2D(w, h, GL_RGB16F_ARB, GL_RGB, GL_FLOAT));
   mIntermediateBuffer.push_back(EmptyTexture::create2D(w/2, h/2, GL_RGB16F_ARB, GL_RGB, GL_FLOAT));
   mIntermediateBuffer.push_back(EmptyTexture::create2D(w/4, h/4, GL_RGB16F_ARB, GL_RGB, GL_FLOAT));


   vector<GLfloat> dataFull;
   for(int i = 0; i < w*h*4; i++)
      dataFull.push_back(0);
   
   mBufferWidth.push_back(w);
   mBufferWidth.push_back(w/2);
   mBufferWidth.push_back(w/4);
   mBufferHeight.push_back(h);
   mBufferHeight.push_back(h/2);
   mBufferHeight.push_back(h/4);


   glGenFramebuffersEXT(1, &fboIndirectLight);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboIndirectLight);

   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mIndirectLightResultBuffer.at(FULL), 0);                                                                                                                                     
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, mHitBuffer.at(FULL), 0);                                                                           
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT4_EXT, GL_TEXTURE_2D, mIntermediateBuffer.at(FULL), 0); 

   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

   std::cout << "  <>   [FBO Status] Indirect Light: " << checkFramebufferStatus()<< endl;

}


void IndirectLight::setBufferSize(BufferSize size)
{
   if( size != mCurrentBufferSize )
   {
      //cout << "Changing IL Buffer Size" << endl;
      mCurrentBufferSize = size;

      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboIndirectLight);
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mIndirectLightResultBuffer.at(mCurrentBufferSize), 0);                                                                                                                                     
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, 0, 0);                                                                                                                                     
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_2D, 0, 0);                                                                                                                                     
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT3_EXT, GL_TEXTURE_2D, 0, 0);                                                                                                                                     
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT4_EXT, GL_TEXTURE_2D, mIntermediateBuffer.at(mCurrentBufferSize), 0);                                                                           
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

   }
}


void IndirectLight::createPIPixelTexture()
{
   GLfloat data[] = {F_PI, F_PI, F_PI};
   GLuint piPixelTex;
	glGenTextures(1, &piPixelTex);
	glBindTexture(GL_TEXTURE_2D, piPixelTex);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, 1, 1, 0, GL_RGB, GL_FLOAT, data);

   TexturePool::addTexture("piPixel", piPixelTex);
}

void IndirectLight::createRandom2DTexture()
{
   if(glIsTexture(mTexRand2D))
      glDeleteTextures(1, &mTexRand2D);

   // 2D Random Texture for random rotation of rays

   int size = SETTINGS->getRandomPatternSize();

   vector<GLfloat> data;
   for(int i = 0; i < size*size; i++)
   {
      float rnd1 = (float)rand() / RAND_MAX; 
      float rnd2 = (float)rand() / RAND_MAX; 
      float rnd3 = (float)rand() / RAND_MAX;  

      data.push_back(rnd1);
      data.push_back(rnd2);
      data.push_back(rnd3);
   }

   // assign random-data to new empty tex
   mTexRand2D = EmptyTexture::create2D(size, size, GL_RGB, GL_RGB, GL_FLOAT, GL_NEAREST, GL_NEAREST, &data[0]);
   V(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
   V(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

}

void IndirectLight::createSamplingSequence()
{
   for(int j = 0; j < MAX_RAYS; j++)
      for(int i = 0; i < 2*MAX_RAYS; i++)
      {
         mSamplingSequence[j][i] = 0; 
      }

   if(mHammersleySequence2D) delete[] mHammersleySequence2D; 
   mHammersleySequence2D = new glm::vec2[MAX_RAYS];

   int startSample = 0;
   for(int j = 0; j < MAX_RAYS; j++)
   {
      Sampling::generateHammersleySequence(mHammersleySequence2D, j+1);

      startSample += MAX_RAYS;

      for(int i = 0; i < 2 * MAX_RAYS; i+=2)
      {
         mSamplingSequence[j][i]   = mHammersleySequence2D[i/2].x;
         mSamplingSequence[j][i+1] = mHammersleySequence2D[i/2].y;
      }
   }
}


const glm::vec2* IndirectLight::getHammersleySequence()
{
   return mHammersleySequence2D;
}

// Average the RSM color textures and use this color as global ambient color
void IndirectLight::computeAmbientTerm() 
{
   GLfloat color[] = {0, 0, 0};

   if(!SCENE->getSpotLights().empty())
   {

      glPushAttrib(GL_VIEWPORT_BIT);

      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboIndirectLight);
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
      glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
      glClearColor(0, 0, 0, 0);
      glClear(GL_COLOR_BUFFER_BIT);

      pSum->useProgram();
      glUniform1i(pSum->getUniformLocation("srcTex"), 0);
      int sRes = SCENE->getSpotMapRenderer()->getSpotMapResolution();

      glDisable(GL_DEPTH_TEST);

      glActiveTexture(GL_TEXTURE0);


      for(unsigned int spot = 0; spot < SCENE->getNumSpotLights(); spot++)
      {
         glViewport(0, 0, sRes, sRes);
         // read from colorSpotMap initially
         glBindTexture(GL_TEXTURE_2D, SCENE->getSpotMapRenderer()->getSpotMap(MAP_DIRECTLIGHT, spot));

         int pass = 0;
         for(int currentWidth = sRes/2; currentWidth >= 1; currentWidth /= 2)
         {
            glViewport(0, 0, currentWidth, currentWidth);

            // Set the size of the input texture
            glUniform1f(pSum->getUniformLocation("texDelta"), 1.0 / (2.0 * currentWidth));

            FullScreenQuad::drawComplete();

            if(pass == 0)
            {
               // read from sumTexture and copy intermediate results to sumTexture
               glBindTexture(GL_TEXTURE_2D, sumTexture);
            }
            // Copy Framebuffer Content to sumTexture
            glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 0, 0, currentWidth, currentWidth, 0);


            pass++;
         }

         GLfloat valFromGPU[3]; // average color from spot map
         glReadPixels(0,0, 1,1, GL_RGB, GL_FLOAT, &valFromGPU);
         color[0] += valFromGPU[0]; color[1] += valFromGPU[1]; color[2] += valFromGPU[2]; 
         //cout << valFromGPU[0] << " " << valFromGPU[1] << " " << valFromGPU[2] << endl;
      }

      glPopAttrib();

   }
   else
   {
      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboIndirectLight);
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

      color[0] = 0.6f;
      color[1] = 0.6f;
      color[2] = 0.5f;
   }

   // Write this color to the indirect light buffer

   glDisable(GL_BLEND);

   glPushAttrib(GL_VIEWPORT_BIT);
   glViewport(0, 0, mBufferWidth.at(mCurrentBufferSize), mBufferHeight.at(mCurrentBufferSize));


   pWriteColorRGB->useProgram();
   glUniform3f(pWriteColorRGB->getUniformLocation("color"), color[0], color[1], color[2]);
   FullScreenQuad::drawComplete();

   glPopAttrib();

}

   void IndirectLight::computeVGIWithMipmapTest(
   GLuint mipmappedBinaryVoxelTexture, int voxelTextureResolution, int maxMipmapLevel,
   const Camera* const voxelCamera, EnvMap* envMap)
{
   // Compute size of a voxel in world coordinates

   Frustum f = voxelCamera->getFrustum();
   float vX = f.width  / voxelTextureResolution;
   float vY = f.height / voxelTextureResolution;
   float vZ = f.zRange / 128;
   float voxelDiagonal = sqrt(vX * vX + vY * vY + vZ * vZ);

   // The contribution of one sample (a ray) 
   // corresponds to a user defines scale factor and 
   // the number of samples (rays) per pixel
   float sampleContrib = SETTINGS->getIndirectLightScaleFactor()/SETTINGS->getNumRays();

   // set uniform variables for RSM lookup shader
   setupSpotMapLookupShader(sampleContrib,
      voxelDiagonal, 7, 8, 9, 6, 0, 11);

   // set uniform variables for mipmap intersection test
   setupMipmapTestShaderUniforms(pIntersectMipmap,
      4, 5, 0, 1, 11,
      maxMipmapLevel, voxelDiagonal, voxelCamera);

   glUniform1i(pIntersectMipmap->getUniformLocation("randTex"), 3);
   glUniform1i(pIntersectMipmap->getUniformLocation("randTexSize"),  SETTINGS->getRandomPatternSize());

   // Bind all textures needed for computation
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(POSITION));
   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(NORMAL));
   glActiveTexture(GL_TEXTURE2);
   glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(DIRECTLIGHT));
   glActiveTexture(GL_TEXTURE11);
   glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(MATERIAL));

   glActiveTexture(GL_TEXTURE3);
   glBindTexture(GL_TEXTURE_2D, mTexRand2D); 
   glActiveTexture(GL_TEXTURE4);
   glBindTexture(GL_TEXTURE_2D, mipmappedBinaryVoxelTexture);
   glActiveTexture(GL_TEXTURE5);
   glBindTexture(GL_TEXTURE_2D, TexturePool::getTexture("bitmaskXORRays")); 

   glActiveTexture(GL_TEXTURE6);
   glBindTexture(GL_TEXTURE_2D, mHitBuffer.at(mCurrentBufferSize));

   glActiveTexture(GL_TEXTURE16);
   glBindTexture(GL_TEXTURE_2D, envMap->getTexture());


   // More uniform variables for the intersection shader
   glUniformMatrix4fv(pIntersectMipmap->getUniformLocation("inverseViewProjToUnitMatrixVoxelCam"), 1, GL_FALSE, &voxelCamera->getInverseViewProjectionToUnitMatrix()[0][0]);
   glUniform3fv(pIntersectMipmap->getUniformLocation("voxelizingDirection"), 1, &voxelCamera->getViewDirection()[0]);
   glUniform2fv(pIntersectMipmap->getUniformLocation("samplingSequence"), SETTINGS->getNumRays(), mSamplingSequence[SETTINGS->getNumRays()-1]);

   glUniform1f(pIntersectMipmap->getUniformLocation("radius"), SETTINGS->getRadius());
   glUniform1f(pIntersectMipmap->getUniformLocation("spread"), SETTINGS->getSpread());
   glUniform1f(pIntersectMipmap->getUniformLocation("useRandRay"),  SETTINGS->useRandomRays());
 
   // Variables that belong to the directional occlusion effect

   V(glUniform1f(pIntersectMipmap->getUniformLocation("occlusionStrength"), SETTINGS->getOcclusionStrength()));
   V(glUniform1f(pIntersectMipmap->getUniformLocation("envMapBrightness"), SETTINGS->getEnvMapBrightness()));
   V(glUniform1f(pIntersectMipmap->getUniformLocation("invRays"), 1.0 / SETTINGS->getNumRays()));
   V(glUniform1i(pIntersectMipmap->getUniformLocation("envMap"), 16));
   V(glUniform1f(pIntersectMipmap->getUniformLocation("lightRotationAngle"), envMap->getRotationAngle()));


   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboIndirectLight);
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, mHitBuffer.at(mCurrentBufferSize), 0);                                                                                                                                     

   glPushAttrib(GL_VIEWPORT_BIT);
   glViewport(0, 0, mBufferWidth.at(mCurrentBufferSize), mBufferHeight.at(mCurrentBufferSize));


   glDrawBuffer(GL_COLOR_ATTACHMENT4_EXT); // intermediate buffer
   glClearColor(0, 0, 0, 0);
   glClear(GL_COLOR_BUFFER_BIT);
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT); // indirect light
   glClear(GL_COLOR_BUFFER_BIT);

   FullScreenQuad::setupRendering();

   glBlendFunc(GL_ONE, GL_ONE); // ADD
   glEnable(GL_BLEND);

   glDisable(GL_DEPTH_TEST);

   for(int i = 0; i < SETTINGS->getNumRays(); i++)
   {
      glDrawBuffer(GL_COLOR_ATTACHMENT1_EXT); // hit buffer
      glClear(GL_COLOR_BUFFER_BIT); // clear hit buffer to zero (because of accumulation)
      glDrawBuffers(2, FBOUtil::buffers14); // hitPos, doContrib

      if(glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT) != GL_FRAMEBUFFER_COMPLETE_EXT)
      {
         cout << "  <>   [FBO Status] computeVGIWithMipmapTest : " << checkFramebufferStatus() << endl;
         system("pause");
      }

      //  (intersection shader sets P.z = 100 if no hit )

      pIntersectMipmap->useProgram();
      glUniform1i(pIntersectMipmap->getUniformLocation("currentRay"), i);
      
      FullScreenQuad::drawOnly();
      
      // hit positions for current ray are now in mHitBuffer
      // and directional occlusion value is accumulated in intermediate buffer 0

      // bounce light:
      if(SETTINGS->getIndirectLightScaleFactor() > 0.0001f)
      {
         // write to indirect light, read from hit buffer
         glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT); // indirect light

         spotLookupDrawOnly(7, 8, 9);
      }
   }

   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT); // indirect light

   // add directional occlusion component to indirect light buffer (is bound)
   // while clamping the RGB value to a minimal value of zero

   ShaderPool::getQuadClampToZero(); // uses program

   // read from mIntermediateBuffer 
   // (holds directional occlusion which has potential negative values 
   //  if high occlusion strength is set)
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mIntermediateBuffer.at(mCurrentBufferSize));

   FullScreenQuad::drawOnly();

   glDisable(GL_BLEND);

   glPopAttrib();

   FullScreenQuad::resetRendering();
}


void IndirectLight::setupSpotMapLookupShader(
   float sampleContrib,
   float voxelDiagonal,
   int texSlotPositionMap,
   int texSlotColorMap,
   int texSlotNormalMap,
   int texSlotHitBuffer,
   int texSlotHitRayOriginBuffer,
   int texSlotMaterialBuffer)
{
   if(!SCENE->getSpotLights().empty())
   {
      pSpotLookup->useProgram();
      V(glUniform1i(pSpotLookup->getUniformLocation("hitBuffer"), texSlotHitBuffer));
      V(glUniform1i(pSpotLookup->getUniformLocation("hitRayOriginBuffer"), texSlotHitRayOriginBuffer)); 
      V(glUniform1i(pSpotLookup->getUniformLocation("positionSpotMap"), texSlotPositionMap));
      V(glUniform1i(pSpotLookup->getUniformLocation("colorSpotMap"), texSlotColorMap));
      V(glUniform1i(pSpotLookup->getUniformLocation("normalSpotMap"), texSlotNormalMap));
      
      glUniform1f(pSpotLookup->getUniformLocation("voxelDiagonal"), voxelDiagonal); 
      glUniform1f(pSpotLookup->getUniformLocation("distanceThresholdScale"), SETTINGS->getDistanceThresholdScale()); 

      V(glUniform1f(pSpotLookup->getUniformLocation("sampleContrib"), sampleContrib));
      V(glUniform1i(pSpotLookup->getUniformLocation("normalCheck"), SETTINGS->normalCheckForLightLookup));

   }
}

void IndirectLight::spotLookupDrawOnly(int texSlotPositionMap,
                                       int texSlotColorMap,
                                       int texSlotNormalMap)
{

   pSpotLookup->useProgram();
   for(unsigned int spot = 0; spot < SCENE->getSpotLights().size(); spot++)
   {
      V(glUniformMatrix4fv(pSpotLookup->getUniformLocation("mapLookupMatrix"),
         1, GL_FALSE, &SCENE->getSpotLights().at(spot)->getMapLookupMatrix()[0][0]));
      glUniform3fv(pSpotLookup->getUniformLocation("spotDirection"), 1, & ((SCENE->getSpotLights().at(spot)->getSpotDirection() * -1.0f)[0])); 
      //cout << "pixelSide_zNear " << SCENE->getSpotLights().at(spot)->getPixelSide_zNear() << endl;
      //cout << "pixelDiag_zNear " << SCENE->getSpotLights().at(spot)->getPixelDiag_zNear() << endl;
      glUniform1f(pSpotLookup->getUniformLocation("pixelSide_zNear"), SCENE->getSpotLights().at(spot)->getPixelSide_zNear()); 

      glActiveTexture(GL_TEXTURE0 + texSlotPositionMap);
      glBindTexture(GL_TEXTURE_2D, SCENE->getSpotMapRenderer()->getSpotMap(MAP_POSITION, spot));
      glActiveTexture(GL_TEXTURE0 + texSlotColorMap);
      glBindTexture(GL_TEXTURE_2D, SCENE->getSpotMapRenderer()->getSpotMap(MAP_DIRECTLIGHT, spot));
      glActiveTexture(GL_TEXTURE0 + texSlotNormalMap);
      glBindTexture(GL_TEXTURE_2D, SCENE->getSpotMapRenderer()->getSpotMap(MAP_NORMAL, spot));

      FullScreenQuad::drawOnly();
   }
}


void IndirectLight::setupMipmapTestShaderUniforms(
   ShaderProgram* prog,
   int texSlotVoxelTexture,
   int texSlotBitmaskXORRays,
   int texSlotPositionBuffer,
   int texSlotNormalBuffer,
   int texSlotMaterialBuffer,
   int maxMipMapLevel,
   float voxelDiagonal,
   const Camera* voxelCamera
   )
{
   V(prog->useProgram());

   // Texture slots
   V(glUniform1i(prog->getUniformLocation("voxelTexture"), texSlotVoxelTexture));
   V(glUniform1i(prog->getUniformLocation("bitmaskXORRays"), texSlotBitmaskXORRays));
   V(glUniform1i(prog->getUniformLocation("positionBuffer"), texSlotPositionBuffer));		
   V(glUniform1i(prog->getUniformLocation("normalBuffer"),   texSlotNormalBuffer));
   V(glUniform1i(prog->getUniformLocation("materialBuffer"), texSlotMaterialBuffer));

   V(glUniform1i(prog->getUniformLocation("maxMipMapLevel"), maxMipMapLevel));

   V(glUniform1f(prog->getUniformLocation("voxelDiagonal"), voxelDiagonal));
   V(glUniform1f(prog->getUniformLocation("dEps"), SETTINGS->getRadius()));
   V(glUniform1i(prog->getUniformLocation("steps"), SETTINGS->getNumSteps()));
   glUniform1f(prog->getUniformLocation("voxelOffsetCosThetaScale"),  SETTINGS->getVoxelOffsetCosThetaScale());
   glUniform1f(prog->getUniformLocation("voxelOffsetNormalScale"),  SETTINGS->getVoxelOffsetNormalScale());
   V(glUniformMatrix4fv(prog->getUniformLocation("viewProjToUnitMatrixVoxelCam"), 1, GL_FALSE, &voxelCamera->getViewProjectionToUnitMatrix()[0][0]));

   V(glUniform3fv(prog->getUniformLocation("worldPosCamera"), 1, &SCENE->getCamera()->getEye()[0])); 

}
