#include "Scene.h"

#include "DebugVisualization/Debugging.h"
#include "DebugVisualization/VoxelVisualization.h"

#include "Scene/Camera.h"
#include "Scene/ObjModel.h"
#include "Scene/ObjectSequence.h"
#include "Scene/ObjectSequenceInstance.h"
#include "Scene/SpotLight.h"
#include "Lighting/EnvMap.h"
#include "Lighting/Sampling.h"
#include "Lighting/SpotMapRenderer.h"

#include "Qt/Settings.h"

#include "Utils/CoutMethods.h"
#include "Utils/EmptyTexture.h"
#include "Utils/FBOUtil.h"
#include "Utils/FullScreenQuad.h"
#include "Utils/GLError.h"
#include "Utils/ShaderPool.h"

Scene* Scene::mInstance = 0;

Scene::Scene()
{
   mName = "unknown";
   
   mWindowWidth  = 0;
   mWindowHeight = 0;

   mHasDynamicElements = false;

   mActiveDynamicElementIndex = -1;   // no dynamic element
   mActiveInstanceIndex = -1;         // no instance
   mActiveSpotLightIndex = -1;        // no spot light
   mCurrentCameraPoseIndex = -1;

   mShowLights = false;
   
   mSpotMapRenderer = 0;

   mShadowMapResolution  = 2048;

   mSpotLightQuadric = gluNewQuadric();
   gluQuadricDrawStyle(mSpotLightQuadric, GLU_FILL);
}

void Scene::initialize(int windowWidth, int windowHeight)
{
   this->mWindowWidth  = windowWidth;
   this->mWindowHeight = windowHeight;
   createShader();
   createFBO();

   mEnvMap = new EnvMap("images/KitchenMediumBlurred.pfm");
   //mEnvMap = new EnvMap("images/grace_latlong-blurred.pfm");
   //mEnvMap = new EnvMap("images/campus_probe_latlong.pfm");
}

void Scene::createShader()
{
   pGBuffer = new ShaderProgram("src/shader/GBuffer.vert", "src/shader/GBuffer.frag");
   pGBuffer->useProgram();
   glUniform1i(pGBuffer->getUniformLocation("diffuseTexture"), 0); // slot 0

   pSpotShadow = new ShaderProgram("src/shader/SpotShadow.vert", "src/shader/SpotShadow.frag"); 
   pSpotShadow->useProgram();
   glUniform1i(pSpotShadow->getUniformLocation("positionBuffer"), 0); // slot 0
   glUniform1i(pSpotShadow->getUniformLocation("shadowMap"), 3); // slot 3
   glUniform1i(pSpotShadow->getUniformLocation("randTex"), 4); // slot 4
   glUniform1i(pSpotShadow->getUniformLocation("randTexSize"), 64); 
   glUniform2f(pSpotShadow->getUniformLocation("pixelOffset"), 1.0f / mShadowMapResolution,
                                                               1.0f / mShadowMapResolution );
   computeJitteredShadowSamples();

   pSpotLighting = new ShaderProgram("src/shader/Quad.vert", "src/shader/SpotLighting.frag");  
   pSpotLighting->useProgram();
   glUniform1i(pSpotLighting->getUniformLocation("positionBuffer"), 0); // slot 0
   glUniform1i(pSpotLighting->getUniformLocation("normalBuffer"), 1); // slot 1
   glUniform1i(pSpotLighting->getUniformLocation("materialBuffer"), 2); // slot 2
   glUniform1i(pSpotLighting->getUniformLocation("envMap"), 5); // slot 5
}

void Scene::createFBO()
{
   // Create textures

   // GBuffer: position, normal, material, direct light (full resolution)
   vector<GLuint> pnmd;
   pnmd.push_back(EmptyTexture::create2D(mWindowWidth, mWindowHeight, GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT));
   pnmd.push_back(EmptyTexture::create2D(mWindowWidth, mWindowHeight, GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT));
   pnmd.push_back(EmptyTexture::create2D(mWindowWidth, mWindowHeight, GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT));
   pnmd.push_back(EmptyTexture::create2D(mWindowWidth, mWindowHeight, GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT));
   mGBuffer.push_back(pnmd);
   // material buffer for spatial upsampling in half and quarter resolution
   vector<GLuint> pnm;
   pnm.push_back(0);
   pnm.push_back(0);
   pnm.push_back(EmptyTexture::create2D(mWindowWidth/2, mWindowHeight/2, GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT));
   mGBuffer.push_back(pnm);
   pnm.clear();
   pnm.push_back(0);
   pnm.push_back(0);
   pnm.push_back(EmptyTexture::create2D(mWindowWidth/4, mWindowHeight/4, GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT));
   mGBuffer.push_back(pnm);

   // Intermediate buffer for direct light computation
   mIntermediateBuffer = EmptyTexture::create2D(mWindowWidth, mWindowHeight, GL_RGBA16F_ARB, GL_RGBA, GL_FLOAT);
   
   // Create FBOs

   V(glGenFramebuffersEXT(1, &fboLowRes));

   V(glGenFramebuffersEXT(1, &fboGBuffer));
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboGBuffer));

	// attach render buffer object "depth buffer" to FBO for depth test
	glGenRenderbuffersEXT(1, &mDepthBuffer);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, mDepthBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, mWindowWidth, mWindowHeight);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, mDepthBuffer);

   // bind texture(s) to color attachment points
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mGBuffer.at(FULL).at(POSITION), 0);                                                                           
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_2D, mGBuffer.at(FULL).at(NORMAL), 0);                                                                           
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_TEXTURE_2D, mGBuffer.at(FULL).at(MATERIAL), 0);     // rhoDiffuse / PI, specExponent                                                                      
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT3_EXT, GL_TEXTURE_2D, mGBuffer.at(FULL).at(DIRECTLIGHT), 0);                                                                           
   glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT4_EXT, GL_TEXTURE_2D, mIntermediateBuffer, 0);                                                                           

   glDrawBuffers(5, FBOUtil::buffers01234);
   glClearColor(0, 0, 0, 0);
   glClear(GL_COLOR_BUFFER_BIT);
   glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

   std::cout << "  <>   [FBO Status] GBuffer: " << checkFramebufferStatus()<< endl;
}

void Scene::setCurrentCameraPose(int index)
{
   mCurrentCameraPoseIndex = index;
   Pose pose = mCameraPoses.at(index);
   mCamera->setUserPosition(pose.userPosition);
   mCamera->setAngleX(pose.angleX);
   mCamera->setAngleY(pose.angleY);
   mCamera->setAngleZ(pose.angleZ);
}

void Scene::addCurrentCameraPose()
{
   Pose pose;
   pose.userPosition = mCamera->getUserPosition();
   pose.viewDirection = mCamera->getViewDirection();
   pose.upVector = mCamera->getUpVector();
   pose.angleX = mCamera->getAngleX();
   pose.angleY = mCamera->getAngleY();
   pose.angleZ = mCamera->getAngleZ();

   mCameraPoses.push_back(pose);
   mCurrentCameraPoseIndex = mCameraPoses.size() -1 ;
}

void Scene::addSpotLight()
{
   // make a copy of the current spot light
   mSpotLights.push_back(new SpotLight(mSpotLights.at(mActiveSpotLightIndex)));
   mActiveSpotLightIndex = mSpotLights.size() - 1 ; // added spot light is active
   // rotate and move a bit
   modifyActiveSpotLightAngles(RAND(0, 5), RAND(0, 5), 0);
   moveActiveSpotLight(0, 0, RAND(0, 0.25f));
   mSpotLights.at(mActiveSpotLightIndex)->updatePixelSide();

   // corrupt all spotlights for lookupMatrixUpdate
   for(unsigned int i = 0; i < mSpotLights.size(); i++)
      mSpotLights.at(i)->setChanged(true);

   mSpotMapRenderer->addSpotLight();
}

void Scene::deleteCurrentSpotLight()
{
   if(mSpotLights.size() > 1)
   {
      int newActiveSpotLightIndex;
      // set new active spot light to previous or next spotlight
      if(mActiveSpotLightIndex == mSpotLights.size() -1) // last
      {
         newActiveSpotLightIndex = mActiveSpotLightIndex-1;
      }
      else
      {
         newActiveSpotLightIndex = mActiveSpotLightIndex;
      }
      // delete spotlight
      mSpotLights.erase(mSpotLights.begin() + mActiveSpotLightIndex);
      
      // corrupt all spotlights for lookupMatrixUpdate
      for(unsigned int i = 0; i < mSpotLights.size(); i++)
         mSpotLights.at(i)->setChanged(true);
      
      mSpotMapRenderer->deleteSpotLight(mActiveSpotLightIndex);


      // set new index:
      mActiveSpotLightIndex = newActiveSpotLightIndex;

   }
}

void Scene::addInstanceToActiveDynamicElement()
{
   if(mActiveDynamicElementIndex >= 0)
   {
      const ObjectSequenceInstance* const copyInstance = mSceneElements.at(mActiveDynamicElementIndex)->getInstance(mActiveInstanceIndex);
      mSceneElements.at(mActiveDynamicElementIndex)->addInstance(copyInstance);
      
      // Added instance becomes current active instance
      mActiveInstanceIndex = mSceneElements.at(mActiveDynamicElementIndex)->getNumInstances() - 1;
      
      ObjectSequenceInstance* addedInstance = mSceneElements.at(mActiveDynamicElementIndex)->getInstance(mActiveInstanceIndex);
      addedInstance->move(0.1f, 0.0f, 0.1f);

   }
}

void Scene::deleteCurrentCameraPose()
{
   if(mCameraPoses.size() > 1)
   {
      Pose newPose;
      int newIndex;
      // set current camera to previous or next pose
      if(mCurrentCameraPoseIndex == mCameraPoses.size() -1) // last
      {
         newPose = mCameraPoses.at(mCurrentCameraPoseIndex-1);
         newIndex = mCurrentCameraPoseIndex-1;
      }
      else
      {
         newPose = mCameraPoses.at(mCurrentCameraPoseIndex+1);
         newIndex = mCurrentCameraPoseIndex;
      }

      // set new pose
      mCamera->setUserPosition(newPose.userPosition);
      mCamera->setAngleX(newPose.angleX);
      mCamera->setAngleY(newPose.angleY);
      mCamera->setAngleZ(newPose.angleZ);

      // delete pose
      mCameraPoses.erase(mCameraPoses.begin() + mCurrentCameraPoseIndex);
      // set new index:
      mCurrentCameraPoseIndex = newIndex;

   }
}


void Scene::computeJitteredShadowSamples()
{
   int wTexRand = 64;
   int hTexRand = 64;

   vector<GLfloat> data;
   for(int y = 0; y < hTexRand; y++)
   for(int x = 0; x < wTexRand; x++)
   {
      float randAngle = 2.0f * F_PI * (rand() / RAND_MAX);
      data.push_back(cos(randAngle));
      data.push_back(sin(randAngle));

   }
   mRandShadowSamplesTex = EmptyTexture::create2D(wTexRand, hTexRand, GL_RG16F, GL_RG, GL_FLOAT, GL_NEAREST, GL_NEAREST, &data[0]);

}

void Scene::computeSceneBoundingBox()
{
   //cout << "mSceneElements.size() " << mSceneElements.size() << endl;
   //mStaticBoundingBox = minX, minY, minZ, maxX, maxY, maxZ

   GLfloat instanceBox[6];

   mSceneBoundingBox[0] = std::numeric_limits<float>::max();
   mSceneBoundingBox[1] = std::numeric_limits<float>::max();
   mSceneBoundingBox[2] = std::numeric_limits<float>::max();
   mSceneBoundingBox[3] = -std::numeric_limits<float>::max();
   mSceneBoundingBox[4] = -std::numeric_limits<float>::max();
   mSceneBoundingBox[5] = -std::numeric_limits<float>::max();

   for(unsigned int e = 0; e < mSceneElements.size(); e++)
   { 
      for(unsigned int i = 0; i < mSceneElements.at(e)->getNumInstances(); i++)
      {
         // gets the current AABB of the first instance (static element has only 1 instance)
         mSceneElements.at(e)->getInstance(i)->getCurrentAABB(instanceBox);
         //cout << "get Current AABB for " << mSceneElements.at(e)->getName() << endl;

         // get min / max
         for(int j = 0; j < 3; j++)
         {
            // min
            if(mSceneBoundingBox[j] > instanceBox[j])
               mSceneBoundingBox[j] = instanceBox[j];

            // max
            if(mSceneBoundingBox[j+3] < instanceBox[j+3])
               mSceneBoundingBox[j+3] = instanceBox[j+3];
         } 
      } // instances
   } // elements


   mSceneBoundingBoxDimension.x = mSceneBoundingBox[3] - mSceneBoundingBox[0];
   mSceneBoundingBoxDimension.y = mSceneBoundingBox[4] - mSceneBoundingBox[1];
   mSceneBoundingBoxDimension.z = mSceneBoundingBox[5] - mSceneBoundingBox[2];

   mSceneBoundingBoxCenter.x = (mSceneBoundingBox[0] + mSceneBoundingBox[3]) * 0.5f;
   mSceneBoundingBoxCenter.y = (mSceneBoundingBox[1] + mSceneBoundingBox[4]) * 0.5f;
   mSceneBoundingBoxCenter.z = (mSceneBoundingBox[2] + mSceneBoundingBox[5]) * 0.5f;

}


void Scene::postLoadProcessing()
{
   mAnimationsRunning = false;

   // There must be 1 light
   assert(!mSpotLights.empty());

   if(!mSpotLights.empty())
   {
      mSpotMapRenderer = new SpotMapRenderer(mShadowMapResolution, SpotLight::spotMapResolution());
   }

   for(unsigned int i = 0; i < mSpotLights.size(); i++)
   {
      mSpotLights.at(i)->updatePixelSide();
   }

   for(unsigned int i = 0; i < mSceneElements.size(); i++)
   {
      mHasDynamicElements |= !mSceneElements.at(i)->isStatic();
      mSceneElements.at(i)->computeOriginalAABB();
      mSceneElements.at(i)->updateAllCurrentAABBs(); // force update
   }

   if(!mSpotLights.empty())
      mActiveSpotLightIndex = 0;

   setupActiveDynamicElementIndex();
   setupActiveInstanceIndex();

   computeSceneBoundingBox();
   // scene bounding box values (minX, minY, minZ, maxX, maxY, maxZ) now in mSceneBoundingBox

   startAllObjAnimations();
}

//---------- Drawing methods --------------

void Scene::drawSceneBoundingBox()
{
   glUseProgram(0);
   glEnable(GL_DEPTH_TEST);

   glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

   glLineWidth(3.0);

   glColor4f(1, 1, 0, 1);
   VoxelVisualization::drawBox(mSceneBoundingBox);

   //std::cout << "mSceneBoundingBox" << std::endl;
   //std::cout << mSceneBoundingBox[0] << " " << mSceneBoundingBox[1] << " " << mSceneBoundingBox[2] << std::endl;
   //std::cout << mSceneBoundingBox[3] << " " << mSceneBoundingBox[4] << " " << mSceneBoundingBox[5] << std::endl;
   
   V(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
}

void Scene::drawDynamicBoundingBoxes()
{
   glUseProgram(0);
   glEnable(GL_DEPTH_TEST);

   glLineWidth(3.0);

   glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

   // all bounding boxes of dynamic scene elements

   glColor4f(0, 1, 0, 1);

   for(unsigned int e = 0 ; e < mSceneElements.size(); e++)
   {
      if(!mSceneElements.at(e)->isStatic())
      {
         for(unsigned int i = 0; i < mSceneElements.at(e)->getNumInstances(); i++)
            VoxelVisualization::drawBox(mSceneElements.at(e)->getInstance(i)->getCurrentAABB());
      }
   }
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void Scene::drawAllElementsDefault() const
{
   for(unsigned int e = 0; e < mSceneElements.size(); e++)
   {
      ObjectSequence* seq = mSceneElements.at(e);
      for(unsigned int i = 0; i < seq->getNumInstances(); i++)
      {
         const ObjectSequenceInstance* inst = seq->getInstance(i);

         glPushMatrix();
         glMultMatrixf(inst->getTransformationMatrix());
         seq->getCurrentObjModel(i)->draw(seq->getDefaultDrawMode());
         glPopMatrix();
      }
   }
}

int Scene::drawElementWithMode(int elementIndex, int instanceIndex, GLuint mode, GeometryType geom) const
{
   const ObjectSequence* const seq = mSceneElements.at(elementIndex);

   int triangles = 0;

   bool condition = true;
   switch(geom)
   {
   case DYNAMIC:
      condition = !seq->isStatic();
      break;

   case ALL:
      condition = true;
      break;

   case STATIC:
      condition = seq->isStatic();
      break;

   default:
      condition = false;
      break;
   }

   if(condition)
   {
      glPushMatrix();
      glMultMatrixf(seq->getInstance(instanceIndex)->getTransformationMatrix());
      seq->getCurrentObjModel(instanceIndex)->draw(mode);
      glPopMatrix();
      triangles += seq->getTriangleCount();

   }

   // return rendered triangle count
   return triangles;

}


int Scene::drawAllElementsWithMode(GLuint mode,  GeometryType geom) const
{
   int triangles = 0;

   bool condition = true;
   for(unsigned int e = 0; e < mSceneElements.size(); e++)
   {
      const ObjectSequence* const seq = mSceneElements.at(e);

      switch(geom)
      {
      case DYNAMIC:
         condition = !seq->isStatic();
         break;

      case ALL:
         condition = true;
         break;

      case STATIC:
         condition = seq->isStatic();
         break;

      default:
         condition = false;
         break;
      }

      if(condition)
      {
         bool draw = true;
         for(unsigned int i = 0; i < seq->getNumInstances(); i++)
         {
            const ObjectSequenceInstance* inst = seq->getInstance(i);
            //cout << "element: " << seq->getName() << " , i " << i << endl;

            if(draw)
            {
               glPushMatrix();
               glMultMatrixf(inst->getTransformationMatrix());
               seq->getCurrentObjModel(i)->draw(mode);
               glPopMatrix();
               triangles += seq->getTriangleCount();
            }
         }
      }
   }

   // return rendered triangle count
   return triangles;
}


void Scene::drawElementWithModeModifyingDefault(int elementIndex, int instanceIndex, GLuint addMode, GLuint removeMode)
{
   const ObjectSequence* const seq = mSceneElements.at(elementIndex);

   GLuint drawMode = seq->getDefaultDrawMode();
   if(drawMode & removeMode)
      drawMode ^= removeMode;
   drawMode |= addMode;

   glPushMatrix();
   glMultMatrixf(seq->getInstance(instanceIndex)->getTransformationMatrix());
   seq->getCurrentObjModel(instanceIndex)->draw(drawMode);
   glPopMatrix();
}

void Scene::drawAllElementsWithModeModifyingDefault(GLuint addMode, GLuint removeMode)
{
   for(unsigned int e = 0; e < mSceneElements.size(); e++)
   {
      const ObjectSequence* const seq = mSceneElements.at(e);

 
      GLuint drawMode = seq->getDefaultDrawMode();
      if(drawMode & removeMode)
         drawMode ^= removeMode;
      drawMode |= addMode;

      for(unsigned int i = 0; i < seq->getNumInstances(); i++)
      {
         glPushMatrix();
         glMultMatrixf(seq->getInstance(i)->getTransformationMatrix());
         seq->getCurrentObjModel(i)->draw(drawMode);
         glPopMatrix();
      }
   }
}

void Scene::generateShadowAndSpotMaps(bool withShadows, bool withSpotMaps, bool renderPosNormMat)
{
   if(!withShadows && !withSpotMaps)
      return;

   // generate all shadow and spot maps for spot lights
   for(unsigned int i = 0; i < mSpotLights.size(); i++)
   {
      if(withShadows)
      {
         mSpotMapRenderer->createShadowMap(i);
         // shadow map fbo is bound
         // program 0 is bound
      }
      if(withSpotMaps)
      {
         mSpotMapRenderer->createSpotMap(i, renderPosNormMat);
      }
   }
}

void Scene::createGBuffer(bool withShadows)
{
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboGBuffer));

   glDrawBuffers(5, FBOUtil::buffers01234);

   // clear all buffers to some big magic number 
   // for "geometry rendered to this pixel?"-check in the shaders later on
   glClearColor(100, 100, 100, 0);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   glDrawBuffers(3, FBOUtil::buffers012); 

   pGBuffer->useProgram();
   glUniformMatrix4fv(pGBuffer->getUniformLocation("inverseViewMatrix"), 1, GL_FALSE, &mCamera->getInverseViewMatrix()[0][0]);

   glEnable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);
   drawAllElementsDefault();


   // render low resolution material for spatial upsampling
   const int s = Settings::Instance()->getCurrentILBufferSize() ;
   if(Settings::Instance()->filterEnabled() && s > 0) 
   {
      V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboLowRes));
      glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, mGBuffer.at(s).at(MATERIAL), 0); 
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

      ShaderPool::getQuad();

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, mGBuffer.at(FULL).at(MATERIAL));

      glPushAttrib(GL_VIEWPORT_BIT);
      int div = pow(2.0f, float(s));
      glViewport(0, 0, mWindowWidth / div, mWindowHeight / div );
      FullScreenQuad::drawComplete();
      glPopAttrib();
   }

   deferredShading(mGBuffer.at(FULL).at(POSITION), mGBuffer.at(FULL).at(NORMAL), mGBuffer.at(FULL).at(MATERIAL),
      mIntermediateBuffer, fboGBuffer, GL_COLOR_ATTACHMENT3_EXT, GL_COLOR_ATTACHMENT4_EXT,
      withShadows, false);


}

void Scene::drawHighlightedModel()
{
   // high light model 
   if(mActiveDynamicElementIndex >= 0 && mActiveInstanceIndex >= 0)
   {
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_LIGHTING);
      glUseProgram(0);
      glColor4f(0.1f, .4f, 0.1f, 1);
      //glBlendFunc(GL_DST_COLOR, GL_ZERO);
      glBlendFunc(GL_ONE, GL_ONE);
      glEnable(GL_BLEND);

      // highlight the active movable instances
      ObjectSequence* seq = mSceneElements.at(mActiveDynamicElementIndex);
      glPushMatrix();
      glMultMatrixf(seq->getInstance(mActiveInstanceIndex)->getTransformationMatrix());
      seq->getCurrentObjModel(mActiveInstanceIndex)->draw(GLM_NONE);
      glPopMatrix();


      glDisable(GL_BLEND);

      // Viewing direction
      glm::vec3 to = SCENE->getActiveDynamicInstance()->getPosition() + 1.5f * SCENE->getActiveDynamicInstance()->getViewDir();
      V(glPolygonMode(GL_FRONT_AND_BACK, GL_LINE));
      glBegin(GL_LINES);
      glColor3f(0,1,1);
      glVertex3fv(&SCENE->getActiveDynamicInstance()->getPosition()[0]);
      glVertex3fv(&to[0]);
      glEnd();

      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

      glEnable(GL_DEPTH_TEST);

   }
}

void Scene::setDrawToDirectLightBuffer()
{
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboGBuffer));
   glDrawBuffer(GL_COLOR_ATTACHMENT3_EXT);

}

void Scene::deferredShading(GLuint positionBuffer, GLuint normalBuffer, GLuint materialBuffer,
                            GLuint mIntermediateBuffer, GLuint fbo,
                            GLenum directLightAttachment, GLenum intermediateAttachment,
                            bool withShadows, bool lowQuality)
{

   ShaderProgram* q = ShaderPool::getQuad(); // for fullscreen quad renderings
   glUniform1i(q->getUniformLocation("tex"), 4);

  // GBuffer has positions, normals, material

   // Now accumulate the illumination contribution from all lights
   // 1a: write illuminance into direct-light-texture
   // 1b: multiply this illuminance with the current lights shadow
   // 1c: add current illumination to direct light buffer

   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, positionBuffer);
   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, normalBuffer);
   glActiveTexture(GL_TEXTURE2);
   glBindTexture(GL_TEXTURE_2D, materialBuffer);

   // Draw only to DirectLight-Buffer
   V(glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo));
   glDrawBuffer(directLightAttachment);

   // DEBUGGING INFO
   if(glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT) != GL_FRAMEBUFFER_COMPLETE_EXT)
   {
      cout << "  <>   [FBO Status] While Deferred Shading: " << checkFramebufferStatus() << endl;
      system("pause");
   }

   FullScreenQuad::setupRendering();
   glDisable(GL_DEPTH_TEST);

   glActiveTexture(GL_TEXTURE5);
   glBindTexture(GL_TEXTURE_2D, mEnvMap->getTexture());

   glActiveTexture(GL_TEXTURE4);
   glBindTexture(GL_TEXTURE_2D, mRandShadowSamplesTex);

   if(!mSpotLights.empty())
   {
      pSpotLighting->useProgram();
      glUniform3fv(pSpotLighting->getUniformLocation("eyePos"), 1, &mCamera->getEye()[0]);         
      const Frustum& f = mCamera->getFrustum();
      glm::vec4 leftBottomNear = mCamera->getInverseViewMatrix() * glm::vec4(f.left, f.bottom, -f.zNear, 1.0);
      glm::vec4 leftTopNear = mCamera->getInverseViewMatrix() * glm::vec4(f.left, f.top, -f.zNear, 1.0);
      glm::vec4 rightBottomNear = mCamera->getInverseViewMatrix() * glm::vec4(f.right, f.bottom, -f.zNear, 1.0);
      glm::vec3 up = glm::vec3(leftTopNear - leftBottomNear);
      glm::vec3 right = glm::vec3(rightBottomNear - leftBottomNear);
      glUniform3fv(pSpotLighting->getUniformLocation("leftBottomNear"), 1, &leftBottomNear[0]);         
      glUniform3fv(pSpotLighting->getUniformLocation("up"), 1, &up[0]);         
      glUniform3fv(pSpotLighting->getUniformLocation("right"), 1, &right[0]);         
      
      glUniform1f(pSpotLighting->getUniformLocation("envMapRotationAngle"), mEnvMap->getRotationAngle());                



      // SPOT LIGHTS
      for(unsigned int i = 0; i < mSpotLights.size(); i++)
      {
         V(glDisable(GL_BLEND)); // overwrite

         if(i == 0) 
         {
            // this is the first spot light, so
            // Draw to DirectLight-Buffer
            glDrawBuffer(directLightAttachment);

         }
         else
         {
            // Draw to intermediate buffer
            V(glDrawBuffer(intermediateAttachment));
         }

         SpotLight* spot = mSpotLights.at(i);

         pSpotLighting->useProgram();

         glUniform3fv(pSpotLighting->getUniformLocation("lightPos"), 1, &spot->getPosition()[0]);
         glUniform3fv(pSpotLighting->getUniformLocation("I"), 1, &spot->getI()[0]);
         glUniform1f(pSpotLighting->getUniformLocation("constantAttenuation"), spot->getConstantAttenuation());
         glUniform1f(pSpotLighting->getUniformLocation("quadraticAttenuation"), spot->getQuadraticAttenuation());

         glUniform3fv(pSpotLighting->getUniformLocation("spotDirection"), 1, &spot->getSpotDirection()[0]);
         glUniform1f(pSpotLighting->getUniformLocation("spotCosCutoff"), spot->getCosCutoffAngle());
         glUniform1f(pSpotLighting->getUniformLocation("spotInnerCosCutoff"), spot->getInnerCosCutoffAngle());
         V(glUniform1f(pSpotLighting->getUniformLocation("spotExponent"), spot->getExponent()));

         FullScreenQuad::drawOnly();

         // 2: multiply with shadows

         V(glEnable(GL_BLEND));
         if(withShadows)
         {
            V(glBlendFunc(GL_DST_COLOR, GL_ZERO)); // MULTIPLY

            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, mSpotMapRenderer->getShadowMap(i));

            pSpotShadow->useProgram();
            glUniform1i(pSpotShadow->getUniformLocation("lowQuality"), lowQuality);
            glUniform1f(pSpotShadow->getUniformLocation("shadowMapEps"), Settings::Instance()->getShadowEpsilon());

            V(glUniformMatrix4fv(pSpotShadow->getUniformLocation("mapLookupMatrix"),
               1, GL_FALSE, &spot->getMapLookupMatrix()[0][0]));

            FullScreenQuad::drawOnly();
         }

         if( !(i == 0 ) ) 
         {
            // ADD TO DirectLight Buffer
            V(glBlendFunc(GL_ONE, GL_ONE));
            // Draw only to DirectLight-Buffer
            V(glDrawBuffer(directLightAttachment));
            V(glActiveTexture(GL_TEXTURE4));
            V(glBindTexture(GL_TEXTURE_2D, mIntermediateBuffer));
            q->useProgram();
            FullScreenQuad::drawOnly();
         }
      }

   }

   glDrawBuffer(directLightAttachment);

   glActiveTexture(GL_TEXTURE0);
   glDisable(GL_BLEND);

   FullScreenQuad::resetRendering();
}


void Scene::drawLights()
{
   for(unsigned int i = 0; i < mSpotLights.size(); i++)
   {
      drawSpotLight(i);
   }
}


void Scene::drawSpotLight(int index)
{
   if(mShowLights)
   {
      float scaleFactor = 0.2f * std::max(mSceneBoundingBoxDimension.z, std::max(mSceneBoundingBoxDimension.x, mSceneBoundingBoxDimension.y));

      const SpotLight* spot = mSpotLights.at(index);
      glUseProgram(0);
      glColor4fv(&glm::vec4(1.5f * glm::normalize(spot->getI()), 1.0)[0]);

      glPushMatrix();
      glTranslatef(spot->getPosition()[0], spot->getPosition()[1], spot->getPosition()[2]);
      gluSphere(mSpotLightQuadric, scaleFactor*0.05, 30, 30);
      glPopMatrix();

      VoxelVisualization::drawCylinderFromTo(mSpotLightQuadric, scaleFactor*0.01f, spot->getPosition(),
         spot->getPosition() + scaleFactor*spot->getSpotDirection()*0.7f);
   }
}


void Scene::drawCamera()
{
   glUseProgram(0);
   Debugging::drawPerspectiveFrustum(mCamera);
}


const vector<ObjectSequence*>& Scene::getSceneElements() const
{
   return mSceneElements;
}

const vector<SpotLight*>&  Scene::getSpotLights() const
{
   return mSpotLights;
}

unsigned int Scene::getNumSpotLights() const
{
   return mSpotLights.size();
}

void Scene::setupActiveDynamicElementIndex()
{
   // sets the value of mActiveDynamicElementIndex
   // to the index of the first scene element
   // which is dynamic 

   if(mHasDynamicElements)
   {

      for(unsigned int e = 0; e < mSceneElements.size(); e++)
      {
         if(!mSceneElements.at(e)->isStatic())
         {
            mActiveDynamicElementIndex = e;
            return;
         }
      }
   }
}

void Scene::setupActiveInstanceIndex()
{
   if(mActiveDynamicElementIndex >= 0) // if there is an active dynamic element
   {
      // set the value of mActiveInstanceIndex to the first instance 
      // of this active dynamic element
      mActiveInstanceIndex = 0;
   }
}

/*
void Scene::gotoNextDynamicElement()
{
   if(mActiveDynamicElementIndex >= 0) // if there is an active dynamic element 
   {
      bool setToNext = false;
      for(unsigned int i = mActiveDynamicElementIndex+1; i < mSceneElements.size(); i++)
      {
         if(!mSceneElements.at(i)->isStatic())
         {
            mActiveDynamicElementIndex = i;
            setToNext = true;
            break;
         }
      }
      if(!setToNext)
      // maybe we are at the end of the scene elements vector
      {
         // start at the beginning
         for(int i = 0; i < mActiveDynamicElementIndex; i++)
         {
            if(!mSceneElements.at(i)->isStatic())
            {
               mActiveDynamicElementIndex = i;
               break;
            }
         }

      }

      // reset movable instance index
      setupActiveInstanceIndex();
   }
}

void Scene::gotoNextInstance()
{
   if(mActiveDynamicElementIndex >= 0) // if there is an active dynamic element with movable instances
   {
      bool setToNext = (mActiveInstanceIndex+1) < (int)mSceneElements.at(mActiveDynamicElementIndex)->getNumInstances();
      if(setToNext)
      {
         mActiveInstanceIndex++;
      }
      else // we are at the end of the instance vector
      {
         mActiveInstanceIndex = 0; // start at the beginning
      }
   }
}
*/


void Scene::setActiveInstance(int element, int instance)
{
   if(mHasDynamicElements && element >= 0 && instance >= 0  
      && unsigned int(instance) < mSceneElements.at(element)->getNumInstances())
   {
      mActiveInstanceIndex = instance;
      mActiveDynamicElementIndex = element;
   }
}

void Scene::gotoNextActiveSpotLight()
{
   if(!mSpotLights.empty())
   {
      mActiveSpotLightIndex++;
      if (mActiveSpotLightIndex >= int(mSpotLights.size()))
         mActiveSpotLightIndex = 0;
   }
}

void Scene::moveActiveSpotLight(float dx, float dy, float dz)
{
   if(!mSpotLights.empty())
   {
      mSpotLights.at(mActiveSpotLightIndex)->move(dx, dy, dz);
   }
}

void Scene::modifyActiveSpotLightAngles(float angleXDelta, float angleYDelta, float angleZDelta)
{
   if(!mSpotLights.empty())
   {
      mSpotLights.at(mActiveSpotLightIndex)->modifyAngles(angleXDelta, angleYDelta, angleZDelta);
   }
}

void Scene::modifyActiveSpotLightCutoffAngle(float angleDelta)
{
   if(!mSpotLights.empty())
   {
      mSpotLights.at(mActiveSpotLightIndex)->modifyCutoffAngle(angleDelta);
      mSpotLights.at(mActiveSpotLightIndex)->updatePixelSide();
   }

}


void Scene::moveActiveInstance(float dx, float dy, float dz)
{
   ObjectSequenceInstance* i = mSceneElements.at(mActiveDynamicElementIndex)->getInstance(mActiveInstanceIndex);
   if(mActiveDynamicElementIndex >=0 && i->isUserMovable())
   {
      i->move(dx, dy, dz);
   }
}

void Scene::rotateActiveInstance(float angleX, float angleY, float angleZ)
{
   if(mActiveDynamicElementIndex >=0 && mSceneElements.at(mActiveDynamicElementIndex)->getInstance(mActiveInstanceIndex)->isUserMovable())
      mSceneElements.at(mActiveDynamicElementIndex)->getInstance(mActiveInstanceIndex)->rotate(angleX, angleY, angleZ);
}

Camera* Scene::getCamera() const
{
   return mCamera; 
}

ObjectSequenceInstance* const Scene::getActiveDynamicInstance()
{
   if(mActiveDynamicElementIndex >= 0)
      return mSceneElements.at(mActiveDynamicElementIndex)->getInstance(mActiveInstanceIndex);
   else return 0;
}



void Scene::startAllObjAnimations()
{
   if(mHasAnimatedElements && !mAnimationsRunning)
   {
      for(unsigned int e = 0; e < mSceneElements.size(); e++)
      {
         for(unsigned int i = 0; i < mSceneElements.at(e)->getNumInstances(); i++)
            mSceneElements.at(e)->startAnimation(i);
      }
      mAnimationsRunning = true;
   }
}

void Scene::stopAllObjAnimations()
{
   if(mHasAnimatedElements && mAnimationsRunning)
   {
      for(unsigned int e = 0; e < mSceneElements.size(); e++)
      {
         for(unsigned int i = 0; i < mSceneElements.at(e)->getNumInstances(); i++)
            mSceneElements.at(e)->stopAnimation(i);
      }
      mAnimationsRunning = false;
   }
}

void Scene::updateAllObjAnimations()
{
   if(mHasAnimatedElements && mAnimationsRunning)
   {
      for(unsigned int e = 0; e < mSceneElements.size(); e++)
      {
         for(unsigned int i = 0; i < mSceneElements.at(e)->getNumInstances(); i++)
            mSceneElements.at(e)->updateCurrentFrameIndex(i);
      }
   }
}


bool Scene::areAnimationsRunning() const
{
   return mAnimationsRunning;
}

void Scene::toggleAllObjAnimations()
{
   if(mAnimationsRunning)
   {
      stopAllObjAnimations();
   }
   else
   {
      startAllObjAnimations();
   }
}


//------------ Loading ----------------------------------------------------

void Scene::addStaticObject(const StaticElementData& elem)
{
   ObjModel* obj = new ObjModel(elem.pathModel);
   if(!elem.pathAtlas.empty())
   {
      obj->addAtlasTextureCoordinates(elem.pathAtlas);
   }
   obj->glmFacetNormals();
   if(elem.computedVertexNormals)
   {
      obj->glmVertexNormals(elem.vertexNormalsAngle, elem.vertexNormalsSmoothingGroups);
   }
   if(elem.centered)
   {
      obj->glmCenter();
   }
   if(elem.unitized)
   {
      obj->glmUnitize();
   }
   if(elem.fixedScaleFactor != 1.0)
   {
      obj->glmScale(elem.fixedScaleFactor);
   }

   mSceneElements.push_back(new ObjectSequence(elem.name, elem.atlasWidth, elem.atlasHeight, obj, false));
   mSceneElements.back()->setDefaultDrawMode(elem.defaultDrawMode);
   cout << "num Instances: " << mSceneElements.back()->getNumInstances() << endl;
   mSceneElements.back()->getInstance(0)->setPose(elem.position.x, elem.position.y, elem.position.z,
      elem.rotation.x, elem.rotation.y, elem.rotation.z);
   mSceneElements.back()->getInstance(0)->setScaleFactor(elem.scaleFactor);

}

void Scene::addSingleFrameObject(const DynamicElementData& elem)
{
   ObjModel* obj = new ObjModel(elem.pathModel);
   if(!elem.pathAtlas.empty())
   {
      obj->addAtlasTextureCoordinates(elem.pathAtlas);
   }
   obj->glmFacetNormals();
   if(elem.computedVertexNormals)
   {
      obj->glmVertexNormals(elem.vertexNormalsAngle, elem.vertexNormalsSmoothingGroups);
   }
   if(elem.centered)
   {
      obj->glmCenter();
   }
   if(elem.unitized)
   {
      obj->glmUnitize();
   }
   if(elem.fixedScaleFactor != 1.0)
   {
      obj->glmScale(elem.fixedScaleFactor);
   }

   mSceneElements.push_back(new ObjectSequence(elem.name, elem.atlasWidth, elem.atlasHeight, obj, true));
   mSceneElements.back()->setDefaultDrawMode(elem.defaultDrawMode);

   for(unsigned int i = 0; i < elem.instances.size(); i++)
   {
      DynamicInstanceData inst = elem.instances.at(i);
      mSceneElements.back()->addInstance(inst.isUserMovable, inst.startAtFrame);
      mSceneElements.back()->getInstance(i)->setLooping(inst.looping);
      mSceneElements.back()->getInstance(i)->setForwards(inst.forwards);
      mSceneElements.back()->getInstance(i)->setStepInterval(inst.stepInterval);

      mSceneElements.back()->getInstance(i)->setPose(inst.position.x, inst.position.y, inst.position.z,
         inst.rotation.x, inst.rotation.y, inst.rotation.z);

      mSceneElements.back()->getInstance(i)->setScaleFactor(inst.scaleFactor);
   }
  

}


void Scene::addSequence(const DynamicElementData& elem)
{
   ObjectSequence* seq = new ObjectSequence(elem.name, elem.atlasWidth, elem.atlasHeight, 0, true,
      elem.pathSequence, elem.sequenceReadInMethod);

   bool firstModel = true;
   for(int i = elem.animFileStartIndex; i <= elem.animFileEndIndex; i++)
   {
      stringstream fileName;
      fileName << elem.pathSequence;
      switch(elem.sequenceReadInMethod)
      {
      case 0:
         break;
      case 1:
         if(i < 10)
            fileName << "0";
         break;
      case 2:
         if(i < 10)
            fileName << "00";
         else if(i < 100)
            fileName << "0";
         break;
      case 3:
         if(i < 10)
            fileName << "000";
         else if(i < 100)
            fileName << "00";
         else if(i < 1000)
            fileName << "0";
         break;
      default:
         cerr << "[Scene addSequence ERROR] Not defined read in method" << endl;
         return;
         break;
      }

      fileName << i;
      fileName << ".obj";

      ObjModel* m;
      if(firstModel)
      {
         m = new ObjModel(fileName.str());
         if(!elem.pathAtlas.empty()) m->addAtlasTextureCoordinates(elem.pathAtlas);
         firstModel = false;
      }
      else
      {
         m = new ObjModel(fileName.str(), seq->getModel(0));
      }
      m->glmFacetNormals();

      if(elem.computedVertexNormals)
      {
         m->glmVertexNormals(elem.vertexNormalsAngle, elem.vertexNormalsSmoothingGroups);
      }
      if(elem.centered)
      {
         m->glmCenter();
      }
      if(elem.unitized)
      {
         m->glmUnitize();
      }
      if(elem.fixedScaleFactor != 1.0)
      {
         m->glmScale(elem.fixedScaleFactor);
      }

      seq->addObjModel(m, i);
   }

   for(unsigned int i = 0; i < elem.instances.size(); i++)
   {
      DynamicInstanceData inst = elem.instances.at(i);
      seq->addInstance(inst.isUserMovable, inst.startAtFrame);
      seq->getInstance(i)->setLooping(inst.looping);
      seq->getInstance(i)->setForwards(inst.forwards);
      seq->getInstance(i)->setStepInterval(inst.stepInterval);

      seq->getInstance(i)->setPose(inst.position.x, inst.position.y, inst.position.z,
         inst.rotation.x, inst.rotation.y, inst.rotation.z);

      seq->getInstance(i)->setScaleFactor(inst.scaleFactor);
   }

   mSceneElements.push_back(seq);
   mSceneElements.back()->setDefaultDrawMode(elem.defaultDrawMode);
}


void Scene::load(const SceneData* const data)
{
   // Models
   for(unsigned int i = 0; i < data->staticElements.size(); i++)
   {
      StaticElementData elem = data->staticElements.at(i);
      addStaticObject(elem);
   }

   mHasAnimatedElements = false;

   for(unsigned int i = 0; i < data->dynamicElements.size(); i++)
   {
      DynamicElementData elem = data->dynamicElements.at(i);
      if((elem.animFileEndIndex - elem.animFileStartIndex + 1) == 1)
      {
         addSingleFrameObject(elem);
      }
      else
      {
         mHasAnimatedElements = true;
         addSequence(elem);
      }

   }


   // Camera
   
   mCamera = new Camera();
   mCamera->setPerspectiveFrustum(data->cameraData.fovh, data->cameraData.aspect/*float(mWindowWidth)/mWindowHeight*/, data->cameraData.zNear, data->cameraData.zFar);
   mCameraPoses = data->cameraData.poses;

   // set camera to "current" pose
   mCurrentCameraPoseIndex = data->cameraData.currentPoseIndex;

   mCamera->setUserPosition(mCameraPoses.at(mCurrentCameraPoseIndex).userPosition);
   mCamera->setAngleX(mCameraPoses.at(mCurrentCameraPoseIndex).angleX);
   mCamera->setAngleY(mCameraPoses.at(mCurrentCameraPoseIndex).angleY);
   mCamera->setAngleZ(mCameraPoses.at(mCurrentCameraPoseIndex).angleZ);

   // Lights 

   for(unsigned int i = 0; i < data->spotLights.size(); i++)
   {
      SpotLightData spot = data->spotLights.at(i);
      mSpotLights.push_back(new SpotLight(spot.position, spot.I, spot.cutoffAngle, spot.spotExponent,
         spot.constantAttenuation, spot.quadraticAttenuation, spot.angleX, spot.angleY, spot.angleZ));
   }

   mAnimationsRunning = false;

   if(data->automaticRotation)
   {
      SETTINGS->toggleAutoRotateModel(true);
   }
}


///////////////////////////////////


Scene* Scene::Instance()
{
	if(mInstance == 0)
	{
		mInstance = new Scene();
	}
	return mInstance;
}