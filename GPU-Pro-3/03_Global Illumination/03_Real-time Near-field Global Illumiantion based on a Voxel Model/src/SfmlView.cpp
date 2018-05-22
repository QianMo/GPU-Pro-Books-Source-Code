///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#include "SfmlView.h"

// Voxels
#include "Voxelization/AtlasRenderer.h"
#include "Voxelization/AtlasVoxelization.h"
#include "Voxelization/Bitmask.h"
#include "Voxelization/PreVoxelization.h"
#include "Voxelization/PreVoxelizationData.h"
#include "Voxelization/MipmapRenderer.h"

// Lighting
#include "Lighting/IndirectLight.h"
#include "Lighting/EnvMap.h"
#include "Lighting/SpotMapRenderer.h"

// Post Process
#include "Postprocess/Filter.h"
#include "Postprocess/ToneMapping.h"

// Utils
#include "Utils/FPSCounter.h"
#include "Utils/GLError.h"
#include "Utils/TexturePool.h"
#include "Utils/TimerQuery.h"
#include "Utils/ShaderPool.h"
#include "Utils/ShaderProgram.h"

// Scene
#include "Scene/Camera.h"
#include "Scene/ObjectSequence.h"
#include "Scene/ObjectSequenceInstance.h"
#include "Scene/ObjModel.h"
#include "Scene/Scene.h"
#include "Scene/SpotLight.h"

// Debugging
#include "DebugVisualization/Debugging.h"
#include "DebugVisualization/VoxelVisualization.h"

const int timingIterationsVoxel = 1;
const int timingIterationsMipmap = 1;

SfmlView::SfmlView(sf::ContextSettings contextSettings,
                   SceneData* data,
                   int timerInterval, QWidget* parent,
                   QTextEdit* timerMonitor)
: QSfmlWidget(contextSettings, timerInterval, parent),
  Input(GetInput()),
  qteTimerMonitor(timerMonitor),
  mSceneData(data)
{

   if(data)
   {
      //setFixedSize(data->windowWidth, data->windowHeight);
      setGeometry(0, 40, data->windowWidth, data->windowHeight);
      setFixedSize(data->windowWidth, data->windowHeight);
      setHidden(true);
   }
   else
   {
      setFixedSize(768, 512);
   }
   mSaveScreenshot = false;
 
   mLastPosX = 0;
   mLastPosY = 0;
   mUserClickPos = glm::ivec2(-1, -1);

   mShowGBuffer = true;
   mShowEnvMap = false;
   mShowAtlas = false;
   mSceneElementsChanged = false;

   mElementIndexShowAtlas = 0;
   mInstanceIndexShowAtlas = 0;

   setWindowTitle("VGI-Demo");
   
   cout << "VGI-Demo starting..." << endl;
}


void SfmlView::initialize()
{
	// Check OpenGL Context Settings
	sf::ContextSettings contextSettings = GetSettings();
	cout << "MajorVersion      = " << contextSettings.MajorVersion << endl;
	cout << "MinorVersion      = " << contextSettings.MinorVersion << endl;
	cout << "DepthBits         = " << contextSettings.DepthBits << endl;
	cout << "StencilBits       = " << contextSettings.StencilBits << endl;
	cout << "AntialiasingLevel = " << contextSettings.AntialiasingLevel << endl <<endl;
   cout << GetWidth() << " x " << GetHeight() << endl << endl;

   initGLEW();

   initGL();

   initShader();

   initQtConnections();

   mFPSCounter = new FPSCounter(this);

   initScene();

   preprocess();

   mSceneElementsChanged = true;

   //---- construction and initialization ----

   mMipmapRenderer = new MipmapRenderer();

   mMipmapRenderer->attachMipmapsTo(mVoxelizerAtlas->getBinaryVoxelTexture(),
      mVoxelizerAtlas->getVoxelTextureResolutionX());

   mMipmapRenderer->attachMipmapsTo(mPreVoxelization->getFinalVoxelTexture(),
      mPreVoxData->getVoxelTextureResolution());

   mVoxelVisualizer = new VoxelVisualization(mPreVoxData);

   mIndirectLight = new IndirectLight();
   mFilter        = new Filter();
   mToneMapping   = new ToneMapping(SCENE->getWindowWidth(), SCENE->getWindowHeight());

   IndirectLight::createPIPixelTexture();

   currentVoxelCamera = mPreVoxData->getVoxelCamera();


   //---- OpenGL Timer Queries -----

   tqGBuffer         = new TimerQuery();
   tqSpotMaps        = new TimerQuery();
   tqAtlasRendering  = new TimerQuery();
   tqVoxelization    = new TimerQuery();
   tqMipmap          = new TimerQuery();
   tqVGI             = new TimerQuery();
   tqFilter          = new TimerQuery();

   //cout << "TimerQueries created" << endl;

   mAccumulatedTimeQuery = 0;

   if(!mSceneData->parameterFilename.empty())
      SceneXMLDocument::loadParameterXML(QString::fromStdString(mSceneData->parameterFilename), this);

}



void SfmlView::initQtConnections()
{
   connect(SETTINGS, SIGNAL(toggledAllObjAnimations()), this, SLOT(forceTogglingAllObjAnimations()));
   connect(SETTINGS, SIGNAL(currentCameraPoseChanged(int)), this, SLOT(forceSettingCameraPose(int)));
   connect(SETTINGS, SIGNAL(cameraFovHChanged(double)), this, SLOT(forceSettingCameraFovH(double)));
   connect(SETTINGS, SIGNAL(forwardedPoseAddingRequest()), this, SLOT(forcePoseAdding()));
   connect(SETTINGS, SIGNAL(forwardedPoseDeletionRequest()), this, SLOT(forcePoseDeletion()));
   connect(SETTINGS, SIGNAL(forwardedInstanceAddingRequest()), this, SLOT(forceInstanceAdding()));
   connect(SETTINGS, SIGNAL(changedILBufferSize(int)), this, SLOT(forceILBufferSizeChange(int)));
   connect(SETTINGS, SIGNAL(forwardedSpotAddingRequest()), this, SLOT(forceSpotAdding()));
   connect(SETTINGS, SIGNAL(forwardedSpotDeletionRequest()), this, SLOT(forceSpotDeletion()));
   connect(SETTINGS, SIGNAL(forwardedLightColorChangeRequest(float, float, float, float)), this, 
      SLOT(forceLightColorChange(float, float, float, float)));
   
   connect(SETTINGS, SIGNAL(randomTextureSizeChanged()), this, SLOT(forceRandomTextureUpdate()));
}

void SfmlView::initScene()
{
   SCENE->initialize(GetWidth(), GetHeight());

   ObjModel::createWhitePixelTexture();

   FullScreenQuad::setupQuadDisplayList(); // display list for a single quad

   // load a predefined scene

   if(mSceneData != 0)
   {
      SCENE->load(mSceneData);
   }
   else
   {
      std::cerr << "[ERROR] No scene data loaded." << std::endl;
   }

   mCurrentViewCamera  = SCENE->getCamera();
}

void SfmlView::preprocess()
{
   tqPreProcess = new TimerQuery();
   tqPreProcess->start(true);

   Bitmask::createBitmasks();

   mAtlasRenderer = new AtlasRenderer(); // constructor renders geometry atlases

   SCENE->postLoadProcessing();

   mPreVoxData = new PreVoxelizationData();
   updatePreVoxData();

   mPreVoxData->coutVoxelTextureData();


   cout << endl;
   mPreVoxelization = new PreVoxelization(mAtlasRenderer, mPreVoxData);
   cout << endl;
   mPreVoxelization->voxelizeStaticSceneElements();


   // atlas voxelizer uses the voxel camera computed for single voxel texture
   mVoxelizerAtlas = new AtlasVoxelization(
      mPreVoxData->getVoxelTextureResolution(),
      mPreVoxData->getVoxelTextureResolution(),
      128);

   tqPreProcess->end(true);
   tqPreProcess->waitAndAccumResult();
   cout << endl << "PRE PROCESS TIME: " << tqPreProcess->getAccumAverageResult() << " ms" << endl << endl;

}

void SfmlView::initGLEW()
{
	// Init glew so that the GLSL functionality will be available
	if(glewInit() != GLEW_OK)
		cout << "GLEW init failed!" << endl;

	// Check for GLSL availability
	if(!GLEW_VERSION_2_0)
		cout << "OpenGL 2.0 not supported!" << endl;
}

void SfmlView::initShader()
{   
   glUseProgram(0);
}


void SfmlView::initGL()
{
}

void SfmlView::queryTimes()
{
   mAccumulatedTimeQuery += GetFrameTime();

   tqVoxelization->waitAndAccumResult();
   //if(!tqMipmap->waitAndAccumResult() && SETTINGS->mipmappingEnabled())
   //   cout << "tqMipmap: 0 time" << endl;

   tqMipmap->waitAndAccumResult();

   tqGBuffer->waitAndAccumResult();
   tqAtlasRendering->waitAndAccumResult();
   //tqSpotMaps->waitAndAccumResult();

   tqVGI->waitAndAccumResult();
   tqFilter->waitAndAccumResult();

   if(qteTimerMonitor != 0 && mAccumulatedTimeQuery > 4.0) // print timing results every 2 seconds
   {
      if(!qteTimerMonitor->isHidden() /*true*//*mWriteToTextEdit*/)
      {
         qteTimerMonitor->clear();

         qteTimerMonitor->append("[Time ms] GBuffer:\t" + QString::number(tqGBuffer->getAccumAverageResult(), 'g', 5));
         qteTimerMonitor->append("[Time ms] Atlases:\t" + QString::number(tqAtlasRendering->getAccumAverageResult(), 'g', 5));
         qteTimerMonitor->append("[Time ms] Voxelization:\t" + QString::number(tqVoxelization->getAccumAverageResult()/timingIterationsVoxel, 'g', 5));
        // qteTimerMonitor->append("[Time ms] VoxelMipmaps:\t" + QString::number(tqMipmap->getAccumAverageResult()/timingIterationsMipmap, 'g', 5));
        // qteTimerMonitor->append("[Time ms] Spot/Shadow Maps:\t" + QString::number(tqSpotMaps->getAccumAverageResult(), 'g', 5));
         qteTimerMonitor->append("[Time ms] IL VGI:\t" + QString::number(tqVGI->getAccumAverageResult(), 'g', 5));
         qteTimerMonitor->append("[Time ms] Filter:\t\t" + QString::number(tqFilter->getAccumAverageResult(), 'g', 5));
      }
      mAccumulatedTimeQuery = 0;
   }
}


void SfmlView::refresh()
{
   if(mSceneElementsChanged)
   {
      emit(sceneLoaded());
      mSceneElementsChanged = false;
   }

   if(SETTINGS->getVoxelTextureResolution() != mPreVoxData->getVoxelTextureResolution())
   { 
      updatePreVoxData();
      mPreVoxelization->update();
      mMipmapRenderer->attachMipmapsTo(mPreVoxelization->getFinalVoxelTexture(),
         mPreVoxData->getVoxelTextureResolution());
   }
   if(SETTINGS->getVoxelTextureResolution() != mVoxelizerAtlas->getVoxelTextureResolutionX())
   {
      mVoxelizerAtlas->changeVoxelTextureResolution(SETTINGS->getVoxelTextureResolution(),
         SETTINGS->getVoxelTextureResolution());
      mMipmapRenderer->attachMipmapsTo(mVoxelizerAtlas->getBinaryVoxelTexture(),
         mVoxelizerAtlas->getVoxelTextureResolutionX());
   }


 //  std::cout << " new frame " << std::endl;

   if(SETTINGS->renderingEnabled())
   {
      float fps = mFPSCounter->getFPS(); // 1.0f / GetFrameTime();
      emit(updatedFPS(fps));
      queryTimes();

      processSfmlEvents();

      SCENE->updateAllObjAnimations();
      if(SETTINGS->autoRotateModel() && SCENE->hasDynamicElements())
      {
         float speed = 35;//45.0f; 
         float angle = mRotationClock.GetElapsedTime() * speed;
         if(SETTINGS->autoRotateOnXAxis())
            SCENE->getActiveDynamicInstance()->setAutoOrientation(angle, 0, 0);
         else if(SETTINGS->autoRotateOnYAxis())
            SCENE->getActiveDynamicInstance()->setAutoOrientation(0, angle, 0);         
         else if(SETTINGS->autoRotateOnZAxis())
            SCENE->getActiveDynamicInstance()->setAutoOrientation(0, 0, angle);
         ;
      }

      render();	
   }

   if(mSaveScreenshot)
      saveCurrentFrame();

}

void SfmlView::saveCurrentFrame()
{
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
   
   unsigned char* imgBuffer = (unsigned char*)malloc(GetWidth()* GetHeight() * 4);
   unsigned char* imgBufferMirror = (unsigned char*)malloc(GetWidth()* GetHeight() * 4);

   if (!imgBuffer || !imgBufferMirror )
      return;

   glReadPixels((GLint)0, (GLint)0,
      (GLint)GetWidth(), (GLint)GetHeight(),
      GL_RGBA, GL_UNSIGNED_BYTE, imgBuffer);

   // mirror
   for(uint y = 0; y < GetHeight(); y++)
      for(uint x = 0; x < GetWidth(); x++)
      {
         int i = y * GetWidth() + x;
         int iMirr = (GetHeight() - 1 - y) * GetWidth() + x;
         imgBufferMirror[4 * i] = imgBuffer[4 * iMirr];
         imgBufferMirror[4 * i + 1] = imgBuffer[4 * iMirr + 1];
         imgBufferMirror[4 * i + 2] = imgBuffer[4 * iMirr + 2];
         imgBufferMirror[4 * i + 3] = 255;
      }

   sf::Image img;
   if(img.LoadFromPixels(GetWidth(), GetHeight(), imgBufferMirror))
   {
       img.SaveToFile(mScreenshotFile.c_str());
   }

   free(imgBuffer);
   free(imgBufferMirror);

   cout << "Written image to disk: " << mScreenshotFile << endl;

   mSaveScreenshot = false;
   
}

void SfmlView::generateGBuffer()
{
   tqGBuffer->start(true);
   SCENE->createGBuffer(SETTINGS->shadowMappingEnabled());
   tqGBuffer->end(true);

   glEnable(GL_DEPTH_TEST);

   if(SCENE->mShowLights)
   {
      SCENE->drawLights();
   }

   if(SETTINGS->displayBoundingBoxes())
   {
      SCENE->drawSceneBoundingBox();
      SCENE->drawDynamicBoundingBoxes();
   }
   // draw voxel camera frustum and voxels
   if(SETTINGS->displayVoxelCamera())
   {
      glUseProgram(0);
      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT + max(3, SETTINGS->getCurrentGBufferTex())); 
      drawVoxelCameraFrustum();
   }

   if(SETTINGS->highlightModel())
   {
      SCENE->drawHighlightedModel();
   }  
}

void SfmlView::generateSpotMaps()
{
   if(!SCENE->getSpotLights().empty())
   {
      bool needSpotMaps
          = !mShowGBuffer
         || SETTINGS->indirectLightEnabled()
         && (SETTINGS->VGIEnabled()
         || SETTINGS->ambientTermEnabled())
         || SETTINGS->binaryRayCastingEnabled();

      tqSpotMaps->start(true);
      SCENE->generateShadowAndSpotMaps(SETTINGS->shadowMappingEnabled(), needSpotMaps, true);
      tqSpotMaps->end(true);
   }
}

void SfmlView::generateAtlasTextures()
{
      // for this we only need the current position atlases
   bool needPositionAtlas 
      =  SETTINGS->VGIEnabled()
      || SETTINGS->voxelCubesEnabled() && SETTINGS->atlasBinaryVoxelizationEnabled()
      || SETTINGS->atlasBinaryVoxelizationEnabled() && SETTINGS->voxelizationEnabled()
      || mShowAtlas
      ;

   if(needPositionAtlas)
   {
      tqAtlasRendering->start(true);
      mAtlasRenderer->updateAllTextureAtlases();
      tqAtlasRendering->end(true);
   }
}

void SfmlView::generateVoxelizations()
{
   bool needAnyVox 
      =  SETTINGS->VGIEnabled()
      || SETTINGS->voxelizationEnabled()
      || SETTINGS->VGIEnabled();


   if(!needAnyVox)
      return;

   if(SETTINGS->voxelizationEnabled())
   {
      userControlledVoxelization(); 
   }

   else
   {
      tqVoxelization->start(true);
      mPreVoxelization->insertDynamicObjects();
      tqVoxelization->end(true);


      tqMipmap->start();
      mMipmapRenderer->renderMipmapsFor(mPreVoxelization->getFinalVoxelTexture(),
         mPreVoxData->getVoxelTextureResolution());
      tqMipmap->end();

   }
}

void SfmlView::updatePreVoxData()
{
   mPreVoxData->computeVoxelTextureData(
      SCENE->getSceneBoundingBoxDimension().x, SCENE->getSceneBoundingBoxDimension().y, SCENE->getSceneBoundingBoxDimension().z,
      SCENE->getSceneBoundingBoxCenter().x, SCENE->getSceneBoundingBoxCenter().y, SCENE->getSceneBoundingBoxCenter().z,
      SETTINGS->getVoxelTextureResolution());
}


void SfmlView::userControlledVoxelization()
{
   // Atlas (Binary)
   if(SETTINGS->atlasBinaryVoxelizationEnabled())
   {
      if(SETTINGS->preVoxelizationEnabled())
      {
         mCurrentBinaryVoxelTexture       = mPreVoxelization->getFinalVoxelTexture();
         mCurrentBinaryVoxelTextureWidth  = mPreVoxData->getVoxelTextureResolution();
         mCurrentBinaryVoxelTextureHeight = mPreVoxData->getVoxelTextureResolution();
         currentVoxelCamera = mPreVoxData->getVoxelCamera();

         tqVoxelization->start(true);
         //for(unsigned int i = 0; i < timingIterationsVoxel; i++)
         {
            mPreVoxelization->insertDynamicObjects();
         }
         tqVoxelization->end(true);

      }
      else
      {
         mCurrentBinaryVoxelTexture       = mVoxelizerAtlas->getBinaryVoxelTexture();
         mCurrentBinaryVoxelTextureWidth  = mVoxelizerAtlas->getVoxelTextureResolutionX();
         mCurrentBinaryVoxelTextureHeight = mVoxelizerAtlas->getVoxelTextureResolutionY();
         currentVoxelCamera = mPreVoxData->getVoxelCamera();

         tqVoxelization->start(true);
         //for(unsigned int i = 0; i < timingIterationsVoxel; i++)
         {
            mVoxelizerAtlas->voxelizeBinary(mAtlasRenderer,
               currentVoxelCamera);

         }
         tqVoxelization->end(true);

      }
      if(SETTINGS->mipmappingEnabled())
      {
         tqMipmap->start(true);
         //for(int i = 0; i < timingIterationsMipmap; i++)
         {
            mMipmapRenderer->renderMipmapsFor(mCurrentBinaryVoxelTexture, mCurrentBinaryVoxelTextureWidth);
         }
         tqMipmap->end(true);
      }

   }   
}


void SfmlView::displayAtlasTexture()
{
   ShaderPool::getQuad()->useProgram();
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, mAtlasRenderer->getTextureAtlas(mElementIndexShowAtlas, mInstanceIndexShowAtlas));
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
   glPushAttrib(GL_VIEWPORT_BIT);
   int w = SCENE->getSceneElements().at(mElementIndexShowAtlas)->getAtlasWidth();
   int h = SCENE->getSceneElements().at(mElementIndexShowAtlas)->getAtlasHeight();
   float scale = min(GetWidth() / float(w), GetHeight()/ float(h));
   glViewport(0, 0, w*scale, h*scale);
   FullScreenQuad::drawComplete();

   // Draw valid pixels
   // glPointSize(1.0f);
   // Debugging::renderPixelDisplayList(mAtlasRenderer->getPixelDisplayList(mElementIndexShowAtlas), w, h);
}

void SfmlView::displayEnvMap()
{
   ShaderPool::getQuad()->useProgram();
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, SCENE->getEnvMap()->getTexture());
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
   glPushAttrib(GL_VIEWPORT_BIT);
   int w = SCENE->getEnvMap()->getTextureWidth();
   int h = SCENE->getEnvMap()->getTextureHeight();
   float scale = min(GetWidth() / float(w), GetHeight()/ float(h));
   glViewport(0, 0, w*scale, h*scale);
   FullScreenQuad::drawComplete();
   glPopAttrib();

}


void SfmlView::drawVoxelCameraFrustum()
{
   Debugging::drawOrthoFrustum(currentVoxelCamera);
}

void SfmlView::rayCasting()
{

   if(SETTINGS->atlasBinaryVoxelizationEnabled())
   {
      if(SETTINGS->mipmappingEnabled())
      {
         mVoxelVisualizer->rayCastMipmappedBinaryVoxelTexture(
            mCurrentBinaryVoxelTexture,
            mCurrentBinaryVoxelTextureWidth, SETTINGS->getMipmapLevel(),
            currentVoxelCamera);
      }
      else
      {
         mVoxelVisualizer->rayCastBinaryVoxelTexture(
            mCurrentBinaryVoxelTexture,
            currentVoxelCamera);
      }
   }


}


void SfmlView::drawVoxelCubes()
{
   bool withLighting = true;
   
   if(withLighting)
   {
      glEnable(GL_LIGHTING);
      glEnable(GL_LIGHT2);
      glEnable(GL_LIGHT3);
      glEnable(GL_LIGHT4);
      glEnable(GL_LIGHT5);
      glEnable(GL_LIGHT6);
      glEnable(GL_LIGHT7);
      glLightfv(GL_LIGHT5, GL_DIFFUSE, &glm::vec4(0.6, 0.6, 0.6, 1)[0]);
      glLightfv(GL_LIGHT5, GL_POSITION, &glm::vec4(1, 0, 0, 0)[0]);
      glLightfv(GL_LIGHT6, GL_DIFFUSE, &glm::vec4(0.4, 0.4, 0.4, 1)[0]);
      glLightfv(GL_LIGHT6, GL_POSITION, &glm::vec4(0, 1, 0, 0)[0]);
      glLightfv(GL_LIGHT7, GL_DIFFUSE, &glm::vec4(0.3, 0.3, 0.3, 1)[0]);
      glLightfv(GL_LIGHT7, GL_POSITION, &glm::vec4(0, 0, 1, 0)[0]);

      glLightfv(GL_LIGHT2, GL_DIFFUSE, &glm::vec4(0.2, 0.2, 0.2, 1)[0]);
      glLightfv(GL_LIGHT2, GL_POSITION, &glm::vec4(-1, 0, 0, 0)[0]);
      glLightfv(GL_LIGHT3, GL_DIFFUSE, &glm::vec4(0.3, 0.3, 0.3, 1)[0]);
      glLightfv(GL_LIGHT3, GL_POSITION, &glm::vec4(0, -1, 0, 0)[0]);
      glLightfv(GL_LIGHT4, GL_DIFFUSE, &glm::vec4(0.4, 0.4, 0.4, 1)[0]);
      glLightfv(GL_LIGHT4, GL_POSITION, &glm::vec4(0, 0, -1, 0)[0]);

      glEnable(GL_COLOR_MATERIAL);
      glColor3f(0.8f, 0.8f, 0.8f);
   }
   else
   {
      glDisable(GL_LIGHTING);
   }

   glUseProgram(0);
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
   glClearColor(1, 1, 1, 1);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   // unbind all textures
   for(int i = 0; i < 8; i++)
   {
      glActiveTexture(GL_TEXTURE0 + i);
      glBindTexture(GL_TEXTURE_1D, 0);
      glBindTexture(GL_TEXTURE_2D, 0);
   }


   glEnable(GL_DEPTH_TEST);


   if(SETTINGS->mipmappingEnabled())
   {
      if(SETTINGS->mipmapTestVisualizationEnabled())
      {
         mVoxelVisualizer->drawVoxelsAsCubesInstancedMipmapTestVis(
            mCurrentBinaryVoxelTexture,
            mCurrentBinaryVoxelTextureWidth, mCurrentBinaryVoxelTextureHeight,
            MipmapRenderer::computeMaxLevel(mCurrentBinaryVoxelTextureWidth),
            currentVoxelCamera, mUserClickPos, mIndirectLight);
      }
      else
      {
         mVoxelVisualizer->drawVoxelsAsCubesInstancedMipmapped(
            mCurrentBinaryVoxelTexture,
            mCurrentBinaryVoxelTextureWidth, mCurrentBinaryVoxelTextureHeight,
            currentVoxelCamera);
      }
   }
   else // no mipmapping
   {
      mVoxelVisualizer->drawVoxelsAsCubesInstanced(
         mCurrentBinaryVoxelTexture,
         mCurrentBinaryVoxelTextureWidth, mCurrentBinaryVoxelTextureHeight,
         currentVoxelCamera);

      glUseProgram(0);
   }


   if(withLighting)
   {
      glDisable(GL_LIGHT2);
      glDisable(GL_LIGHT3);
      glDisable(GL_LIGHT4);

      glDisable(GL_LIGHT5);
      glDisable(GL_LIGHT6);
      glDisable(GL_LIGHT7);

      glDisable(GL_LIGHTING);
      glDisable(GL_COLOR_MATERIAL);
   }
   glDisable(GL_DEPTH_TEST);
}


void SfmlView::render()
{
   glViewport(0, 0, GetWidth(), GetHeight());
   glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
   glClearColor(1, 1, 1, 1);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   glMatrixMode(GL_PROJECTION);
   glLoadMatrixf(mCurrentViewCamera->getProjectionMatrix());

   glMatrixMode(GL_MODELVIEW);
   glLoadMatrixf(&mCurrentViewCamera->getViewMatrix()[0][0]);


   // Generate Spot Maps if needed
   generateSpotMaps();

   // Generate Atlas Textures if needed
   generateAtlasTextures();

   // Generate Voxelizations if needed
   generateVoxelizations();

   // Generate GBuffer if needed
   generateGBuffer();

   // No Voxel Visualization
   if(!SETTINGS->voxelVisualizationEnabled())
   {
      // Texture Atlas Visualization
      if(mShowAtlas)
      {
         displayAtlasTexture();
      }

      // display loaded env map
      else if(mShowEnvMap)
      {
         displayEnvMap();
      }

      // Default Scene and Indirect Light drawing
      else
      {
         // Compute Indirect Light
         if(SETTINGS->indirectLightEnabled() && SETTINGS->getCurrentGBufferTex() == 4)
         {
            computeIndirectLight();
            combineAndShowIndirectLightResult();
         }

         // Choose one of the G-Buffer Textures to display
         else 
         {
            displayGBufferOrSpotMap();
         }

         SCENE->drawLights();
      }

   } 

   if(SETTINGS->binaryRayCastingEnabled())
   {
      rayCasting();
   }

   else if(SETTINGS->voxelCubesEnabled())
   {
      drawVoxelCubes();
   }

   if(SETTINGS->displayVoxelCamera())
   {
      drawVoxelCameraFrustum();
   }

   if(SETTINGS->displayWorldSpaceAxes())
   {
      Debugging::drawWorldSpaceAxes();
   }


}

void SfmlView::displayGBufferOrSpotMap()
{
   glActiveTexture(GL_TEXTURE0);         
   Buffer b;
   if(SETTINGS->getCurrentGBufferTex() == 4)
   {
      b = DIRECTLIGHT;
   }
   else
   {
      b = static_cast<Buffer>(SETTINGS->getCurrentGBufferTex());
   }
   if(mShowGBuffer)
   {
      glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(b));

      if(b != DIRECTLIGHT)
      {
         glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

         // no tone mapping or gamma correction
         //if(b == POSITION)
         //   ShaderPool::getQuadAlpha(); 
         //else
         ShaderPool::getQuad(); 
         FullScreenQuad::drawComplete();
      }
      else // b == DIRECTLIGHT
      {
         mToneMapping->renderToFBO();

         bool TONEMAP_LINEAR = SETTINGS->linearToneMappingEnabled();
         bool TONEMAP_LOG    = SETTINGS->logToneMappingEnabled();

         ShaderPool::getQuad(); // no tone mapping or gamma correction
         FullScreenQuad::drawComplete();

         if(SETTINGS->linearToneMappingEnabled())
            mToneMapping->tonemapLinear();
         else if(SETTINGS->logToneMappingEnabled())
            mToneMapping->tonemapLog();
         else 
            mToneMapping->onlyGammaCorrection();

      }

   }
   else //if(!mShowGBuffer)
   {

      // render spot map 
      // std::cout << "Render Spot map for light " << SCENE->getActiveSpotLightIndex()<< std::endl;

      glPushAttrib(GL_VIEWPORT_BIT);

      glViewport(0, 0, std::min<int>(GetHeight(), std::max<int>(512, SCENE->getSpotMapRenderer()->getSpotMapResolution())), std::min<int>(GetHeight(), std::max<int>(512, SCENE->getSpotMapRenderer()->getSpotMapResolution())));
      //glBindTexture(GL_TEXTURE_2D, mIndirectLight->getRandTex());
      glBindTexture(GL_TEXTURE_2D, SCENE->getSpotMapRenderer()->getSpotMap(static_cast<SpotBuffer>(b), SCENE->getActiveSpotLightIndex()));
      ShaderPool::getQuadGamma(); // gamma correction

      glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
      FullScreenQuad::drawComplete();

      glUseProgram(0);

      glPopAttrib();
   }
}


void SfmlView::computeIndirectLight()
{
   if(SETTINGS->VGIEnabled())
   {
      tqVGI->start(true);

      mIndirectLight->computeVGIWithMipmapTest(
         mPreVoxelization->getFinalVoxelTexture(),
         mPreVoxData->getVoxelTextureResolution(),
         MipmapRenderer::computeMaxLevel(mPreVoxData->getVoxelTextureResolution()),
         mPreVoxData->getVoxelCamera(), SCENE->getEnvMap());

      tqVGI->end(true);
   }

   else if(SETTINGS->ambientTermEnabled())
   {
      mIndirectLight->computeAmbientTerm();
   }
}

void SfmlView::combineAndShowIndirectLightResult()
{
   // Combination
   GLuint resultTex = mIndirectLight->getResult();

   GLuint inputTex = resultTex;

   if(SETTINGS->filterEnabled())
   {
      tqFilter->start(true);

      mFilter->upsampleSpatial(inputTex, mIndirectLight->getCurrentBufferWidth(), mIndirectLight->getCurrentBufferHeight());

      tqFilter->end(true);

      resultTex = mFilter->getResult();
   }

   if(SETTINGS->surfaceDetailEnabled())
   {
      mFilter->addSurfaceDetail(resultTex);
      resultTex = mFilter->getResult();
   }

   mToneMapping->renderToFBO();

   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, resultTex);   
   V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_NEAREST));
   V(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_NEAREST));
   glActiveTexture(GL_TEXTURE3);
   glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(POSITION));

   //L_ind(diffuse) = E_ind * rho/PI (=material gbuffer)

   if(SETTINGS->indirectLight_L_ind_Enabled()) 
   {
      if(SETTINGS->VGIEnabled())
      {
         ShaderProgram* q = ShaderPool::getQuadCombine();
         q->useProgram();
         glUniform1i(q->getUniformLocation("addDirectLight"), 0);
         glUniform1f(q->getUniformLocation("scaleIL"), 1.0f /*SETTINGS->getIndirectLightScaleFactor()*/);
         glActiveTexture(GL_TEXTURE0);
         glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(DIRECTLIGHT));
         glActiveTexture(GL_TEXTURE1);
         glBindTexture(GL_TEXTURE_2D, resultTex);
         glActiveTexture(GL_TEXTURE2);
         glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(MATERIAL));

      }
      else
      {
         ShaderProgram* q = ShaderPool::getQuad2Tex();
         q->useProgram();
         glActiveTexture(GL_TEXTURE0);
         glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(MATERIAL));
         glActiveTexture(GL_TEXTURE1);
         glBindTexture(GL_TEXTURE_2D, resultTex);
         glUniform1i(q->getUniformLocation("tex1SingleChannel"), 0); 
         glUniform1i(q->getUniformLocation("operation"), 0); // 0 = Multiply, 1 = Add

      }

   }
   // L_dir + L_ind
   else if(SETTINGS->indirectLightCombinationEnabled())
   {
      ShaderProgram* q = ShaderPool::getQuadCombine();
      q->useProgram();
      glUniform1i(q->getUniformLocation("addDirectLight"), 1);
      glUniform1f(q->getUniformLocation("scaleLdir"), SETTINGS->getDirectLightScaleFactor());
      glUniform1f(q->getUniformLocation("scaleIL"), 1.0f/*SETTINGS->getIndirectLightScaleFactor()*/);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(DIRECTLIGHT));
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, resultTex);
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, SCENE->getBuffer(MATERIAL));
      
   }
   // E_ind 
   // "raw result", keep what is accumulated in indirect light buffer
   else
   {
      ShaderProgram* q = ShaderPool::getWriteContribTex();
      glUniform1f(q->getUniformLocation("contrib"), 1.0f /*SETTINGS->getIndirectLightScaleFactor()*/);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, resultTex);
   }

   FullScreenQuad::drawComplete();
   if(SETTINGS->linearToneMappingEnabled())
      mToneMapping->tonemapLinear();
   else if(SETTINGS->logToneMappingEnabled())
      mToneMapping->tonemapLog();
   else 
      mToneMapping->onlyGammaCorrection();
}


void SfmlView::processSfmlEvents()
{
   sf::Event Event;
   bool camManipulated = false;
   bool lightManipulated = false;
   bool objectManipulated = false;
   while (GetEvent(Event))
   {
      switch(Event.Type)
      {
      case sf::Event::MouseWheelMoved:
         if(SETTINGS->spotLightActive())
         {
            // move active spot light
            SCENE->moveActiveSpotLight(0, 0, Event.MouseWheel.Delta/8.0);
            lightManipulated = true;
         }
         else if(SETTINGS->modelActive())
         {
            // move model (scene element tagged as movable by user)
            SCENE->moveActiveInstance(SCENE->getCamera()->getViewDirection() * glm::vec3(1, 0, 1) * (Event.MouseWheel.Delta/8.0f));
            objectManipulated = true;
         }
         else
         {
            // move back and forwards
            mCurrentViewCamera ->move(0, 0, Event.MouseWheel.Delta/8.0);
            camManipulated = true;
         }
         break;


      case sf::Event::MouseButtonPressed:
         if(Input.IsMouseButtonDown(sf::Mouse::Right))
         {
            mUserClickPos.x = Event.MouseButton.X;
            mUserClickPos.y = GetHeight() - 1 - Event.MouseButton.Y;
            cout << mUserClickPos.x << " " << mUserClickPos.y << endl;
         }
         mLastPosX = Event.MouseButton.X;
         mLastPosY = Event.MouseButton.Y;
         break;

      case sf::Event::MouseMoved:
         if(Input.IsMouseButtonDown(sf::Mouse::Middle))
         {
            int currentX = Event.MouseMove.X < 65000 ? Event.MouseMove.X : 0;
            int currentY = Event.MouseMove.Y < 65000 ? Event.MouseMove.Y : 0;

            int dx = mLastPosX - currentX;
            int dy = mLastPosY - currentY;

            if(SETTINGS->cameraActive())
            {
               // rotate camera
               mCurrentViewCamera->modifyAngleX(-dy/8.0);
               mCurrentViewCamera->modifyAngleY(-dx/8.0);
               camManipulated = true;
            }
            else if(SETTINGS->modelActive())
            {
               if(Input.IsKeyDown(sf::Key::LShift))
                  SCENE->rotateActiveInstance(dy/8.0, -dx/8.0, 0);
               else
                  SCENE->rotateActiveInstance(0, -dx/8.0, 0);
               objectManipulated = true;
            }
            else if(SETTINGS->spotLightActive())
            {
               SCENE->modifyActiveSpotLightAngles(-dy/4.0, -dx/4.0, 0);
               lightManipulated = true;
            }

            mLastPosX = currentX;
            mLastPosY = currentY;
         }
         else if(Input.IsMouseButtonDown(sf::Mouse::Left))
         {
            int currentX = Event.MouseMove.X < 65000 ? Event.MouseMove.X : 0;
            int currentY = Event.MouseMove.Y < 65000 ? Event.MouseMove.Y : 0;
            // cout << currentX << " " << currentY << endl;
            int dx = mLastPosX - currentX;
            int dy = mLastPosY - currentY;

            if(SETTINGS->modelActive())
            {
               if(Input.IsKeyDown(sf::Key::LShift))
               {
                  SCENE->moveActiveInstance(0.0f, dy/64.0f, 0.0f);
               }
               else
               {
                  SCENE->moveActiveInstance(SCENE->getCamera()->getRightVector() * glm::vec3(1, 0, 1) * (-dx/64.0f)
                     + SCENE->getCamera()->getViewDirection() * glm::vec3(1, 0, 1) * (dy/64.0f));
               }
               objectManipulated = true;
            }
            else if(SETTINGS->spotLightActive())
            {
               SCENE->moveActiveSpotLight(-dx/64.0, dy/64.0, 0.0f);
               lightManipulated = true;
            }
            else
            {
               if(Input.IsKeyDown(sf::Key::LShift))
               {
                  mCurrentViewCamera->move(0, 0, -dy/256.0);
               }
               else
               {
                  mCurrentViewCamera->move(dx/256.0, -dy/256.0, 0.0);
               }
               camManipulated = true;
            }

            mLastPosX = currentX;
            mLastPosY = currentY;
         }
         else if(Input.IsMouseButtonDown(sf::Mouse::Right))
         {
            int currentX = Event.MouseMove.X < 65000 ? Event.MouseMove.X : 0;
            int currentY = Event.MouseMove.Y < 65000 ? Event.MouseMove.Y : 0;
            //cout << currentX << " " << currentY << endl;
         }
         break;

      case sf::Event::KeyPressed:
         switch(Event.Key.Code)
         {
         case sf::Key::A:
            if(!Input.IsKeyDown(sf::Key::Add))
               SCENE->toggleAllObjAnimations();
            break;
         case sf::Key::Num2:
            VoxelVisualization::voxelAlpha -= 0.05f;
            VoxelVisualization::voxelAlpha = max<float>(0.0f, VoxelVisualization::voxelAlpha );
            break;
         case sf::Key::Num4:
            VoxelVisualization::voxelAlpha += 0.05f;
            VoxelVisualization::voxelAlpha = min<float>(1.0f, VoxelVisualization::voxelAlpha );
            break;       


         case sf::Key::Left:
            //if(Input.IsKeyDown(sf::Key::LShift))
            //{
            //}
            //else if(Input.IsKeyDown(sf::Key::LControl))
            //{
            //}
            if(Input.IsKeyDown(sf::Key::Up))
            {
               SCENE->getActiveDynamicInstance()->moveAlongViewingDirection(0.2f);
            }
            else if(Input.IsKeyDown(sf::Key::Down))
            {
               SCENE->getActiveDynamicInstance()->moveAlongViewingDirection(-0.2f);
            }
            SCENE->getActiveDynamicInstance()->rotate(0, 5, 0);
         
            break;
         case sf::Key::Right:
            if(SETTINGS->spotLightActive())
            {  
               SCENE->gotoNextActiveSpotLight();
            }
            else if(SETTINGS->modelActive())
            {
               if(Input.IsKeyDown(sf::Key::LShift))
               {
                  SCENE->getActiveDynamicInstance()->rotateViewingDirection(5); 
               }
               else
               {
                  if(Input.IsKeyDown(sf::Key::Up))
                  {
                     SCENE->getActiveDynamicInstance()->moveAlongViewingDirection(0.2f);
                  }
                  else if(Input.IsKeyDown(sf::Key::Down))
                  {
                     SCENE->getActiveDynamicInstance()->moveAlongViewingDirection(-0.2f);
                  }
                  SCENE->getActiveDynamicInstance()->rotate(0, -5, 0); 
               }

               
               //if(Input.IsKeyDown(sf::Key::LShift))
               //{
               //   SCENE->gotoNextInstance();
               //}
               //else
               //{
               //   SCENE->gotoNextDynamicElement();
               //}
            }
            else
            {
               if(Input.IsKeyDown(sf::Key::LShift))
                  SCENE->getEnvMap()->rotate(-0.1f);
               else
                  SCENE->getEnvMap()->rotate(0.1f);
            }
            break;

         case sf::Key::Up:
            if(Input.IsKeyDown(sf::Key::Left))
            {
               SCENE->getActiveDynamicInstance()->rotate(0, 5, 0); 
            }
            else if(Input.IsKeyDown(sf::Key::Right))
            {
               SCENE->getActiveDynamicInstance()->rotate(0, -5, 0); 
            }
            SCENE->getActiveDynamicInstance()->moveAlongViewingDirection(0.2f);

            break;
         case sf::Key::Down:
            if(Input.IsKeyDown(sf::Key::Left))
            {
               SCENE->getActiveDynamicInstance()->rotate(0, 5, 0); 
            }
            else if(Input.IsKeyDown(sf::Key::Right))
            {
               SCENE->getActiveDynamicInstance()->rotate(0, -5, 0); 
            }
            SCENE->getActiveDynamicInstance()->moveAlongViewingDirection(-0.2f);

            break;
         case sf::Key::F:
            if(Input.IsKeyDown(sf::Key::LShift))
               SCENE->getSpotMapRenderer()->modifyOffsetFactor(0.1f);
            else
               SCENE->getSpotMapRenderer()->modifyOffsetFactor(-0.1f);
			cout << SCENE->getSpotMapRenderer()->getOffsetFactor() << endl;
            break;

         case sf::Key::Add:
            if(Input.IsKeyDown(sf::Key::A))
            {
               // atlas resolution
               mAtlasRenderer->changeAtlasResolutionRelative(mElementIndexShowAtlas, 16, 16);
            }
            break;
         case sf::Key::Subtract:
            if(Input.IsKeyDown(sf::Key::A))
            {
               // atlas resolution
               mAtlasRenderer->changeAtlasResolutionRelative(mElementIndexShowAtlas, -16, -16);
            }
            break;

         case sf::Key::T:
            mShowAtlas =  !mShowAtlas;
            break;

         case sf::Key::M:
            if(SETTINGS->spotLightActive())
            {
               if(Input.IsKeyDown(sf::Key::LShift))
                  SCENE->getSpotMapRenderer()->changeMapResolution(64);
               else
                  SCENE->getSpotMapRenderer()->changeMapResolution(-64);
            }
            break;
         case sf::Key::S:
            if(SETTINGS->spotLightActive())
            {
               if(Input.IsKeyDown(sf::Key::LShift))
                  SCENE->modifyActiveSpotLightCutoffAngle(1);
               else
                  SCENE->modifyActiveSpotLightCutoffAngle(-1);
            }
            break;

         case  sf::Key::L:
            SCENE->mShowLights = !SCENE->mShowLights;
            break;
         case sf::Key::C:
             SETTINGS->normalCheckForLightLookup = !SETTINGS->normalCheckForLightLookup;
             cout << "NormalCheck For SpotLookup: " << (SETTINGS->normalCheckForLightLookup ? "ON" : "OFF") << endl;
            break;
          case sf::Key::G:
            if(!SCENE->getSpotLights().empty())
               mShowGBuffer = !mShowGBuffer;
            break;

         case sf::Key::Y:
            mShowEnvMap = !mShowEnvMap;
            break;
         case sf::Key::E:
            if(Input.IsKeyDown(sf::Key::LShift))
            { // go to next instance of current element
               mInstanceIndexShowAtlas++;
               if(mInstanceIndexShowAtlas >= static_cast<int>(SCENE->getSceneElements().at(mElementIndexShowAtlas)->getNumInstances()))
                  mInstanceIndexShowAtlas = 0;
            }
            else // go to next element
            {
               mElementIndexShowAtlas++;
               if(mElementIndexShowAtlas >= static_cast<int>(SCENE->getSceneElements().size()))
                  mElementIndexShowAtlas = 0;
               if(mInstanceIndexShowAtlas >= static_cast<int>(SCENE->getSceneElements().at(mElementIndexShowAtlas)->getNumInstances()))
                  mInstanceIndexShowAtlas = 0;
            }
            break;

         default: break;
         }

         break;



      default: break;
      }


   }
   //if(camManipulated || lightManipulated || objectManipulated)
   //{

   //}

   if(objectManipulated)
   {
      SCENE->computeSceneBoundingBox();

      // updates the voxel camera frustum according to current scene bounding box
      if(SETTINGS->displayVoxelCamera())
      {
         updatePreVoxData();
         mPreVoxelization->voxelizeStaticSceneElements();
      }
   }

}


void SfmlView::closeEvent(QCloseEvent* evt)
{
	Q_UNUSED( evt );

	foreach (QWidget* widget, QApplication::topLevelWidgets())
	{
		widget->close();
	}

   QApplication::quit();
}


// SLOTS

void SfmlView::toggleTimerMonitor(bool show)
{
   qteTimerMonitor->setHidden(!show);
}

void SfmlView::forceTogglingAllObjAnimations()   { SCENE->toggleAllObjAnimations(); }
void SfmlView::forceSettingCameraPose(int index) { SCENE->setCurrentCameraPose(index); }
void SfmlView::forceSettingCameraFovH(double f)  { SCENE->getCamera()->setFovH(f); }
void SfmlView::forcePoseDeletion()               { SCENE->deleteCurrentCameraPose(); }
void SfmlView::forcePoseAdding()                 { SCENE->addCurrentCameraPose(); }

void SfmlView::forceInstanceAdding()
{
   SCENE->addInstanceToActiveDynamicElement();
   mAtlasRenderer->createAtlas(SCENE->getActiveDynamicElementIndex(), false);
   mSceneElementsChanged = true;
}

void SfmlView::forceILBufferSizeChange(int i)
{ 
   mIndirectLight->setBufferSize(static_cast<BufferSize>(i));
}

void SfmlView::forceRandomTextureUpdate()
{
   mIndirectLight->createRandom2DTexture(); 
} 


void SfmlView::forceSpotAdding()
{
   if(SETTINGS->spotLightActive())
   {
      SCENE->addSpotLight();
   }
}

void SfmlView::forceSpotDeletion()
{
   if(SETTINGS->spotLightActive())
   {
      SCENE->deleteCurrentSpotLight();
   }
}

void SfmlView::forceLightColorChange(float r, float g, float b, float scale)
{
   //if(SETTINGS->spotLightActive())
   {
      SCENE->getSpotLights().at(SCENE->getActiveSpotLightIndex())->setI(r, g, b);
      SCENE->getSpotLights().at(SCENE->getActiveSpotLightIndex())->scaleI(scale);
   }

}