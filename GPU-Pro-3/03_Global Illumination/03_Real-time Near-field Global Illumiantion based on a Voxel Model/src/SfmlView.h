///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#ifndef SFMLVIEW_H
#define SFMLVIEW_H

// STL
#include <iostream>
#include <string>

// Qt
#include <QApplication>
#include <QStatusBar>
#include <QTextEdit>

// SFML
#include "SFML/Window.hpp"

// OpenGL
#include "OpenGL.h"

// Qt
#include "Qt/QSfmlWidget.h"
#include "Qt/Settings.h"
#include "Qt/SceneXMLDocument.h"


#include "Scene/SceneDataStructs.h"

using namespace std;

class AtlasRenderer;
class AtlasVoxelization;
class Camera;
class Debugging;
class Filter;
class FPSCounter;
class IndirectLight;
class MipmapRenderer;
class PreVoxelization;
class PreVoxelizationData;
class ShaderProgram;
class SpotMapRenderer;
class TimerQuery;
class ToneMapping;
class VoxelVisualization;

///
/// A custom QSfmlWidget that overrides base class methods initialize(), refresh()  
///

class SfmlView : public QSfmlWidget
{
   Q_OBJECT


public:
   SfmlView(sf::ContextSettings contextSettings,
      SceneData* data = 0,
      int timerInterval = 0, QWidget* parent = 0,
      QTextEdit* timerMonitor = 0);

   void saveScreenshotNextFrame(string filename)
   { 
      mSaveScreenshot = true;
      mScreenshotFile = filename;
   }
   public slots:
      void forceRandomTextureUpdate(); 
      void forceTogglingAllObjAnimations();
      void forceSettingCameraPose(int index);
      void forceSettingCameraFovH(double f);
      void forcePoseDeletion();
      void forcePoseAdding();
      void forceInstanceAdding();
      void forceILBufferSizeChange(int i);
      void forceSpotAdding();
      void forceSpotDeletion();
      void forceLightColorChange(float r, float g, float b, float scale);

      void toggleTimerMonitor(bool show);

signals:
   void updatedFPS(float fps);
   void sceneLoaded();

protected:

   ///
   /// Reimplements QWidget::closeEvent, forces all 
   /// top-level widgets to be closed and the application to quit.
   /// closeEvent() is called when the user closes the widget
   /// (or when close() is called).
   ///
   void closeEvent(QCloseEvent* e);


private:

   void refresh();

   void initialize();
   
   void initQtConnections();
   void initScene();
   void preprocess();

   ///
   /// Initializations for OpenGL Rendering:
   /// setups camera, light, scene, etc.
   ///
   void initGL();

   ///
   /// Init glew so that the GLSL functionality will be available
   ///
   void initGLEW();

   ///
   /// Load, compile, link shader programs
   ///
   void initShader();


   ///
   /// Uses SFML polling system for getting events.
   /// We ask events to the window at each loop.
   /// Events are stored in a stack at each frame, 
   /// and getting an event pops the top of this stack
   ///
   void processSfmlEvents();

   ///
   /// Render a new frame with OpenGL.
   ///
   void render();

   /// 
   /// Query and output OpenGL rendering times.
   ///
   void queryTimes();

   ///
   /// Saves the current system framebuffer content (after render())
   /// to a given filename.
   ///
   void saveCurrentFrame();

   /// Updates the current voxel camera according to the scene bounding box
   void updatePreVoxData();


   /// Rendering methods
   void generateSpotMaps();
   void generateAtlasTextures();
   void generateVoxelizations();
   void userControlledVoxelization();
   void generateGBuffer();
   void displayAtlasTexture();
   void displayEnvMap();  ///< loaded environment map in latlong format
   void displayGBufferOrSpotMap();

   void rayCasting();
   void drawVoxelCubes();
   void drawVoxelCameraFrustum();

   void computeIndirectLight();
   void combineAndShowIndirectLightResult();

   // Member data
   FPSCounter* mFPSCounter;
  
   // Screenshot
   bool mSaveScreenshot;
   string mScreenshotFile;
   
   // Scene
   SceneData* const mSceneData; ///< data from loaded XML 
   bool mSceneElementsChanged;

   // Cameras
   Camera* mCurrentViewCamera; ///< points to current camera (scene) for event handling
   const Camera* currentVoxelCamera; ///< the active voxelization camera for user defined voxelization

   // Displaying options
   bool mShowGBuffer; ///< show gbuffer texture (true) or spot map (false)
   bool mShowEnvMap; ///< show an environment map if loaded
   bool mShowAtlas; ///< Display the atlas texture for one of the scene elements (default: first)

   // Mouse and Keyboard interaction
   int mLastPosX;
   int mLastPosY;
   const sf::Input& Input; ///< Abbrev. for GetInput() Method

   // Voxelization
   PreVoxelizationData* mPreVoxData; ///< holds current voxelization camera and voxel texture resolution
   PreVoxelization* mPreVoxelization;
   AtlasVoxelization* mVoxelizerAtlas;
   MipmapRenderer* mMipmapRenderer; ///< renders custom mipmaps for voxel texture

   GLuint mCurrentBinaryVoxelTexture;
   int mCurrentBinaryVoxelTextureWidth, mCurrentBinaryVoxelTextureHeight;

   // Atlas Rendering
   AtlasRenderer* mAtlasRenderer;
   int mElementIndexShowAtlas;
   int mInstanceIndexShowAtlas;
   
   // Indirect Light and Post Processing
   IndirectLight* mIndirectLight;
   Filter* mFilter;
   ToneMapping* mToneMapping;

   // Debugging visualization
   VoxelVisualization* mVoxelVisualizer; 

   // Timers
   TimerQuery* tqGBuffer;
   TimerQuery* tqVoxelization;
   TimerQuery* tqMipmap;
   TimerQuery* tqVGI;
   TimerQuery* tqPreProcess;
   TimerQuery* tqSpotMaps;
   TimerQuery* tqAtlasRendering;
   TimerQuery* tqFilter;
   double mAccumulatedTimeQuery;

   // Qt Stuff
   QTextEdit* const qteTimerMonitor;
   glm::ivec2 mUserClickPos; 
   // ---


   /// Animation related
   sf::Clock mRotationClock; // for automatically rotating the active model instance
 
};

#endif
