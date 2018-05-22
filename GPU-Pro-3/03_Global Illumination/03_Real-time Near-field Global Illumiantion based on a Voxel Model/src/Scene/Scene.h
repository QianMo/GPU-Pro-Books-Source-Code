#ifndef SCENE_H
#define SCENE_H

#include "OpenGL.h"
#include "Scene/SceneDataStructs.h"

#include <limits>
#include <vector>

using std::vector;

class Camera;
class Debugging;
class EnvMap;
class ObjModel;
class ObjectSequence;
class ObjectSequenceInstance;
class ShaderProgram;
class SpotLight;
class SpotMapRenderer;
class VoxelVisualization;

/// Class that defines the scene to be rendered.
/// Defines the spatial placement of the scene geometry elements.
class Scene
{
public:
   static Scene* Instance();

   void initialize(int windowWidth, int windowHeight);

   /// Returns the name of the scene
   string getName() { return mName; }

   /// Adds models, a camera and lights with given parameters.
   void load(const SceneData* const data);

   /// Computes axis-aligned bounding box for all static elements of the scene
   /// starts the precomputed voxelization of the static elements
   void postLoadProcessing();

   void setupActiveDynamicElementIndex();
   void setupActiveInstanceIndex();

   ///// User control and manipulation
   void setActiveInstance(int element, int instance);
   void gotoNextActiveSpotLight();
   
   void moveActiveInstance(glm::vec3& delta) { moveActiveInstance(delta.x, delta.y, delta.z); }
   void moveActiveInstance(float dx, float dy, float dz);
   void rotateActiveInstance(float angleX, float angleY, float angleZ);

   void moveActiveSpotLight(float dx, float dy, float dz);
   void modifyActiveSpotLightAngles(float angleXDelta, float angleYDelta, float angleZDelta);
   void modifyActiveSpotLightCutoffAngle(float angleDelta);

   /// Render scene to gbuffer fbo (normals, positions, direct light)
   void createGBuffer(bool withShadows);

   /// Deferred shading with given position buffer, normal buffer, material buffer
   /// information about lights and shadow maps
   void deferredShading(GLuint positionBuffer, GLuint normalBuffer, GLuint materialBuffer,
                            GLuint intermediateBuffer, GLuint fbo,
                            GLenum directLightAttachment, GLenum intermediateAttachment,
                            bool withShadows, bool lowQuality);

   void generateShadowAndSpotMaps(bool withShadows, bool withSpotMaps, bool renderPosNormMat);


   void setDrawToDirectLightBuffer();

   /// Compute bounding box of scene elements
   void computeSceneBoundingBox();

   /// Draws the (world) axis-aligned bounding box of the scene
   /// as a wireframe box
   void drawSceneBoundingBox();

   /// Draws the (world) axis-aligned bounding boxes of the dynamic elements of the scene
   /// as a wireframe box
   void drawDynamicBoundingBoxes();

   /// Draw all scene elements with their default draw mode.
   void drawAllElementsDefault() const;

   /// Draw all scene elements with a given draw mode
   /// (replaces default draw mode of all elements).
   /// \param mode The draw mode (GLM_FLAT etc)
   /// \param GeometryType Draw only static, only dynamic or all scene elements
   /// \param selectedLayer If selectedLayer >= 0, 
   ///        draw only objects whose bbox overlaps with given layer (t.i. voxel camera) index
   /// \return Number of rendered triangles
   int drawAllElementsWithMode(GLuint mode, GeometryType geom = ALL) const;
   /// The same method but for a single object
   int drawElementWithMode(int elementIndex, int instanceIndex, GLuint mode, GeometryType geom = ALL) const;


   /// Draw all scene elements with their default draw mode,
   /// but modify this draw mode by or-ing the given addMode
   /// and removing the removeMode with xor-ing.
   void drawAllElementsWithModeModifyingDefault(GLuint addMode, GLuint removeMode);
   /// The same method but for a single object
   void drawElementWithModeModifyingDefault(int elementIndex, int instanceIndex, GLuint addMode, GLuint removeMode);

   /// Draws all lights in this scene.
   void drawLights();

   void drawCamera();

   /// Marks the active model / instance with a green overlay.
   void drawHighlightedModel();

   /// Draws given spot light as a point and its direction as a line.
   void drawSpotLight(int index);

   void startAllObjAnimations();
   void stopAllObjAnimations();
   void toggleAllObjAnimations();
   void updateAllObjAnimations();

   void setCurrentCameraPose(int index);
   void addCurrentCameraPose();
   void deleteCurrentCameraPose();

   void addInstanceToActiveDynamicElement();
   void addSpotLight();
   void deleteCurrentSpotLight();

   // GETTER

   GLuint getBuffer(Buffer b) const {return mGBuffer.at(0).at(b); }
   GLuint getBuffer(Buffer b, int resolutionIndex) const {return mGBuffer.at(resolutionIndex).at(b); }

   const vector<ObjectSequence*>& getSceneElements() const;
   const vector<SpotLight*>& getSpotLights() const;
   unsigned int getNumSpotLights() const;
   EnvMap* getEnvMap() const { return mEnvMap; }
   
   int getActiveDynamicElementIndex() const { return mActiveDynamicElementIndex; }
   int getActiveSpotLightIndex() const { return mActiveSpotLightIndex; }
   ObjectSequenceInstance* const getActiveDynamicInstance();

   bool hasDynamicElements() const { return mHasDynamicElements; }

   const GLfloat* getSceneBoundingBox() const { return mSceneBoundingBox; }
   const glm::vec3 getSceneBoundingBoxDimension() const { return mSceneBoundingBoxDimension; }
   const glm::vec3 getSceneBoundingBoxCenter() const { return mSceneBoundingBoxCenter; }

   Camera* getCamera() const;
   const vector<Pose>& getCameraPoses() const { return mCameraPoses; }
   int getCurrentCameraPoseIndex() const { return mCurrentCameraPoseIndex; }
   
   bool areAnimationsRunning() const;

   SpotMapRenderer* getSpotMapRenderer() const { return mSpotMapRenderer; }

   int getWindowWidth() { return mWindowWidth; }
   int getWindowHeight() { return mWindowHeight; }

   // PUBLIC MEMBER VARIABLE
   bool mShowLights;


private:
   Scene();

   /// MEMBER METHODS

   void createShader();
   void createFBO(); ///< for Gbuffer

   void computeJitteredShadowSamples();

   // loading
   void addStaticObject(const StaticElementData& elem); // add to sceneElements
   void addSingleFrameObject(const DynamicElementData& elem);
   void addSequence(const DynamicElementData& elem);


   /// MEMBER DATA

   static Scene* mInstance; ///< Singleton instance
   string mName; ///< scene name 

   /// Scene elements: geometry
   vector<ObjectSequence*> mSceneElements; /// one object is be a sequence of animation frames
                                           /// and it may have multiple instances
   bool mHasDynamicElements;        /// does this scene contain dynamic objects
   int mActiveDynamicElementIndex;  /// which dynamic scene element is active (for user manipulation)
   int mActiveInstanceIndex;        /// which instance of the active dynamic scene element is active
   
   /// Lights
   vector<SpotLight*> mSpotLights; ///< Scene may have several spot lights.
   GLUquadricObj* mSpotLightQuadric; // spot light cone 
   int mActiveSpotLightIndex;
   /// Environment map for directional occlusion
   EnvMap* mEnvMap;

   /// Camera
   Camera* mCamera; ///< The user controllable scene camera.
   vector<Pose> mCameraPoses; ///< All saved scene camera poses.
   int mCurrentCameraPoseIndex;
   
   /// Scene bounding box
   GLfloat mSceneBoundingBox[6]; //  bounding box of statics parts of the loaded scene in world coordinates
   glm::vec3 mSceneBoundingBoxDimension;
   glm::vec3 mSceneBoundingBoxCenter;

   // Shader for rendering the scene to the gbuffer
   ShaderProgram* pGBuffer; ///< renders normals, positions, direct lighting to gBuffer FBO
   ShaderProgram* pSpotShadow; ///< spot light shadow mapping
   ShaderProgram* pSpotLighting; ///< computes illuminance for a spot light
   SpotMapRenderer* mSpotMapRenderer; ///< Renders shadow and spot maps (RSM) for the scene

   // Shadow Map resolution
   int mShadowMapResolution;
   GLuint mRandShadowSamplesTex;

   // Render Window Resolution
   int mWindowWidth; ///< OpenGL window width
   int mWindowHeight; ///< OpenGL window height

   /// Frame buffer objects and textures
   GLuint fboGBuffer;
   GLuint fboLowRes;
   GLuint mDepthBuffer; ///< for depth test
   vector<vector<GLuint> > mGBuffer; ///< contains G Buffer textures

   GLuint mIntermediateBuffer; ///< for direct illumination accumulation

   /// Animation related
   bool mHasAnimatedElements; ///< Indicates whether vector sceneElements contains obj-animations.
   bool mAnimationsRunning; ///< Indicates whether the animations have been started for animated scene elements.
};

#endif
