///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#ifndef SETTINGS_H
#define SETTINGS_H

#include <QObject>

#include <iostream>
#include <glm/glm.hpp>

using namespace std;

class ObjModel;

/// Singleton that stores all parameters needed by simulation and
/// visualization. All GUI elements are connected to its instance.
class Settings : public QObject
{
   Q_OBJECT

public:

   static Settings* Instance();

   /// Getter for querying current states
   //@{

   // Scene
   int getCurrentGBufferTex() const     { return mCurrentGBufferTex; }
   bool shadowMappingEnabled() const     { return mShadowMappingEnabled; }
   bool renderingEnabled() const      { return mRenderingEnabled; }
   bool highlightModel() const      { return mHighlightModel; }

   // Post Process
   float getGammaExponent() const       { return mGammaExponent; }
   bool toneMappingEnabled() const    { return mToneMappingEnabled; }
   bool linearToneMappingEnabled() const { return mToneMappingEnabled && mLinearToneMappingEnabled; }
   bool logToneMappingEnabled() const    { return mToneMappingEnabled && mLogToneMappingEnabled; }
   float getSimpleMaxRadiance() const        { return mSimpleMaxRadiance; }

   // Animation
   bool autoRotateModel() const     { return mAutoRotateModel; }
   bool autoRotateOnXAxis() const   { return mAutoRotateXAxis; }
   bool autoRotateOnYAxis() const   { return mAutoRotateYAxis; }
   bool autoRotateOnZAxis() const   { return mAutoRotateZAxis; }

   // Filter
   bool filterEnabled() const        { return mFilterEnabled; }
   int getFilterRadius() const           { return mFilterRadius; }
   int getFilterIterationRadius() const  { return mFilterIterationRadius; }
   float getFilterDistanceLimit () const { return mFilterDistanceLimit; }
   float getFilterNormalLimit() const    { return mFilterNormalLimit; }
   float getFilterNormalLimitAngle() const { return mFilterNormalLimitAngle; }
   float getFilterMaterialLimit() const  { return mFilterMaterialLimit; }
   int getFilterIterations() const       { return mFilterIterations; }
   int getRandomPatternSize() const      { return mRandomPatternSize; }

   bool surfaceDetailEnabled() const { return mSurfaceDetailEnabled; }
   float getSurfaceDetailAlpha() const { return mSurfaceDetailAlpha; }

   // Debugging
   bool displayVoxelCamera() const    { return mDisplayVoxelCamera; }
   bool displayWorldSpaceAxes() const { return mDisplayWorldSpaceAxes; }
   bool displayBoundingBoxes() const  { return mDisplayBoundingBoxes; }

   // Voxelization
   bool voxelizationEnabled() const            { return mVoxelizationEnabled; }
   bool atlasBinaryVoxelizationEnabled() const { return mVoxelizationEnabled && mAtlasBinaryVoxelizationEnabled; }
   bool mipmappingEnabled() const              { return mVoxelizationEnabled && mMipmappingEnabled; }
   bool preVoxelizationEnabled() const         { return mVoxelizationEnabled && mPreVoxelizationEnabled; }
   int  getMipmapLevel() const                 { return mMipmapLevel; }
   int getVoxelTextureResolution() const       { return mVoxelTextureResolution; }

   // Voxel Visualization
   bool voxelVisualizationEnabled() const  { return mVoxelizationEnabled && (mBinaryRayCastingEnabled || mVoxelCubesEnabled);}
   bool binaryRayCastingEnabled() const    { return mBinaryRayCastingEnabled && mVoxelizationEnabled ; }
   bool voxelCubesEnabled() const { return mVoxelCubesEnabled && mVoxelizationEnabled; } 
   int  getMipmapTestCycleIndex() const { return mMipmapTestCycleIndex; }
   bool mipmapTestVisualizationEnabled() const { return mVoxelizationEnabled && mMipmapTestVisualizationEnabled; }
   bool mustDisplayVoxelCubesDuringMipmapTest() const { return mDisplayVoxelCubesDuringMipmapTest; }
   const glm::vec3& getStartPoint() const { return mStartPoint; }
   const glm::vec3& getEndPoint() const { return mEndPoint; }

   // Indirect Light 
   bool indirectLightEnabled() const { return mIndirectLightEnabled; }
   bool VGIEnabled() const       { return mIndirectLightEnabled && mVGIEnabled; }
   bool ambientTermEnabled()   const { return mIndirectLightEnabled && mAmbientTermEnabled; }
   bool indirectLightCombinationEnabled() const { return mIndirectLightEnabled && mIndirectLightCombinationEnabled; }
   bool indirectLight_E_ind_Enabled() const     { return mIndirectLightEnabled && mIndirectLight_E_ind_Enabled; }
   bool indirectLight_L_ind_Enabled() const     { return mIndirectLightEnabled && mIndirectLight_L_ind_Enabled; }
   
   int getCurrentILBufferSize() const  { return mCurrentILBufferSize; }
   int getNumRays() const              { return mNumRays; }
   int getNumSteps() const             { return mNumSteps; }
   bool useRandomRays() const          { return mUseRandomRays; }
   float getRadius() const             { return mRadius; }
   float getSpread() const             { return mSpread; }
   float getIndirectLightScaleFactor() const { return mIndirectLightScaleFactor; }
   float getDirectLightScaleFactor() const   { return mDirectLightScaleFactor; }
   float getDistanceThresholdScale() const   { return mDistanceThresholdScale; }
   float getVoxelOffsetCosThetaScale() const { return mVoxelOffsetCosThetaScale; }
   float getVoxelOffsetNormalScale() const   { return mVoxelOffsetNormalScale; }

   float getOcclusionStrength() const    { return mOcclusionStrength; }
   float getEnvMapBrightness() const { return mEnvMapBrightness; }

   // Interaction (mouse)
   bool cameraActive() const    { return mCameraActive; }
   bool modelActive() const     { return mModelActive; }
   bool spotLightActive() const { return mSpotLightActive; }

   float getShadowEpsilon() const { return mShadowEpsilon; }

   bool normalCheckForLightLookup;

   // GUI elements send values to these slots
   public slots:
      
      void setCameraPose(int index) { emit(currentCameraPoseChanged(index)); }
      void setCameraFovH(double value) { emit(cameraFovHChanged(value)); }
      void forwardPoseDeletionRequest() { emit(forwardedPoseDeletionRequest()); }
      void forwardPoseAddingRequest() { emit(forwardedPoseAddingRequest()); }
      void forwardInstanceAddingRequest() { emit(forwardedInstanceAddingRequest()); }
      void forwardLightColorChangeRequest(float r, float g, float b, float scale)
      { emit(forwardedLightColorChangeRequest(r, g, b, scale)); }

      // Voxelization
      void toggleVoxelizationEnabled(bool checked)            { mVoxelizationEnabled = checked; }
      void toggleAtlasBinaryVoxelizationEnabled(bool checked) { mAtlasBinaryVoxelizationEnabled = checked; }
      
      void setMipmapLevel(int level)              { mMipmapLevel = level;  }
      void toggleMipmapping(bool checked)         { mMipmappingEnabled = checked; }
      void togglePreVoxelization(bool checked)    { mPreVoxelizationEnabled = checked; }
      void setVoxelTextureResolution(int res)     { mVoxelTextureResolution = res; emit(voxelTextureResolutionChanged(res)); }
      
      // Voxel Visualization
      void toggleBinaryRayCasting(bool checked)    { mBinaryRayCastingEnabled = checked; }
      void toggleVoxelCubes(bool checked)          { mVoxelCubesEnabled = checked; } 
      void setMipmapTestCycleIndex(int index)   { mMipmapTestCycleIndex = index; }
      void toggleShowVoxelCubesDuringMipmapTest( bool c) { mDisplayVoxelCubesDuringMipmapTest = c; }
      void toggleMipmapTestVisualization(bool c)   { mMipmapTestVisualizationEnabled = c; }
      void setStartPointX(double x) { mStartPoint[0] = x; }
      void setStartPointY(double y) { mStartPoint[1] = y; }
      void setStartPointZ(double z) { mStartPoint[2] = z; }
      void setEndPointX(double x) { mEndPoint[0] = x; }
      void setEndPointY(double y) { mEndPoint[1] = y; }
      void setEndPointZ(double z) { mEndPoint[2] = z; }

      // Scene
      void setCurrentGBufferTex(int index) { mCurrentGBufferTex = index; }
      void toggleShadowMapping(bool checked) { mShadowMappingEnabled = checked; }
      void toggleHighlightModel(bool checked)  { mHighlightModel = checked; }
      void toggleAutoRotateModel(bool checked) { mAutoRotateModel = checked; }
      void toggleAutoRotateXAxis(bool checked) { mAutoRotateXAxis = checked; }
      void toggleAutoRotateYAxis(bool checked) { mAutoRotateYAxis = checked; }
      void toggleAutoRotateZAxis(bool checked) { mAutoRotateZAxis = checked; }
      void addSpotLight() { emit(forwardedSpotAddingRequest()); }
      void deleteCurrentSpotLight() { emit(forwardedSpotDeletionRequest()); }
      void toggleRenderingEnabled(bool checked) { mRenderingEnabled = checked; }
      void setShadowEpsilon(double value) { mShadowEpsilon = static_cast<float>(value); }
      
      // Animation
      void toggleAllObjAnimations()    { emit(toggledAllObjAnimations()); }

      // Filter
      void toggleFilterEnabled(bool checked)     { mFilterEnabled = checked; emit(filterToggled(checked)); }
      void setFilterRadius(int value)           { mFilterRadius = value; emit(filterRadiusChanged(value)); }
      void setFilterIterations(int value)       { mFilterIterations = value; emit(filterIterationsChanged(value)); }
      void setFilterIterationRadius(int value)  { mFilterIterationRadius = value; emit(filterIterationRadiusChanged(value)); }
      void setFilterDistanceLimit(double value) { mFilterDistanceLimit = float(value); emit(filterDistanceLimitChanged(value)); }
      void setFilterNormalLimit(double value)   { mFilterNormalLimitAngle = float(value);
                                                  mFilterNormalLimit = float(cos((90.0 - value)/180.0*3.14159)); 
                                                  emit(filterNormalLimitChanged(value)); }
      void setFilterMaterialLimit(double value) { mFilterMaterialLimit = float(value); emit(filterMaterialLimitChanged(value)); }
      void toggleSurfaceDetailEnabled(bool c)   { mSurfaceDetailEnabled = c; emit(surfaceDetailToggled(c));}
      void setSurfaceDetailAlpha(double value)  { mSurfaceDetailAlpha = float(value); emit(surfaceDetailAlphaChanged(value));}
      void setRandomPatternSize(int value)           { mRandomPatternSize = value; emit(randomTextureSizeChanged(value)); }
     
      // Post Process
      void setGammaExponent(double value)         { mGammaExponent = float(value); }
      void toggleToneMappingEnabled(bool checked) { mToneMappingEnabled = checked; emit(toneMappingToggled(checked)); }
      void toggleLinearToneMappingEnabled(bool c)  { mLinearToneMappingEnabled = c; emit(linearToneMappingToggled(c)); }
      void toggleLogToneMappingEnabled(bool c)     { mLogToneMappingEnabled = c; emit(logToneMappingToggled(c)); }
       void setSimpleMaxRadiance(double value)     { mSimpleMaxRadiance = static_cast<float>(value); emit(simpleMaxRadianceChanged(value)); }

      // Debugging
      void toggleDisplayVoxelCamera(bool checked)    { mDisplayVoxelCamera = checked; }
      void toggleDisplayWorldSpaceAxes(bool checked) { mDisplayWorldSpaceAxes = checked; }
      void toggleDisplayBoundingBoxes(bool on)       { mDisplayBoundingBoxes = on; }

      // Indirect Light
      void toggleIndirectLightEnabled( bool checked) { mIndirectLightEnabled = checked; }
      void toggleAmbientTermEnabled(bool checked)    { mAmbientTermEnabled = checked; }
      void toggleIndirectLightCombinationEnabled( bool c)     { mIndirectLightCombinationEnabled = c; emit(toggledIndirectLightCombination(c)); }
      void toggleIndirectLight_E_ind_Enabled(bool c)          { mIndirectLight_E_ind_Enabled = c; emit(toggledIndirectLight_E_ind(c)); }
      void toggleIndirectLight_L_ind_Enabled(bool c)          { mIndirectLight_L_ind_Enabled = c; emit(toggledIndirectLight_L_ind(c)); }
      void toggleVGIEnabled(bool c)                           { mVGIEnabled = c; }  

      // IL Parameters
      void setCurrentILBufferSize(int index)         { mCurrentILBufferSize = index; emit(changedILBufferSize(index));}
      void setNumRays(int value)                     { if(value == mNumRays) return; mNumRays = value; emit(numRaysChanged(value)); }
      void setNumSteps(int value)                    { if(value == mNumSteps) return; mNumSteps = value; emit(numStepsChanged(value));}
      void setRadius(double value)                   { mRadius = static_cast<float>(value); emit(radiusChanged(value)); }
      void setSpread(double value)                   { mSpread = static_cast<float>(value); emit(spreadChanged(value)); }
      void setIndirectLightScaleFactor(double value) { mIndirectLightScaleFactor = static_cast<float>(value); emit(indirectLightScaleFactorChanged(value)); }
      void setDirectLightScaleFactor(double value)   { mDirectLightScaleFactor = static_cast<float>(value); emit(directLightScaleFactorChanged(value));}
      void setDistanceThresholdScale(double value)   { mDistanceThresholdScale = static_cast<float>(value); emit(distanceThresholdScaleChanged(value));}
      void setVoxelOffsetCosThetaScale(double value) { mVoxelOffsetCosThetaScale = static_cast<float>(value); emit(voxelOffsetCosThetaScaleChanged(value));}
      void setVoxelOffsetNormalScale(double value)   { mVoxelOffsetNormalScale = static_cast<float>(value); emit(voxelOffsetNormalScaleChanged(value));}

      void toggleRandomRays(bool c)                  { mUseRandomRays = c; }

      void setOcclusionStrength(double value)        { mOcclusionStrength = static_cast<float>(value); emit(occlusionStrengthChanged(value));}
      void setEnvMapBrightness(double value)         { mEnvMapBrightness = static_cast<float>(value); emit(envMapBrightnessChanged(value));}

      // Interaction (Mouse)
      void toggleCameraActive(bool checked) { mCameraActive = checked; }
      void toggleModelActive(bool checked)  { mModelActive = checked; }
      void toggleSpotLightActive(bool checked) { mSpotLightActive = checked; }
      
signals:

      // for GUI update
      void numRaysChanged(int value);
      void numStepsChanged(int value);
      void radiusChanged(double value);
      void spreadChanged(double value);

      void indirectLightScaleFactorChanged(double value);
      void directLightScaleFactorChanged(double value);
      void envMapBrightnessChanged(double value);
      void occlusionStrengthChanged(double value);

      void distanceThresholdScaleChanged(double value);
      void voxelOffsetNormalScaleChanged(double value);
      void voxelOffsetCosThetaScaleChanged(double value);
      void voxelTextureResolutionChanged(int res);

      void filterToggled(bool enabled);
      void filterRadiusChanged(int value);
      void filterIterationsChanged(int value);
      void filterIterationRadiusChanged(int value);
      void filterDistanceLimitChanged(double value);
      void filterNormalLimitChanged(double value);
      void filterMaterialLimitChanged(double value);
      void surfaceDetailToggled(bool enabled);
      void surfaceDetailAlphaChanged(double value);

      void toggledIndirectLightCombination(bool enabled);
      void toggledIndirectLight_E_ind(bool enabled);
      void toggledIndirectLight_L_ind(bool enabled);

      void toneMappingToggled(bool enabled);
      void linearToneMappingToggled(bool enabled);
      void logToneMappingToggled(bool enabled);
      void simpleMaxRadianceChanged(double value);

      void randomTextureSizeChanged(int value = 0);
      void changedILBufferSize(int index);

      void toggledAllObjAnimations();
      void currentCameraPoseChanged(int index);
      void cameraFovHChanged(double fovH);
      void forwardedPoseDeletionRequest();
      void forwardedPoseAddingRequest();
      void forwardedSpotDeletionRequest();
      void forwardedSpotAddingRequest();
      void forwardedInstanceAddingRequest();
      void forwardedLightColorChangeRequest(float r, float g, float b, float scale);

private:

   /// Constructor initializes all values.
   Settings();
   static Settings* mInstance;

   // Parameter values //

   // Scene
   int mCurrentGBufferTex; ///< Which of the GBuffer Textures should be displayed?
   bool mShadowMappingEnabled; ///< Indicates whether the direct lighting uses a shadow map for hard shadows.
   bool mRenderingEnabled; ///< Enabled or disables rendering completely
   float mShadowEpsilon;

   // Filter
   bool mFilterEnabled;
   int mFilterRadius;
   int mFilterIterationRadius;
   float mFilterDistanceLimit;
   float mFilterNormalLimit;
   float mFilterNormalLimitAngle;
   float mFilterMaterialLimit;
   int mFilterIterations;
   float mSurfaceDetailAlpha;
   bool mSurfaceDetailEnabled;

   // random texture
   int mRandomPatternSize;

   // Debugging
   bool mDisplayVoxelCamera; ///< Indicates whether the frustum of a voxelizing ortho cam should be drawn
   bool mDisplayWorldSpaceAxes; ///< Indicates whether the three world space coordinate axes should be drawn
   bool mDisplayBoundingBoxes; ///< Static and dynamic bounding boxes
   bool mHighlightModel; ///< is current user movable model / instance highlighted 
   bool mAutoRotateModel; ///< is automatic rotation of the current selected instance enabled?
   bool mAutoRotateXAxis, mAutoRotateYAxis, mAutoRotateZAxis;

   // Voxel Visualization
   bool mBinaryRayCastingEnabled; ///< Indicates whether voxel visualization via ray casting should be done
   bool mVoxelCubesEnabled; ///< Indicates whether voxel visualization via cube drawing should be done
   bool mMipmapTestVisualizationEnabled;
   int mMipmapTestCycleIndex; ///< visualization of the mipmap-test for this for-loop's cycle
   bool mDisplayVoxelCubesDuringMipmapTest; ///< show the voxel cubes at level 0 during the mipmap test visualization
   glm::vec3 mStartPoint;
   glm::vec3 mEndPoint;
  
   // Voxelization
   // Parameters for visualization of voxels
   bool mVoxelizationEnabled;
   bool mAtlasBinaryVoxelizationEnabled;
   int  mMipmapLevel;
   bool mMipmappingEnabled;
   bool mPreVoxelizationEnabled;
   int mVoxelTextureResolution;

   // Interaction
   bool mCameraActive;  
   bool mModelActive; 
   bool mSpotLightActive; 

   // Indirect Light
   int mCurrentILBufferSize; // FULL / HALF / QUARTER
   bool mIndirectLightEnabled;
   bool mAmbientTermEnabled;
   bool mVGIEnabled;

   bool mIndirectLightCombinationEnabled;
   bool mIndirectLight_E_ind_Enabled;
   bool mIndirectLight_L_ind_Enabled;
   
   // IL Paramters

   int mNumRays;
   int mNumSteps;
   float mRadius;
   float mSpread;

   bool mUseRandomRays;

   // Scales brightness
   float mIndirectLightScaleFactor;
   float mDirectLightScaleFactor;

   float mOcclusionStrength;
   float mEnvMapBrightness;

   // Offsets for voxel intersection 
   float mDistanceThresholdScale;
   float mVoxelOffsetCosThetaScale;
   float mVoxelOffsetNormalScale;

   // Post process (tone mapping, gamma..)
   bool mToneMappingEnabled;
   bool mLinearToneMappingEnabled;
   bool mLogToneMappingEnabled;

   float mSimpleMaxRadiance;
   float mGammaExponent; ///< for Gamma Correction

};


#endif
