///////////////////////////////////////////////////////////////////////
// Real-time Near-Field Global Illumination based on a Voxel Model
// GPU Pro 3
///////////////////////////////////////////////////////////////////////
// Authors : Sinje Thiedemann, Niklas Henrich, 
//           Thorsten Grosch, Stefan Mueller
// Created:  August 2011
///////////////////////////////////////////////////////////////////////

#include "Settings.h"

#include "Scene/ObjModel.h"

Settings* Settings::mInstance = 0;

Settings::Settings()
{
   // Scene
   mCurrentGBufferTex = 4; // Combined
   mShadowMappingEnabled = true;
   mHighlightModel = false;
   mAutoRotateModel = false;
   mAutoRotateXAxis = false; mAutoRotateYAxis = false; mAutoRotateZAxis = false;
   mRenderingEnabled = true;

   // Filter
   mFilterEnabled = false;
   mFilterRadius = 8;
   mFilterIterationRadius = 4;
   mFilterDistanceLimit = 0.8f;
   mFilterIterations = 2;
   mRandomPatternSize  = 16;
   mSurfaceDetailAlpha = 0.4f;
   mSurfaceDetailEnabled = false;

   // Post Process
   mGammaExponent = 2.2f;
   mToneMappingEnabled = false;
   mLinearToneMappingEnabled = false;
   mLogToneMappingEnabled = false;
   mSimpleMaxRadiance = 2.0;

   // Debugging
   mDisplayWorldSpaceAxes = false;
   mDisplayBoundingBoxes = false;
   mDisplayVoxelCamera = false;

   // Voxelization
   mVoxelizationEnabled = false;
   mAtlasBinaryVoxelizationEnabled = false;
   mMipmappingEnabled = false;
   mPreVoxelizationEnabled = false;
   mMipmapLevel = 0;
   mVoxelTextureResolution = 128;

   // Voxel Visualization
   normalCheckForLightLookup = true;

   mBinaryRayCastingEnabled = false;
   mVoxelCubesEnabled = false;
   mMipmapTestCycleIndex = 0;
   mDisplayVoxelCubesDuringMipmapTest = false;
   mMipmapTestVisualizationEnabled = false;

   // Interaction
   mCameraActive = true;
   mModelActive = false; 
   mSpotLightActive = false;


   // Indirect Light
   mIndirectLightEnabled = true;
   mCurrentILBufferSize = 0;
   mUseRandomRays = true;
   
   mAmbientTermEnabled = false;
   mVGIEnabled = false;

   mIndirectLightCombinationEnabled = true;
   mIndirectLight_E_ind_Enabled = false;
   mIndirectLight_L_ind_Enabled = false;

   mNumRays = 8;
   mNumSteps = 42;
   mRadius = 4.0;
   mSpread = 1.0;

   mOcclusionStrength = 1.0f;
   mEnvMapBrightness = 1.0f;

   mIndirectLightScaleFactor = 2.0;
   mDirectLightScaleFactor = 1.0;

   mDistanceThresholdScale = 1.0;
   mVoxelOffsetCosThetaScale = 0.6f;
   mVoxelOffsetNormalScale = 0.8f;

   mShadowEpsilon = 0.0001f; // Shadow mapping offset
}


Settings* Settings::Instance()
{
	if(mInstance == 0)
	{
		mInstance = new Settings();
	}
	return mInstance;
}

