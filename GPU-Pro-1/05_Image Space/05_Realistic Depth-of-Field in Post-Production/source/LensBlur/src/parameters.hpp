/* ******************************************************************************
* Description: Application parameters including DOF and GPU settings.
*
*  Version 1.0.0
*  Date: Sep 19, 2009
*  Author: David Illes, Peter Horvath
*          www.postpipe.hu
*
* GPUPro
***************************************************************************** */

#ifndef _PARAMETERS_
#define _PARAMETERS_

#include "settings.hpp"

#define MAX_PIXEL 255.0f

//==============================================================================
class Parameters {
//==============================================================================
public:
	int width, height;
	int size;

	// mode
	int mode;

	// focus
	int focus;
	// z settings
	int zEpsilon;
	int samplingRadius;
	float overlap;

	// Artist Mode
	float strength;			// percentage value
	
	// Physically-based Mode
	float fLength;
	float ms;
	float fStop;
	float distance;
	float cameraDistance;

	float physicsParam;

	// bloom
	int threshold;
	int bloomAmount;

	bool interpolation;

	// gpu
	int gpuBlocks;
	int gpuThreads;
								
public:

//-----------------------------------------------------------------
// Summary: Constructs the parameters class with default values.
//-----------------------------------------------------------------
Parameters() {
//-----------------------------------------------------------------
	mode = MODE_DEFAULT;

	focus = FOCALPOINT_DEFAULT;

	setEpsilon(EPSILON_DEFAULT);
	samplingRadius = SAMPLING_DEFAULT;
	overlap = OVERLAP_DEFAULT;
	setStrength((float)BLUR_DEFAULT / 50.0f * 100.0f);

	fLength = FLENGTH_DEFAULT;
	ms = MS_DEFAULT;
	fStop = FSTOP_DEFAULT;
	distance = DISTANCE_DEFAULT;
	cameraDistance = CAMERA_DISTANCE_DEFAULT;
	computePhysicsParam();

	threshold = BLOOMTHRESHOLD_DEFAULT;
	bloomAmount = BLOOMAMOUNT_DEFAULT;

	setInterpolation(INTERPOLATION_DEFAULT);

	gpuBlocks = NUM_BLOCKS;
	gpuThreads = NUM_THREADS;
}

//-----------------------------------------------------------------
// Summary: Calculates coefficient from the lens parameters used in physics mode.
//-----------------------------------------------------------------
void computePhysicsParam() {
//-----------------------------------------------------------------
	float N = pow(2,fStop/2);
	physicsParam = fLength*ms/N;
}

//-----------------------------------------------------------------
// Summary: Calculates the blur size (in mm) from the physical parameters.
// Arguments: zValue - z value (0 - 255)
// Returns: blur length.
//-----------------------------------------------------------------
float getPhysicalStrength(int zValue) {
//-----------------------------------------------------------------
	// focal point distance
	//float s = cameraDistance + (1.0f - (float)focusValue / maxPixelValue) * distance;
	// subject distance from the camera
	float s = cameraDistance + (1.0f - (float)zValue / MAX_PIXEL) * distance;
	// subject distance from the focal point
	float xd = abs(focus - zValue) / MAX_PIXEL * distance;
	//float d = zValue < focusValue ? xd / (s + xd) : xd / (s - xd);
	float d = xd / s;

	return physicsParam * d;
}

//-----------------------------------------------------------------
// Returns: Image width.
//-----------------------------------------------------------------
int getWidth() {
//-----------------------------------------------------------------
	return width;
}

//-----------------------------------------------------------------
// Returns: Image height.
//-----------------------------------------------------------------
int getHeight() {
//-----------------------------------------------------------------
	return height;
}

//-----------------------------------------------------------------
// Returns: DOF calculation mode.
//-----------------------------------------------------------------
int getMode() {
//-----------------------------------------------------------------
	return mode;
}

//-----------------------------------------------------------------
// Summary: Sets DOF calculation mode.
//-----------------------------------------------------------------
void setMode(int m) {
//-----------------------------------------------------------------
	mode = m;
}

//-----------------------------------------------------------------
// Returns: true if the kernel interpolation is enabled
//          false otherwise
//-----------------------------------------------------------------
bool isInterpolation() {
//-----------------------------------------------------------------
	return interpolation;
}

//-----------------------------------------------------------------
// Summary: Enable / disable kernel interpolation.
//-----------------------------------------------------------------
void setInterpolation(int interpolation) {
//-----------------------------------------------------------------
	this->interpolation = interpolation != 0;
}

//-----------------------------------------------------------------
// Returns: the blur strength scaled to 0 - 50 range.
//-----------------------------------------------------------------
inline int getStrengthAsControl()
//-----------------------------------------------------------------
{
	return (int)(50.0f * strength);
}

//-----------------------------------------------------------------
// Summary: sets the blur strength (percentage value).
//-----------------------------------------------------------------
inline void setStrength(float value)
//-----------------------------------------------------------------
{
	strength = value / 100.0f;
	if (strength < 0.0f) {
		strength = 0.0f;
	} else if (1.0f < strength) {
		strength = 1.0f;
	}
}

//-----------------------------------------------------------------
// Returns: f length.
//-----------------------------------------------------------------
inline float getFLength()
//-----------------------------------------------------------------
{
	return fLength;
}

//-----------------------------------------------------------------
// Summary: sets f length lens parameter.
//-----------------------------------------------------------------
inline void setFLength(float value)
//-----------------------------------------------------------------
{
	fLength = value;
	computePhysicsParam();
}

//-----------------------------------------------------------------
// Returns: f stop.
//-----------------------------------------------------------------
inline int getFStop()
//-----------------------------------------------------------------
{
	return (int)fStop;
}

//-----------------------------------------------------------------
// Summary: sets f length lens parameter.
//-----------------------------------------------------------------
inline void setFStop(float value)
//-----------------------------------------------------------------
{
	fStop = value;
	computePhysicsParam();
}

//-----------------------------------------------------------------
// Returns: ms.
//-----------------------------------------------------------------
inline float getMS()
//-----------------------------------------------------------------
{
	return ms;
}

//-----------------------------------------------------------------
// Summary: sets ms lens parameter.
//-----------------------------------------------------------------
inline void setMS(float value)
//-----------------------------------------------------------------
{
	ms = value;
	computePhysicsParam();
}

//-----------------------------------------------------------------
// Returns: image range (in meter).
//-----------------------------------------------------------------
inline float getDistance()
//-----------------------------------------------------------------
{
	return distance;
}

//-----------------------------------------------------------------
// Summary: sets image range.
//-----------------------------------------------------------------
inline void setDistance(float value)
//-----------------------------------------------------------------
{
	distance = value;
}

//-----------------------------------------------------------------
// Returns: distance of the camera from the front clip of the image (in meter).
//-----------------------------------------------------------------
inline float getCameraDistance()
//-----------------------------------------------------------------
{
	return cameraDistance;
}

//-----------------------------------------------------------------
// Summary: camera - front clip distance.
//-----------------------------------------------------------------
inline void setCameraDistance(float value)
//-----------------------------------------------------------------
{
	cameraDistance = value;
}

//-----------------------------------------------------------------
// Returns: focal point depth (0 - 255)
//-----------------------------------------------------------------
int getFocus() {
//-----------------------------------------------------------------
	return focus;
}

//-----------------------------------------------------------------
// Summary: reads the z value from the given pixel and sets as the focal point.
//-----------------------------------------------------------------
void setFocus(Image& zMap,int x,int y) {
//-----------------------------------------------------------------
	int pos = x+y*width;
	focus = (int)zMap.pixels[pos].r;
}

//-----------------------------------------------------------------
// Summary: sets the focal point.
//-----------------------------------------------------------------
void setFocus(int value) {
//-----------------------------------------------------------------
	focus = value;
}

//-----------------------------------------------------------------
// Returns: bloom threshold.
//-----------------------------------------------------------------
int getThreshold() {
//-----------------------------------------------------------------
	return threshold;
}

//-----------------------------------------------------------------
// Summary: sets bloom threshold.
//-----------------------------------------------------------------
void setThreshold(int th) {
//-----------------------------------------------------------------
	threshold = th;
}

//-----------------------------------------------------------------
// Returns: bloom amount.
//-----------------------------------------------------------------
int getBloomAmount() {
//-----------------------------------------------------------------
	return bloomAmount;
}

//-----------------------------------------------------------------
// Summary: sets bloom amount.
//-----------------------------------------------------------------
void setBloomAmount(int ba) {
//-----------------------------------------------------------------
	bloomAmount = ba;
}

//-----------------------------------------------------------------
// Returns: z epsilon parameter used in edge detection.
//          When the difference of two neighboring pixels' z value
//          is above the epsilon limit they belong to different objects.
//-----------------------------------------------------------------
int getEpsilon() {
//-----------------------------------------------------------------
	return zEpsilon;
}

//-----------------------------------------------------------------
// Returns: z epsilon value in 0.0 - 1.0 range.
//-----------------------------------------------------------------
float getEpsilonScale() {
//-----------------------------------------------------------------
	return (float)zEpsilon / MAX_PIXEL;
}

//-----------------------------------------------------------------
// Summary: sets the z epsilon.
//-----------------------------------------------------------------
void setEpsilon(float eps) {
//-----------------------------------------------------------------
	zEpsilon = (int)(eps * MAX_PIXEL);
}

//-----------------------------------------------------------------
// Returns: edge overlap (0.0 - 1.0). This parameter defines how to blend
//          the accumulated and reaccumulated colors.
//-----------------------------------------------------------------
float getOverlap() {
//-----------------------------------------------------------------
	return overlap;
}

//-----------------------------------------------------------------
// Summary: sets the edge overlap.
//-----------------------------------------------------------------
void setOverlap(float o) {
//-----------------------------------------------------------------
	overlap = o;
}

//-----------------------------------------------------------------
// Returns: sampling radius parameter used in edge refining.
//-----------------------------------------------------------------
int getSamplingRadius() {
//-----------------------------------------------------------------
	return samplingRadius;
}

//-----------------------------------------------------------------
// Summary: sets the edge sampling radius.
//-----------------------------------------------------------------
void setSamplingRadius(int sr) {
//-----------------------------------------------------------------
	samplingRadius = sr;
}

//-----------------------------------------------------------------
// Returns: number of blocks used on GPU.
//-----------------------------------------------------------------
int getGPUBlocks() {
//-----------------------------------------------------------------
	return gpuBlocks;
}

//-----------------------------------------------------------------
// Returns: number of threads used on GPU.
//-----------------------------------------------------------------
int getGPUThreads() {
//-----------------------------------------------------------------
	return gpuThreads;
}

};

#endif
