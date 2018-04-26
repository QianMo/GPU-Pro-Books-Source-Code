/* ******************************************************************************
* Description: This class controls the calculation process. Stores the input and output
*              images and starts rendering.
*
*  Version 1.0.0
*  Date: Nov 22, 2008
*  Author: David Illes, Peter Horvath
*          www.postpipe.hu
*
* GPUPro
***************************************************************************** */

#ifndef _LENSBLURAPP_
#define _LENSBLURAPP_

#include <time.h>

#include "image.hpp"
#include "imageBuffer.hpp"
#include "apKernel.hpp"
#include "defaultKernels.hpp"
#include "parameters.hpp"

#if COMPILE_GPU
	//#include "gpudof.h"
#endif

#define KERNEL_DIRECTORY "resources/kernels/"

//===============================================================
class LensBlurApp {
//===============================================================
public:
	Image* image;
	Image* previewImage;
	Image* zMap;
	Image* previewZMap;
	Image* resultImage;
	Image* iris;
	
	ImageBuffer* imageBuffer;
	ApKernel* apKernel;

	// parameters
	Parameters parameters;

	double accumulationTime;
	double reaccumulationTime;
	double bloomingTime;
	double calculationTime;

//-----------------------------------------------------------------
// Summary: Constructs the controller instance. Loads the default kernel.
//-----------------------------------------------------------------
LensBlurApp()
//-----------------------------------------------------------------
{
	image = 0;
	zMap = 0;
	resultImage = 0;
	iris = 0;
	imageBuffer = 0;
	apKernel = getKernel(APERTURE_DEFAULT);//new ApKernel();
}

//-----------------------------------------------------------------
// Summary: Frees memory.
//-----------------------------------------------------------------
~LensBlurApp()
//-----------------------------------------------------------------
{
	if (image) delete image;
	if (zMap) delete zMap;
	if (resultImage) delete resultImage;
	if (iris) delete iris;
	if (imageBuffer) delete imageBuffer;
	if (apKernel) delete apKernel;
}

//-----------------------------------------------------------------
// Summary: Creates a new source and z map image from the orignals 
//          with smaller size defined in the argument. 
//          Calculation is faster for this preview image.
// Arguments: n - ratio of the original and preview image
//-----------------------------------------------------------------
inline void createPreviewImage(int n)
//-----------------------------------------------------------------
{
	previewImage = new Image(image->getWidth() / n, image->getHeight() / n, image->getBpp());
	previewZMap = new Image(image->getWidth() / n, image->getHeight() / n, image->getBpp());
	
	for (int i=0;i<previewImage->getSize();i++) {
		int px = i % previewImage->getWidth();
		int py = i / previewImage->getWidth();
		int ix = n*px;
		int iy = n*py;

		for(int y=0;y<n;y++) {
			for(int x=0;x<n;x++) {
				int imageCoord = (ix+x) + (iy+y)*image->getWidth();
				previewImage->pixels[i] += image->pixels[imageCoord];
				previewZMap->pixels[i] += zMap->pixels[imageCoord];
			}
		}
		previewImage->pixels[i].r /= n*n;
		previewImage->pixels[i].g /= n*n;
		previewImage->pixels[i].b /= n*n;
		previewImage->pixels[i].luminance();
		previewZMap->pixels[i].r /= n*n;
		previewZMap->pixels[i].g /= n*n;
		previewZMap->pixels[i].b /= n*n;
		previewZMap->pixels[i].luminance();
	}
}

//-----------------------------------------------------------------
// Summary: Sets a new kernel for rendering.
// Arguments: new kernel - new kernel instance to use
//-----------------------------------------------------------------
inline void changeKernel(ApKernel* newKernel)
//-----------------------------------------------------------------
{
//	imageBuffer->uninitialize(true);

	if (apKernel) delete apKernel;
	apKernel = newKernel;
	//imageBuffer->kernelWidth = apKernel->getWidth();

	// reinit GPU
	if (imageBuffer->useGPU) {
		imageBuffer->uninitGPU();
		imageBuffer->initGPU(*image,*zMap,apKernel);
	}
}

//-----------------------------------------------------------------
// Summary: Sets a new kernel for rendering.
// Arguments: kernelIndex - type of the new kernel (see defaultKernels.hpp)
//-----------------------------------------------------------------
inline void changeKernel(int kernelIndex)
//-----------------------------------------------------------------
{
	changeKernel(getKernel(kernelIndex));
}

//-----------------------------------------------------------------
// Summary: Increases the given value.
// Arguments: value - value to increase
//            with - increase with
//            max - maximum value
//-----------------------------------------------------------------
inline void increaseValue(int* value, int with, int max=255) {
//-----------------------------------------------------------------
	*value += with;
	if (*value > max) *value = max;
}

//-----------------------------------------------------------------
// Summary: Decreases the given value.
// Arguments: value - value to decrease
//            with - decrease with
//            max - minimum value
//-----------------------------------------------------------------
inline void decreaseValue(int* value, int with, int min=0) {
//-----------------------------------------------------------------
	*value -= with;
	if (*value < min) *value = min;
}

//-----------------------------------------------------------------
// Summary: Smoothes the edges on the z map image.
// Arguments: zMap - grey-scaled z map image
//-----------------------------------------------------------------
inline void initZMap(Image* zMap)
//-----------------------------------------------------------------
{
	int size = zMap->getSize();
	int width = zMap->getWidth();
	int height = zMap->getHeight();

//	float* newValues = new float[size];
//	for(int i=0;i<size;i++) {
//		newValues[i] = -1;
//	}
	float newValue = -1;

	int RADIUS = 4;

	int* pixelGroupNum = new int[PIXEL_GROUPS];
	float* pixelGroupMaxZ = new float[PIXEL_GROUPS];

	for(int i=0;i<size;i++) {
		newValue = -1;
		float z = zMap->pixels[i].r;
		int zX = i%width;
		int zY = i/width;

		int zGroupId = (int)z * PIXEL_GROUPS / (MAX_PIXEL+1);

		for(int j=0;j<PIXEL_GROUPS;j++) {
			pixelGroupNum[j] = 0;
			pixelGroupMaxZ[j] = 0.0f;
		}

		for(int x=-RADIUS;x<RADIUS;x++) {
			for(int y=-RADIUS;y<RADIUS;y++) {
				int localX = zX + x;
				int localY = zY + y;

				if (localX<0 || localY<0) {
					continue;
				}
				if (width<=localX || height<=localY) {
					break;
				}
				int localCoord = localX + localY*width;
				float localZ = zMap->pixels[localCoord].r;

				int groupId = (int)localZ * PIXEL_GROUPS / (MAX_PIXEL+1);
				pixelGroupNum[groupId]++;
				if (localZ > pixelGroupMaxZ[groupId]) {
					pixelGroupMaxZ[groupId] = localZ;
				}
			}
		}

		// get the two groups with the most members
		int mostNum1 = -1;
		int mostNum2 = -1;
		for(int j=0;j<PIXEL_GROUPS;j++) {
#if TRACE_PIXEL
	if (i==PIXEL_X+width*PIXEL_Y) {
		pixelLogFile << j << ".) num: " << pixelGroupNum[j] << " - " << pixelGroupMaxZ[j] << "\n";
	}
#endif
			if (mostNum1 == -1 || pixelGroupNum[j] >= pixelGroupNum[mostNum1]) {
				mostNum2 = mostNum1;
				mostNum1 = j;
			} 
			else if (mostNum2 == -1 || pixelGroupNum[j] > pixelGroupNum[mostNum2]) {
				mostNum2 = j;
			}
		}

		// correct z value
		if (zGroupId != mostNum1 && zGroupId != mostNum2) {
			//if (abs(z-pixelGroupMaxZ[mostNum1]) < abs(z-pixelGroupMaxZ[mostNum2])) {
			if (pixelGroupMaxZ[mostNum1] > pixelGroupMaxZ[mostNum2]) {
				newValue = pixelGroupMaxZ[mostNum1];
			} else {
				newValue = pixelGroupMaxZ[mostNum2];
			}
		}

		if (newValue > -1) {
			zMap->pixels[i].r = newValue;
			zMap->pixels[i].g = newValue;
			zMap->pixels[i].b = newValue;
		}
	}

	delete [] pixelGroupNum;
	delete [] pixelGroupMaxZ;
}

//-----------------------------------------------------------------
// Summary: Renders the output image by the render parameters.
//-----------------------------------------------------------------
inline void render()
//-----------------------------------------------------------------
{
	long start_calc = clock();

	imageBuffer->clear();
//	imageBuffer->initialize(image,zMap,apKernel);

	imageBuffer->preProcess(*image,*zMap, apKernel);

//	if (!imageBuffer->reaccInitialized) {
//		imageBuffer->initReaccAll(apKernel);
//	}
	
	long start_acc = clock();
		imageBuffer->accumulateAll(image,zMap,apKernel/*,resultImage*/);
	if (LOG_TIME) {
		accumulationTime = (double)(clock() - start_acc) / CLOCKS_PER_SEC;
		printf("   accumulation time: %f\n",accumulationTime);
	}
	long start_reacc = clock();
		imageBuffer->reaccumulateAll(image,zMap,apKernel/*,resultImage*/);
	if (LOG_TIME) {
		reaccumulationTime = (double)(clock() - start_reacc) / CLOCKS_PER_SEC;
		printf("   reaccumulation time: %f\n",reaccumulationTime);
	}
	long start_bloom = clock();
		imageBuffer->bloomingAll(image,apKernel);
	if (LOG_TIME) {
		bloomingTime = (double)(clock() - start_bloom) / CLOCKS_PER_SEC;
		printf("   blooming time: %f\n",bloomingTime);
	}
 
	imageBuffer->postProcess();

	calculationTime = (double)(clock() - start_calc) / CLOCKS_PER_SEC;
	if (LOG_TIME) {
		printf("   calculation time: %f\n",time);
	}
}

//-----------------------------------------------------------------
// Summary: Renders the given range of the image by the render parameters.
//          Used for parallel rendering.
// Arguments: offset - pixel offset (index of the first pixel to render)
//            pixels - number of pixels to render
//-----------------------------------------------------------------
inline void render(int offset, int pixels)
//-----------------------------------------------------------------
{
	long start_calc = clock();

		// accumulation
		long start_acc = clock();

		for (register int i = 0; i < pixels; i++) {
			int pos = offset + i;
			imageBuffer->accumulate(*image, *zMap, *apKernel, pos);
		}

		if (LOG_TIME) {
			double accTime = (double)(clock() - start_acc) / CLOCKS_PER_SEC;
			printf("   accumulation time from %d to %d: %f\n",offset,offset+pixels,accTime);
		}

		// reaccumulation
		long start_reacc = clock();

		for (register int i = 0; i < pixels; i++) {
			int pos = offset + i;
			imageBuffer->reaccumulate(*image, apKernel, pos);
			//resultImage->pixels[pos] = imageBuffer->imagePixels[pos] / imageBuffer->imagePixels[pos].getLumValue();
		}

		if (LOG_TIME) {
			double reaccTime = (double)(clock() - start_reacc) / CLOCKS_PER_SEC;
			printf("   reaccumulation time from %d to %d: %f\n",offset,offset+pixels,reaccTime);
		}
	if (LOG_TIME) {
		double calcTime = (double)(clock() - start_calc) / CLOCKS_PER_SEC;
		printf("   calculation time from %d to %d: %f\n",offset,offset+pixels,calcTime);
	}
}

//-----------------------------------------------------------------
// Summary: Apply bloom effect the given range of the image.
//          Used for parallel rendering.
// Arguments: offset - pixel offset (index of the first pixel to calculate)
//            pixels - number of pixels to calculate
//-----------------------------------------------------------------
inline void bloom(int offset, int pixels) {
//-----------------------------------------------------------------
	// blooming
	long start_blooming = clock();

		for (register int i = 0; i < pixels; i++) {
			int pos = offset + i;
			imageBuffer->addBloomingValue(pos);
			resultImage->pixels[pos] = imageBuffer->imagePixels[pos];
		}

	if (LOG_TIME) {
		double bloomingtime = (double)(clock() - start_blooming) / CLOCKS_PER_SEC;
		printf("   blooming time from %d to %d: %f\n",offset,offset+pixels,bloomingtime);
	}
}

//-----------------------------------------------------------------
char** getGPUDevices() {
//-----------------------------------------------------------------
	int deviceCount = 0;
	char** deviceNames = 0;
	
#if COMPILE_GPU
	deviceNames = gpuGetDeviceNames();
#endif

	return deviceNames;
}

};

#endif