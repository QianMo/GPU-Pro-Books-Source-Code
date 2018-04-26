/* ******************************************************************************
* Description: This class implements the dof calculation algorithms and 
*              collects the result pixel values in a buffer.
*
*  Version 1.0.0
*  Date: Nov 22, 2008
*  Author: David Illes, Peter Horvath
*          www.postpipe.hu
*
* GPUPro
***************************************************************************** */

#ifndef _IMAGEBUFFER_
#define _IMAGEBUFFER_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <time.h>
using namespace std;

//#if COMPILE_GPU
	#include "gpudof.h"
//#include <cutil.h>
//#include "gpu/lensblur.cu"
//#endif

#include "imagePixel.hpp"
#include "apKernel.hpp"
#include "parameters.hpp"

//===============================================================
class ImageBuffer {
//===============================================================
public:
	int width, height;
	int size;

	// parameters
	Parameters* parameters;

	float* recalcZDepth;
	float* recalcDist;
	int* recalcZValue;

	float* bloomValues;

	int* pixelCategoriesNum;
	float* pixelCategoriesDist;
	float* pixelCategoriesZDepth;
	int* pixelCategoriesZValue;

	bool useGPU;

#if TRACE_PIXEL
	ofstream pixelLogFile; 
#endif

public:
	ImagePixel* imagePixels;

#ifndef _INC_MATH
inline int abs(int x) {
	if(x<0) return -x;
	return x;
}

inline float abs(float x) {
	if(x<0) return -x;
	return x;
}

#endif

//-----------------------------------------------------------------
// Summary: Constructs default image buffer.
//-----------------------------------------------------------------
ImageBuffer() {
//-----------------------------------------------------------------
	width = 0;
	height = 0;
	size = width*height;
}

//-----------------------------------------------------------------
// Summary: Constructs image buffer.
// Arguments: image - source image
//            zMap - z map image
//            kernel - kernel mask
//            resultImage - result image
//-----------------------------------------------------------------
ImageBuffer(Image* image, Image* zMap, ApKernel* kernel, Image* resultImage, Parameters* parameters) {
//-----------------------------------------------------------------
	this->parameters = parameters;

	width = image->getWidth();
	height = image->getHeight();
	size = width*height;
	parameters->width = width;
	parameters->height = height;
	parameters->size = size;
	//imagePixels = new ImagePixel[size];
	imagePixels = resultImage->pixels;

	recalcZDepth = new float[size];
	recalcDist = new float[size];
	recalcZValue = new int[size];

	bloomValues = new float[size];

	pixelCategoriesNum = new int[PIXEL_GROUPS];
	pixelCategoriesDist = new float[PIXEL_GROUPS];
	pixelCategoriesZDepth = new float[PIXEL_GROUPS];
	pixelCategoriesZValue = new int[PIXEL_GROUPS];

#if TRACE_PIXEL
	// open log file
	pixelLogFile.open(LOG_FILE, ios::out);
	pixelLogFile << "focal point: " << focusValue << "\n";
	pixelLogFile << "z epsilon: " << zEpsilon << "\n";
	pixelLogFile << "bloom amount: " << bloomAmount << "\n";
	pixelLogFile << "bloom threshold: " << threshold << "\n";
	pixelLogFile << "\n";
#endif

#if COMPILE_GPU
	//int deviceCount = gpuGetDeviceCount();
	//cudaGetDeviceCount(&deviceCount);
	useGPU = false;//deviceCount > 0;

	if (useGPU) {
		clock_t start_init = clock();
			gpuInit(*image,*zMap, kernel, parameters);
		if (LOG_TIME) {
			double time_init = (double)(clock() - start_init) / CLOCKS_PER_SEC;
			printf("   GPU init time: %f\n",time_init); 
		}
	}
#else
	useGPU = false;
#endif
}

//-----------------------------------------------------------------
// Summary: Destructs the image buffer.
//-----------------------------------------------------------------
~ImageBuffer() {
//-----------------------------------------------------------------
	//delete [] imagePixels;

	delete [] recalcZDepth;
	delete [] recalcDist;
	delete [] recalcZValue;

	delete [] bloomValues;

	delete [] pixelCategoriesNum;
	delete [] pixelCategoriesDist;
	delete [] pixelCategoriesZDepth;
	delete [] pixelCategoriesZValue;

#if TRACE_PIXEL
	pixelLogFile.close();
#endif
}

//-----------------------------------------------------------------
// Summary: gets the kernel dimension by the pixel z value.
// Arguments: pixel - image pixel
//            kernel - basic kernel mask
// Returns: kernel width
//-----------------------------------------------------------------
float calculateKernelWidth(ImagePixel* pixel, ApKernel& kernel) {
//-----------------------------------------------------------------
	return calculateKernelWidth(pixel->zDepth,pixel->zValue,kernel);
}

//-----------------------------------------------------------------
// Summary: gets the kernel dimension by z value. For physical mode we
//          need the z depth value (0.0 - 1.0).
// Arguments: zDepth - z depth (physical mode)
//            zValue - z value (artist mode)
//            kernel - basic kernel mask
// Returns: kernel width
//-----------------------------------------------------------------
float calculateKernelWidth(float zDepth, int zValue, ApKernel& kernel) {
//-----------------------------------------------------------------
	// Artist Mode
	if (parameters->mode != 1) {
		return parameters->strength*zDepth * (float)kernel.width;
	}
	// Physically-based Mode
	else {
		float blurRadius = parameters->getPhysicalStrength(zValue);
		return (blurRadius / 11.0f) * 64.0f;
	}
}

//-----------------------------------------------------------------
// Summary: gets the kernel mask defined the pixel z value. The kernel
//          is transformed from the original kernel.
// Arguments: pixel - image pixel
//            kernel - basic kernel mask
// Returns: kernel mask for the pixel
//-----------------------------------------------------------------
ApKernel* getZKernel(ImagePixel* pixel, ApKernel& kernel) {
//-----------------------------------------------------------------
	return getZKernel(pixel->zDepth,pixel->zValue,kernel);
}

//-----------------------------------------------------------------
// Summary: gets the kernel mask defined the pixel z value. The kernel
//          is transformed from the original kernel.
// Arguments: zDepth - z depth (physical mode)
//            zValue - z value (artist mode)
//            kernel - basic kernel mask
// Returns: kernel mask for the pixel
//-----------------------------------------------------------------
ApKernel* getZKernel(float zDepth, int zValue, ApKernel& kernel) {
//-----------------------------------------------------------------
	float kernelWidth = calculateKernelWidth(zDepth,zValue,kernel);
	if (parameters->interpolation) {
		return kernel.getInterpolatedKernel(kernelWidth,kernelWidth);
	} else {
		return kernel.getKernel(kernelWidth,kernelWidth);
	}
}

//-----------------------------------------------------------------
// Summary: clears the pixels and helper arrays.
//-----------------------------------------------------------------
void clear() {
//-----------------------------------------------------------------
	for(int i=0;i<size;i++) {
		imagePixels[i].reset();

		recalcZDepth[i]=0.0f;
		recalcDist[i]=0.0f;
		recalcZValue[i]=0;

		bloomValues[i]=0.0f;
	}
}

//-----------------------------------------------------------------
// Returns: image width.
//-----------------------------------------------------------------
int getWidth() {
//-----------------------------------------------------------------
	return width;
}

//-----------------------------------------------------------------
// Returns: image height.
//-----------------------------------------------------------------
int getHeight() {
//-----------------------------------------------------------------
	return height;
}

//-----------------------------------------------------------------
// Summary: gets the z value of the given pixel.
// Arguments: zMap - z map image
//            pos - pixel index
// Returns: pixel z value from the grey-scaled image
//-----------------------------------------------------------------
inline int getZValue(Image& zMap, int pos) {
//-----------------------------------------------------------------
	return (int)zMap.pixels[pos].r;
}

//-----------------------------------------------------------------
// Summary: gets the z value of the given pixel.
// Arguments: zMap - z map image
//            x - pixel x coordinate
//            y - pixel y coordinate
// Returns: pixel z value from the grey-scaled image
//-----------------------------------------------------------------
inline int getZValue(Image& zMap,int x,int y) {
//-----------------------------------------------------------------
	return getZValue(zMap,x+y*width);
}

//-----------------------------------------------------------------
// Summary: executes pre-calculations.
// Arguments: image - source image
//            zMap - z map image
//            kernel - kernel mask
//-----------------------------------------------------------------
void preProcess(Image& image, Image& zMap, ApKernel* kernel) {
//-----------------------------------------------------------------
	for(int i=0;i<image.getSize();i++) {
		int zValue = getZValue(zMap,i);
		int focusDistance = abs(parameters->focus - zValue);
		float zDepth = (float)focusDistance / MAX_PIXEL;

		imagePixels[i].zValue = zValue;
		imagePixels[i].focusDistance = focusDistance;
		imagePixels[i].zDepth = zDepth;

		imagePixels[i].group = zValue * PIXEL_GROUPS / (MAX_PIXEL+1);

		#if TRACE_PIXEL
			if (i == PIXELX + PIXELY*width) {
				pixelLogFile << "z value: " << imagePixels[i].zValue << "\n";
				pixelLogFile << "z depth: " << imagePixels[i].zDepth << "\n";
				pixelLogFile << "group: " << imagePixels[i].group << "\n";
			}
		#endif
	}
}

//-----------------------------------------------------------------
// Summary: finalizes the calculation process.
//-----------------------------------------------------------------
void postProcess() {
//-----------------------------------------------------------------
}

//-----------------------------------------------------------------
// Summary: initializes GPU device.
// Arguments: image - source image
//            zMap - z map image
//            kernel - kernel mask
//-----------------------------------------------------------------
void initGPU(Image& image, Image& zMap, ApKernel* kernel) {
//-----------------------------------------------------------------
#if COMPILE_GPU
	gpuInit(image,zMap,kernel,parameters);
#endif
}

//-----------------------------------------------------------------
// Summary: uninitializes the GPU device.
//-----------------------------------------------------------------
void uninitGPU() {
//-----------------------------------------------------------------
#if COMPILE_GPU
	gpuUninit();
#endif
}

//-----------------------------------------------------------------//
//                         ACCUMULATION
//-----------------------------------------------------------------//

//-----------------------------------------------------------------
// Summary: executes accumulation for each pixel. In the accumulation 
//          phase we calculate pixel color from the neighboring pixels
//          using the kernel as a mask. We marks the the edge pixels.
// Arguments: image - source image
//            zMap - z map image
//            kernel - kernel mask
//-----------------------------------------------------------------
void accumulateAll(Image* image, Image* zMap, ApKernel* kernel) {
//-----------------------------------------------------------------
	// separate gpu - cpu calculation
	#if COMPILE_GPU
		if (useGPU) {
			accumulateOnGPU(image->getSize(), kernel);
		} else {
			accumulateOnCPU(*image,*zMap,*kernel);
		}
	#else
		accumulateOnCPU(*image,*zMap,*kernel);
	#endif
}

//-----------------------------------------------------------------
// Summary: executes pixel accumulation on the GPU device.
// Arguments: numPixels - number of pixels
//            kernel - kernel mask
//-----------------------------------------------------------------
void accumulateOnGPU(int numPixels, ApKernel* kernel) {
//-----------------------------------------------------------------
#if COMPILE_GPU
	gpuAccumulate(numPixels,kernel,parameters);
/*
	for(int i=0;i<numPixels;i++) {
		int pos = i;//pixels[i];
		int index = 4*i;
		imagePixels[pos].r = resultData[index];
		imagePixels[pos].g = resultData[index+1];
		imagePixels[pos].b = resultData[index+2];
		imagePixels[pos].lum = resultData[index+3];
	}
*/
	//delete [] resultData;
#endif
}

//-----------------------------------------------------------------
// Summary: executes pixel accumulation on the CPU.
// Arguments: image - source image
//            zMap - z map image
//            kernel - kernel mask
//-----------------------------------------------------------------
void accumulateOnCPU(Image& image, Image& zMap, ApKernel& kernel) {
//-----------------------------------------------------------------
	for(int i=0;i<size;i++) {
		accumulate(image,zMap,kernel,i);
	}
}

//-----------------------------------------------------------------
// Summary: executes accumulation for a pixel. In the accumulation 
//          phase we calculate pixel color from the neighboring pixels
//          using the kernel as a mask. We marks the the edge pixels.
// Arguments: image - source image
//            zMap - z map image
//            zKernel - transformed kernel mask
//            pixel - result pixel
//            pos - pixel index
//            recalc - called in re-calcualtion phase or not
//            recalcGroup - z group index of the pixel
//-----------------------------------------------------------------
void accumulatePixel(Image& image, Image& zMap, ApKernel* zKernel, ImagePixel* pixel, int pos, bool recalc = false, int recalcGroup = -1) {
//-----------------------------------------------------------------
	int imageX = pos % width;
	int imageY = pos / width;

	#if TRACE_PIXEL
		if (imageX == PIXELX && imageY == PIXELY) {
			pixelLogFile << "\n   --- PIXEL ACCUMULATION ---   \n\n";
		}
	#endif

	ImagePixel* source = &(imagePixels[pos]);

	int zValue = source->zValue;

	int topLeftX = imageX - zKernel->halfwidth;
	int topLeftY = imageY - zKernel->halfheight;

	int kernelCoord = -1;
	for(register int y=0;y<zKernel->getHeight();y++) {
		for(register int x=0;x<zKernel->getWidth();x++) {
			kernelCoord++;
			float zKernelValue = zKernel->values[kernelCoord];

			if (zKernelValue == 0.0f) {
				continue;
			}

			int localX = (topLeftX+x);
			int localY = (topLeftY+y);

			// skip the pixels from the other size of the image
			if (localX<0 || localY<0 || width<=localX || height<=localY) {
				continue;
			}

			int localCoord = localX + width*localY;
			ImagePixel localColor = image.pixels[localCoord];
			int localZValue = imagePixels[localCoord].zValue;

			#if TRACE_PIXEL
				if (pos == PIXELX + PIXELY*width) {
					pixelLogFile << localX << " - " << localY << " z value: " << localZValue << " diff: " << abs(zValue - localZValue) << " group: " << imagePixels[localCoord].group << "\n";
				}
			#endif

			if (!recalc) {
				if (abs(zValue - localZValue) > parameters->zEpsilon) {
					// calculating background
					if (zValue < localZValue) {
						if (abs(localX - imageX) > parameters->samplingRadius || abs(localY - imageY) > parameters->samplingRadius) {
							#if TRACE_PIXEL
								if (pos == PIXELX + PIXELY*width) {
									pixelLogFile << "skip " << localX << " - " << localY << "\n";
								}
							#endif
							continue;
						}
					}
				}
			}
			// reaccumulate by group
			else {
				// skip pixels from the closer layers
				if (imagePixels[localCoord].group > recalcGroup) {
					#if TRACE_PIXEL
						if (pos == PIXELX + PIXELY*width) {
							pixelLogFile << "skip " << localX << " - " << localY << " group: " << imagePixels[localCoord].group << "\n";
						}
					#endif
					continue;
				}
			}

			float multiplicator = zKernelValue * localColor.getLumValue();
			pixel->r += localColor.r * multiplicator;
			pixel->g += localColor.g * multiplicator;
			pixel->b += localColor.b * multiplicator;
			pixel->lum += multiplicator;
		}
	}

	if (pixel->lum > 0) {
		pixel->r /= pixel->lum;
		pixel->g /= pixel->lum;
		pixel->b /= pixel->lum;
	}

	#if TRACE_PIXEL
		if (imageX == PIXELX && imageY == PIXELY) {
			pixelLogFile << "color: " << pixel->r << " - " << pixel->g << " - " << pixel->b << "\n\n" ;
		}
	#endif
}

//-----------------------------------------------------------------
// Summary: executes accumulation for a pixel. In the accumulation 
//          phase we calculate pixel color from the neighboring pixels
//          using the kernel as a mask. We marks the edge pixels.
// Arguments: image - source image
//            zMap - z map image
//            kernel - kernel mask
//            pos - pixel index
//-----------------------------------------------------------------
void accumulate(Image& image, Image& zMap, ApKernel& kernel, int pos) {
//-----------------------------------------------------------------
	// init pixel groups
	for(int i=0;i<PIXEL_GROUPS;i++) {
		pixelCategoriesNum[i] = -1;
		pixelCategoriesDist[i] = 0.0f;
		pixelCategoriesZDepth[i] = (float)INT_MAX;
		pixelCategoriesZValue[i] = 0;
	}

	int imageX = pos % width;
	int imageY = pos / width;

	#if TRACE_PIXEL
		if (imageX == PIXELX && imageY == PIXELY) {
			pixelLogFile << "\n   --- ACCUMULATION ---   \n\n";
		}
	#endif

	ImagePixel* pixel = &(imagePixels[pos]);
	int zValue = pixel->zValue;
	float zDepth = pixel->zDepth;

	// get z kernel
	ApKernel* zKernel = getZKernel(pixel,kernel);
	// use default color when no kernel is defined
	if (!zKernel) {
		return;
	}

//	ImagePixel origColor = image.pixels[pos];

	int topLeftX = imageX - zKernel->halfwidth;
	int topLeftY = imageY - zKernel->halfheight;

	int kernelCoord = -1;
	for(register int y=0;y<zKernel->getHeight();y++) {
		for(register int x=0;x<zKernel->getWidth();x++) {
			//int kernelCoord = x + y * zKernel->getWidth();
			kernelCoord++;
			float kernelValue = zKernel->values[kernelCoord];

			if (kernelValue == 0.0f) {
				continue;
			}

			int localX = (topLeftX+x);
			int localY = (topLeftY+y);

			// skip the pixels from the other size of the image
			if (localX<0 || localY<0 || width<=localX || height<=localY) {
				continue;
			}

			int localCoord = localX + width*localY;

			ImagePixel* localColor = &(image.pixels[localCoord]);
			ImagePixel* localPixel = &(imagePixels[localCoord]);

			//int localZValue = getZValue(zMap,localCoord);
			int localZValue = localPixel->zValue;

			if (abs(zValue - localZValue) > parameters->zEpsilon) {
				// calculating background
				if (zValue < localZValue) {
					if (localPixel->zDepth > 0.1f && localZValue < parameters->focus) {
						//if ((1<localCoord && localCoord<size-1 && abs(imagePixels[localCoord-1].zDepth - imagePixels[localCoord+1].zDepth) < 0.04f) && (width<localCoord && localCoord < size-width-1 && abs(imagePixels[localCoord-width].zDepth - imagePixels[localCoord+width].zDepth) < 0.04f)) {
							int dist = abs(imageX-localX)*abs(imageX-localX) + abs(imageY-localY)*abs(imageY-localY);
							// original formula: 1.0f - 0.028125f*(float)kernel.width* ((float)dist / (strength * localPixel->zDepth*(float)kernel.halfwidth*localPixel->zDepth*(float)kernel.halfwidth));
							float distScale = 1.0f - 0.1125f* ((float)dist / (parameters->strength * localPixel->zDepth*localPixel->zDepth*(float)kernel.width));

							// get pixel group
							int group = localPixel->group;
							//int group = localZValue * PIXEL_GROUPS / (int)maxPixelValue;

							pixelCategoriesNum[group]++;
							if (distScale > pixelCategoriesDist[group]) {
								pixelCategoriesDist[group] = distScale;
							}
							if (localPixel->zDepth < pixelCategoriesZDepth[group]) {
								pixelCategoriesZDepth[group] = localPixel->zDepth;
								pixelCategoriesZValue[group] = localPixel->zValue;
							}
						//}
					}

					if (abs(localX - imageX) > parameters->samplingRadius || abs(localY - imageY) > parameters->samplingRadius) {
						continue;
					}
				}

				// calculating foreground
				else {
					// background is in focus
					if (zValue > parameters->focus) {
						float dist = (float)(abs(imageX - localX)*abs(imageX - localX) + abs(imageY - localY)*abs(imageY - localY));
						//float r = (float)(zKernel->halfwidth*zKernel->halfwidth + zKernel->halfheight*zKernel->halfheight) / 4;
						float distScale = 1.0f - dist / zKernel->r;
						if (distScale > recalcDist[localCoord]) {
							recalcDist[localCoord] = distScale;
						}
						if (zValue > recalcZValue[localCoord]) {
							recalcZDepth[localCoord] = zDepth;
							recalcZValue[localCoord] = zValue;
						}
					}
				}
			}

			float multiplicator = kernelValue * localColor->getLumValue();
			pixel->r += localColor->r * multiplicator;
			pixel->g += localColor->g * multiplicator;
			pixel->b += localColor->b * multiplicator;
			pixel->lum += multiplicator;
		}
	}
	
	if (pixel->lum > 0) {
		pixel->r /= pixel->lum;
		pixel->g /= pixel->lum;
		pixel->b /= pixel->lum;
	}

	for(int i=0;i<PIXEL_GROUPS;i++) {
#if TRACE_PIXEL
	if (pos==PIXEL_X+width*PIXEL_Y) {
		pixelLogFile << i << ".) category num: " << pixelCategoriesNum[i] << " - " << pixelCategoriesDist[i] << " - " << pixelCategoriesZDepth[i] << "\n";
	}
#endif

		if (pixelCategoriesDist[i] > 0.05f && pixelCategoriesNum[i] > zKernel->size / 40) {
			reaccumulateBG(image,&kernel,pixelCategoriesZDepth[i],pixelCategoriesZValue[i],pixelCategoriesDist[i],pos,i);
		}
	}

	blooming(&image,zKernel,pos);

	if (parameters->interpolation) delete zKernel;
}

//-----------------------------------------------------------------//
//                        REACCUMULATION
//-----------------------------------------------------------------//

//-----------------------------------------------------------------
// Summary: executes re-accumulation for each pixel. In the re-accumulation 
//          phase we calculate more realistic pixel color for the edge pixels.
// Arguments: image - source image
//            zMap - z map image
//            kernel - kernel mask
//-----------------------------------------------------------------
void reaccumulateAll(Image* image, Image* zMap, ApKernel* kernel) {
//-----------------------------------------------------------------
	#if COMPILE_GPU
		if (useGPU) {
			reaccumulateOnGPU(image->getSize(),kernel);
		} else {
			reaccumulateOnCPU(image,zMap,kernel);
		}
	#else
		reaccumulateOnCPU(image,zMap,kernel);
	#endif
}

//-----------------------------------------------------------------
// Summary: executes pixel re-accumulation on the GPU device.
// Arguments: numPixels - number of pixels
//            kernel - kernel mask
//-----------------------------------------------------------------
void reaccumulateOnGPU(int numPixels, ApKernel* kernel) {
//-----------------------------------------------------------------
#if COMPILE_GPU
	float* resultData = gpuReaccumulate(numPixels,kernel,parameters);

	// read pixels
	for(int i=0;i<numPixels;i++) {
		int pos = i;//pixels[i];
		int index = 4*i;
		imagePixels[pos].r = resultData[index];
		imagePixels[pos].g = resultData[index+1];
		imagePixels[pos].b = resultData[index+2];
		imagePixels[pos].lum = resultData[index+3];
	}

	delete [] resultData;
#endif
}

//-----------------------------------------------------------------
// Summary: executes pixel re-accumulation on the CPU.
// Arguments: image - source image
//            zMap - z map image
//            kernel - kernel mask
//-----------------------------------------------------------------
void reaccumulateOnCPU(Image* image, Image* zMap, ApKernel* kernel) {
//-----------------------------------------------------------------
	int imageSize = image->getSize();
	for(int i=0;i<imageSize;i++) {
		reaccumulate(*image,kernel,i);
	}
}

//-----------------------------------------------------------------
// Summary: gets overlap scale value from the distance scale of the edge pixel.
// Arguments: value - distance scale
//-----------------------------------------------------------------
float getOverlapScale(float value) {
//-----------------------------------------------------------------
	if (parameters->overlap == 0.0f) {
		return 0.0f;
	}
	if (parameters->overlap < 1.0f) {
		if (value > (1.0f-parameters->overlap)) {
			return (value - (1.0f - parameters->overlap)) /parameters->overlap;
		} else {
			return 0;
		}
	} 
	if (parameters->overlap > 1.0f) {
		float rec = 1.0f / parameters->overlap;
		return rec * value + (1.0f - rec);
	}
	return value;
}

//-----------------------------------------------------------------
// Summary: executes re-accumulation for a pixel before the focal point.
//          In the re-accumulation phase we calculate more realistic pixel color
//          for the edge pixels.
// Arguments: image - source image
//            kernel - kernel mask
//            pos - pixel index
//-----------------------------------------------------------------
void reaccumulate(Image& image, ApKernel* kernel, int pos) {
//-----------------------------------------------------------------
	if (recalcZDepth[pos]) {
		#if TRACE_PIXEL
			if (pos == PIXELX + PIXELY*width) {
				pixelLogFile << "\n   --- REACCUMULATE FOREGROUND ---   \n\n";
				pixelLogFile << "recalc dist: " << recalcDist[pos] << " recalc z depth: " << recalcZDepth[pos] << " recalc z value: " << recalcZValue[pos] << "\n";
			}
		#endif

		ApKernel* zKernel = getZKernel(recalcZDepth[pos],recalcZValue[pos],*kernel);

		if (zKernel) {
			ImagePixel* pixel = &imagePixels[pos];
			int imageX = pos % width;
			int imageY = pos / width;

			#if TRACE_PIXEL
				if (imageX == PIXELX && imageY == PIXELY) {
					pixelLogFile << "reacc z kernel: " << zKernel->width << "\n";
				}
			#endif

			//imagePixels[pos].reset();
			ImagePixel newColor;

			//float lumScale = 1.0f - image->pixels[i].getLumValue() / maxPixelValue;

			int topLeftX = imageX - zKernel->halfwidth;
			int topLeftY = imageY - zKernel->halfheight;

			int kernelCoord = -1;
			for(register int y=0;y<zKernel->getHeight();y++) {
				for(register int x=0;x<zKernel->getWidth();x++) {
					kernelCoord++;
					//int kernelCoord = x + y * kernel.getWidth();
					float kernelValue = zKernel->values[kernelCoord];

					if (kernelValue == 0.0f) {
						continue;
					}

					int localX = (topLeftX+x);
					int localY = (topLeftY+y);

					// skip the pixels from the other size of the image
					if (localX<0 || localY<0 || width<=localX || height<=localY) {
						continue;
					}

					int localCoord = localX + width*localY;
					ImagePixel localColor = image.pixels[localCoord];

					#if TRACE_PIXEL
						if (pos == PIXELX + PIXELY*width) {
							pixelLogFile << localX << " - " << localY << "\n";
						}
					#endif

					float multiplicator = kernelValue * localColor.getLumValue();
					newColor += localColor * multiplicator;
					newColor.lum += multiplicator;
				}
			}

			float scale = getOverlapScale(recalcDist[pos]);
			newColor.r /= newColor.lum;
			newColor.g /= newColor.lum;
			newColor.b /= newColor.lum;
			pixel->r = pixel->r*(1.0f-scale) + newColor.r*scale;
			pixel->g = pixel->g*(1.0f-scale) + newColor.g*scale;
			pixel->b = pixel->b*(1.0f-scale) + newColor.b*scale;

			#if TRACE_PIXEL
				if (pos == PIXELX + PIXELY*width) {
					pixelLogFile << "reacc color: " << imagePixels[pos].r << " - " << imagePixels[pos].g << " - " << imagePixels[pos].b << "\n\n" ;
				}
			#endif

			blooming(&image,zKernel,pos);

			if (parameters->interpolation) delete zKernel;
		}
	}
}

//-----------------------------------------------------------------
// Summary: executes re-accumulation for a pixel after the focal point.
//          In the re-accumulation phase we calculate more realistic pixel color
//          for the edge pixels.
// Arguments: image - source image
//            kernel - kernel mask
//            zDepth - z depth
//            zValue - z value
//            distScale - edge distance scale 
//            pos - pixel index
//            group - z group index
//-----------------------------------------------------------------
void reaccumulateBG(Image& image, ApKernel* kernel, float zDepth, int zValue, float distScale, int pos, int group) {
//-----------------------------------------------------------------
	#if TRACE_PIXEL
		if (pos == PIXELX + PIXELY*width) {
			pixelLogFile << "\n   --- REACCUMULATE BACKGROUND ---   \n\n";
			pixelLogFile << "recalc dist: " << distScale << " recalc z depth: " << zDepth << " recalc z value: " << zValue << " group: " << group << "\n";
		}
	#endif

	ApKernel* zKernel = getZKernel(zDepth,zValue,*kernel);
	if (!zKernel) return;

	#if TRACE_PIXEL
		if (pos == PIXELX + PIXELY*width) {
			pixelLogFile << "reacc z kernel: " << zKernel->width << "\n";
		}
	#endif

	ImagePixel newColor;
	accumulatePixel(image,image,zKernel,&newColor,pos,true,group);
	distScale = getOverlapScale(distScale);
	imagePixels[pos].r = imagePixels[pos].r*(1.0f-distScale) + newColor.r*distScale;
	imagePixels[pos].g = imagePixels[pos].g*(1.0f-distScale) + newColor.g*distScale;
	imagePixels[pos].b = imagePixels[pos].b*(1.0f-distScale) + newColor.b*distScale;

	if (parameters->interpolation) delete zKernel;

	#if TRACE_PIXEL
		if (pos == PIXELX + PIXELY*width) {
			pixelLogFile << "reacc color: " << imagePixels[pos].r << " - " << imagePixels[pos].g << " - " << imagePixels[pos].b << "\n\n" ;
		}
	#endif
}

//-----------------------------------------------------------------//
//                           BLOOMING
//-----------------------------------------------------------------//

//-----------------------------------------------------------------
// Summary: executes blooming calculation for each pixel.
// Arguments: image - source image
//            kernel - kernel mask
//-----------------------------------------------------------------
void bloomingAll(Image* image, ApKernel* kernel) {
//-----------------------------------------------------------------
	#if COMPILE_GPU
		if (useGPU) {
			if (bloomValues) delete [] bloomValues;
			bloomValues = gpuGetBloomValues(image->getSize());
		}
	#else
		reaccumulateOnCPU(image,zMap,kernel);
	#endif

	int imageSize = image->getSize();
	for(int i=0;i<imageSize;i++) {
		addBloomingValue(i);
	}
}

//-----------------------------------------------------------------
// Summary: apply blooming on a pixel.
// Arguments: pos - pixel index
//-----------------------------------------------------------------
void addBloomingValue(int pos) {
//-----------------------------------------------------------------
	if (bloomValues[pos]) {
		float bloomScale = bloomValues[pos] * (float)parameters->bloomAmount / 1000.0f;
		#if TRACE_PIXEL
			if (pos == PIXELX + PIXELY*width) {
				pixelLogFile << "blooming value: " << bloomValues[pos] << " bloom scale: " << bloomScale << "\n";
			}
		#endif

		imagePixels[pos].r = bloomScale * MAX_PIXEL + (1.0f - bloomScale) * imagePixels[pos].r;
		imagePixels[pos].g = bloomScale * MAX_PIXEL + (1.0f - bloomScale) * imagePixels[pos].g;
		imagePixels[pos].b = bloomScale * MAX_PIXEL + (1.0f - bloomScale) * imagePixels[pos].b;
		if (imagePixels[pos].r > MAX_PIXEL) imagePixels[pos].r = MAX_PIXEL;
		if (imagePixels[pos].g > MAX_PIXEL) imagePixels[pos].g = MAX_PIXEL;
		if (imagePixels[pos].b > MAX_PIXEL) imagePixels[pos].b = MAX_PIXEL;
	}
}

//-----------------------------------------------------------------
// Summary: accumulates bloom value for a pixel calculated from the neighboring pixels.
// Arguments: image - source image
//            zKernel - transformed kernel mask
//            pos - pixel index
//-----------------------------------------------------------------
void blooming(Image* image, ApKernel* zKernel, int pos) {
//-----------------------------------------------------------------
	if (parameters->bloomAmount > 0 && image->pixels[pos].avg() > parameters->threshold) {
		int imageX = pos % width;
		int imageY = pos / width;

		#if TRACE_PIXEL
			if (imageX == PIXELX && imageY == PIXELY) {
				pixelLogFile << "\n   --- BLOOMING ---   \n\n";
			}
		#endif

		int topLeftX = imageX - zKernel->halfwidth;
		int topLeftY = imageY - zKernel->halfheight;

		int kernelCoord = -1;
		for(register int y=0;y<zKernel->getHeight();y++) {
			for(register int x=0;x<zKernel->getWidth();x++) {
				kernelCoord++;
				float kernelValue = zKernel->values[kernelCoord];

				if (kernelValue == 0.0f) {
					continue;
				}

				int localX = (topLeftX+x);
				int localY = (topLeftY+y);

				// skip the pixels from the other size of the image
				if (localX<0 || localY<0 || width<=localX || height<=localY) {
					continue;
				}

				int localCoord = localX + width*localY;
				//ImagePixel* localPixel = &(imagePixels[localCoord]);

				float distance = (abs(localX-imageX)*abs(localX-imageX) + abs(localY-imageY)*abs(localY-imageY)) / 1.2f;
				float multiplicator = (1.0f - distance / zKernel->r) * (recalcZDepth[pos] ? recalcZDepth[pos] : imagePixels[pos].zDepth);
				//if (multiplicator > bloomValues[localCoord]) {
				if (multiplicator > 0) {
					#if TRACE_PIXEL
						if (imageX == PIXELX && imageY == PIXELY) {
							pixelLogFile << "increase blooming value: " << multiplicator << " at " << localX << " - " << localY << "\n";
						}
					#endif
					bloomValues[localCoord] += multiplicator;
				}
				//}
			}
		}
	}
}

};

#endif
