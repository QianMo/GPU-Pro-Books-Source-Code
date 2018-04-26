#ifndef _LENSBLUR_KERNEL_
#define _LENSBLUR_KERNEL_

//-----------------------------------------------------------------
__device__ static float getPhysicalStrengthGPU(int zValue, float* params) {
//-----------------------------------------------------------------
	float maxPixelValue = params[14];
	float cameraDistance = params[16];
	float distance = params[11];
	int focusValue = (int)params[2];
	float physicsParam = params[18];

	// focal point distance
	//float s = cameraDistance + (1.0f - (float)focusValue / maxPixelValue) * distance;
	// subject distance from the camera
	float s = cameraDistance + (1.0f - (float)zValue / maxPixelValue) * distance;
	// subject distance from the focal point
	float xd = (float)abs(focusValue - zValue) / maxPixelValue * distance;
	//float d = zValue < focusValue ? xd / (s + xd) : xd / (s - xd);
	float d = xd / s;

	return physicsParam * d;
}

//-----------------------------------------------------------------
__device__ static int getKernelWidth(int kernelWidth, int mainWidth) {
//-----------------------------------------------------------------
	if (1 > kernelWidth) return 1;
	if (kernelWidth > mainWidth) kernelWidth = mainWidth;
	if (kernelWidth % 2 == 0) return kernelWidth-1;
	return kernelWidth;
}

//-----------------------------------------------------------------
__device__ float calculateKernelWidth(float zDepth, int zValue, float* params) {
//-----------------------------------------------------------------
	// Artist Mode
	if (params[20] == 0.0f) {
		float strength = params[5];
		float mainKernelWidth = params[15];
		return strength * zDepth * mainKernelWidth;
	}
	// Physically-based Mode
	else {
		float blurRadius = getPhysicalStrengthGPU(zValue,params);
		return (blurRadius / 11.0f) * 64.0f;
	}
}

//-----------------------------------------------------------------
__device__ static float getKernelValue(int y, int kernelCoord, int smallOffset, int bigOffset, float bigScale, float* kernels, int smallSize, int kernelWidth) {
//-----------------------------------------------------------------
	if (smallOffset < 0) {
		return 1.0f;
	}
	if (bigOffset == smallOffset) {
		return kernels[bigOffset + kernelCoord];
	}

	int smallCoord = kernelCoord - kernelWidth - 2 * y + 1;
	if (smallCoord < 0 || smallCoord >= smallSize) {
		return bigScale * kernels[bigOffset + kernelCoord];
	} else {
		return (1.0f - bigScale)*kernels[smallOffset+smallCoord] + bigScale*kernels[bigOffset+kernelCoord];
	}
}

//-----------------------------------------------------------------//
//                           BLOOMING
//-----------------------------------------------------------------//

//-----------------------------------------------------------------
__global__ static void bloomingGPU(float* bloomValues,int* positions,float* params, float* output) {
//-----------------------------------------------------------------
	int tid = threadIdx.x + blockDim.x * threadIdx.y;
    int bid = blockIdx.x + gridDim.x * blockIdx.y;
    int id = bid * blockDim.x*blockDim.x + tid;

    int width = (int)params[0];
    int height = (int)params[1];

	if (id > width*height) {
		return;
	}

	int pos = positions[id];

	// scene parameters
    float maxPixelValue = (int)params[14];
	float bloomAmount = params[13]; 

	if (bloomValues[pos]) {
		float bloomScale = bloomValues[pos] * (float)bloomAmount / 1000.0f;

		int bufferPos = 4*pos;
		output[bufferPos] = bloomScale * maxPixelValue + (1.0f - bloomScale) * output[bufferPos];
		output[bufferPos+1] = bloomScale * maxPixelValue + (1.0f - bloomScale) * output[bufferPos+1];
		output[bufferPos+2] = bloomScale * maxPixelValue + (1.0f - bloomScale) * output[bufferPos+2];
		if (output[bufferPos] > maxPixelValue) output[bufferPos] = maxPixelValue;
		if (output[bufferPos+1] > maxPixelValue) output[bufferPos+1] = maxPixelValue;
		if (output[bufferPos+2] > maxPixelValue) output[bufferPos+2] = maxPixelValue;
	}
}

//-----------------------------------------------------------------
__device__ static void accumulateBloomingGPU(float* image, float newWidth, int kernelWidth, float* kernels, float* params, float zDepth, int pos, float* bloomValues) {
//-----------------------------------------------------------------
	// scene parameters
	int threshold = (int)params[12]; 
	int bloomAmount = (int)params[13]; 
	int mainKernelWidth = (int)params[15];
	float interpolation = params[19];

	int bufferPos = 4*pos;
	float avg = (image[bufferPos] + image[bufferPos+1] + image[bufferPos+2]) / 3.0f;
	// calculation
	if (bloomAmount > 0 && avg > threshold) {
		int kernelHalfwidth = kernelWidth / 2;
		
		int width = (int)params[0];
		int height = (int)params[1];
		int imageX = pos % width;
		int imageY = pos / width;
		int topLeftX = imageX - kernelHalfwidth;
		int topLeftY = imageY - kernelHalfwidth;

		int smallOffset = 0;
		int bigOffset = 0;
		int smallSize = 0;
		float bigScale = 0.0f;

		if (interpolation) {
			if (newWidth > 1) {
				int smallWidth = getKernelWidth((int)newWidth,mainKernelWidth);
				bigScale = (newWidth - (float)smallWidth) / 2.0f;

				kernelWidth = getKernelWidth(smallWidth+2,mainKernelWidth);

				smallOffset = 0;
				for(int i=1;i<smallWidth;i+=2) {
					smallOffset += i*i;
				}
				smallSize = smallWidth*smallWidth;
				bigOffset = smallWidth == kernelWidth ? smallOffset : smallOffset + smallSize;
			} else {
				smallOffset = -1.0f;
			}
		}
		int kernelCoord = -1;
		if (!interpolation) {
			for(int i=1;i<kernelWidth;i+=2) {
				kernelCoord += i*i;
			}
		}
		for(int y=0;y<kernelWidth;y++) {
			for(int x=0;x<kernelWidth;x++) {
				kernelCoord++;
				//int kernelCoord = x + y * kernelWidth;
				float kernelValue = interpolation ? getKernelValue(y,kernelCoord,smallOffset,bigOffset,bigScale,kernels,smallSize,kernelWidth) : kernels[kernelCoord];

				if (kernelValue != 0.0f) {
					int localX = (topLeftX+x);
					int localY = (topLeftY+y);

					// skip the pixels from the other size of the image
					if (0 <= localX && 0 <= localY && localX < width && localY < height) {
						int localCoord = localX + width*localY;

						float distance = (abs(localX-imageX)*abs(localX-imageX) + abs(localY-imageY)*abs(localY-imageY)) / 1.2f;
						float r = (float)(kernelHalfwidth*kernelHalfwidth + kernelHalfwidth*kernelHalfwidth) / 4.0f;
						//float multiplicator = (1.0f - distance / r) * (recalcZDepth[pos] ? recalcZDepth[pos] : imagePixels[pos].zDepth);
						float multiplicator = (1.0f - distance / r) * zDepth;
						if (multiplicator > 0) {
							bloomValues[localCoord] += multiplicator;
						}
					}
				}
			}
		}
	}
}

//-----------------------------------------------------------------//
//                        REACCUMULATION
//-----------------------------------------------------------------//

//-----------------------------------------------------------------
__device__ static float getOverlapScaleGPU(float overlap,float value) {
//-----------------------------------------------------------------
	if (overlap == 0.0f) {
		return 0.0f;
	}
	if (overlap < 1.0f) {
		if (value > (1.0f-overlap)) {
			return (value - (1.0f - overlap)) /overlap;
		} else {
			return 0;
		}
	} 
	if (overlap > 1.0f) {
		float rec = 1.0f / overlap;
		return rec * value + (1.0f - rec);
	}
	return value;
}

//-----------------------------------------------------------------
__global__ static void reaccumulateGPU(float* image, float* kernels, float* params, float* recalcDist, float* recalcZDepth, float* recalcZValue,float* bloomValues, float* output) {
//-----------------------------------------------------------------
	int tid = threadIdx.x + blockDim.x * threadIdx.y;
    int bid = blockIdx.x + gridDim.x * blockIdx.y;
    int id = bid * blockDim.x*blockDim.x + tid;
    
    int width = (int)params[0];
    int height = (int)params[1];

	if (id >= width*height) {
		return;
	}

	int pos = id;
    
	// scene parameters
    int mainKernelWidth = (int)params[15]; 
	float overlap = params[17];
	float interpolation = params[19];

	// calculation

	//int zValue = recalcZValue[pos];

	//float newWidth = strength * recalcZDepth[pos] * (float)mainKernelWidth;
	float newWidth = calculateKernelWidth(recalcZDepth[pos],recalcZValue[pos],params);
	int kernelWidth = getKernelWidth((int)newWidth + 2, mainKernelWidth);

	// do not do anything when no kernel is defined
	if (kernelWidth < 3) {
		return;
	}

	int smallOffset = 0;
	int bigOffset = 0;
	int smallSize = 0;
	float bigScale = 0.0f;

	if (interpolation) {
		if (newWidth > 1) {
			int smallWidth = getKernelWidth((int)newWidth,mainKernelWidth);
			bigScale = (newWidth - (float)smallWidth) / 2.0f;

			kernelWidth = getKernelWidth(smallWidth+2,mainKernelWidth);
	
			smallOffset = 0;
			for(int i=1;i<smallWidth;i+=2) {
				smallOffset += i*i;
			}
			smallSize = smallWidth*smallWidth;
			bigOffset = smallWidth == kernelWidth ? smallOffset : smallOffset + smallSize;
		} else {
			smallOffset = -1.0f;
		}
	}

	int kernelHalfwidth = kernelWidth / 2;
	
	int imageX = pos % width;
	int imageY = pos / width;
	int topLeftX = imageX - kernelHalfwidth;
	int topLeftY = imageY - kernelHalfwidth;

	float newR=0.0f, newG=0.0f, newB=0.0f, newLum=0.0f;

	int kernelCoord = -1;
	if (!interpolation) {
		for(int i=1;i<kernelWidth;i+=2) {
			kernelCoord += i*i;
		}
	}
	for(int y=0;y<kernelWidth;y++) {
		for(int x=0;x<kernelWidth;x++) {
			kernelCoord++;
			//int kernelCoord = x + y * kernelWidth;
			float kernelValue = interpolation ? getKernelValue(y,kernelCoord,smallOffset,bigOffset,bigScale,kernels,smallSize,kernelWidth) : kernels[kernelCoord];

			if (kernelValue != 0.0f) {
				int localX = (topLeftX+x);
				int localY = (topLeftY+y);

				// skip the pixels from the other size of the image
				if (0 <= localX && 0 <= localY && localX < width && localY < height) {
					int localCoord = localX + width*localY;
					int localBufferPos = 4*localCoord;
					float localR = image[localBufferPos];
					float localG = image[localBufferPos+1];
					float localB = image[localBufferPos+2];
					float localLum = image[localBufferPos+3];

					// accumulate pixel color
					float multiplicator = kernelValue * localLum;
					newR += localR * multiplicator;
					newG += localG * multiplicator;
					newB += localB * multiplicator;
					newLum += multiplicator;
				}
			}
		}
	}

	float scale = getOverlapScaleGPU(overlap,recalcDist[pos]);
	newR /= newLum;
	newG /= newLum;
	newB /= newLum;
	int bufferPos = 4*pos;
	output[bufferPos] = output[bufferPos]*(1.0f-scale) + newR*scale;
	output[bufferPos+1] = output[bufferPos+1]*(1.0f-scale) + newG*scale;
	output[bufferPos+2] = output[bufferPos+2]*(1.0f-scale) + newB*scale;

	accumulateBloomingGPU(image,newWidth,kernelWidth,kernels,params,recalcZDepth[pos],pos,bloomValues);
}

//-----------------------------------------------------------------
__device__ static void reaccumulateBackgroundGPU(float* image, float* zMap, float* kernels, float* params, float zDepth, int zValue, float distScale, int pos, int group, float* output) {
//-----------------------------------------------------------------    
	// scene parameters
    float maxPixelValue = (int)params[14];
    int mainKernelWidth = (int)params[15]; 
	float overlap = params[17];
	float interpolation = params[19];

	// calculation
	//float newWidth = strength * zDepth * (float)mainKernelWidth;
	float newWidth = calculateKernelWidth(zDepth,zValue,params);
	int kernelWidth = getKernelWidth((int)newWidth + 2, mainKernelWidth);

	// do not do anything when no kernel is defined
	if (kernelWidth < 3) {
		return;
	}

	int smallOffset = 0;
	int bigOffset = 0;
	int smallSize = 0;
	float bigScale = 0.0f;

	if (interpolation) {
		if (newWidth > 1) {
			int smallWidth = getKernelWidth((int)newWidth,mainKernelWidth);
			bigScale = (newWidth - (float)smallWidth) / 2.0f;

			kernelWidth = getKernelWidth(smallWidth+2,mainKernelWidth);
	
			smallOffset = 0;
			for(int i=1;i<smallWidth;i+=2) {
				smallOffset += i*i;
			}
			smallSize = smallWidth*smallWidth;
			bigOffset = smallWidth == kernelWidth ? smallOffset : smallOffset + smallSize;
		} else {
			smallOffset = -1.0f;
		}
	}

	int kernelHalfwidth = kernelWidth / 2;
	
    int width = (int)params[0];
    int height = (int)params[1];
	int imageX = pos % width;
	int imageY = pos / width;
	int topLeftX = imageX - kernelHalfwidth;
	int topLeftY = imageY - kernelHalfwidth;

	float newR=0.0f, newG=0.0f, newB=0.0f, newLum=0.0f;

	int kernelCoord = -1;
	if (!interpolation) {
		for(int i=1;i<kernelWidth;i+=2) {
			kernelCoord += i*i;
		}
	}
	for(int y=0;y<kernelWidth;y++) {
		for(int x=0;x<kernelWidth;x++) {

			kernelCoord++;
			//int kernelCoord = x + y * kernelWidth;
			float kernelValue = interpolation ? getKernelValue(y,kernelCoord,smallOffset,bigOffset,bigScale,kernels,smallSize,kernelWidth) : kernels[kernelCoord];

			if (kernelValue != 0.0f) {
				int localX = (topLeftX+x);
				int localY = (topLeftY+y);

				// skip the pixels from the other size of the image
				if (0 <= localX && 0 <= localY && localX < width && localY < height) {
					int localCoord = localX + width*localY;
					int localZValue = (int)zMap[localCoord];
//					float localZDepth = (float)(abs(focusValue - localZValue)) / maxPixelValue;	
					int localGroup = localZValue * PIXEL_GROUPS / ((int)maxPixelValue+1);

					// TODO - could we use localZValue < zValue?
					// skip pixels from the closer layers
					if (localGroup <= group) {
						int localBufferPos = 4*localCoord;
						float localR = image[localBufferPos];
						float localG = image[localBufferPos+1];
						float localB = image[localBufferPos+2];
						float localLum = image[localBufferPos+3];

						// accumulate pixel color
						float multiplicator = kernelValue * localLum;
						newR += localR * multiplicator;
						newG += localG * multiplicator;
						newB += localB * multiplicator;
						newLum += multiplicator;
					}
				}
			}
		}
	}

	if (newLum > 0) {
		float scale = getOverlapScaleGPU(overlap,distScale);
		newR /= newLum;
		newG /= newLum;
		newB /= newLum;
		int bufferPos = 4*pos;
		output[bufferPos] = output[bufferPos]*(1.0f-scale) + newR*scale;
		output[bufferPos+1] = output[bufferPos+1]*(1.0f-scale) + newG*scale;
		output[bufferPos+2] = output[bufferPos+2]*(1.0f-scale) + newB*scale;
	}
}

//-----------------------------------------------------------------//
//                          ACCUMULATION
//-----------------------------------------------------------------//

//-----------------------------------------------------------------
__global__ static void accumulateGPU(float* image, float* zMap, float* kernels, float* params, float* output, float* recalcDist, float* recalcZDepth, float* recalcZValue,float* bloomValues) {
//-----------------------------------------------------------------
	int tid = threadIdx.x + blockDim.x * threadIdx.y;
    int bid = blockIdx.x + gridDim.x * blockIdx.y;
    int id = bid * blockDim.x*blockDim.y + tid;
    
    int width = (int)params[0];
    int height = (int)params[1];
	float interpolation = params[19];

	if (id >= width*height) {
		return;
	}

	int pos = id;
    int bufferPos = 4*pos;
    
    output[bufferPos] = 0.0f;
	output[bufferPos+1] = 0.0f;
	output[bufferPos+2] = 0.0f;
	output[bufferPos+3] = 0.0f;

	// pixel groups for the background recalculation
	int	pixelCategoriesNum[PIXEL_GROUPS];
	float pixelCategoriesDist[PIXEL_GROUPS];
	float pixelCategoriesZDepth[PIXEL_GROUPS];
	float pixelCategoriesZValue[PIXEL_GROUPS];
	for(int i=0;i<PIXEL_GROUPS;i++) {
		pixelCategoriesNum[i] = 0;
		pixelCategoriesDist[i] = 0.0f;
		pixelCategoriesZDepth[i] = (float)INT_MAX;
		pixelCategoriesZValue[i] = 0;
	}
	
	// scene parameters
    int focusValue = (int)params[2];
    float maxPixelValue = (int)params[14];
    float strength = params[5];
    
    int zEpsilon = (int)params[3];
    int samplingRadius = (int)params[4];
     
    int mainKernelWidth = (int)params[15];
    
	int imageX = pos % width;
	int imageY = pos / width;

	int zValue = (int)zMap[pos];
	float zDepth = (float)(abs(focusValue - zValue)) / maxPixelValue;	

	// Artist Mode
	//float newWidth = strength * zDepth * (float)mainKernelWidth;
	float newWidth = calculateKernelWidth(zDepth,zValue,params);
	int kernelWidth = getKernelWidth((int)newWidth + 2, mainKernelWidth);

	// use default color when no kernel is defined
	if (kernelWidth < 3) {
		output[bufferPos] = image[bufferPos];
		output[bufferPos+1] = image[bufferPos+1];
		output[bufferPos+2] = image[bufferPos+2];
		output[bufferPos+3] = 1.0f;
		return;
	}

	int smallOffset = 0;
	int bigOffset = 0;
	int smallSize = 0;
	float bigScale = 0.0f;

	if (interpolation) {
		if (newWidth > 1) {
			int smallWidth = getKernelWidth((int)newWidth,mainKernelWidth);
			bigScale = (newWidth - (float)smallWidth) / 2.0f;

			kernelWidth = getKernelWidth(smallWidth+2,mainKernelWidth);

			smallOffset = 0;
			for(int i=1;i<smallWidth;i+=2) {
				smallOffset += i*i;
			}
			smallSize = smallWidth*smallWidth;
			bigOffset = smallWidth == kernelWidth ? smallOffset : smallOffset + smallSize;
		} else {
			smallOffset = -1.0f;
		}
	}

	int kernelHalfwidth = kernelWidth / 2;
	
	int topLeftX = imageX - kernelHalfwidth;
	int topLeftY = imageY - kernelHalfwidth;

	int kernelCoord = -1;
	// no interpolation
	if (!interpolation) {
		for(int i=1;i<kernelWidth;i+=2) {
			kernelCoord += i*i;
		}
	}
	for(int y=0;y<kernelWidth;y++) {
		for(int x=0;x<kernelWidth;x++) {
			kernelCoord++;
			//int kernelCoord = x + y * kernelWidth;
			float kernelValue = interpolation ? getKernelValue(y,kernelCoord,smallOffset,bigOffset,bigScale,kernels,smallSize,kernelWidth) : kernels[kernelCoord];

			if (kernelValue != 0.0f) {
				int localX = (topLeftX+x);
				int localY = (topLeftY+y);

				// skip the pixels from the other size of the image
				if (0 <= localX && 0 <= localY && localX < width && localY < height) {
					int localCoord = localX + width*localY;
					int localZValue = (int)zMap[localCoord];
					float localZDepth = (float)(abs(focusValue - localZValue)) / maxPixelValue;	

					if (abs(zValue - localZValue) > zEpsilon) {
						// calculating background
						if (zValue < localZValue) {
							if (localZDepth > 0.1f && localZValue < focusValue) {
								int dist = abs(imageX-localX)*abs(imageX-localX) + abs(imageY-localY)*abs(imageY-localY);
								// original formula: 1.0f - 0.028125f*(float)kernel.width* ((float)dist / (strength * localPixel->zDepth*(float)kernel.halfwidth*localPixel->zDepth*(float)kernel.halfwidth));
								float distScale = 1.0f - 0.1125f* ((float)dist / (strength * localZDepth*localZDepth*(float)kernelWidth));

								// get pixel group
								int group = localZValue * PIXEL_GROUPS / ((int)maxPixelValue+1);

								pixelCategoriesNum[group]++;
								if (distScale > pixelCategoriesDist[group]) {
									pixelCategoriesDist[group] = distScale;
								}
								if (localZDepth < pixelCategoriesZDepth[group]) {
									pixelCategoriesZDepth[group] = localZDepth;
									pixelCategoriesZValue[group] = localZValue;
								}
							}

							if (abs(localX - imageX) > samplingRadius || abs(localY - imageY) > samplingRadius) {
								continue;
							}
						}

						// calculating foreground
						else if (zValue > localZValue) {
							// background is in focus
							if (zValue > focusValue) {
								float dist = (float)(abs(imageX - localX)*abs(imageX - localX) + abs(imageY - localY)*abs(imageY - localY));
								float r = (float)(kernelHalfwidth*kernelHalfwidth + kernelHalfwidth*kernelHalfwidth) / 4.0f;
								float distScale = 1.0f - dist / r;
								if (distScale > recalcDist[localCoord]) {
									recalcDist[localCoord] = distScale;
								}
								if (!recalcZDepth[localCoord] || zDepth < recalcZDepth[localCoord]) {
									recalcZDepth[localCoord] = zDepth;
									recalcZValue[localCoord] = zValue;
								}
							}
						}
					}

					int localBufferPos = 4*localCoord;
					float localR = image[localBufferPos];
					float localG = image[localBufferPos+1];
					float localB = image[localBufferPos+2];
					float localLum = image[localBufferPos+3];

					// accumulate pixel color
					float multiplicator = kernelValue * localLum;
					output[bufferPos] += localR * multiplicator;
					output[bufferPos+1] += localG * multiplicator;
					output[bufferPos+2] += localB * multiplicator;
					output[bufferPos+3] += multiplicator;
				}
			}
		} // for x
	} // for y

	// final pixel color
	if (output[bufferPos+3] > 0) {
		output[bufferPos] /= output[bufferPos+3];
		output[bufferPos+1] /= output[bufferPos+3];
		output[bufferPos+2] /= output[bufferPos+3];
	}

	// background recalculation
	for(int i=0;i<PIXEL_GROUPS;i++) {
		if (pixelCategoriesDist[i] > 0.05f && pixelCategoriesNum[i] > kernelWidth*kernelWidth / 40) {
			reaccumulateBackgroundGPU(image,zMap,kernels,params,pixelCategoriesZDepth[i],pixelCategoriesZValue[i],pixelCategoriesDist[i],pos,i,output);
		}
	}

	accumulateBloomingGPU(image,newWidth,kernelWidth,kernels,params,recalcZDepth[pos] ? recalcZDepth[pos] : zDepth,pos,bloomValues);
}

#endif // _LENSBLUR_KERNEL_
