/* ******************************************************************************
* Description: Aperture kernel. Kernel is a mask used in dof calculation by a pixel.
*              The pixel's color is defined from the neighbor pixels. Kernel is 
*              a two dimensional matrix which says the weight how a neighbor pixel
*              takes part in the calculation.
*
*  Version 1.0.0
*  Date: Nov 22, 2008
*  Author: David Illes, Peter Horvath
*          www.postpipe.hu
*
* GPUPro
***************************************************************************** */

#ifndef _APKERNEL_
#define _APKERNEL_

#include <stdio.h>
#include <math.h>
#include "image.hpp"

static const float defaultKernel[] = { 
	0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.388f, 0.808f, 0.431f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f,
	0.000f, 0.000f, 0.000f, 0.000f, 0.255f, 0.761f, 1.000f, 1.000f, 1.000f, 0.792f, 0.282f, 0.000f, 0.000f, 0.000f, 0.000f,
	0.000f, 0.000f, 0.129f, 0.612f, 0.966f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.647f, 0.153f, 0.000f, 0.000f,
	0.000f, 0.455f, 0.922f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.941f, 0.494f, 0.000f,
	0.000f, 0.957f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.976f, 0.000f,
	0.000f, 0.910f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.929f, 0.000f,
	0.000f, 0.910f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.929f, 0.000f,
	0.000f, 0.910f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.929f, 0.000f,
	0.000f, 0.910f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.929f, 0.000f,
	0.000f, 0.910f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.929f, 0.000f,
	0.000f, 0.957f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.976f, 0.000f,
	0.000f, 0.455f, 0.922f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.941f, 0.494f, 0.000f,
	0.000f, 0.000f, 0.129f, 0.612f, 0.966f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 1.000f, 0.647f, 0.153f, 0.000f, 0.000f,
	0.000f, 0.000f, 0.000f, 0.000f, 0.255f, 0.761f, 1.000f, 1.000f, 1.000f, 0.792f, 0.282f, 0.000f, 0.000f, 0.000f, 0.000f,
	0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.388f, 0.808f, 0.431f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000	
};

//===============================================================
class ApKernel {
//===============================================================
public:
	int		type;
	int		width;
	int		height;
	int		size;
	int		halfwidth;
	int		halfheight;
	float	r;

	float*	values;

	ApKernel*	parent;
	ApKernel*** cachedKernels;

//-----------------------------------------------------------------
// Summary: Copies an aperture kernel.
// Arguments: p - kernel to copy
// Returns: a new kernel instance with the same values as the source.
//-----------------------------------------------------------------
inline ApKernel operator=(const ApKernel& p) const {
//-----------------------------------------------------------------
	ApKernel newKernel(p.width,p.height);
	for(int i=0;i<p.size;i++) {
		newKernel.values[i] = p.values[i];
	}
	return newKernel;
}

//-----------------------------------------------------------------
// Summary: Prints the kernel values to the standard output.
//          (development function)
//-----------------------------------------------------------------
inline void printValues() const {
//-----------------------------------------------------------------
	for(int y=height-1;y>=0;y--) {
		for(int x=width-1;x>=0;x--) {
			printf("%0.4f ",values[x+y*width]);
		}
		printf("\n");
	}
}
	
//-----------------------------------------------------------------
// Summary: Constructs a new kernel.
// Arguments: type - type of the kernel (see defaultKernels.hpp)
//-----------------------------------------------------------------
ApKernel(int type=0) {
//-----------------------------------------------------------------
	this->type = type;
	init(15,15, true);
	for(int i=0;i<225;i++)
		values[i] = defaultKernel[i];
}

//-----------------------------------------------------------------
// Summary: Frees memory. Removes all sub-kernels (see initKernels()).
//-----------------------------------------------------------------
~ApKernel() {
//-----------------------------------------------------------------
	destructKernels();
	delete [] values;
}

//-----------------------------------------------------------------
// Summary: Constructs a kernel.
// Arguments: width - kernel width
//            height - kernel height
//            withCache - create cache for sub-kernels or not (see initKernels())
//            type - type of the kernel (see defaultKernels.hpp)
//-----------------------------------------------------------------
ApKernel(int width, int height, bool withCache = false,int type=0) {
//-----------------------------------------------------------------
	this->type = type;
	init(width,height,withCache);
}

//-----------------------------------------------------------------
// Summary: Loads kernel values from an image.
// Arguments: iris - grey-scale kernel image
//-----------------------------------------------------------------
void init(Image* iris) {
//-----------------------------------------------------------------
	init(iris->getWidth(),iris->getHeight(),true);
	for(int i=0;i<size;i++) {
		values[i] = (float)iris->pixels[i].r / 255.0f;
	}

	//printValues();
}

//-----------------------------------------------------------------
// Summary: Initializes the kernel.
// Arguments: width - kernel width
//            height - kernel height
//            withCache - create cache for sub-kernels or not (see initKernels())
//-----------------------------------------------------------------
void init(int width, int height, bool withCache = false) {
//-----------------------------------------------------------------
	this->width = width;
	this->height = height;
	this->size = width*height;
	this->halfwidth = width / 2;
	this->halfheight = height / 2;
	this->r = (float)(halfwidth*halfwidth + halfheight*halfheight) / 4.0f;

	values = new float[size];

	parent = 0;

	if (withCache) {
		initKernels();
	} else {
		cachedKernels = 0;
	}
}

//-----------------------------------------------------------------
// Summary: During the calculation we are using sub-kernels transformed from
//          the original kernel by the z depth value. To faster running
//          the sub-kernels are computed only once and stored in a cache array.
//          Only the original kernel has sub-kernels, a sub-kernel does not
//          need cache. We can tell in the constuctor that we need cache or not.
//-----------------------------------------------------------------
void initKernels() {
//-----------------------------------------------------------------
	cachedKernels = new ApKernel**[width+1];
	for(register int x=0;x<=width;x++) {
		cachedKernels[x] = new ApKernel*[height+1];
		for(register int y=0;y<=height;y++) {
			cachedKernels[x][y] = 0;
		}
	}
	cachedKernels[width][height] = this;
}

//-----------------------------------------------------------------
// Summary: Removes the sub-kernels and the cache from the memory.
//-----------------------------------------------------------------
void destructKernels() {
//-----------------------------------------------------------------
	if (cachedKernels) {
		for(register int x=0;x<=width;x++) {
			for(register int y=0;y<=height;y++) {
				// do not delete the main kernel
				if (cachedKernels[x][y] && (x<width || y<height)) {
					delete cachedKernels[x][y];
				}
			}
			delete [] cachedKernels[x];
		}
		delete [] cachedKernels;
	}
}

//-----------------------------------------------------------------
// Summary: Returns the required sub-kernel from the cache. When it is
//          not found in the cache, calcualtes it and adds to the cache.
// Arguments: scale - scale value (sub-kernel width / original kernel width)
// Returns: Sub-kernel transformed from the original.
//-----------------------------------------------------------------
ApKernel* getKernel(float scale) {
//-----------------------------------------------------------------
	int newWidth = (int)(width*scale);
	int newHeight = (int)(height*scale);

	return getKernel(newWidth, newHeight);
}

//-----------------------------------------------------------------
// Summary: Returns the required sub-kernel from the cache. When it is
//          not found in the cache, calcualtes it and adds to the cache.
// Arguments: newWidth - sub-kernel width
//            newHeight - sub-kernel height
// Returns: Sub-kernel transformed from the original.
//-----------------------------------------------------------------
ApKernel* getKernel(float newWidth, float newHeight) {
//-----------------------------------------------------------------
	return getKernel((int)newWidth,(int)newHeight);
}

//-----------------------------------------------------------------
// Summary: Returns the required sub-kernel from the cache. When it is
//          not found in the cache, calcualtes it and adds to the cache.
// Arguments: newWidth - sub-kernel width
//            newHeight - sub-kernel height
// Returns: Sub-kernel transformed from the original.
//-----------------------------------------------------------------
ApKernel* getKernel(int newWidth, int newHeight) {
//-----------------------------------------------------------------
	if (parent) return parent->getKernel(newWidth,newHeight);

	if (0 > newWidth) newWidth = 0;
	else if (newWidth > width) newWidth = width;
	if (0 > newHeight) newHeight = 0;
	else if (newHeight > height) newHeight = height;

	if (newWidth && newWidth % 2 == 0) {
		newWidth--;
	}
	if (newHeight && newHeight % 2 == 0) {
		newHeight--;
	}

	// check cached kernels
	if (!cachedKernels[newWidth][newHeight]) {
		ApKernel* newKernel = transform(newWidth,newHeight);
		cachedKernels[newWidth][newHeight] = newKernel;
		newKernel->parent = this;
	}

	return cachedKernels[newWidth][newHeight];
}

//-----------------------------------------------------------------
// Summary: For smooth result in the animation we need to use interpolated
//          kernel values. Kernels are matrices with integer width and height.
//          For kernel interpolation we take the closest lower and higher 
//          sub-kernels and calculate values with linear interpolation.
// Arguments: newWidth - float sub-kernel width
//            newHeight - float sub-kernel height
// Returns: Interpolated sub-kernel.
//-----------------------------------------------------------------
ApKernel* getInterpolatedKernel(float newWidth, float newHeight) {
//-----------------------------------------------------------------
	ApKernel* smallKernel = getKernel((int)newWidth,(int)newHeight);
	if (!smallKernel) {
		return 0;
	}
	ApKernel* bigKernel = getKernel((int)newWidth+2,(int)newHeight+2);
	if (bigKernel->getWidth() == smallKernel->getWidth()) {
		ApKernel* kernel = new ApKernel(bigKernel->getWidth(),bigKernel->getHeight());
		for(int i=0;i<kernel->getSize();kernel->values[i]=bigKernel->values[i++]);
		return kernel;
	}

	ApKernel* kernel = new ApKernel(bigKernel->getWidth(),bigKernel->getHeight());
	for(int i=0;i<kernel->getSize();kernel->values[i++]=-1);

	float bigScale = (newWidth - (float)smallKernel->getWidth()) / 2.0f;
	float smallScale = 1.0f - bigScale;

	// x x x x x         x: big
	// x o o o x         o: small
	// x o o o x
	// x o o o x
	// x x x x x

	for(int smallCoord=0;smallCoord<smallKernel->getSize();smallCoord++) {
		int row = (int)(smallCoord/smallKernel->getWidth());
		int bigCoord = smallCoord + bigKernel->getWidth() + 2*row + 1;
		kernel->values[bigCoord] = smallScale * smallKernel->values[smallCoord] + bigScale * bigKernel->values[bigCoord];
	}
	for(int i=0;i<kernel->getSize();i++) {
		if (kernel->values[i] < 0) {
			kernel->values[i] = bigScale * bigKernel->values[i];
		}
	}

	return kernel;
}

//-----------------------------------------------------------------
// Summary: For smooth result in the animation we need to use interpolated
//          kernel values. Kernels are matrices with integer width and height.
//          For kernel interpolation we take the closest lower and higher 
//          sub-kernels and calculate values with linear interpolation.
// Arguments: scale - scale value (sub-kernel width / original kernel width)
// Returns: Interpolated sub-kernel.
//-----------------------------------------------------------------
ApKernel* getInterpolatedKernel(float scale) {
//-----------------------------------------------------------------
	float newWidth = ((float)width)*scale;
	float newHeight = ((float)height)*scale;

	return getInterpolatedKernel(newWidth, newHeight);
}

//-----------------------------------------------------------------
// Returns: width of the kernel matrix.
//-----------------------------------------------------------------
int getWidth() {
//-----------------------------------------------------------------
	return width;
}

//-----------------------------------------------------------------
// Returns: height of the kernel matrix.
//-----------------------------------------------------------------
int getHeight() {
//-----------------------------------------------------------------
	return height;
}

//-----------------------------------------------------------------
// Returns: size (width*height) of the kernel matrix.
//-----------------------------------------------------------------
int getSize() {
//-----------------------------------------------------------------
	return size;
}

//-----------------------------------------------------------------
// Summary: Calculates sub-kernel values from the original values
//          with linear transformation. Transformes the original
//          matrix indices along the x and y axis. So we translate
//          the original matrix coordinates to sub-kernel matrix
//          coordinates. Where more values belong to a sub-kernel value
//          we take the average.
// Arguments: scale - scale value (sub-kernel width / original kernel width)
// Returns: Sub-kernel transformed from the original.
//-----------------------------------------------------------------
ApKernel* transform(float scale) {
//-----------------------------------------------------------------
	return transform((int)(width*scale),(int)(height*scale));
}

//-----------------------------------------------------------------
// Summary: Calculates sub-kernel values from the original values
//          with linear transformation. Transformes the original
//          matrix indices along the x and y axis. So we translate
//          the original matrix coordinates to sub-kernel matrix
//          coordinates. Where more values belong to a sub-kernel value
//          we take the average.
// Arguments: newWidth - sub-kernel matrix width
//            newHeight - sub-kernel matrix height
// Returns: Sub-kernel transformed from the original.
//-----------------------------------------------------------------
ApKernel* transform(int newWidth, int newHeight) {
//-----------------------------------------------------------------
	ApKernel* newKernel = this;
	
	if (newWidth == 0 && newHeight == 0) {
		newKernel = new ApKernel(1,1);
		newKernel->values[0] = 1.0f;
		return newKernel;
	}

	// transform when size changed
	if (newWidth != width || newHeight != height) {
		// calculate scale from the new width
		float scale = (float)(newWidth-1)/(width-1);

		newKernel = new ApKernel(newWidth,newHeight);

		// sum of kernel values
		float* sum = new float[newKernel->getSize()];
		for(int i=0;i<newKernel->getSize();sum[i++]=0);
		// number of values from the original kernel
		float* count = new float[newKernel->getSize()];
		for(int i=0;i<newKernel->getSize();count[i++]=0);

		for(int y=0;y<height;y++) {
			for(int x=0;x<width;x++) {
				int kernelCoord = x + y*width;

				// floor(a + 0.5) means round(a)
				int newX = (int)floor((float)x*scale+0.5);
				int newY = (int)floor((float)y*scale+0.5);
				int newCoord = newX + newY*newWidth;

				sum[newCoord] += values[kernelCoord];
				count[newCoord]++;
			}
		}

		for(int i=0;i<newKernel->getSize();i++) {
			if (count[i] > 0) {
				newKernel->values[i] = sum[i] / count[i];
			}
		}

		// verification
//		newKernel->printValues();
	}

	return newKernel;
}

float getR() {
	return r;
}

};

#endif
