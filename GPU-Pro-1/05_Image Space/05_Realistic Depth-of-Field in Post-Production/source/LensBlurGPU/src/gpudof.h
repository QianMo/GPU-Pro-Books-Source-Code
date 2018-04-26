/* ******************************************************************************
* Description: Library for GPU access.
*
*  Version 1.0.0
*  Date: Sep 19, 2009
*  Author: David Illes, Peter Horvath
*          www.postpipe.hu
*
* GPUPro
***************************************************************************** */

#ifndef _GPUDOF_
#define _GPUDOF_

#include <stdio.h>
#include <stdlib.h>

#include "image.hpp"
#include "apKernel.hpp"
#include "parameters.hpp"

#include <cuda.h>
#include <cutil.h>

/*
//===============================================================
class GPUDof {
//===============================================================
public:
	char** getDeviceNames();
};
*/

extern "C" int gpuGetDeviceCount();
extern "C" char** gpuGetDeviceNames();

extern "C" void gpuInit(Image& image, Image& zMap, ApKernel* kernel, Parameters* parameters);
extern "C" void gpuParamsToGPU(int kernelWidth, Parameters* parameters);
extern "C" void gpuUninit();

extern "C" void gpuAccumulate(int numPixels, ApKernel* kernel, Parameters* parameters);
extern "C" float* gpuReaccumulate(int numPixels, ApKernel* kernel, Parameters* parameters);
extern "C" float* gpuGetBloomValues(int numPixels);

#endif
