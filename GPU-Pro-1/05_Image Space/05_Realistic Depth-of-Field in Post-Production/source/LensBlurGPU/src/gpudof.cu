#include "gpudof.h"

#include "lensblur.cu"

float* imageGPU;
float* zMapGPU;
float* kernelsGPU;
float* paramsGPU;

float* outputGPU;
float* recalcDistGPU;
float* recalcZDepthGPU;
float* recalcZValueGPU;
float* bloomValuesGPU;

//-----------------------------------------------------------------
// Returns: number of available GPU devices.
//-----------------------------------------------------------------
extern "C" int gpuGetDeviceCount() {
//-----------------------------------------------------------------
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	return deviceCount;
}

//-----------------------------------------------------------------
// Returns: names of the available GPU devices.
//-----------------------------------------------------------------
extern "C" char** gpuGetDeviceNames() {
//-----------------------------------------------------------------
	int deviceCount = 0;
	char** deviceNames = 0;

	cudaGetDeviceCount(&deviceCount);
	deviceNames = new char*[deviceCount+1];
	deviceNames[deviceCount] = 0;
	// read device names
	for(int i=0;i<deviceCount;i++) {
		deviceNames[i] = new char[64];
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		//CUdevice device;
		//cuDeviceGet(&device,i);
		//cudaGetDeviceName(deviceNames[i],64,&device);
		strcpy_s(deviceNames[i],64,deviceProp.name);
	}

	return deviceNames;
}

//-----------------------------------------------------------------
// Summary: initializes the GPU device with the image data and DoF parameters.
//          Loads the source image pixels, the z map and kernel values and 
//          the parameters to the GPU device.
// Arguments: image - source image
//            z map - z map image
//            kernel - kernel mask
//            parameters - DoF parameters
//-----------------------------------------------------------------
extern "C" void gpuInit(Image& image, Image& zMap, ApKernel* kernel, Parameters* parameters) {
//-----------------------------------------------------------------
//	int argc = 1;
	char* argv[1];
	argv[0] = "LensBlur v0.1";
/*    if ( cutCheckCmdLineFlag(argc, (const char **)argv, "device")) {
		cutilDeviceInit (argc, argv);
    } else {
        cudaSetDevice (cutGetMaxGflopsDeviceId() );
    }
*/	CUT_DEVICE_INIT(1, argv);

	// image
	float* imageData = new float[4*image.getSize()];
	for(int i=0;i<image.getSize();i++) {
		imageData[4*i] = image.pixels[i].r;
		imageData[4*i+1] = image.pixels[i].g;
		imageData[4*i+2] = image.pixels[i].b;
		imageData[4*i+3] = image.pixels[i].lum;
	}
	// z map
	float* zMapData = new float[zMap.getSize()];
	for(int i=0;i<zMap.getSize();i++) {
		zMapData[i] = zMap.pixels[i].r;
	}
	// transform kernels
	int kernelBufferSize = 0;
	for(int i=1;i<=kernel->getWidth();i+=2) {
		kernelBufferSize += i*i;
	}
	float* kernels = new float[kernelBufferSize];
	int actualPos = 0;
	for(int i=1;i<=kernel->getWidth();i+=2) {
		ApKernel* transformed = kernel->getKernel(i,i);
		for(int j=0;j<transformed->getSize();j++) {
			kernels[actualPos++] = transformed->values[j];
		}
	}
	// parameters
	CUDA_SAFE_CALL(cudaMalloc((void**)&paramsGPU, sizeof(float) * 25));
	gpuParamsToGPU(kernel->getWidth(),parameters);

	CUDA_SAFE_CALL(cudaMalloc((void**)&imageGPU, sizeof(float) * 4 * image.getSize()));
	CUDA_SAFE_CALL(cudaMalloc((void**)&zMapGPU, sizeof(float) * zMap.getSize()));
	CUDA_SAFE_CALL(cudaMalloc((void**)&kernelsGPU, sizeof(float) * kernelBufferSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)&outputGPU, sizeof(float) * 4 * image.getSize()));
	CUDA_SAFE_CALL(cudaMalloc((void**)&recalcDistGPU, sizeof(float) * image.getSize()));
	CUDA_SAFE_CALL(cudaMalloc((void**)&recalcZDepthGPU, sizeof(float) * image.getSize()));
	CUDA_SAFE_CALL(cudaMalloc((void**)&recalcZValueGPU, sizeof(float) * image.getSize()));
	CUDA_SAFE_CALL(cudaMalloc((void**)&bloomValuesGPU, sizeof(float) * image.getSize()));

	// load buffers to the GPU device
    CUDA_SAFE_CALL(cudaMemcpy(imageGPU, imageData, sizeof(float) * 4 * image.getSize(), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(zMapGPU, zMapData, sizeof(float) * zMap.getSize(), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(kernelsGPU, kernels, sizeof(float) * kernelBufferSize, cudaMemcpyHostToDevice));

	delete [] imageData;
	delete [] zMapData;
	delete [] kernels;
}

//-----------------------------------------------------------------
// Summary: Loads the DoF parameters to the GPU device.
// Arguments: kernelWidth - basic kernel dimension
//            parameters - DoF parameters
//-----------------------------------------------------------------
extern "C" void gpuParamsToGPU(int kernelWidth, Parameters* parameters) {
//-----------------------------------------------------------------
	// parameters
	float* params = new float[25];
	params[0] = (float)parameters->width;
	params[1] = (float)parameters->height;
	params[2] = (float)parameters->focus;
	params[3] = (float)parameters->zEpsilon;
	params[4] = (float)parameters->samplingRadius;
	params[5] = parameters->strength;
	params[8] = parameters->fLength;
	params[9] = parameters->ms;
	params[10] = parameters->fStop;
	params[11] = parameters->distance;
	params[12] = (float)parameters->threshold;
	params[13] = (float)parameters->bloomAmount;
	params[14] = (float)MAX_PIXEL;
	params[15] = (float)kernelWidth;
	params[16] = parameters->cameraDistance;
	params[17] = parameters->overlap;
	params[18] = parameters->physicsParam;
	params[19] = parameters->interpolation ? 1.0f : 0.0f;
	params[20] = (float)parameters->mode;

	CUDA_SAFE_CALL(cudaMemcpy(paramsGPU, params, sizeof(float) * 25, cudaMemcpyHostToDevice));

	delete [] params;
}

//-----------------------------------------------------------------
// Summary: frees the allocated memory on the GPU device.
//-----------------------------------------------------------------
extern "C" void gpuUninit() {
//-----------------------------------------------------------------
	CUDA_SAFE_CALL(cudaFree(imageGPU));
	CUDA_SAFE_CALL(cudaFree(zMapGPU));
    CUDA_SAFE_CALL(cudaFree(kernelsGPU));
	CUDA_SAFE_CALL(cudaFree(paramsGPU));
	CUDA_SAFE_CALL(cudaFree(outputGPU));
	CUDA_SAFE_CALL(cudaFree(recalcZDepthGPU));
	CUDA_SAFE_CALL(cudaFree(recalcZValueGPU));
	CUDA_SAFE_CALL(cudaFree(recalcDistGPU));
	CUDA_SAFE_CALL(cudaFree(bloomValuesGPU));

	cudaThreadExit();
}

//-----------------------------------------------------------------
// Summary: executes accumulation on the GPU device.
// Arguments: numPixels - number of pixels
//            kernel - kernel mask
//            parameters - DoF parameters
//-----------------------------------------------------------------
extern "C" void gpuAccumulate(int numPixels, ApKernel* kernel, Parameters* parameters) {
//-----------------------------------------------------------------
	gpuParamsToGPU(kernel->getWidth(),parameters);
	float* clearArray = new float[numPixels];
	for(int i=0;i<numPixels;clearArray[i++]=0);
	CUDA_SAFE_CALL(cudaMemcpy(recalcDistGPU, clearArray, sizeof(float) * numPixels, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(recalcZDepthGPU, clearArray, sizeof(float) * numPixels, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(recalcZValueGPU, clearArray, sizeof(float) * numPixels, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(bloomValuesGPU, clearArray, sizeof(float) * numPixels, cudaMemcpyHostToDevice));

	// calculate
	dim3 threadBlock(parameters->gpuThreads,parameters->gpuThreads);
	int blockCount = (int)(numPixels / (parameters->gpuThreads*parameters->gpuThreads)) + 1;
	accumulateGPU<<<blockCount, threadBlock>>>(imageGPU,zMapGPU,kernelsGPU,paramsGPU,outputGPU,recalcDistGPU,recalcZDepthGPU,recalcZValueGPU,bloomValuesGPU);

	// TODO - we don't need this
	//float* resultData = new float[4*numPixels];
	//CUDA_SAFE_CALL(cudaMemcpy(resultData, outputGPU, sizeof(float) * 4 * numPixels, cudaMemcpyDeviceToHost));
	//return resultData;
}

//-----------------------------------------------------------------
// Summary: executes re-accumulation on the GPU device.
// Arguments: numPixels - number of pixels
//            kernel - kernel mask
//            parameters - DoF parameters
//-----------------------------------------------------------------
extern "C" float* gpuReaccumulate(int numPixels, ApKernel* kernel, Parameters* parameters) {
//-----------------------------------------------------------------
	//paramsToGPU(kernel->getWidth());
	//CUDA_SAFE_CALL(cudaMemcpy(pixelsGPU, pixels, sizeof(int) * numPixels, cudaMemcpyHostToDevice));

	// calculate
	dim3 threadBlock(parameters->gpuThreads,parameters->gpuThreads);
	int blockCount = (int)(numPixels / (parameters->gpuThreads*parameters->gpuThreads)) + 1;
	reaccumulateGPU<<<blockCount, threadBlock>>>(imageGPU,kernelsGPU,paramsGPU,recalcDistGPU,recalcZDepthGPU,recalcZValueGPU,bloomValuesGPU,outputGPU);

	// get result
	float* resultData = new float[4*numPixels];
	CUDA_SAFE_CALL(cudaMemcpy(resultData, outputGPU, sizeof(float) * 4 * numPixels, cudaMemcpyDeviceToHost));
	return resultData;
}

//-----------------------------------------------------------------
// Summary: executes re-accumulation on the GPU device.
// Arguments: numPixels - number of pixels
//            kernel - kernel mask
//            parameters - DoF parameters
//-----------------------------------------------------------------
extern "C" float* gpuGetBloomValues(int numPixels) {
//-----------------------------------------------------------------
	float* bloomValues = new float[numPixels];
	CUDA_SAFE_CALL(cudaMemcpy(bloomValues, bloomValuesGPU, sizeof(float) * numPixels, cudaMemcpyDeviceToHost));
	return bloomValues;
}
