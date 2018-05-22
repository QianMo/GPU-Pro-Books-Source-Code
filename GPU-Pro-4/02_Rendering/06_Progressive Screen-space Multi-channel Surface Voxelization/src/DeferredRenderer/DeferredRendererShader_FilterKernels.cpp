#include "shaders/DeferredRendererShader_FilterKernels.h"


char DRShader_Kernels::DRSH_Gauss_Samples[] = "\n\
float kernel[32]; \n\
void initGaussKernel () \n\
{ \n\
	kernel[ 0] =  0.5000; kernel[ 1] =  0.0000; \n\
	kernel[ 2] =  0.3536; kernel[ 3] =  0.3536; \n\
	kernel[ 4] =  0.0000; kernel[ 5] =  0.5000; \n\
	kernel[ 6] = -0.3536; kernel[ 7] =  0.3536; \n\
	kernel[ 8] = -0.5000; kernel[ 9] =  0.0000; \n\
	kernel[10] = -0.3536; kernel[11] = -0.3536; \n\
	kernel[12] = -0.0000; kernel[13] = -0.5000; \n\
	kernel[14] =  0.3536; kernel[15] = -0.3536; \n\
	kernel[16] =  0.9808; kernel[17] =  0.1951; \n\
	kernel[18] =  0.5556; kernel[19] =  0.8315; \n\
	kernel[20] = -0.1951; kernel[21] =  0.9808; \n\
	kernel[22] = -0.8315; kernel[23] =  0.5556; \n\
	kernel[24] = -0.9808; kernel[25] = -0.1951; \n\
	kernel[26] = -0.5556; kernel[27] = -0.8315; \n\
	kernel[28] =  0.1951; kernel[29] = -0.9808; \n\
	kernel[30] =  0.8315; kernel[31] = -0.5556; \n\
}";

