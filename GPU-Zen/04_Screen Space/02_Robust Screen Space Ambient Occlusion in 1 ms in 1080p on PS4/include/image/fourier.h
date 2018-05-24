#pragma once


#include <essentials/types.h>


using namespace NEssentials;


namespace NImage
{
	float* DiscreteFourierTransform(uint8* input, uint16 width, uint16 height); // input must be R8, output is RG32
	float* InverseDiscreteFourierTransform(float* input, uint16 width, uint16 height);
	float* DiscreteFourierTransform_Separable(uint8* input, uint16 width, uint16 height); // input must be R8, output is RG32
	float* InverseDiscreteFourierTransform_Separable(float* input, uint16 width, uint16 height);
}
