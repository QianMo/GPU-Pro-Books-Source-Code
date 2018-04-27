#ifndef MATH_UTILS_H_INCLUDED
#define MATH_UTILS_H_INCLUDED

#include "Forw.h"

namespace Mod
{
	namespace Math
	{
		void CalculateBiTanSpaceVectors( const float3& position1, const float3& position2, const float3& position3,	float u1, float v1, float u2, float v2, float u3, float v3, float3& oTang, float3& oBinorm);
	}
}

#endif