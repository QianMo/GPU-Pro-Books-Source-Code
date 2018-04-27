#ifndef MATH_FRUSTUM_H_INCLUDED
#define MATH_FRUSTUM_H_INCLUDED

#include "Types.h"

namespace Mod
{
	namespace Math
	{
		struct Frustum
		{
			enum
			{
				LEFT,
				RIGHT,
				BOTTOM,
				TOP,
				NEAR,
				FAR,
				NUM_PLANES
			};

			float4 planes[ NUM_PLANES ];
		};

		struct FrustumPoints
		{
			enum
			{
				NEAR_LEFT_BOTTOM,
				NEAR_RIGHT_BOTTOM,
				NEAR_LEFT_TOP,
				NEAR_RIGHT_TOP,

				FAR_LEFT_BOTTOM,
				FAR_RIGHT_BOTTOM,
				FAR_LEFT_TOP,
				FAR_RIGHT_TOP,

				NUM_POINTS
			};

			float4 points[ NUM_POINTS ];
		};
	}
}

#endif