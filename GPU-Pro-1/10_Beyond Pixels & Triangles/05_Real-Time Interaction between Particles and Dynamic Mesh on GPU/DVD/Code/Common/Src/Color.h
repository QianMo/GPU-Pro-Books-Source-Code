#ifndef COMMON_COLOR_H_INCLUDED
#define COMMON_COLOR_H_INCLUDED

#include "Math/Src/Forw.h"

namespace Mod
{

	struct Color
	{
		union
		{
			struct
			{
				UINT8 r;
				UINT8 g;
				UINT8 b;
				UINT8 a;
			};

			UINT32 value;
		};
	};

	Color			ToColor( const Math::float4& colr );
	Color			ToColor( const Math::float3& colr );
	Math::float4	ToFloat4( Color colr );
}

#endif