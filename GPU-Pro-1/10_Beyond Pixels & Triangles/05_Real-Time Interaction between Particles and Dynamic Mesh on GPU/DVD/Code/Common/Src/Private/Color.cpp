#include "Precompiled.h"

#include "Math/Src/Types.h"
#include "Math/Src/Operations.h"

#include "Color.h"

namespace Mod
{

	Color ToColor( const Math::float4& colr )
	{
		Color result;

		Math::float4 colrSat = saturate( colr );

		result.r = UINT8( colrSat.x * 255 );
		result.g = UINT8( colrSat.g * 255 );
		result.b = UINT8( colrSat.b * 255 );
		result.a = UINT8( colrSat.a * 255 );

		return result;
	}

	//------------------------------------------------------------------------

	Color ToColor( const Math::float3& colr )
	{
		Color result;

		Math::float3 colrSat = saturate( colr );

		result.r = UINT8( colrSat.x * 255 );
		result.g = UINT8( colrSat.g * 255 );
		result.b = UINT8( colrSat.b * 255 );
		result.a = 255;

		return result;
	}

	//------------------------------------------------------------------------

	Math::float4 ToFloat4( Color colr )
	{
		Math::float4 res;

		res.x = colr.r / 255.f;
		res.y = colr.g / 255.f;
		res.z = colr.b / 255.f;
		res.w = colr.a / 255.f;

		return res;
	}
}