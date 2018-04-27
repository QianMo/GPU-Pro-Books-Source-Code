#include "Precompiled.h"
#include "Operations.h"
#include "Utils.h"

namespace Mod
{
	namespace Math
	{
		void CalculateBiTanSpaceVectors(	const float3& position1, const float3& position2, const float3& position3,
											float u1, float v1, float u2, float v2, float u3, float v3, float3& oTang, float3& oBinorm	)
		{
			float3 side0 = position1 - position2;
			float3 side1 = position3 - position1;

			float3 normal = normalize( cross( side1, side0 ) );
			
			float deltaV0 = v1 - v2;
			float deltaV1 = v3 - v1;

			float3 tangent = deltaV1 * side0 - deltaV0 * side1;

			tangent = normalize( tangent );

			float deltaU0 = u1 - u2;
			float deltaU1 = u3 - u1;

			float3 binormal = deltaU1 * side0 - deltaU0 * side1;

			binormal = normalize( binormal );

			float3 tangentCross = cross( tangent, binormal );

			float dotProd = dot( tangentCross, normal );

			if (dotProd < 0.0f)
			{
				tangent = -tangent;
				binormal = -binormal;
			}

			oTang	= tangent;
			oBinorm = binormal;
		}
	}
}