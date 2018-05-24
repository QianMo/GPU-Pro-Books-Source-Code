#pragma once


#include "types.h"
#include "vector.h"


namespace NMath
{
	float Distance(const Vector2& v1, const Vector2& v2);
	float Distance(const Vector3& v1, const Vector3& v2);
	float DistanceSquared(const Vector2& v1, const Vector2& v2);
	float DistanceSquared(const Vector3& v1, const Vector3& v2);
	float DistanceSigned(const Plane& plane, const Vector3& point);

	//

	inline float Distance(const Vector2& v1, const Vector2& v2)
	{
		return Length(Sub(v1, v2));
	}

	inline float Distance(const Vector3& v1, const Vector3& v2)
	{
		return Length(Sub(v1, v2));
	}

	inline float DistanceSquared(const Vector2& v1, const Vector2& v2)
	{
		return LengthSquared(Sub(v1, v2));
	}

	inline float DistanceSquared(const Vector3& v1, const Vector3& v2)
	{
		return LengthSquared(Sub(v1, v2));
	}

	inline float DistanceSigned(const Plane& plane, const Vector3& point)
	{
		return plane.a*point.x + plane.b*point.y + plane.c*point.z + plane.d;
	}
}
