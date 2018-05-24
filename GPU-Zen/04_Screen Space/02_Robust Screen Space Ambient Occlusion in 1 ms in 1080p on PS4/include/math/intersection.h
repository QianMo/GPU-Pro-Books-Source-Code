#pragma once


#include "types.h"


namespace NMath
{
	bool IntersectionRayPlane(
		const Vector3& rayStart,
		const Vector3& rayDir,
		const Plane& plane,
		Vector3& intersectionPoint,
		float& distance);

	bool IntersectionRayTriangle(
		const Vector3& rayStart,
		const Vector3& rayDir,
		const Vector3& p1,
		const Vector3& p2,
		const Vector3& p3,
		Vector3& intersectionPoint,
		float& distance);

	bool IntersectionRaySphere(
		const Vector3& rayStart,
		const Vector3& rayDir,
		const Vector3& sphereCenter,
		float sphereRadius,
		Vector3& intersectionPoint,
		float& distance);

	//

	inline bool IntersectionRayPlane(
		const Vector3& rayStart,
		const Vector3& rayDir,
		const Plane& plane,
		Vector3& intersectionPoint,
		float& distance)
	{
		distance = -(plane.a*rayStart.x + plane.b*rayStart.y + plane.c*rayStart.z + plane.d) / (plane.a*rayDir.x + plane.b*rayDir.y + plane.c*rayDir.z);
		intersectionPoint = rayStart + distance*rayDir;
		return distance >= 0.0f;
	}

	inline bool IntersectionRayTriangle(
		const Vector3& rayStart,
		const Vector3& rayDir,
		const Vector3& p1,
		const Vector3& p2,
		const Vector3& p3,
		Vector3& intersectionPoint,
		float& distance)
	{
		Vector3 e1 = p2 - p1;
		Vector3 e2 = p3 - p1;
		Vector3 s1 = Cross(rayDir, e2);
		float divisor = Dot(s1, e1);

		if (divisor == 0.0f)
			return false;

		float oneOverDivisor = 1.0f / divisor;

		Vector3 s = rayStart - p1;
		float b1 = Dot(s, s1) * oneOverDivisor;
		if (b1 < 0.0f || b1 > 1.0f)
			return false;

		Vector3 s2 = Cross(s, e1);
		float b2 = Dot(rayDir, s2) * oneOverDivisor;
		if (b2 < 0.0f || b1 + b2 > 1.0f)
			return false;

		distance = Dot(e2, s2) * oneOverDivisor;
		intersectionPoint = rayStart + distance*rayDir;
		return true;
	}

	inline bool IntersectionRaySphere(
		const Vector3& rayStart,
		const Vector3& rayDir,
		const Vector3& sphereCenter,
		float sphereRadius,
		Vector3& intersectionPoint,
		float& distance)
	{
		float a = Dot(rayDir, rayDir);
		float b = Dot(rayDir, 2.0f * (rayStart - sphereCenter));
		float c = Dot(sphereCenter, sphereCenter) + Dot(rayStart, rayStart) - 2.0f*Dot(rayStart, sphereCenter) - sphereRadius*sphereRadius;
		float delta = b*b - 4.0f*a*c;

		if (delta < 0)
			return false;

		delta = Sqrt(delta);
		float t1 = -0.5f * (b + delta) / a;
		float t2 = -0.5f * (b - delta) / a;

		if (t1 > t2)
			Swap(t1, t2);

		if (t1 > 0.0f)
		{
			distance = Sqrt(a) * t1;
			intersectionPoint = rayStart + t1*rayDir;
			return true;
		}
		else if (t2 > 0.0f)
		{
			distance = Sqrt(a) * t2;
			intersectionPoint = rayStart + t2*rayDir;
			return true;
		}
		else
		{
			return false;
		}
	}
}
