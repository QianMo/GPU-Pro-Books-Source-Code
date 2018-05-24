#pragma once


#include "types.h"
#include "common.h"


namespace NMath
{
	Plane PlaneFromPointAndNormal(const Vector3& point, const Vector3& normal);
	Plane PlaneFromPoints(const Vector3& point1, const Vector3& point2, const Vector3& point3);
	Vector2 PlaneSize(float distance, float fovY, float aspect);
	void NormalizeIn(Plane& plane);

	//

	inline Plane PlaneFromPointAndNormal(const Vector3& point, const Vector3& normal)
	{
		Plane temp;

		temp.a = normal.x;
		temp.b = normal.y;
		temp.c = normal.z;
		temp.d = -Dot(point, normal);

		return temp;
	}

	inline Plane PlaneFromPoints(const Vector3& point1, const Vector3& point2, const Vector3& point3)
	{
		Vector3 v1 = point2 - point1;
		Vector3 v2 = point3 - point1;
		Vector3 normal = Normalize(Cross(v1, v2));

		Plane temp;

		temp.a = normal.x;
		temp.b = normal.y;
		temp.c = normal.z;
		temp.d = -Dot(point1, normal);

		return temp;
	}

	inline Vector2 PlaneSize(float distance, float fovY, float aspect)
	{
		Vector2 size;

		size.y = 2.0f * distance * Tan(0.5f * fovY);
		size.x = aspect * size.y;

		return size;
	}

	inline void NormalizeIn(Plane& plane)
	{
		float length = Sqrt(plane.a*plane.a + plane.b*plane.b + plane.c*plane.c);

		plane.a /= length;
		plane.b /= length;
		plane.c /= length;
	}
}
