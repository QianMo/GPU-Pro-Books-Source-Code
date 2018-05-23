#ifndef MATH_PLANE_HPP
#define MATH_PLANE_HPP

#include <cmath>

#include <math/vector.hpp>



class Matrix;



class Plane
{
public:
	Plane() {}

	Plane(float a, float b, float c, float d)
	{
		this->a = a;
		this->b = b;
		this->c = c;
		this->d = d;
	}

	Plane(const Vector3& normal, float d = 0.0f)
	{
		this->a = normal.x;
		this->b = normal.y;
		this->c = normal.z;
		this->d = d;
	}

	Plane(const Vector3& point, const Vector3& normal)
	{
		a = normal.x;
		b = normal.y;
		c = normal.z;
		d = -(point % normal);
	}

	Plane(const Vector3& point1, const Vector3& point2, const Vector3& point3)
	{
		Vector3 v1 = point2 - point1;
		Vector3 v2 = point3 - point1;
		Vector3 normal = (v1 ^ v2).GetNormalized();

		a = normal.x;
		b = normal.y;
		c = normal.z;
		d = -(point1 % normal);
	}

	void Normalize()
	{
		float length = sqrtf(a*a + b*b + c*c);

		a /= length;
		b /= length;
		c /= length;
	}

	float GetSignedDistanceFromPoint(const Vector3& point) const
	{
		return a*point.x + b*point.y + c*point.z + d;
	}

	float GetSignedDistanceFromPoint(const Vector4& point) const
	{
		return a*point.x + b*point.y + c*point.z + d;
	}

	void Transform(Matrix transform);

public:
	float a, b, c;
	float d;
};



typedef Plane plane;



#endif
