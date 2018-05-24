#pragma once


#include "constants.h"
#include "common.h"
#include "matrix.h"
#include "quaternion.h"


namespace NMath
{
	float DegToRad(float degrees);
	float RadToDeg(float radians);

	Vector3 SphericalToCartesian(const Spherical& spherical);
	Spherical CartesianToSpherical(const Vector3& cartesian);

	Vector3 QuaternionToEulerAngles(const Quaternion& q);
	Matrix QuaternionToMatrix(const Quaternion& q);

	//

	inline float DegToRad(float degrees)
	{
		return (degrees * (Pi / 180.0f));
	}

	inline float RadToDeg(float radians)
	{
		return (radians * (180.0f / Pi));
	}

	inline Vector3 SphericalToCartesian(const Spherical& spherical)
	{
		Vector3 cartesian;

		cartesian.x = sinf(spherical.theta) * cosf(spherical.phi);
		cartesian.y = sinf(spherical.theta) * sinf(spherical.phi);
		cartesian.z = cosf(spherical.theta);

		return cartesian;
	}

	inline Spherical CartesianToSpherical(const Vector3& cartesian)
	{
		Spherical spherical;

		spherical.theta = acosf(cartesian.z / Length(cartesian));
		spherical.phi = atan2f(cartesian.y, cartesian.x);

		return spherical;
	}

	inline Vector3 QuaternionToEulerAngles(const Quaternion& q)
	{
		Vector3 temp;

		temp.x = ATan2(2.0f*(q.x*q.w + q.y*q.z), 1.0f - 2.0f*(q.x*q.x + q.y*q.y));
		temp.y = ASin(2.0f*(q.y*q.w - q.x*q.z));
		temp.z = ATan2(2.0f*(q.z*q.w + q.x*q.y), 1.0f - 2.0f*(q.y*q.y + q.z*q.z));

		return temp;
	}

	inline Matrix QuaternionToMatrix(const Quaternion& q)
	{
		return MatrixRotate(Axis(q), Angle(q));
	}
}
