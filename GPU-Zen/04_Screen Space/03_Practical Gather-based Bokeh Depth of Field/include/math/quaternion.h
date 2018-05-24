#pragma once


#include "types.h"
#include "common.h"
#include "vector.h"


namespace NMath
{
	Quaternion QuaternionCustom(float x, float y, float z, float w);
	Quaternion QuaternionCopy(const Quaternion& q);

	Quaternion Mul(const Quaternion& q1, const Quaternion& q2);
	void MulIn(Quaternion& q1, const Quaternion& q2);
	Quaternion Conjugate(const Quaternion& q);
	float Dot(const Quaternion& q1, const Quaternion& q2);
	float LengthSquared(const Quaternion& q);
	float Length(const Quaternion& q);
	Quaternion Normalize(const Quaternion& q);
	void NormalizeIn(Quaternion& q);
	Quaternion Pow(const Quaternion& q, float p); // it's simply a multiplication of angle by p
	void PowIn(Quaternion& q, float p);
	float Angle(const Quaternion& q);
	Vector3 Axis(const Quaternion& q);
	Quaternion Slerp(const Quaternion& q1, const Quaternion q2, float t);
	Quaternion operator * (const Quaternion& q1, const Quaternion q2);

	void SetIdentity(Quaternion& q);
	void SetRotate(Quaternion& q, float x, float y, float z, float angle);
	void SetRotate(Quaternion& q, const Vector3& axis, float angle);
	void SetRotateX(Quaternion& q, float angle);
	void SetRotateY(Quaternion& q, float angle);
	void SetRotateZ(Quaternion& q, float angle);

	Quaternion QuaternionIdentity();
	Quaternion QuaternionRotate(float x, float y, float z, float angle);
	Quaternion QuaternionRotate(const Vector3& axis, float angle);
	Quaternion QuaternionRotateX(float angle);
	Quaternion QuaternionRotateY(float angle);
	Quaternion QuaternionRotateZ(float angle);

	//

	inline Quaternion QuaternionCustom(float x, float y, float z, float w)
	{
		Quaternion temp;

		temp.x = x;
		temp.y = y;
		temp.z = z;
		temp.w = w;

		return temp;
	}

	inline Quaternion QuaternionCopy(const Quaternion& q)
	{
		return QuaternionCustom(q.x, q.y, q.z, q.w);
	}

	inline Quaternion Mul(const Quaternion& q1, const Quaternion& q2)
	{
		Quaternion temp;

		temp.x = q1.w*q2.x + q1.x*q2.w + q1.z*q2.y - q1.y*q2.z;
		temp.y = q1.w*q2.y + q1.y*q2.w + q1.x*q2.z - q1.z*q2.x;
		temp.z = q1.w*q2.z + q1.z*q2.w + q1.y*q2.x - q1.x*q2.y;
		temp.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z;

		return temp;
	}

	inline void MulIn(Quaternion& q1, const Quaternion& q2)
	{
		q1 = Mul(q1, q2);
	}

	inline Quaternion Conjugate(const Quaternion& q)
	{
		Quaternion temp;

		temp.x = -q.x;
		temp.y = -q.y;
		temp.z = -q.z;
		temp.w = q.w;

		return temp;
	}

	inline float Dot(const Quaternion& q1, const Quaternion& q2)
	{
		return q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w;
	}

	inline float LengthSquared(const Quaternion& q)
	{
		return Dot(q, q);
	}

	inline float Length(const Quaternion& q)
	{
		return Sqrt(LengthSquared(q));
	}

	inline Quaternion Normalize(const Quaternion& q)
	{
		Quaternion temp = QuaternionCopy(q);
		NormalizeIn(temp);
		return temp;
	}

	inline void NormalizeIn(Quaternion& q)
	{
		float oneOverLength = 1.0f / Length(q);

		q.x *= oneOverLength;
		q.y *= oneOverLength;
		q.z *= oneOverLength;
		q.w *= oneOverLength;
	}

	inline Quaternion Pow(const Quaternion& q, float p)
	{
		float angleOver2 = ACos(q.w);
		float newAngleOver2 = angleOver2 * p;
		float ratio = Sin(newAngleOver2) / Sin(angleOver2);

		Quaternion temp;

		temp.x = q.x * ratio;
		temp.y = q.y * ratio;
		temp.z = q.z * ratio;
		temp.w = Cos(newAngleOver2);

		return temp;
	}

	inline void PowIn(Quaternion& q, float p)
	{
		q = Pow(q, p);
	}

	inline float Angle(const Quaternion& q)
	{
		return 2.0f * ACos_Clamped(q.w);
	}

	inline Vector3 Axis(const Quaternion& q)
	{
		float squaredSinAngleOver2 = 1.0f - q.w*q.w;

		if (squaredSinAngleOver2 <= 0.0f)
		{
			return Vector3EX;
		}
		else
		{
			float oneOverSinAngleOver2 = 1.0f / (Sqrt(squaredSinAngleOver2));
			return VectorCustom(oneOverSinAngleOver2*q.x, oneOverSinAngleOver2*q.y, oneOverSinAngleOver2*q.z);
		}
	}

	inline Quaternion Slerp(const Quaternion& q1, const Quaternion q2, float t)
	{
		float omega = Cos(Dot(q1, q2));

		float s1 = Sin((1.0f - t)*omega);
		float s2 = Sin(t*omega);
		float s3 = Sin(omega);
		float r1 = s1 / s3;
		float r2 = s2 / s3;

		Quaternion temp;

		temp.x = r1*q1.x + r2*q2.x;
		temp.y = r1*q1.y + r2*q2.y;
		temp.z = r1*q1.z + r2*q2.z;
		temp.w = r1*q1.w + r2*q2.w;

		return temp;
	}

	inline Quaternion operator * (const Quaternion& q1, const Quaternion q2)
	{
		return Mul(q1, q2);
	}

	inline void SetIdentity(Quaternion& q)
	{
		q.x = 0.0f;
		q.y = 0.0f;
		q.z = 0.0f;
		q.w = 1.0f;
	}

	inline void SetRotate(Quaternion& q, float x, float y, float z, float angle)
	{
		angle = 0.5f * angle;
		float c = Cos(angle);
		float s = Sin(angle);

		q.x = x * s;
		q.y = y * s;
		q.z = z * s;
		q.w = c;
	}

	inline void SetRotate(Quaternion& q, const Vector3& axis, float angle)
	{
		SetRotate(q, axis.x, axis.y, axis.z, angle);
	}

	inline void SetRotateX(Quaternion& q, float angle)
	{
		SetRotate(q, 1.0f, 0.0f, 0.0f, angle);
	}

	inline void SetRotateY(Quaternion& q, float angle)
	{
		SetRotate(q, 0.0f, 1.0f, 0.0f, angle);
	}


	inline void SetRotateZ(Quaternion& q, float angle)
	{
		SetRotate(q, 0.0f, 0.0f, 1.0f, angle);
	}

	inline Quaternion QuaternionIdentity()
	{
		Quaternion temp;
		SetIdentity(temp);
		return temp;
	}

	inline Quaternion QuaternionRotate(float x, float y, float z, float angle)
	{
		Quaternion temp;
		SetRotate(temp, x, y, z, angle);
		return temp;
	}

	inline Quaternion QuaternionRotate(const Vector3& axis, float angle)
	{
		Quaternion temp;
		SetRotate(temp, axis, angle);
		return temp;
	}

	inline Quaternion QuaternionRotateX(float angle)
	{
		Quaternion temp;
		SetRotateX(temp, angle);
		return temp;
	}

	inline Quaternion QuaternionRotateY(float angle)
	{
		Quaternion temp;
		SetRotateY(temp, angle);
		return temp;
	}

	inline Quaternion QuaternionRotateZ(float angle)
	{
		Quaternion temp;
		SetRotateZ(temp, angle);
		return temp;
	}
}
