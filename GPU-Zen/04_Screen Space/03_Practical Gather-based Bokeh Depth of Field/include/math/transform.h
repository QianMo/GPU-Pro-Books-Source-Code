#pragma once


#include "types.h"
#include "vector.h"
#include "quaternion.h"


namespace NMath
{
	Vector2 Transform(const Vector2& v, const Matrix& m);
	Vector3 Transform(const Vector3& v, const Matrix& m);
	Vector4 Transform(const Vector4& v, const Matrix& m);
	Vector2 Transform(const Matrix& m, const Vector2& v);
	Vector3 Transform(const Matrix& m, const Vector3& v);
	Vector4 Transform(const Matrix& m, const Vector4& v);
	Vector2 operator * (const Vector2& v, const Matrix& m);
	Vector3 operator * (const Vector3& v, const Matrix& m);
	Vector4 operator * (const Vector4& v, const Matrix& m);
	Vector2 operator * (const Matrix& m, const Vector2& v);
	Vector3 operator * (const Matrix& m, const Vector3& v);
	Vector4 operator * (const Matrix& m, const Vector4& v);

	Vector3 TransformPoint(const Vector3& p, const Quaternion& q);
	Vector3 TransformPoint(const Vector3& p, const Matrix& m);

	Plane Transform(const Plane& plane, Matrix matrix);

	//

	inline Vector2 Transform(const Vector2& v, const Matrix& m)
	{
		Vector2 temp;

		temp.x = v.x*m.m[0][0] + v.y*m.m[1][0];
		temp.y = v.x*m.m[0][1] + v.y*m.m[1][1];

		return temp;
	}

	inline Vector3 Transform(const Vector3& v, const Matrix& m)
	{
		Vector3 temp;

		temp.x = v.x*m.m[0][0] + v.y*m.m[1][0] + v.z*m.m[2][0];
		temp.y = v.x*m.m[0][1] + v.y*m.m[1][1] + v.z*m.m[2][1];
		temp.z = v.x*m.m[0][2] + v.y*m.m[1][2] + v.z*m.m[2][2];

		return temp;
	}

	inline Vector4 Transform(const Vector4& v, const Matrix& m)
	{
		Vector4 temp;

		temp.x = v.x*m.m[0][0] + v.y*m.m[1][0] + v.z*m.m[2][0] + v.w*m.m[3][0];
		temp.y = v.x*m.m[0][1] + v.y*m.m[1][1] + v.z*m.m[2][1] + v.w*m.m[3][1];
		temp.z = v.x*m.m[0][2] + v.y*m.m[1][2] + v.z*m.m[2][2] + v.w*m.m[3][2];
		temp.w = v.x*m.m[0][3] + v.y*m.m[1][3] + v.z*m.m[2][3] + v.w*m.m[3][3];

		return temp;
	}

	inline Vector2 Transform(const Matrix& m, const Vector2& v)
	{
		return Transform(v, m);
	}

	inline Vector3 Transform(const Matrix& m, const Vector3& v)
	{
		return Transform(v, m);
	}

	inline Vector4 Transform(const Matrix& m, const Vector4& v)
	{
		return Transform(v, m);
	}

	inline Vector2 operator * (const Vector2& v, const Matrix& m)
	{
		return Transform(v, m);
	}

	inline Vector3 operator * (const Vector3& v, const Matrix& m)
	{
		return Transform(v, m);
	}

	inline Vector4 operator * (const Vector4& v, const Matrix& m)
	{
		return Transform(v, m);
	}

	inline Vector2 operator * (const Matrix& m, const Vector2& v)
	{
		return Transform(m, v);
	}

	inline Vector3 operator * (const Matrix& m, const Vector3& v)
	{
		return Transform(m, v);
	}

	inline Vector4 operator * (const Matrix& m, const Vector4& v)
	{
		return Transform(m, v);
	}

	inline Vector3 TransformPoint(const Vector3& p, const Quaternion& q)
	{
		Quaternion temp = (Conjugate(q)) * (QuaternionCustom(p.x, p.y, p.z, 0.0f)) * q;
		return VectorCustom(temp.x, temp.y, temp.z);
	}

	inline Vector3 TransformPoint(const Vector3& p, const Matrix& m)
	{
		Vector4 temp = VectorCustom(p.x, p.y, p.z, 1.0f);
		temp = Transform(temp, m);
		DivideByWIn(temp);
		return VectorCustom(temp.x, temp.y, temp.z);
	}

	inline Plane Transform(const Plane& plane, Matrix matrix)
	{
		Vector4 v = VectorCustom(plane.a, plane.b, plane.c, plane.d);
		Invert(matrix);
		Transpose(matrix);
		v = v * matrix;

		Plane temp;

		temp.a = v.x;
		temp.b = v.y;
		temp.c = v.z;
		temp.d = v.w;

		NormalizeIn(temp);

		return temp;
	}
}
